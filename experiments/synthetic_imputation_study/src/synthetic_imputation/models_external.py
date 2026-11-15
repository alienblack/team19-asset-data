"""Models from the other two Team 19 branches, adapted to the study's task interface.

Each class has the interface of ``ChainedColumns``: it is built from settings and a seed,
fitted on the measured training rows, and returns numeric estimates for every row with
``model_id`` class probabilities. The model code itself is in ``vendor`` and is unchanged.

The study gives no model any relationship between columns. Subteam A's code has three
optional parts that encode the generator: a gas-decay correction, an installation day
estimated from neighbouring coordinates, and exact formulas applied before the model. All
three are switched off here by describing the table as a generic asset table
(``transformer=False``), which also stops their code from blanking the placebo column.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .columns import CATEGORY_TARGET
from .vendor.pipeline import mice_pmm_logistic as mice
from .vendor.pipeline.comparison_models import ChainedConditionalImputer, MedianMode, ModelSpec
from .vendor.pipeline.predictive_missforest import PredictiveMissForest, PredictiveMissForestConfig
from .vendor.subteam_a.data import Schema
from .vendor.subteam_a.features import Features
from .vendor.subteam_a.mixture import MaskedMixture
from .vendor.subteam_a.models import Autoencoder

CLASS_LEVELS = (0, 1, 2, 3)


class _ExternalBase:
    """Shared steps: drop a predictor block, keep the column order, return the study's output form."""

    def __init__(self, settings, seed, drop=()):
        self.settings, self.seed, self.drop = dict(settings), seed, tuple(drop)
        self.classes_ = np.array(CLASS_LEVELS)

    def _frame(self, frame):
        return frame.drop(columns=list(self.drop))

    def _output(self, frame, completed, probabilities):
        numeric = pd.DataFrame(completed, index=frame.index, columns=self.columns).drop(columns=CATEGORY_TARGET)
        return numeric.astype(float), np.asarray(probabilities, dtype=float)


class _SubteamA(_ExternalBase):
    """Subteam A's models on the measured matrix, with their own train-only features."""

    def _model(self):
        raise NotImplementedError

    def fit(self, measured):
        frame = self._frame(measured)
        self.columns = list(frame.columns)
        self.category = self.columns.index(CATEGORY_TARGET)
        self.schema = Schema(tuple(self.columns), {self.category: CLASS_LEVELS}, transformer=False)
        matrix = frame.to_numpy(dtype=float)
        self.features = Features(self.schema, domain=False).fit(matrix)
        self.model = self._model().fit(matrix, self.features)
        return self

    def predict(self, frame):
        frame = self._frame(frame)
        prediction = self.model.predict(frame.to_numpy(dtype=float))
        return self._output(frame, prediction.point, prediction.proba[self.category])


class GaussianEM(_SubteamA):
    """Subteam A's Gaussian mixture: shared-covariance components fitted by EM on incomplete rows."""

    def _model(self):
        return MaskedMixture(self.settings, self.seed)


class MaskedMLP(_SubteamA):
    """Subteam A's masked MLP: learns to rebuild observed cells that are hidden during training."""

    def _model(self):
        return Autoencoder(self.settings, self.seed)


class LightGBMChained(_ExternalBase):
    """The pipeline's chained LightGBM: median start, then fixed passes of stored per-column models."""

    def fit(self, measured):
        frame = self._frame(measured)
        self.columns = list(frame.columns)
        spec = ModelSpec(tuple(c for c in self.columns if c != CATEGORY_TARGET), (CATEGORY_TARGET,), self.seed)
        self.model = ChainedConditionalImputer(spec, family="lightgbm", **self.settings).fit(frame)
        return self

    def predict(self, frame):
        frame = self._frame(frame)
        # The vendored transform, with the class probabilities of the last model_id model kept.
        model = self.model
        missing = frame[list(model.spec.columns)].isna()
        state = MedianMode.transform(model, frame)
        probabilities = np.zeros((len(frame), len(CLASS_LEVELS)))
        for target, estimator in model.models_:
            needs_fill = missing[target]
            if estimator is not None and needs_fill.any():
                x = model._inputs(state, target)
                state.loc[needs_fill, target] = estimator.predict(x.loc[needs_fill])
                if target == CATEGORY_TARGET:
                    probabilities = np.zeros((len(frame), len(CLASS_LEVELS)))
                    shares = estimator.predict_proba(x)
                    probabilities[:, np.asarray(estimator.classes_, dtype=float).astype(int)] = shares
        observed = frame[CATEGORY_TARGET].notna().to_numpy()
        probabilities[observed] = np.eye(len(CLASS_LEVELS))[frame[CATEGORY_TARGET].to_numpy()[observed].astype(int)]
        return self._output(frame, state[self.columns].to_numpy(dtype=float), probabilities)


class _ProbabilityForest(PredictiveMissForest):
    """Keeps the full class probabilities of the final model_id forest, which the vendored code scores."""

    def _score_rows(self, pipeline, X_rows, predictions, target, positions, index_values):
        if target == CATEGORY_TARGET:
            forest = pipeline.named_steps["forest"]
            labels = np.array([float(self._labels[str(c)]) for c in forest.classes_]).astype(int)
            shares = np.zeros((len(X_rows), len(CLASS_LEVELS)))
            shares[:, labels] = pipeline.predict_proba(X_rows)
            self.captured_ = (np.asarray(positions), shares)
        return super()._score_rows(pipeline, X_rows, predictions, target, positions, index_values)


class PredictiveMissForestModel(_ExternalBase):
    """The pipeline's Predictive MissForest: stored per-pass forests replayed on new rows."""

    def fit(self, measured):
        frame = self._frame(measured)
        self.columns = list(frame.columns)
        data = frame.astype({CATEGORY_TARGET: object})
        config = PredictiveMissForestConfig(**self.settings, random_state=self.seed, n_jobs=1)
        self.model = _ProbabilityForest([c for c in self.columns if c != CATEGORY_TARGET], [CATEGORY_TARGET], config)
        self.model.fit(data)
        self.model._labels = {str(v): v for v in frame[CATEGORY_TARGET].dropna().unique()}
        return self

    def predict(self, frame):
        frame = self._frame(frame)
        data = frame.astype({CATEGORY_TARGET: object})
        self.model.captured_ = None
        result = self.model.transform_detailed(data, with_scores=True)
        completed = result.imputed[self.columns].astype(float)
        probabilities = np.zeros((len(frame), len(CLASS_LEVELS)))
        observed = frame[CATEGORY_TARGET].notna().to_numpy()
        probabilities[observed] = np.eye(len(CLASS_LEVELS))[frame[CATEGORY_TARGET].to_numpy()[observed].astype(int)]
        if self.model.captured_ is not None:
            positions, shares = self.model.captured_
            probabilities[positions] = shares
        return self._output(frame, completed.to_numpy(), probabilities)


class MICEPMM(_ExternalBase):
    """The team's MICE with predictive mean matching and bootstrapped logistic regression.

    The shared code completes a whole table, so its conditional models see the visible cells of
    the rows being scored. Here every conditional model, scaler and encoder is fitted on training
    rows only, and PMM draws donors from training rows only. Test rows give their visible cells as
    predictors and receive draws for their gaps. The vendored ``_pmm`` and ``_categorical`` steps
    do the fitting and drawing unchanged; this class only decides which rows they see.

    The estimate is the mean of the chains for numeric columns, as in the shared code, and the
    share of chains choosing each class for ``model_id``.
    """

    def fit(self, measured):
        self.train = self._frame(measured)
        self.columns = list(self.train.columns)
        settings = dict(self.settings)
        self.engine = mice.MICEPMMLogistic(
            mice.Schema(targets=self.columns, categorical=[CATEGORY_TARGET]),
            n_imputations=settings["chains"], n_iterations=settings["sweeps"], donors=settings["donors"],
            seed=self.seed, ridge=settings["ridge"], logistic_C=settings["logistic_C"],
            logistic_max_iter=settings["logistic_max_iter"], threads=1)
        return self

    def _design(self, work, columns, train):
        categories = [c for c in columns if c == CATEGORY_TARGET]
        numbers = [c for c in columns if c not in categories]
        steps = []
        if numbers:
            steps.append(("numeric", StandardScaler(), numbers))
        if categories:
            steps.append(("category", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False), categories))
        transformer = ColumnTransformer(steps, remainder="drop").fit(work.loc[train, columns])
        return transformer.transform(work[columns])

    def predict(self, frame):
        test = self._frame(frame)
        table = pd.concat([self.train, test], ignore_index=True)
        train = np.r_[np.ones(len(self.train), bool), np.zeros(len(test), bool)]
        original = table.copy()
        original[CATEGORY_TARGET] = original[CATEGORY_TARGET].map(lambda v: str(v) if pd.notna(v) else np.nan).astype(object)
        masks = original.isna()
        targets = sorted([c for c in self.columns if masks[c].any()], key=lambda c: masks[c].sum())
        predictors = {c: self.engine.schema.predictors(table.loc[train], c) for c in targets}
        rows = np.flatnonzero(~train)
        numeric = [c for c in self.columns if c != CATEGORY_TARGET]
        totals = np.zeros((len(rows), len(numeric)))
        votes = np.zeros((len(rows), len(CLASS_LEVELS)))
        with mice.threadpool_limits(limits=1):
            for child in np.random.SeedSequence(self.seed).spawn(self.engine.n_imputations):
                rng = np.random.default_rng(child)
                work = original.copy()
                for c in targets:
                    donors = original.loc[~masks[c].to_numpy() & train, c].to_numpy()
                    work.loc[masks[c], c] = rng.choice(donors, size=int(masks[c].sum()))
                for _ in range(self.engine.n_iterations):
                    for c in targets:
                        fit_rows = ~masks[c].to_numpy() & train
                        fill_rows = masks[c].to_numpy()
                        design = self._design(work, predictors[c], train)
                        stacked = np.vstack([design[fit_rows], design[fill_rows]])
                        observed = np.r_[np.ones(fit_rows.sum(), bool), np.zeros(fill_rows.sum(), bool)]
                        y = original.loc[fit_rows, c].to_numpy()
                        if c == CATEGORY_TARGET:
                            values, _ = self.engine._categorical(stacked, observed, y.astype(str), rng)
                        else:
                            values = self.engine._pmm(stacked, observed, y.astype(float), rng)
                        work.loc[fill_rows, c] = values
                completed = work.iloc[rows]
                totals += completed[numeric].to_numpy(dtype=float)
                labels = completed[CATEGORY_TARGET].astype(float).astype(int).to_numpy()
                votes[np.arange(len(rows)), labels] += 1
        estimates = pd.DataFrame(totals / self.engine.n_imputations, index=test.index, columns=numeric)
        estimates[CATEGORY_TARGET] = np.nan
        return self._output(test, estimates[self.columns].to_numpy(), votes / votes.sum(axis=1, keepdims=True))


EXTERNAL_CLASSES = {
    "Gaussian EM": GaussianEM,
    "Masked MLP": MaskedMLP,
    "LightGBM chained": LightGBMChained,
    "Predictive MissForest": PredictiveMissForestModel,
    "MICE-PMM": MICEPMM,
}
# Where each model comes from, for tables and figures.
SOURCES = {
    "Gaussian EM": "Subteam A", "Masked MLP": "Subteam A",
    "LightGBM chained": "team pipeline", "Predictive MissForest": "team pipeline", "MICE-PMM": "team pipeline",
}
