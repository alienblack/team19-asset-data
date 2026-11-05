"""Imputation methods, one class per model. Every model sees measured training data only."""

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

from team19_asset_data.transformers.miceforest_compat import CorrectedKernel

from .columns import CATEGORY_TARGET

NUMERIC_METHODS = (
    "Median", "KNN imputer", "IterativeImputer", "Random Forest",
    "HistGradientBoosting", "CatBoost", "miceforest direct",
)
CATEGORY_METHODS = (
    "Mode", "KNN", "Random Forest", "HistGradientBoosting", "CatBoost", "miceforest direct",
)
CLASS_COUNT = 4
THREADS = 1  # parallelism is across tasks, so each model stays single-threaded


class Features:
    """Fold-fitted predictor matrix.

    Missing predictors stay NaN: every method here handles them natively, which
    avoids inventing values. Masks are random, so missing-value indicators would
    add columns without information. Scaling is applied only for methods that
    compare distances or fit linear coefficients.
    """

    def __init__(self, scale):
        self.scale = scale

    def fit(self, frame):
        self.columns = list(frame.columns)
        self.numeric = [c for c in self.columns if c != CATEGORY_TARGET]
        self.center = frame[self.numeric].mean()
        self.spread = frame[self.numeric].std(ddof=0).replace(0, 1.0)
        self.levels = (
            sorted(frame[CATEGORY_TARGET].dropna().unique()) if CATEGORY_TARGET in frame else []
        )
        return self

    def transform(self, frame):
        if list(frame.columns) != self.columns:
            raise ValueError("Predictor columns changed between fit and transform")
        values = frame[self.numeric]
        if self.scale:
            values = (values - self.center) / self.spread
        parts = [values.to_numpy(dtype=float)]
        if CATEGORY_TARGET in self.columns:
            codes = frame[CATEGORY_TARGET].to_numpy()
            parts.append(np.column_stack([codes == level for level in self.levels]).astype(float))
            parts.append(np.isnan(codes).astype(float)[:, None])
        return np.column_stack(parts)


class SingleTargetModel:
    """Shared steps for models of one target: choose predictors, labels and features."""

    scaled = False

    def __init__(self, target, config, settings, seed, drop=()):
        self.target, self.config, self.seed = target, config, seed
        self.settings, self.drop = dict(settings), tuple(drop)
        self.categorical = target == CATEGORY_TARGET

    def fit(self, measured):
        self.predictors = [c for c in measured if c != self.target and c not in self.drop]
        self.known = measured[self.target].notna().to_numpy()
        labels = measured.loc[self.known, self.target].to_numpy()
        if self.categorical:
            labels = labels.astype(int)
            self.classes_ = np.unique(labels)
        self.features = Features(self.scaled).fit(measured[self.predictors])
        self._fit(self.features.transform(measured[self.predictors]), labels, measured[self.target].to_numpy())
        return self

    def _matrix(self, frame):
        return self.features.transform(frame[self.predictors])

    def predict(self, frame):
        """Point prediction from a row's predictors; the row's own target is never read."""
        if self.categorical:
            return self.classes_[self.predict_proba(frame).argmax(axis=1)].astype(float)
        return self._predict(self._matrix(frame))

    def predict_proba(self, frame):
        return self.model.predict_proba(self._matrix(frame))


class BaselineModel(SingleTargetModel):
    """Median for numeric targets and Mode for model_id, from observed training labels."""

    def _fit(self, matrix, labels, target):
        if self.categorical:
            self.constant = pd.Series(labels).mode().iloc[0]
            self.prior = pd.Series(labels).value_counts(normalize=True).reindex(self.classes_).to_numpy()
        else:
            self.constant = float(np.median(labels))

    def predict(self, frame):
        return np.repeat(float(self.constant), len(frame))

    def predict_proba(self, frame):
        return np.tile(self.prior, (len(frame), 1))


class KNNModel(SingleTargetModel):
    """Nearest neighbours on scaled columns, comparing only the cells two rows both have."""

    scaled = True

    def _fit(self, matrix, labels, target):
        if self.categorical:
            self.model = KNeighborsClassifier(**self.settings, metric="nan_euclidean").fit(matrix[self.known], labels)
            return
        self.center, self.spread = float(np.mean(labels)), float(np.std(labels)) or 1.0
        # The target joins the matrix with its own gaps, so no pseudo-labels are created.
        stacked = np.column_stack((matrix, (target - self.center) / self.spread))
        self.model = KNNImputer(**self.settings, keep_empty_features=True).fit(stacked)

    def _predict(self, matrix):
        stacked = np.column_stack((matrix, np.full(len(matrix), np.nan)))
        return self.model.transform(stacked)[:, -1] * self.spread + self.center


class EstimatorModel(SingleTargetModel):
    """A supervised estimator trained on the rows where the target was observed."""

    def _fit(self, matrix, labels, target):
        self.model = self._estimator().fit(matrix[self.known], labels)

    def _predict(self, matrix):
        return self.model.predict(matrix)


class RandomForestModel(EstimatorModel):
    """Averages many deep trees grown on bootstrap samples."""

    def _estimator(self):
        kind = RandomForestClassifier if self.categorical else RandomForestRegressor
        return kind(**self.settings, random_state=self.seed, n_jobs=THREADS)


class HistGradientBoostingModel(EstimatorModel):
    """Adds shallow trees one at a time, each correcting the remaining error."""

    def _estimator(self):
        kind = HistGradientBoostingClassifier if self.categorical else HistGradientBoostingRegressor
        return kind(**self.settings, random_state=self.seed)


class CatBoostModel(EstimatorModel):
    """Gradient boosting on symmetric trees, where one split rule is shared across each level."""

    def _estimator(self):
        kind = CatBoostClassifier if self.categorical else CatBoostRegressor
        return kind(**self.settings, random_seed=self.seed, thread_count=THREADS,
                    verbose=False, allow_writing_files=False)


class MiceforestModel(SingleTargetModel):
    """LightGBM through miceforest: a direct prediction, or donor-matched draws when ``draws`` is set."""

    def __init__(self, target, config, settings, seed, drop=(), draws=None):
        super().__init__(target, config, settings, seed, drop)
        self.draws = draws

    def _fit(self, matrix, labels, target):
        settings = dict(self.settings)
        rounds = settings.pop("rounds")
        self.names = [f"f{i}" for i in range(matrix.shape[1])]
        data = pd.DataFrame(matrix[self.known], columns=self.names)
        data["target"] = (
            pd.Categorical(labels.astype(str), categories=self.classes_.astype(str))
            if self.categorical else labels.astype(float)
        )
        self.kernel = CorrectedKernel(
            data, num_datasets=self.draws or 1, variable_schema={"target": self.names},
            mean_match_candidates=self.config.pmm_candidates if self.draws else 0,
            random_state=self.seed, save_all_iterations_data=True,
        )
        self.kernel.mice(self.config.mice_iterations, num_iterations=rounds,
                         num_threads=THREADS, verbosity=-1, **settings)

    def _frame(self, matrix):
        return pd.DataFrame(matrix, columns=self.names)

    def _predict(self, matrix):
        return self.kernel.get_model("target", 0).predict(self._frame(matrix), num_threads=THREADS)

    def predict_proba(self, frame):
        booster = self.kernel.get_model("target", 0)
        result = np.asarray(booster.predict(self._frame(self._matrix(frame)), num_threads=THREADS))
        return np.column_stack((1 - result, result)) if result.ndim == 1 else result

    def draw(self, frame):
        """Donor-matched draws: one column per completed dataset."""
        data = self._frame(self._matrix(frame))
        data["target"] = np.nan
        result = self.kernel.impute_new_data(
            data, iterations=self.config.mice_iterations, random_state=self.seed + 1000
        )
        return np.column_stack([
            result.complete_data(dataset=d).target.to_numpy() for d in result.datasets
        ])


MODEL_CLASSES = {
    "Median": BaselineModel,
    "Mode": BaselineModel,
    "KNN imputer": KNNModel,
    "KNN": KNNModel,
    "Random Forest": RandomForestModel,
    "HistGradientBoosting": HistGradientBoostingModel,
    "CatBoost": CatBoostModel,
    "miceforest direct": MiceforestModel,
}


def make_model(method, target, config, settings, seed, drop=()):
    return MODEL_CLASSES[method](target, config, settings, seed, drop)


class ChainedLinearImputer:
    """IterativeImputer over all columns at once.

    Chained equations update every incomplete column in turn, so one fit imputes
    every numeric column. This avoids repeating the same work per target and
    lets the imputer run until it converges.
    """

    def __init__(self, settings, seed, drop=(), max_iter=None):
        self.settings, self.seed, self.drop = dict(settings), seed, tuple(drop)
        if max_iter is not None:
            self.settings["max_iter"] = max_iter

    def fit(self, measured):
        self.columns = [c for c in measured if c not in self.drop]
        self.features = Features(True).fit(measured[self.columns])
        self.model = IterativeImputer(**self.settings, random_state=self.seed, keep_empty_features=True)
        self.model.fit(self.features.transform(measured[self.columns]))
        self.n_iter_ = self.model.n_iter_
        return self

    def predict(self, frame):
        """Filled numeric columns in their original units, aligned to ``frame``."""
        filled = self.model.transform(self.features.transform(frame[self.columns]))
        numeric = self.features.numeric
        values = filled[:, : len(numeric)] * self.features.spread.to_numpy() + self.features.center.to_numpy()
        return pd.DataFrame(values, index=frame.index, columns=numeric)


class JointModel:
    """All incomplete columns imputed together by chained equations with donor matching."""

    def __init__(self, config, settings, seed):
        self.config, self.seed = config, seed
        self.settings = dict(settings)

    def _prepare(self, frame):
        data = frame[self.columns].reset_index(drop=True).copy()
        data[CATEGORY_TARGET] = pd.Categorical(
            data[CATEGORY_TARGET].map(lambda v: str(int(v)) if pd.notna(v) else np.nan),
            categories=self.levels,
        )
        return data

    def fit(self, measured):
        self.columns = list(measured.columns)
        self.levels = sorted(str(int(v)) for v in measured[CATEGORY_TARGET].dropna().unique())
        incomplete = [c for c in self.columns if measured[c].isna().any()]
        settings = dict(self.settings)
        rounds = settings.pop("rounds")
        self.kernel = CorrectedKernel(
            self._prepare(measured), num_datasets=self.config.pmm_draws,
            variable_schema={c: [p for p in self.columns if p != c] for c in incomplete},
            mean_match_candidates=self.config.pmm_candidates,
            random_state=self.seed, save_all_iterations_data=True,
        )
        self.kernel.mice(self.config.mice_iterations, num_iterations=rounds,
                         num_threads=THREADS, verbosity=-1, **settings)
        return self

    def predict(self, frame):
        """One completed copy of ``frame`` per draw, observed cells unchanged."""
        result = self.kernel.impute_new_data(
            self._prepare(frame), iterations=self.config.mice_iterations, random_state=self.seed + 1000
        )
        completed = []
        for dataset in result.datasets:
            filled = result.complete_data(dataset=dataset)
            filled[CATEGORY_TARGET] = filled[CATEGORY_TARGET].astype(object).astype(float)
            filled.index = frame.index
            completed.append(filled.where(frame.isna(), frame))
        return completed


class CalibratedForest:
    """Random Forest for model_id with sigmoid calibration on inner folds of measured labels."""

    def __init__(self, settings, seed, drop=()):
        self.settings, self.seed, self.drop = dict(settings), seed, tuple(drop)

    def fit(self, measured):
        self.predictors = [c for c in measured if c != CATEGORY_TARGET and c not in self.drop]
        self.features = Features(False).fit(measured[self.predictors])
        known = measured[CATEGORY_TARGET].notna().to_numpy()
        matrix = self.features.transform(measured[self.predictors])[known]
        labels = measured.loc[known, CATEGORY_TARGET].to_numpy().astype(int)
        forest = RandomForestClassifier(**self.settings, n_jobs=THREADS, random_state=self.seed)
        self.model = CalibratedClassifierCV(
            forest, method="sigmoid", cv=StratifiedKFold(3, shuffle=True, random_state=self.seed)
        ).fit(matrix, labels)
        self.classes_ = self.model.classes_
        return self

    def predict_proba(self, frame):
        return self.model.predict_proba(self.features.transform(frame[self.predictors]))
