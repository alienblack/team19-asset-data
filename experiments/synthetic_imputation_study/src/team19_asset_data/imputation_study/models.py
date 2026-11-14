"""Imputation methods, one class per model. Every model sees measured training data only."""

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures

from team19_asset_data.transformers.miceforest_compat import CorrectedKernel

from .columns import CATEGORY_TARGET

NUMERIC_METHODS = (
    "Median", "KNN imputer", "IterativeImputer", "Random Forest",
    "HistGradientBoosting", "CatBoost", "miceforest direct",
)
CATEGORY_METHODS = (
    "Mode", "KNN imputer", "Random Forest", "HistGradientBoosting", "CatBoost", "miceforest direct",
)
CLASS_COUNT = 4
THREADS = 1  # parallelism is across tasks, so each model stays single-threaded
# miceforest defaults to a depth-limited random forest and ignores the learning rate.
# These settings turn that off, so LightGBM runs as gradient boosting.
BOOSTING = {"boosting": "gbdt", "max_depth": -1, "bagging_fraction": 1.0, "bagging_freq": 0,
            "feature_fraction_bynode": 1.0}


class Features:
    """Fold-fitted predictor matrix.

    Missing predictors stay NaN: every method here handles them natively, which
    avoids inventing values. Masks are random, so missing-value indicators would
    add columns without information. Scaling is applied only for methods that
    compare distances or fit linear coefficients. With ``open_category`` a missing
    model_id leaves its one-hot cells empty, so a chained imputer can estimate them.
    """

    def __init__(self, scale, open_category=False):
        self.scale, self.open_category = scale, open_category

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
            hot = np.column_stack([codes == level for level in self.levels]).astype(float)
            if self.open_category:
                hot[np.isnan(codes)] = np.nan
                parts.append(hot)
            else:
                parts += [hot, np.isnan(codes).astype(float)[:, None]]
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


class EstimatorModel(SingleTargetModel):
    """A supervised estimator trained on the rows where the target was observed."""

    def _fit(self, matrix, labels, target):
        self.model = self._estimator().fit(matrix[self.known], labels)

    def _predict(self, matrix):
        return self.model.predict(matrix)


class RandomForestModel(EstimatorModel):
    """Single-target forest, used to explain which columns matter and for calibration."""

    def _estimator(self):
        kind = RandomForestClassifier if self.categorical else RandomForestRegressor
        return kind(**self.settings, random_state=self.seed, n_jobs=THREADS)


class CatBoostModel(EstimatorModel):
    """Single-target CatBoost, used to explain which columns matter."""

    def _estimator(self):
        kind = CatBoostClassifier if self.categorical else CatBoostRegressor
        return kind(**self.settings, random_seed=self.seed, thread_count=THREADS,
                    verbose=False, allow_writing_files=False)


MODEL_CLASSES = {
    "Median": BaselineModel,
    "Mode": BaselineModel,
    "Random Forest": RandomForestModel,
    "CatBoost": CatBoostModel,
}


def make_model(method, target, config, settings, seed, drop=()):
    return MODEL_CLASSES[method](target, config, settings, seed, drop)


class BlockRidge(BaseEstimator, RegressorMixin):
    """One ridge fit on a column's predictors and their second-order terms.

    The two blocks have separate penalties, so straight-line relationships are not
    shrunk by the strong penalty the many second-order terms need. Scaling the
    second-order block by sqrt(linear / curve) lets a single solve apply both.
    """

    def __init__(self, linear_alpha=30.0, curve_alpha=300.0):
        self.linear_alpha, self.curve_alpha = linear_alpha, curve_alpha

    def _terms(self, matrix):
        terms = self.expansion_.transform(matrix)
        terms[:, matrix.shape[1]:] *= (self.linear_alpha / self.curve_alpha) ** 0.5
        return terms

    def fit(self, matrix, labels):
        self.expansion_ = PolynomialFeatures(2, include_bias=False).fit(matrix)
        self.model_ = Ridge(alpha=self.linear_alpha).fit(self._terms(matrix), labels)
        return self

    def predict(self, matrix):
        return self.model_.predict(self._terms(matrix))


class RidgeThenLocal:
    """A block ridge on complete predictors, with a local model fitted to what it leaves.

    Trees and neighbours approximate a sum with many small steps. The ridge takes the
    smooth part, so the local model only needs to learn thresholds and what else remains.
    """

    def __init__(self, local, category_index, linear_alpha, curve_alpha):
        self.local, self.category_index = local, category_index
        self.ridge = BlockRidge(linear_alpha, curve_alpha)

    def _smooth_inputs(self, matrix):
        numeric = (np.delete(matrix, self.category_index, axis=1) - self.center) / self.spread
        codes = matrix[:, self.category_index]
        return np.column_stack([numeric] + [codes == level for level in range(CLASS_COUNT)]).astype(float)

    def fit(self, matrix, labels):
        numeric = np.delete(matrix, self.category_index, axis=1)
        self.center, self.spread = numeric.mean(axis=0), numeric.std(axis=0) + 1e-12
        smooth = self._smooth_inputs(matrix)
        self.ridge.fit(smooth, labels)
        self.local.fit(matrix, labels - self.ridge.predict(smooth))
        return self

    def predict(self, matrix):
        return self.ridge.predict(self._smooth_inputs(matrix)) + self.local.predict(matrix)


class ChainedColumns:
    """A model for every column, each pass using the latest estimates of the others.

    The first pass starts from the gaps themselves, which each method handles natively,
    so no median is invented. Later passes see complete rows, which is what allows a
    ridge to take the smooth part of each column first. model_id is reported by a
    separate classifier on the original gaps: a filled-in estimate looks as certain
    as a measurement, which costs accuracy when judging a category.
    """

    def __init__(self, settings, seed, drop=()):
        self.settings, self.seed, self.drop = dict(settings), seed, tuple(drop)
        self.passes = self.settings.pop("passes")
        self.linear_alpha = self.settings.pop("linear_alpha")
        self.curve_alpha = self.settings.pop("curve_alpha")
        # "none" leaves the local model alone in every pass, to measure what the ridge base adds.
        self.base = self.settings.pop("base", "ridge")

    def _regressor(self, columns, seed):
        raise NotImplementedError

    def _classifier(self, columns, seed):
        raise NotImplementedError

    def _column_model(self, target, columns, step):
        seed = self.seed + step
        if target == CATEGORY_TARGET:
            return self._classifier(columns, seed)
        local = self._regressor(columns, seed)
        if step == 0 or self.base == "none" or CATEGORY_TARGET not in columns:
            return local
        return RidgeThenLocal(local, columns.index(CATEGORY_TARGET), self.linear_alpha, self.curve_alpha)

    def fit(self, measured):
        frame = measured.drop(columns=list(self.drop))
        self.report_columns = [c for c in frame if c != CATEGORY_TARGET]
        known = frame[CATEGORY_TARGET].notna()
        self.reporter = self._classifier(self.report_columns, self.seed).fit(
            frame.loc[known, self.report_columns].to_numpy(float), frame.loc[known, CATEGORY_TARGET].to_numpy().astype(int))
        self.classes_ = self.reporter.classes_
        self.order = list(frame.isna().mean().sort_values(kind="stable").index)
        filled, gaps = frame.copy(), frame.isna()
        self.sequence = []
        for step in range(self.passes):
            for target in self.order:
                columns = [c for c in filled if c != target]
                matrix = filled[columns].to_numpy(float)
                seen = ~gaps[target].to_numpy()
                labels = frame.loc[seen, target].to_numpy()
                labels = labels.astype(int) if target == CATEGORY_TARGET else labels
                model = self._column_model(target, columns, step).fit(matrix[seen], labels)
                self.sequence.append((target, columns, model))
                if not seen.all():
                    filled.loc[gaps[target], target] = model.predict(matrix[~seen])
        return self

    def predict(self, frame):
        """Numeric estimates for every cell of ``frame``, and model_id class probabilities."""
        frame = frame.drop(columns=list(self.drop))
        filled, gaps = frame.copy(), frame.isna()
        estimates = {}
        for target, columns, model in self.sequence:
            values = model.predict(filled[columns].to_numpy(float))
            estimates[target] = values
            if gaps[target].any():
                filled.loc[gaps[target], target] = values[gaps[target].to_numpy()]
        numeric = pd.DataFrame(estimates, index=frame.index).drop(columns=CATEGORY_TARGET)
        return numeric, self.reporter.predict_proba(frame[self.report_columns].to_numpy(float))


class ChainedHistGradientBoosting(ChainedColumns):
    """Boosted trees added one at a time, each correcting the error left by the last.

    Regressors stop when held-out error stops falling, so a column with no signal
    gets few trees and is not fitted to noise. The classifier keeps a fixed size:
    stopping early costs it accuracy.
    """

    def __init__(self, settings, seed, drop=()):
        super().__init__(settings, seed, drop)
        self.classifier_rounds = self.settings.pop("classifier_rounds")
        self.stopping = {name: self.settings.pop(name) for name in
                         ("early_stopping", "n_iter_no_change", "validation_fraction")}

    def _categorical(self, columns):
        return [columns.index(CATEGORY_TARGET)] if CATEGORY_TARGET in columns else None

    def _regressor(self, columns, seed):
        return HistGradientBoostingRegressor(**self.settings, **self.stopping, random_state=seed,
                                             categorical_features=self._categorical(columns))

    def _classifier(self, columns, seed):
        settings = {**self.settings, "max_iter": self.classifier_rounds}
        return HistGradientBoostingClassifier(**settings, early_stopping=False, random_state=seed,
                                              categorical_features=self._categorical(columns))


class ScaledNeighbours:
    """Neighbours on scaled numeric columns and one-hot model_id, comparing the cells two rows both have.

    Distances are found for a block of rows at a time, which bounds memory when
    many workers run at once. A row that shares no cell with any training row has no
    neighbours, so it gets the training mean, or the training class shares.
    """

    block = 500

    def __init__(self, model, category_index):
        self.model, self.category_index = model, category_index

    def _inputs(self, matrix):
        if self.category_index is None:
            return (matrix - self.center) / self.spread
        numeric = (np.delete(matrix, self.category_index, axis=1) - self.center) / self.spread
        codes = matrix[:, self.category_index]
        hot = np.column_stack([codes == level for level in range(CLASS_COUNT)]).astype(float)
        hot[np.isnan(codes)] = np.nan
        return np.column_stack([numeric, hot])

    def fit(self, matrix, labels):
        numeric = matrix if self.category_index is None else np.delete(matrix, self.category_index, axis=1)
        self.center, self.spread = np.nanmean(numeric, axis=0), np.nanstd(numeric, axis=0) + 1e-12
        self.model.fit(self._inputs(matrix), labels)
        self.classes_ = getattr(self.model, "classes_", None)
        if self.classes_ is None:
            self.fallback = float(np.mean(labels))
        else:
            self.fallback = (labels[:, None] == self.classes_).mean(axis=0)
        return self

    def _blocks(self, method, matrix):
        inputs = self._inputs(matrix)
        result = np.concatenate([method(inputs[i:i + self.block]) for i in range(0, len(inputs), self.block)])
        lost = np.isnan(result) if result.ndim == 1 else np.isnan(result).any(axis=1)
        result[lost] = self.fallback
        return result

    def predict(self, matrix):
        if self.classes_ is None:
            return self._blocks(self.model.predict, matrix)
        return self.classes_[self.predict_proba(matrix).argmax(axis=1)].astype(float)

    def predict_proba(self, matrix):
        return self._blocks(self.model.predict_proba, matrix)


class ChainedNeighbours(ChainedColumns):
    """Averages the most similar rows. Distances become more reliable as gaps are filled."""

    def _index(self, columns):
        return columns.index(CATEGORY_TARGET) if CATEGORY_TARGET in columns else None

    def _regressor(self, columns, seed):
        return ScaledNeighbours(KNeighborsRegressor(**self.settings, metric="nan_euclidean"), self._index(columns))

    def _classifier(self, columns, seed):
        return ScaledNeighbours(KNeighborsClassifier(**self.settings, metric="nan_euclidean"), self._index(columns))


class ChainedForest(ChainedColumns):
    """Many deep trees, each grown on a sample of the rows, averaged."""

    def _regressor(self, columns, seed):
        return RandomForestRegressor(**self.settings, random_state=seed, n_jobs=THREADS)

    def _classifier(self, columns, seed):
        return RandomForestClassifier(**self.settings, random_state=seed, n_jobs=THREADS)


class FlatCatBoost:
    """CatBoost's classifier with one-dimensional predictions, as the chain expects."""

    def __init__(self, model):
        self.model = model

    def fit(self, matrix, labels):
        self.model.fit(matrix, labels)
        self.classes_ = np.asarray(self.model.classes_)
        return self

    def predict(self, matrix):
        return np.ravel(self.model.predict(matrix)).astype(float)

    def predict_proba(self, matrix):
        return self.model.predict_proba(matrix)


class StoppedCatBoost:
    """A CatBoost regressor that stops when error on held-out rows stops falling."""

    held_out = 0.15

    def __init__(self, model, seed):
        self.model, self.seed = model, seed

    def fit(self, matrix, labels):
        order = np.random.default_rng(self.seed).permutation(len(labels))
        cut = int(len(labels) * self.held_out)
        held, kept = order[:cut], order[cut:]
        self.model.fit(matrix[kept], labels[kept], eval_set=(matrix[held], labels[held]), use_best_model=True)
        return self

    def predict(self, matrix):
        return self.model.predict(matrix)


class ChainedCatBoost(ChainedColumns):
    """Gradient boosting on symmetric trees, where one split rule is shared across each level."""

    def __init__(self, settings, seed, drop=()):
        super().__init__(settings, seed, drop)
        self.classifier_rounds = self.settings.pop("classifier_rounds")
        self.patience = self.settings.pop("patience")

    def _common(self, seed):
        return {"random_seed": seed, "thread_count": THREADS, "verbose": False, "allow_writing_files": False}

    def _regressor(self, columns, seed):
        model = CatBoostRegressor(**self.settings, **self._common(seed), early_stopping_rounds=self.patience)
        return StoppedCatBoost(model, seed)

    def _classifier(self, columns, seed):
        settings = {**self.settings, "iterations": self.classifier_rounds}
        return FlatCatBoost(CatBoostClassifier(**settings, **self._common(seed)))


def kernel_frame(frame, levels):
    """A frame in the form miceforest needs, with model_id as a category."""
    data = frame.reset_index(drop=True).copy()
    data[CATEGORY_TARGET] = pd.Categorical(
        data[CATEGORY_TARGET].map(lambda v: str(int(v)) if pd.notna(v) else np.nan), categories=levels,
    )
    return data


class MiceforestKernel:
    """Every column in one miceforest kernel, updated in turn by chained equations.

    The first pass starts from the gaps, which LightGBM handles natively. The
    default random start would feed the first models with noise. Predictions are
    used directly, with no donor matching, and one completed dataset is produced.
    """

    candidates, datasets = 0, 1

    def _fit_kernel(self, measured, settings, seed):
        settings = dict(settings)
        self.iterations, rounds = settings.pop("iterations"), settings.pop("rounds")
        self.columns = list(measured.columns)
        self.levels = sorted(str(int(v)) for v in measured[CATEGORY_TARGET].dropna().unique())
        incomplete = [c for c in self.columns if measured[c].isna().any()]
        self.kernel = CorrectedKernel(
            kernel_frame(measured, self.levels), num_datasets=self.datasets,
            variable_schema={c: [p for p in self.columns if p != c] for c in incomplete},
            mean_match_candidates=self.candidates, initialize_empty=True,
            random_state=seed, save_all_iterations_data=True,
        )
        self.kernel.mice(self.iterations, num_iterations=rounds, num_threads=THREADS, verbosity=-1,
                         **BOOSTING, **settings)

    def _impute(self, frame, seed):
        return self.kernel.impute_new_data(kernel_frame(frame[self.columns], self.levels),
                                           iterations=self.iterations, random_state=seed)


class ChainedMiceforest(MiceforestKernel):
    """Boosted predictions from one kernel, with no donor matching.

    model_id is reported by a LightGBM classifier on the original gaps, for the
    same reason as in ``ChainedColumns``.
    """

    def __init__(self, settings, seed, drop=()):
        if drop:
            raise ValueError("Predictor blocks are not used with this model")
        self.settings, self.seed = dict(settings), seed

    def fit(self, measured):
        self._fit_kernel(measured, self.settings, self.seed)
        self.report_columns = [c for c in self.columns if c != CATEGORY_TARGET]
        known = measured[CATEGORY_TARGET].notna()
        self.reporter = LGBMClassifier(
            n_estimators=self.settings["rounds"], learning_rate=self.settings["learning_rate"],
            num_leaves=self.settings["num_leaves"], min_child_samples=self.settings["min_data_in_leaf"],
            random_state=self.seed, n_jobs=THREADS, verbosity=-1,
        ).fit(measured.loc[known, self.report_columns], measured.loc[known, CATEGORY_TARGET].to_numpy().astype(int))
        self.classes_ = self.reporter.classes_
        return self

    def predict(self, frame):
        """Numeric estimates for every cell of ``frame``, and model_id class probabilities."""
        result = self._impute(frame, self.seed + 1000).complete_data(dataset=0)
        numeric = result.drop(columns=CATEGORY_TARGET).astype(float)
        numeric.index = frame.index
        return numeric, self.reporter.predict_proba(frame[self.report_columns])


CHAINED_CLASSES = {
    "KNN imputer": ChainedNeighbours,
    "Random Forest": ChainedForest,
    "HistGradientBoosting": ChainedHistGradientBoosting,
    "CatBoost": ChainedCatBoost,
    "miceforest direct": ChainedMiceforest,
}


class ChainedRidgeImputer:
    """IterativeImputer over all columns at once, with a block ridge for each column.

    Chained equations update every incomplete column in turn, so one fit imputes
    every numeric column. model_id is estimated in the same chain. Every estimate
    stays inside its column's observed training range: the squared terms would
    otherwise feed on their own estimates from one pass to the next and diverge.
    """

    def __init__(self, settings, seed, drop=(), max_iter=None):
        self.settings, self.seed, self.drop = dict(settings), seed, tuple(drop)
        self.linear_alpha = self.settings.pop("linear_alpha")
        self.curve_alpha = self.settings.pop("curve_alpha")
        if max_iter is not None:
            self.settings["max_iter"] = max_iter

    def _limits(self, measured):
        numeric = self.features.numeric
        scaled = (measured[numeric] - self.features.center) / self.features.spread
        hot = len(self.features.levels)
        return np.r_[scaled.min().to_numpy(), np.zeros(hot)], np.r_[scaled.max().to_numpy(), np.ones(hot)]

    def fit(self, measured):
        self.columns = [c for c in measured if c not in self.drop]
        self.features = Features(True, open_category=True).fit(measured[self.columns])
        lower, upper = self._limits(measured)
        self.model = IterativeImputer(
            estimator=BlockRidge(self.linear_alpha, self.curve_alpha), **self.settings,
            min_value=lower, max_value=upper, random_state=self.seed, keep_empty_features=True,
        )
        self.model.fit(self.features.transform(measured[self.columns]))
        self.n_iter_ = self.model.n_iter_
        return self

    def predict(self, frame):
        """Filled numeric columns in their original units, aligned to ``frame``."""
        filled = self.model.transform(self.features.transform(frame[self.columns]))
        numeric = self.features.numeric
        values = filled[:, : len(numeric)] * self.features.spread.to_numpy() + self.features.center.to_numpy()
        return pd.DataFrame(values, index=frame.index, columns=numeric)


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
