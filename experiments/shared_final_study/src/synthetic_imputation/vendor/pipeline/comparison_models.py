"""Isolated, inductive candidate imputers for the transformer model comparison.

Only originally observed TRAINING targets are used to fit a conditional model.
The stored per-pass models are replayed on held-out records without fitting.
TabPFN instead uses Prior Labs' experimental unsupervised imputation extension,
with its fitting context strictly restricted to training records.

This file does NOT change any existing production/pipeline imputer.
"""
from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version as installed_version
import os
import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class ModelSpec:
    numeric: tuple[str, ...]
    categorical: tuple[str, ...]
    seed: int = 219

    @property
    def columns(self) -> tuple[str, ...]:
        return self.numeric + self.categorical


def _check_input(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("Input must be a nonempty pandas DataFrame")
    if set(columns) - set(frame.columns):
        raise ValueError(f"Missing model columns: {sorted(set(columns) - set(frame.columns))}")
    if not frame.index.is_unique:
        raise ValueError("Input indices must be unique")


def _mode(observed: pd.Series) -> Any:
    values = observed.mode(dropna=True)
    if values.empty:
        raise ValueError("A model column has no observed fitting values")
    return sorted(values.tolist(), key=lambda x: str(x))[0]


class MedianMode:
    name = "median_mode"

    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec
        self.statistics_: dict[str, Any] = {}

    def fit(self, train: pd.DataFrame) -> "MedianMode":
        _check_input(train, self.spec.columns)
        for c in self.spec.numeric:
            known = pd.to_numeric(train[c].dropna())
            if known.empty:
                raise ValueError(f"No observed training values in {c}")
            self.statistics_[c] = float(known.median())
        for c in self.spec.categorical:
            self.statistics_[c] = _mode(train[c].dropna())
        return self

    def transform(self, records: pd.DataFrame) -> pd.DataFrame:
        if not self.statistics_:
            raise RuntimeError("Call fit before transform")
        _check_input(records, self.spec.columns)
        result = records.copy(deep=True)
        for c in self.spec.columns:
            missing = result[c].isna()
            if missing.any():
                result.loc[missing, c] = self.statistics_[c]
        return result


class ChainedConditionalImputer(MedianMode):
    """Stored conditional models trained in sequential passes, then replayed.

    Ridge uses linear regression and multinomial logistic classification with
    scaled numeric and one-hot categorical predictors. LightGBM uses nonlinear
    gradient-boosted trees with categorical predictors declared explicitly.
    The approach is *not* an implementation of MICE-PMM or original MissForest.
    """

    def __init__(
        self,
        spec: ModelSpec,
        *,
        family: str,
        passes: int = 3,
        ridge_alpha: float = 10.0,
        lgbm_estimators: int = 100,
        lgbm_learning_rate: float = 0.07,
        lgbm_num_leaves: int = 31,
    ) -> None:
        super().__init__(spec)
        if family not in {"ridge", "lightgbm"}:
            raise ValueError(f"Unknown family: {family}")
        if passes < 1:
            raise ValueError("passes must be positive")
        self.family = family
        self.name = "chained_ridge" if family == "ridge" else "lightgbm_chained"
        self.passes = int(passes)
        self.ridge_alpha = float(ridge_alpha)
        self.lgbm_estimators = int(lgbm_estimators)
        self.lgbm_learning_rate = float(lgbm_learning_rate)
        self.lgbm_num_leaves = int(lgbm_num_leaves)
        self.models_: list[tuple[str, Any]] = []
        self.vocab_: dict[str, list[Any]] = {}
        self.training_rows_ = 0

    def _inputs(self, state: pd.DataFrame, target: str) -> pd.DataFrame:
        features = [c for c in self.spec.columns if c != target]
        x = state[features].copy()
        if self.family == "lightgbm":
            for c in self.spec.categorical:
                if c in features:
                    x[c] = pd.Categorical(x[c], categories=self.vocab_[c])
            for c in self.spec.numeric:
                if c in features:
                    x[c] = pd.to_numeric(x[c], errors="raise").astype(float)
        return x

    def _estimator(self, target: str):
        target_index = self.spec.columns.index(target)
        seed = self.spec.seed + target_index
        if self.family == "lightgbm":
            try:
                from lightgbm import LGBMClassifier, LGBMRegressor
            except ImportError as exc:
                raise ImportError(
                    "LightGBM comparison requires: pip install lightgbm"
                ) from exc
            params = dict(
                n_estimators=self.lgbm_estimators,
                learning_rate=self.lgbm_learning_rate,
                num_leaves=self.lgbm_num_leaves,
                random_state=seed,
                n_jobs=1,
                verbosity=-1,
            )
            if target in self.spec.categorical:
                return LGBMClassifier(**params)
            return LGBMRegressor(**params)

        features = [c for c in self.spec.columns if c != target]
        numeric = [c for c in features if c in self.spec.numeric]
        categorical = [c for c in features if c in self.spec.categorical]
        transformers = []
        if numeric:
            transformers.append(("numeric", StandardScaler(), numeric))
        if categorical:
            transformers.append((
                "categorical",
                OneHotEncoder(
                    categories=[self.vocab_[c] for c in categorical],
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                categorical,
            ))
        preprocessing = ColumnTransformer(transformers, remainder="drop")
        if target in self.spec.categorical:
            estimator = LogisticRegression(max_iter=350, random_state=seed)
        else:
            estimator = Ridge(alpha=self.ridge_alpha)
        return Pipeline([("preprocess", preprocessing), ("model", estimator)])

    def fit(self, train: pd.DataFrame) -> "ChainedConditionalImputer":
        super().fit(train)
        self.vocab_ = {
            c: sorted(train[c].dropna().unique().tolist(), key=str)
            for c in self.spec.categorical
        }
        original_missing = train[list(self.spec.columns)].isna()
        state = super().transform(train)
        self.models_ = []
        self.training_rows_ = len(train)
        for _pass in range(self.passes):
            for target in self.spec.columns:
                observed = ~original_missing[target]
                y = train.loc[observed, target]
                if len(y) < 2 or y.nunique(dropna=True) == 1:
                    self.models_.append((target, None))
                    if original_missing[target].any():
                        state.loc[original_missing[target], target] = self.statistics_[target]
                    continue
                estimator = self._estimator(target)
                predictors = self._inputs(state, target)
                estimator.fit(predictors.loc[observed], y)
                self.models_.append((target, estimator))
                missing = original_missing[target]
                if missing.any():
                    state.loc[missing, target] = estimator.predict(predictors.loc[missing])
        return self

    def transform(self, records: pd.DataFrame) -> pd.DataFrame:
        if not self.models_:
            raise RuntimeError("Call fit before transform")
        _check_input(records, self.spec.columns)
        missing = records[list(self.spec.columns)].isna()
        state = MedianMode.transform(self, records)
        for target, estimator in self.models_:
            needs_fill = missing[target]
            if estimator is not None and needs_fill.any():
                x = self._inputs(state, target)
                state.loc[needs_fill, target] = estimator.predict(x.loc[needs_fill])
        return state


class TabPFNExperimentalImputer(MedianMode):
    """Optional Prior Labs unsupervised extension; fit context stays train-only.

    Not equivalent to stored-model PMF. TabPFN's extension may perform expensive
    in-context classifier/regressor fitting inside each `impute` call.  We use
    the official imputation API, not a simulated substitute.  The context-size
    limit is disclosed by the runner: reduced context is a feasibility pilot,
    not a full 8,000-row training comparison.
    """

    name = "tabpfn_unsupervised"

    def __init__(
        self, spec: ModelSpec, *, device: str = "cpu",
        max_context_rows: int = 2000, n_permutations: int = 1,
        model_version: str = "v3.5-fast",
        mps_batch_rows: int = 128,
        n_estimators: int | None = None,
        mps_force_float32: bool = True,
    ) -> None:
        super().__init__(spec)
        if mps_batch_rows < 1:
            raise ValueError("mps_batch_rows must be positive")
        if n_estimators is not None and n_estimators < 1:
            raise ValueError("n_estimators must be a positive integer or null")
        if max_context_rows < 20 or n_permutations < 1:
            raise ValueError("Invalid TabPFN context / permutation count")
        self.device = device
        self.max_context_rows = int(max_context_rows)
        self.n_permutations = int(n_permutations)
        self.model_version = model_version
        self.mps_batch_rows = int(mps_batch_rows)
        # None keeps official ensemble defaults. Set 1 explicitly for the first
        # MPS diagnostic; log it as an experimental hyperparameter change.
        self.n_estimators = n_estimators
        self.mps_force_float32 = bool(mps_force_float32)
        self.context_rows_ = 0
        self.batch_count_ = 0
        self.model_ = None
        self.vocab_: dict[str, list[Any]] = {}

    def _encode(self, frame: pd.DataFrame) -> np.ndarray:
        matrix = np.empty((len(frame), len(self.spec.columns)), dtype=np.float32)
        for j, c in enumerate(self.spec.columns):
            if c in self.spec.categorical:
                mapping = {v: i for i, v in enumerate(self.vocab_[c])}
                mapped = frame[c].map(mapping)
                unknown = mapped.isna() & frame[c].notna()
                if unknown.any():
                    raise ValueError(f"Unknown observed category in {c}: {frame.loc[unknown, c].unique().tolist()}")
                matrix[:, j] = mapped.to_numpy(dtype=float)
            else:
                matrix[:, j] = pd.to_numeric(frame[c], errors="raise").to_numpy(dtype=float)
        return matrix

    def fit(self, train: pd.DataFrame) -> "TabPFNExperimentalImputer":
        super().fit(train)
        try:
            from tabpfn import TabPFNClassifier, TabPFNRegressor
            from tabpfn_extensions.unsupervised import TabPFNUnsupervisedModel
        except ImportError as exc:
            raise ImportError(
                "TabPFN is optional. Install a compatible pair of `tabpfn` "
                "and `tabpfn-extensions` and review the model-weight licence."
            ) from exc
        self.vocab_ = {
            c: sorted(train[c].dropna().unique().tolist(), key=str)
            for c in self.spec.categorical
        }
        rng = np.random.default_rng(self.spec.seed)
        if len(train) > self.max_context_rows:
            selected = np.sort(rng.choice(len(train), self.max_context_rows, replace=False))
            context = train.iloc[selected]
        else:
            context = train
        self.context_rows_ = len(context)
        # An MPS device being available does not mean a particular TabPFN
        # checkpoint / native attention kernel is stable. Native SIGSEGV cannot
        # be caught in this Python process. Run diagnose_tabpfn_mps.py first.
        overrides: dict[str, Any] = {"device": self.device}
        if self.n_estimators is not None:
            overrides["n_estimators"] = self.n_estimators
        if self.device == "mps":
            import torch
            if not torch.backends.mps.is_available():
                raise RuntimeError("device='mps' requested but PyTorch MPS is unavailable")
            if self.mps_force_float32:
                overrides["inference_precision"] = torch.float32
            try:
                from packaging.version import Version
                current = installed_version("tabpfn")
                if Version(current) < Version("9.0.0"):
                    warnings.warn(
                        f"TabPFN {current} predates several upstream Apple Silicon "
                        "MPS fixes. Upgrade tabpfn and tabpfn-extensions before "
                        "the MPS diagnostic.", RuntimeWarning, stacklevel=2,
                    )
            except (PackageNotFoundError, ImportError, ValueError):
                pass
            if not os.getenv("TABPFN_MPS_MEMORY_FRACTION"):
                warnings.warn(
                    "For an MPS diagnostic, set TABPFN_MPS_MEMORY_FRACTION=0.5 "
                    "in Terminal BEFORE starting Python. This limits GPU memory "
                    "but cannot repair native backend crashes.",
                    RuntimeWarning, stacklevel=2,
                )
        # Important: declare known categorical indices; without this the numeric
        # encoding of model_id could be misidentified as a continuous variable.
        if self.model_version == "default":
            clf = TabPFNClassifier(**overrides)
            reg = TabPFNRegressor(**overrides)
        else:
            try:
                from tabpfn.constants import ModelVersion
                version_map = {"v3.5-fast": "V3_5_FAST", "v3.5": "V3_5", "v2.5": "V2_5"}
                if self.model_version not in version_map:
                    raise ValueError(f"Unsupported TabPFN version: {self.model_version}")
                version = getattr(ModelVersion, version_map[self.model_version])
                clf = TabPFNClassifier.create_default_for_version(version, **overrides)
                reg = TabPFNRegressor.create_default_for_version(version, **overrides)
            except (ImportError, AttributeError, TypeError) as exc:
                raise RuntimeError(
                    "Installed TabPFN version is incompatible with the selected model_version. "
                    "Select a supported version or set model_version='default' and record the package version."
                ) from exc
        # The upstream unsupervised extension checks feature type twice:
        # once on observed target labels during `density_` and again on the
        # original (possibly NaN-containing) context during prediction. Its
        # class-count check can therefore choose a classifier for fitting but
        # a regressor-style predict(output_type="full") call afterwards.
        # For this experiment the column roles are known from ModelSpec, so
        # keep that decision stable at both call sites. No TabPFN package files
        # are modified, and all other candidate imputers remain unchanged.
        categorical_indices = [
            self.spec.columns.index(c) for c in self.spec.categorical
        ]

        # The extension returns classifier samples as *indices in its fitted
        # class order*. Our decoding below requires the full 0..K-1 training
        # vocabulary; otherwise a missing class could silently shift labels.
        for column in self.spec.categorical:
            actual = set(context[column].dropna().tolist())
            required = set(self.vocab_[column])
            if actual != required:
                absent = sorted(required - actual, key=str)
                raise ValueError(
                    f"TabPFN training context of {len(context)} rows does not "
                    f"include every observed category for {column}: {absent}. "
                    "Increase max_context_rows or use a class-covering context sample."
                )

        self_outer_device = self.device

        class SchemaFixedTabPFNModel(TabPFNUnsupervisedModel):
            """Use declared feature roles at every extension decision point."""

            def __init__(self, tabpfn_clf, tabpfn_reg, categorical_indices):
                super().__init__(tabpfn_clf=tabpfn_clf, tabpfn_reg=tabpfn_reg)
                self.categorical_indices = tuple(categorical_indices)
                self._categorical_indices = frozenset(categorical_indices)
                self._align_samples_for_mps = (self_outer_device == "mps")

            def use_classifier_(self, column_idx, y):
                # The extension calls this on both filtered labels and an
                # unfiltered context that may contain NaNs. Ignore y here.
                return column_idx in self._categorical_indices

            def sample_from_model_prediction_(
                self, column_idx, X_fit, model, X_predict, t,
            ):
                """Normalize sampled values to the extension's *input* tensor device.

                TabPFN-3.5-Fast can sample numerical predictions on MPS while
                tabpfn-extensions 0.6.3 holds its imputation matrix on CPU.
                Upstream `.to(y_predict.dtype)` changes dtype but NOT device.
                """
                raw_prediction, sample = super().sample_from_model_prediction_(
                    column_idx, X_fit, model, X_predict, t,
                )
                if self._align_samples_for_mps:
                    import torch
                    sample = torch.as_tensor(sample).to(
                        device=X_predict.device, dtype=X_predict.dtype,
                    )
                return raw_prediction, sample

            def impute_(
                self, X, t=1e-9, n_permutations=10,
                condition_on_all_features=True, dag=None, fast_mode=False,
            ):
                """MPS-safe version of the extension 0.6.3 imputation loop.

                Same original density fits, permutation order and stochastic
                sampling. Only difference is aligning sampled tensors with the
                imputation matrix device at the final assignment. The official
                extension doesn't align them and `.to(dtype)` is insufficient.
                """
                if not self._align_samples_for_mps:
                    return super().impute_(
                        X, t=t, n_permutations=n_permutations,
                        condition_on_all_features=condition_on_all_features,
                        dag=dag, fast_mode=fast_mode,
                    )
                if dag is not None:
                    raise NotImplementedError(
                        "The MPS compatibility adapter does not support DAG imputation. "
                        "Use CPU or extend the adapter with a tested DAG implementation."
                    )
                if not condition_on_all_features:
                    raise NotImplementedError(
                        "MPS compatibility adapter currently supports the default "
                        "condition_on_all_features=True protocol only."
                    )
                import torch
                from tqdm.auto import tqdm
                from tabpfn_extensions.unsupervised.unsupervised import (
                    efficient_random_permutation,
                )
                # The extension's impute() receives numpy and converts it to a
                # CPU tensor. Its conditional TabPFN models may run on MPS.
                state = X.clone()
                all_features = list(range(state.shape[1]))
                columns_with_nan = [
                    c for c in all_features if bool(torch.isnan(state[:, c]).any())
                ]
                X_fit = self.X_
                for column_idx in tqdm(columns_with_nan):
                    column_missing = torch.isnan(state[:, column_idx])
                    condition_features = list(set(all_features) - {column_idx})
                    rows_to_impute = state[column_missing].reshape(-1, state.shape[1])
                    densities = []
                    for permutation in efficient_random_permutation(
                        condition_features,
                        1 if fast_mode else n_permutations,
                    ):
                        _, distribution = self.impute_single_permutation_(
                            rows_to_impute,
                            (*permutation, column_idx),
                            t,
                            condition_on_all_features,
                        )
                        densities.append(distribution)
                    if not densities:
                        raise RuntimeError("TabPFN produced no conditional distributions")
                    if self.use_classifier_(column_idx, X_fit[:, column_idx]):
                        probabilities = torch.stack([
                            torch.as_tensor(d, dtype=torch.float32, device=state.device)
                            for d in densities
                        ]).mean(dim=0)
                        sample = torch.distributions.Categorical(
                            probs=probabilities
                        ).sample().float()
                    else:
                        criterion = densities[0]["criterion"]
                        merged = criterion.average_bar_distributions_into_this(
                            [d["criterion"] for d in densities],
                            [
                                d["logits"].clone().detach()
                                if torch.is_tensor(d["logits"])
                                else torch.as_tensor(d["logits"])
                                for d in densities
                            ],
                        )
                        sample = criterion.sample(merged, t=t)
                    # The MPS-specific repair. Upstream only casts the dtype,
                    # leaving samples on mps:0 while `state` is on cpu.
                    sample = torch.as_tensor(sample).to(
                        device=state.device, dtype=state.dtype
                    )
                    if sample.numel() != int(column_missing.sum()):
                        raise RuntimeError(
                            f"TabPFN sample shape mismatch for column {column_idx}"
                        )
                    state[column_missing, column_idx] = sample.reshape(-1)
                return state
        self.model_ = SchemaFixedTabPFNModel(
            tabpfn_clf=clf,
            tabpfn_reg=reg,
            categorical_indices=categorical_indices,
        )
        self.model_.set_categorical_features(categorical_indices)
        self.model_.fit(self._encode(context))
        # Upstream fit may infer additional categorical columns from numeric
        # values; restore our explicit schema after its inference step.
        self.model_.set_categorical_features(categorical_indices)
        return self

    def transform(self, records: pd.DataFrame) -> pd.DataFrame:
        if self.model_ is None:
            raise RuntimeError("Call fit before transform")
        _check_input(records, self.spec.columns)
        import torch
        torch.manual_seed(self.spec.seed)
        np.random.seed(self.spec.seed)
        matrix = self._encode(records)
        # TabPFN's official unsupervised extension fits many conditional
        # predictors during impute(). On MPS, limiting inference rows reduces
        # peak memory pressure. It is slower and may yield different stochastic
        # samples, so both the batch size and MPS settings belong in provenance.
        chunks = []
        batch_rows = self.mps_batch_rows if self.device == "mps" else len(matrix)
        for start in range(0, len(matrix), batch_rows):
            part = matrix[start:start + batch_rows]
            part_output = self.model_.impute(part, n_permutations=self.n_permutations)
            if hasattr(part_output, "detach"):
                part_output = part_output.detach().cpu().numpy()
            part_output = np.asarray(part_output)
            if part_output.shape != part.shape:
                raise RuntimeError("TabPFN produced an unexpected shape in an inference batch")
            chunks.append(part_output)
            if self.device == "mps":
                # Empty *unused* cached allocations between batches; retains
                # tensors referenced by TabPFN and cannot prevent kernel faults.
                import torch
                torch.mps.empty_cache()
        self.batch_count_ = len(chunks)
        output = np.concatenate(chunks, axis=0)
        if output.shape != matrix.shape:
            raise RuntimeError("TabPFN returned an unexpected shape")
        completed = records.copy(deep=True)
        for j, c in enumerate(self.spec.columns):
            missing = records[c].isna().to_numpy()
            if not missing.any():
                continue
            values = output[missing, j]
            if not np.isfinite(values).all():
                raise RuntimeError(f"TabPFN returned non-finite imputed values in {c}")
            if c in self.spec.categorical:
                codes = np.rint(values).astype(int)
                if np.any(abs(values - codes) > 1e-5) or np.any(codes < 0) or np.any(codes >= len(self.vocab_[c])):
                    raise RuntimeError(f"TabPFN returned invalid category codes in {c}")
                values = [self.vocab_[c][k] for k in codes]
            completed.loc[missing, c] = values
        return completed


def make_candidate(name: str, spec: ModelSpec, config: dict[str, Any]):
    if name == "median_mode":
        return MedianMode(spec)
    if name == "chained_ridge":
        params = config.get("chained_ridge", {})
        return ChainedConditionalImputer(
            spec, family="ridge", passes=params.get("passes", 3),
            ridge_alpha=params.get("alpha", 10.0),
        )
    if name == "lightgbm_chained":
        params = config.get("lightgbm_chained", {})
        return ChainedConditionalImputer(
            spec, family="lightgbm", passes=params.get("passes", 3),
            lgbm_estimators=params.get("n_estimators", 100),
            lgbm_learning_rate=params.get("learning_rate", 0.07),
            lgbm_num_leaves=params.get("num_leaves", 31),
        )
    if name == "tabpfn_unsupervised":
        params = config.get("tabpfn_unsupervised", {})
        return TabPFNExperimentalImputer(
            spec, device=params.get("device", "cpu"),
            max_context_rows=params.get("max_context_rows", 2000),
            n_permutations=params.get("n_permutations", 1),
            model_version=params.get("model_version", "v3.5-fast"),
            mps_batch_rows=params.get("mps_batch_rows", 128),
            n_estimators=params.get("n_estimators"),
            mps_force_float32=params.get("mps_force_float32", True),
        )
    raise ValueError(f"Unsupported comparison method: {name}")
