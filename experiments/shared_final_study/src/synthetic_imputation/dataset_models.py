"""Adapters for new CSV schemas. The archived benchmark modules stay unchanged."""
from __future__ import annotations

import numpy as np
import pandas as pd

METHODS = {
    'median': 'Median / mode', 'forest': 'Predictive MissForest',
    'gaussian': 'Gaussian EM', 'ridge': 'Chained ridge', 'knn': 'k-NN',
    'mlp': 'Masked MLP', 'lightgbm': 'Chained LightGBM',
    'hgb': 'Chained histogram boosting', 'catboost': 'Chained CatBoost',
    'iterative': 'Iterative ridge + category mode',
    'mice': 'MICE-PMM', 'miceforest': 'miceforest / LightGBM',
}


class Baseline:
    def __init__(self, numeric, categorical):
        self.numeric, self.categorical = numeric, categorical

    def fit(self, train):
        self.statistics = {}
        for c in self.numeric + self.categorical:
            known = train[c].dropna()
            if known.empty:
                raise ValueError(f'{c}: no observed training values in this fold')
            self.statistics[c] = float(known.median()) if c in self.numeric else sorted(
                known.mode().tolist(), key=str)[0]
        return self

    def transform(self, frame):
        return frame.fillna(self.statistics).copy()


class MatrixModel:
    """Train-only category vocabularies and explicit, schema-free output formatting."""
    def __init__(self, method, numeric, categorical, params, seed):
        self.method, self.numeric, self.categorical = method, numeric, categorical
        self.params, self.seed = params, seed

    def encode(self, frame):
        arrays = []
        for c in self.columns:
            arrays.append(frame[c].map(self.encoders[c]).to_numpy(dtype=float)
                          if c in self.encoders else frame[c].to_numpy(dtype=float))
        return np.column_stack(arrays)

    def fit(self, train):
        from .vendor.subteam_a.data import Schema
        from .vendor.subteam_a.features import Features, Prediction
        from .vendor.subteam_a.mixture import MaskedMixture
        from .vendor.subteam_a.models import Autoencoder, Neighbours

        class DatasetFeatures(Features):
            def finish(self, raw, point, sd, prob, trace=()):
                point = point.copy()
                for j, p in prob.items():
                    p = np.maximum(np.nan_to_num(p), 1e-12)
                    p /= p.sum(axis=1, keepdims=True)
                    point[:, j] = np.argmax(p, axis=1)
                seen = np.isfinite(raw)
                point[seen] = raw[seen]
                if not np.isfinite(point).all():
                    raise ValueError('The model returned non-finite fills')
                return Prediction(point, np.maximum(np.nan_to_num(sd), 1e-8), prob, tuple(trace))

        self.columns = list(train.columns)
        self.levels = {c: sorted(train[c].dropna().unique().tolist(), key=str)
                       for c in self.categorical}
        self.encoders = {c: {v: i for i, v in enumerate(values)} for c, values in self.levels.items()}
        schema = Schema(tuple(self.columns), {self.columns.index(c): tuple(range(len(v)))
                        for c, v in self.levels.items()}, transformer=False)
        matrix = self.encode(train)
        features = DatasetFeatures(schema, domain=False).fit(matrix)
        if self.method == 'gaussian':
            dimensions = len(self.numeric) + sum(map(len, self.levels.values()))
            if dimensions > 80:
                raise ValueError('Gaussian EM exceeds 80 encoded dimensions; exclude identifiers or high-cardinality categories')
            patterns=len(np.unique(np.isfinite(matrix),axis=0))
            estimated=8*(6*patterns*dimensions**2+3*len(train)*min(self.params['clusters'],len(train))*dimensions)
            if estimated>512*1024**2:
                raise ValueError('Gaussian EM working-memory estimate exceeds 512 MB; reduce the dataset or fields for this method')
            p = dict(clusters=min(self.params['clusters'], len(train)), steps=self.params['em_steps'],
                     reg=self.params['regularisation'], class_weight=1.8)
            self.model = MaskedMixture(p, self.seed)
        elif self.method == 'mlp':
            if not self.numeric:
                raise ValueError('The masked MLP adapter requires at least one numeric field')
            if 2*len(self.numeric)+sum(map(len,self.levels.values()))>1000:
                raise ValueError('Masked MLP exceeds 1,000 encoded inputs; exclude high-cardinality categories')
            p = dict(width=64, epochs=self.params['epochs'], dropout=.1, lr=.0015,
                     decay=.0001, remask=self.params['mask_rate'])
            self.model = Autoencoder(p, self.seed)
        else:
            p = dict(donors=min(3000, len(train)), k=self.params['neighbors'],
                     penalty=.2, power=1.)
            self.model = Neighbours(p, self.seed)
        self.model.fit(matrix, features)
        return self

    def transform(self, frame):
        prediction = self.model.predict(self.encode(frame)).point
        result = frame.copy()
        for j, c in enumerate(self.columns):
            missing = frame[c].isna()
            values = prediction[:, j]
            if c in self.levels:
                values = np.asarray(self.levels[c], dtype=object)[values.astype(int)]
            result.loc[missing, c] = values[missing]
        return result


class SequentialBoosting(Baseline):
    """Fit per-target models once per pass; replay on held-out rows without refitting."""
    def __init__(self, method, numeric, categorical, params, seed):
        super().__init__(numeric, categorical)
        self.method, self.params, self.seed = method, params, seed

    def fit(self, train):
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        from sklearn.pipeline import Pipeline
        super().fit(train)
        self.models = []
        state, missing = super().transform(train), train.isna()
        for iteration in range(self.params['passes']):
            for target in train:
                predictors = [c for c in train if c != target]
                known = ~missing[target]
                if not predictors or train.loc[known, target].nunique() < 2:
                    continue
                parts = []
                nums = [c for c in predictors if c in self.numeric]
                cats = [c for c in predictors if c in self.categorical]
                if nums: parts.append(('numeric', StandardScaler(), nums))
                if cats: parts.append(('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cats))
                classification = target in self.categorical
                if self.method == 'hgb':
                    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
                    kind = HistGradientBoostingClassifier if classification else HistGradientBoostingRegressor
                    estimator = kind(max_iter=self.params['trees'], max_leaf_nodes=15, random_state=self.seed)
                else:
                    from catboost import CatBoostClassifier, CatBoostRegressor
                    kind = CatBoostClassifier if classification else CatBoostRegressor
                    estimator = kind(iterations=self.params['trees'], depth=4, verbose=False,
                                     thread_count=1, allow_writing_files=False, random_seed=self.seed)
                model = Pipeline([('prepare', ColumnTransformer(parts)), ('model', estimator)])
                model.fit(state.loc[known, predictors], train.loc[known, target])
                self.models.append((target, predictors, model))
                if missing[target].any():
                    state.loc[missing[target], target] = model.predict(state.loc[missing[target], predictors]).reshape(-1)
        return self

    def transform(self, frame):
        result = super().transform(frame)
        for target, predictors, model in self.models:
            missing = frame[target].isna()
            if missing.any():
                result.loc[missing, target] = model.predict(result.loc[missing, predictors]).reshape(-1)
        return result


class NumericIterative(Baseline):
    def __init__(self, numeric, categorical, params, seed):
        super().__init__(numeric, categorical)
        self.params, self.seed = params, seed

    def fit(self, train):
        from sklearn.experimental import enable_iterative_imputer  # noqa: F401
        from sklearn.impute import IterativeImputer
        super().fit(train)
        if self.numeric:
            self.model = IterativeImputer(max_iter=self.params['passes'], initial_strategy='median',
                                         random_state=self.seed).fit(train[self.numeric])
        return self

    def transform(self, frame):
        result = super().transform(frame)
        if self.numeric:
            result.loc[:, self.numeric] = self.model.transform(frame[self.numeric])
            result[frame.notna()] = frame[frame.notna()]
        return result


class DonorMICE(Baseline):
    """Reuse the team's PMM/logistic draws with training-only fits and training donors."""
    def __init__(self, numeric, categorical, params, seed):
        super().__init__(numeric, categorical)
        self.params, self.seed = params, seed

    def fit(self, train):
        from .vendor.pipeline.mice_pmm_logistic import MICEPMMLogistic, Schema
        super().fit(train)
        self.train = train.copy()
        self.engine = MICEPMMLogistic(Schema(list(train), self.categorical),
            n_imputations=self.params['chains'], n_iterations=self.params['passes'],
            donors=self.params['donors'], seed=self.seed, threads=1)
        return self

    def transform(self, frame):
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        table = pd.concat([self.train, frame], ignore_index=True)
        training = np.arange(len(table)) < len(self.train)
        original = table.copy()
        decoders = {c: {str(v): v for v in self.train[c].dropna().unique()} for c in self.categorical}
        for c in self.categorical:
            original[c] = original[c].map(lambda v: str(v) if pd.notna(v) else np.nan).astype(object)
        missing = original.isna()
        targets = sorted([c for c in table if missing[c].any()], key=lambda c: missing[c].sum())
        samples = []
        for child in np.random.SeedSequence(self.seed).spawn(self.params['chains']):
            rng = np.random.default_rng(child)
            work = original.copy()
            for c in targets:
                donors = original.loc[training & ~missing[c].to_numpy(), c].to_numpy()
                work.loc[missing[c], c] = rng.choice(donors, int(missing[c].sum()))
            for iteration in range(self.params['passes']):
                for target in targets:
                    predictors = [c for c in table if c != target and self.train[c].nunique() > 1]
                    if not predictors: continue
                    parts = []
                    nums = [c for c in predictors if c in self.numeric]
                    cats = [c for c in predictors if c in self.categorical]
                    if nums: parts.append(('numeric', StandardScaler(), nums))
                    if cats: parts.append(('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cats))
                    design = ColumnTransformer(parts).fit(work.loc[training, predictors]).transform(work[predictors])
                    known = training & ~missing[target].to_numpy()
                    fill = missing[target].to_numpy()
                    stacked = np.vstack([design[known], design[fill]])
                    flags = np.r_[np.ones(known.sum(), bool), np.zeros(fill.sum(), bool)]
                    y = original.loc[known, target].to_numpy()
                    if target in self.categorical:
                        values, _ = self.engine._categorical(stacked, flags, y.astype(str), rng)
                    else:
                        values = self.engine._pmm(stacked, flags, y.astype(float), rng)
                    work.loc[fill, target] = values
            samples.append(work.iloc[len(self.train):].reset_index(drop=True))
        result = frame.copy()
        for c in frame:
            mask = frame[c].isna()
            if c in self.numeric:
                values = np.column_stack([s[c].astype(float) for s in samples]).mean(axis=1)
            else:
                values = pd.concat([s[c] for s in samples], axis=1).mode(axis=1).iloc[:, 0].map(decoders[c]).to_numpy()
            result.loc[mask, c] = values[mask]
        return result


class ForestMICE:
    def __init__(self, numeric, categorical, params, seed):
        self.numeric, self.categorical, self.params, self.seed = numeric, categorical, params, seed

    def fit(self, train):
        from .miceforest_compat import CorrectedKernel
        self.levels = {c: sorted(train[c].dropna().unique().tolist(), key=str) for c in self.categorical}
        data = self.prepare(train)
        self.model = CorrectedKernel(data, num_datasets=1, variable_schema=list(train),
                                    save_all_iterations_data=True, random_state=self.seed)
        self.model.mice(self.params['passes'], num_iterations=self.params['trees'], boosting='gbdt',
                        num_leaves=15, min_data_in_leaf=5, num_threads=1, verbosity=-1)
        return self

    def prepare(self, frame):
        data = frame.reset_index(drop=True).copy()
        for c, levels in self.levels.items(): data[c] = pd.Categorical(data[c], categories=levels)
        return data

    def transform(self, frame):
        result = self.model.impute_new_data(self.prepare(frame), random_state=self.seed).complete_data(0)
        result.index = frame.index
        for c in self.categorical: result[c] = result[c].astype(object)
        result[frame.notna()] = frame[frame.notna()]
        return result


def make_model(method, numeric, categorical, params, seed):
    if len(numeric) + len(categorical) < 2 and method not in ('median', 'gaussian', 'knn', 'iterative'):
        raise ValueError('This method needs at least two modelled columns')
    if method == 'median': return Baseline(numeric, categorical)
    if method in ('gaussian', 'mlp', 'knn'): return MatrixModel(method, numeric, categorical, params, seed)
    if method in ('hgb', 'catboost'): return SequentialBoosting(method, numeric, categorical, params, seed)
    if method == 'iterative': return NumericIterative(numeric, categorical, params, seed)
    if method == 'mice': return DonorMICE(numeric, categorical, params, seed)
    if method == 'miceforest': return ForestMICE(numeric, categorical, params, seed)
    if method == 'forest':
        from .vendor.pipeline.predictive_missforest import PredictiveMissForest, PredictiveMissForestConfig
        return PredictiveMissForest(numeric, categorical, PredictiveMissForestConfig(
            n_estimators=params['trees'], n_passes=params['passes'], min_samples_leaf=2,
            random_state=seed, n_jobs=1))
    if method in ('ridge', 'lightgbm'):
        from .vendor.pipeline.comparison_models import ChainedConditionalImputer, ModelSpec
        return ChainedConditionalImputer(ModelSpec(tuple(numeric), tuple(categorical), seed), family=method,
            passes=params['passes'], ridge_alpha=params['ridge_alpha'], lgbm_estimators=params['trees'])
    raise ValueError(f'Unknown method: {method}')
