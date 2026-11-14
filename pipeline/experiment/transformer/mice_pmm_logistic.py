"""Reusable, transductive MICE: Bayesian linear PMM + bootstrap logistic regression.

Only incomplete input tables enter the imputer. Truth is a separate evaluation input.
This is a Python implementation of the stated variant, not a wrapper around R mice.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from collections import Counter
import hashlib
import json
import platform
import time
import warnings

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.spatial import cKDTree
from scipy.stats import chi2_contingency, wasserstein_distance
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from threadpoolctl import threadpool_limits


@dataclass
class Schema:
    targets: list[str]
    categorical: list[str]
    exclude: list[str] = field(default_factory=list)
    # destination: {source: column, scale: number, offset: number}
    derived: dict = field(default_factory=dict)
    # Optional target-specific predictor lists. Omitted targets use all eligible columns.
    predictor_map: dict = field(default_factory=dict)
    max_categories: int = 200
    max_design_columns: int = 600

    def validate(self, frame):
        if frame.empty or not frame.columns.is_unique:
            raise ValueError("Input must have rows and unique column names.")
        if not frame.index.is_unique:
            raise ValueError("Input index must be unique.")
        if not self.targets or len(set(self.targets)) != len(self.targets):
            raise ValueError("Choose a nonempty, unique target list.")
        named = set(self.targets) | set(self.categorical) | set(self.exclude) | set(self.derived)
        for rule in self.derived.values():
            named.add(rule['source'])
        absent = named - set(frame.columns)
        if absent:
            raise ValueError(f"Configured columns not in input: {sorted(absent)}")
        if set(self.targets) & (set(self.exclude) | set(self.derived)):
            raise ValueError("Targets cannot also be excluded or derived.")
        for dest, rule in self.derived.items():
            if rule['source'] in self.derived:
                raise ValueError("Derived fields must refer directly to an independent column.")
            both = frame[dest].notna() & frame[rule['source']].notna()
            expected = frame.loc[both, rule['source']] * rule.get('scale', 1) + rule.get('offset', 0)
            if not np.allclose(frame.loc[both, dest].astype(float), expected.astype(float)):
                raise ValueError(f"Observed {dest} does not follow its configured derivation.")
            # Avoid silently ignoring observed derived information when its source is absent.
            if (frame[dest].notna() & frame[rule['source']].isna()).any():
                raise ValueError(f"{dest} is observed where its source is missing; reconcile explicitly.")
        eligible = self.model_columns(frame)
        for c in eligible:
            seen = frame[c].dropna()
            if seen.empty:
                raise ValueError(f"{c}: entirely missing; provide external information or exclude it.")
            if frame[c].isna().any() and c not in self.targets:
                raise ValueError(f"{c} has missing predictors; add it to targets or exclude it.")
            if c in self.categorical:
                if seen.nunique() > self.max_categories:
                    raise ValueError(f"{c}: too many categories; exclude IDs or revise max_categories.")
                if len(set(map(str, seen.unique()))) != seen.nunique():
                    raise ValueError(f"{c}: labels collide when converted to strings.")
            elif not pd.api.types.is_numeric_dtype(frame[c]):
                raise ValueError(f"{c}: declare as categorical or explicitly convert to numeric.")
            elif not np.isfinite(seen.astype(float)).all():
                raise ValueError(f"{c}: infinite numeric values are not supported.")
        for target, predictors in self.predictor_map.items():
            if target not in self.targets or target in predictors or set(predictors) - set(eligible):
                raise ValueError(f"Invalid predictor_map entry for {target}.")
        return self

    def model_columns(self, frame):
        return [c for c in frame if c not in set(self.exclude) | set(self.derived)]

    def predictors(self, frame, target):
        choices = self.predictor_map.get(target, self.model_columns(frame))
        return [c for c in choices if c != target and frame[c].nunique(dropna=True) > 1]


def infer_schema(frame, targets=None, categorical=None, exclude=None, derived=None, **kwargs):
    """Inference uses the masked table only; numeric category codes need an override."""
    exclude, derived = list(exclude or []), dict(derived or {})
    if categorical is None:
        categorical = [c for c in frame if not pd.api.types.is_numeric_dtype(frame[c])
                       or pd.api.types.is_bool_dtype(frame[c])]
    if targets is None:
        targets = [c for c in frame if frame[c].isna().any() and c not in exclude and c not in derived]
    return Schema(list(targets), list(categorical), exclude, derived, **kwargs).validate(frame)


def load_inputs(truth_path, masked_paths, id_column=None, align_by_row_order=False, na_values=None):
    """Align truth to each masked table by a unique ID; reject dropped/extra/duplicate rows."""
    truth_path = Path(truth_path)
    paths = [Path(p) for p in masked_paths]
    if not paths or len({p.stem for p in paths}) != len(paths):
        raise ValueError("Supply masked CSVs with unique file stems.")
    if truth_path.resolve() in [p.resolve() for p in paths]:
        raise ValueError("The truth file must not also be a masked input.")
    def read(path):
        # Pandas may mangle duplicate headers: check the raw header first.
        import csv
        with path.open(newline='', encoding='utf-8-sig') as handle:
            header = next(csv.reader(handle))
        if len(set(header)) != len(header):
            raise ValueError(f"Duplicate column names in {path.name}")
        return pd.read_csv(path, na_values=na_values, float_precision='round_trip')
    truth = read(truth_path)
    measured, aligned_truth, audit = {}, {}, []
    if id_column is None and not align_by_row_order:
        raise ValueError("Set id_column, or explicitly enable align_by_row_order.")
    if id_column is not None:
        if id_column not in truth or truth[id_column].isna().any() or truth[id_column].duplicated().any():
            raise ValueError("Truth ID must be present, complete, and unique.")
    for path in paths:
        data = read(path)
        if set(data.columns) != set(truth.columns) or len(data) != len(truth):
            raise ValueError(f"{path.name}: truth and masked schemas/row counts differ.")
        if id_column is not None:
            if data[id_column].isna().any() or data[id_column].duplicated().any():
                raise ValueError(f"{path.name}: missing or duplicate IDs.")
            if set(data[id_column]) != set(truth[id_column]):
                raise ValueError(f"{path.name}: ID sets differ.")
            ref = truth.set_index(id_column).loc[data[id_column]].reset_index()[data.columns]
        else:
            ref = truth[data.columns].copy()
        measured[path.stem], aligned_truth[path.stem] = data, ref
        audit.append(dict(scenario=path.stem, rows=len(data), columns=len(data.columns),
                          missing_cells=int(data.isna().sum().sum()),
                          sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    return measured, aligned_truth, pd.DataFrame(audit)


def apply_derived(frame, schema):
    """Fill missing derived cells only. Never overwrite an observed cell."""
    out = frame.copy()
    for c, rule in schema.derived.items():
        missing = out[c].isna()
        value = out[rule['source']] * rule.get('scale', 1) + rule.get('offset', 0)
        out.loc[missing, c] = value.loc[missing]
    return out


def baseline_impute(frame, schema):
    schema.validate(frame)
    out = frame.copy()
    for c in schema.targets:
        observed = frame[c].dropna()
        value = observed.mode().iloc[0] if c in schema.categorical else observed.median()
        out[c] = out[c].fillna(value)
    return apply_derived(out, schema)


class MICEPMMLogistic:
    """Complete an entire table, retaining independent final states from multiple chains.

    PMM uses a weak proper Gaussian/inverse-gamma linear model and type-1 matching:
    fitted donor predictions versus a posterior coefficient draw for recipients.
    Categorical updates bootstrap observed rows, fit regularized logistic regression,
    and sample labels from predicted probabilities. This is an approximate MI variant.
    There is intentionally no fit/transform claim for unseen tables.
    """
    def __init__(self, schema, n_imputations=5, n_iterations=10, donors=5, seed=219,
                 ridge=1e-4, logistic_C=1.0, logistic_max_iter=500, threads=1, verbose=False):
        if min(n_imputations, n_iterations, donors) < 1 or ridge <= 0 or logistic_C <= 0:
            raise ValueError("Positive imputation, iteration, donor, ridge and C values required.")
        self.schema = schema
        self.n_imputations, self.n_iterations = n_imputations, n_iterations
        self.donors, self.seed, self.ridge = donors, seed, ridge
        self.logistic_C, self.logistic_max_iter, self.threads = logistic_C, logistic_max_iter, threads
        self.verbose = verbose

    def _design(self, work, columns):
        if not columns:
            return np.ones((len(work), 1))
        cats = [c for c in columns if c in self.schema.categorical]
        nums = [c for c in columns if c not in cats]
        steps = []
        if nums:
            steps.append(('numeric', StandardScaler(), nums))
        if cats:
            steps.append(('category', OneHotEncoder(handle_unknown='ignore', drop='first',
                                                    sparse_output=False), cats))
        # Count before allocating a dense matrix to catch accidentally included row IDs.
        count = len(nums) + sum(max(work[c].nunique()-1, 0) for c in cats)
        if count > self.schema.max_design_columns:
            raise ValueError("Design matrix too wide; remove IDs or restrict predictor_map.")
        return ColumnTransformer(steps, remainder='drop').fit_transform(work[columns])

    def _pmm(self, design, observed, y, rng):
        x = np.column_stack([np.ones(len(design)), design])
        xo, xm = x[observed], x[~observed]
        center, scale = float(np.mean(y)), float(np.std(y))
        if scale < 1e-12:
            return np.full(len(xm), y[0])
        z = (y-center)/scale
        # Proper weak normal prior, including a very weak intercept prior.
        precision = np.full(x.shape[1], self.ridge)
        precision[0] = self.ridge * 0.01
        a = xo.T @ xo + np.diag(precision)
        factor, lower = cho_factor(a, lower=True)
        mean = cho_solve((factor, lower), xo.T @ z)
        residual = z - xo @ mean
        shape = (len(y) + 1e-3) / 2
        rate = (float(residual @ residual + np.dot(precision * mean, mean)) + 1e-3) / 2
        sigma2 = rate / rng.gamma(shape)
        perturbation = solve_triangular(np.tril(factor).T, rng.normal(size=x.shape[1]), lower=False)
        draw = mean + np.sqrt(sigma2) * perturbation
        donor_prediction, recipient_prediction = xo @ mean, xm @ draw
        # Shuffle donor order so tied scores do not systematically favor early records.
        order = rng.permutation(len(y))
        tree = cKDTree(donor_prediction[order, None])
        k = min(self.donors, len(y))
        _, neighbors = tree.query(recipient_prediction[:, None], k=k)
        neighbors = np.asarray(neighbors).reshape(len(xm), k)
        selected = neighbors[np.arange(len(xm)), rng.integers(k, size=len(xm))]
        return y[order[selected]]

    def _categorical(self, design, observed, y, rng):
        classes = np.unique(y)
        if len(classes) == 1:
            return np.full((~observed).sum(), classes[0]), 0
        # Ordinary bootstrap, not class balancing: preserve the observed prevalence.
        sampled = rng.integers(len(y), size=len(y))
        boot_y = y[sampled]
        if np.unique(boot_y).size == 1:
            return np.full((~observed).sum(), boot_y[0]), 0
        model = LogisticRegression(C=self.logistic_C, solver='lbfgs',
                                   max_iter=self.logistic_max_iter, tol=1e-4)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', ConvergenceWarning)
            model.fit(design[observed][sampled], boot_y)
        failures = sum(issubclass(w.category, ConvergenceWarning) for w in caught)
        probabilities = model.predict_proba(design[~observed])
        positions = (rng.random(len(probabilities))[:, None] > probabilities.cumsum(axis=1)).sum(axis=1)
        return model.classes_[np.minimum(positions, len(model.classes_)-1)], failures

    def fit_impute(self, frame):
        """Takes ONLY a masked DataFrame, never ground truth."""
        self.schema.validate(frame)
        if not frame.index.equals(pd.RangeIndex(len(frame))):
            raise ValueError("Use a RangeIndex (reset_index(drop=True)); retain IDs in a column.")
        start = time.perf_counter()
        columns = self.schema.model_columns(frame)
        original = frame[columns].copy()
        # Model-space category labels are strings; restore original label types on export.
        decoders = {c: {str(v): v for v in frame[c].dropna().unique()}
                    for c in columns if c in self.schema.categorical}
        for c in decoders:
            original[c] = original[c].map(lambda v: str(v) if pd.notna(v) else np.nan).astype(object)
        masks = original.isna()
        targets = sorted([c for c in self.schema.targets if masks[c].any()],
                         key=lambda c: masks[c].sum())
        predictors = {c: self.schema.predictors(frame, c) for c in targets}
        records, imputations = [], []
        seeds = np.random.SeedSequence(self.seed).spawn(self.n_imputations)
        with threadpool_limits(limits=self.threads):
            for chain, child in enumerate(seeds, 1):
                rng = np.random.default_rng(child)
                work = original.copy()
                for c in targets:
                    observed_values = original.loc[~masks[c], c].to_numpy()
                    work.loc[masks[c], c] = rng.choice(observed_values, size=masks[c].sum())
                for iteration in range(1, self.n_iterations+1):
                    for c in targets:
                        observed = ~masks[c].to_numpy()
                        design = self._design(work, predictors[c])
                        y = original.loc[observed, c].to_numpy()
                        if c in self.schema.categorical:
                            values, failures = self._categorical(design, observed, y.astype(str), rng)
                        else:
                            values = self._pmm(design, observed, y.astype(float), rng)
                            failures = 0
                        work.loc[~observed, c] = values
                        row = dict(chain=chain, iteration=iteration, target=c,
                                   convergence_warnings=failures)
                        if c in self.schema.categorical:
                            shares = pd.Series(values).value_counts(normalize=True)
                            for label in decoders[c]:
                                records.append({**row, 'statistic': f'share:{label}',
                                                'value': float(shares.get(label, 0))})
                        else:
                            records.extend([{**row, 'statistic': 'mean', 'value': float(np.mean(values))},
                                            {**row, 'statistic': 'std', 'value': float(np.std(values))}])
                completed = frame.copy()
                for c in targets:
                    values = work.loc[masks[c], c]
                    if c in decoders:
                        values = values.map(decoders[c])
                    completed.loc[masks[c], c] = values.to_numpy()
                completed = apply_derived(completed, self.schema)
                verify_completion(frame, completed, self.schema)
                imputations.append(completed)
                if self.verbose:
                    print(f'  Chain {chain}/{self.n_imputations} complete '
                          f'({time.perf_counter()-start:.1f} seconds elapsed)', flush=True)
        self.imputations_ = imputations
        self.trace_ = pd.DataFrame(records, columns=['chain', 'iteration', 'target',
                                                   'convergence_warnings', 'statistic', 'value'])
        self.elapsed_seconds_ = time.perf_counter()-start
        self.predictors_ = predictors
        return imputations


def verify_completion(original, completed, schema):
    if not completed.columns.equals(original.columns) or not completed.index.equals(original.index):
        raise AssertionError("Completion changed the row/column structure.")
    for c in original:
        seen = original[c].notna()
        if not original.loc[seen, c].eq(completed.loc[seen, c]).all():
            raise AssertionError(f"Observed values changed in {c}")
    for c in schema.targets + list(schema.derived):
        if completed[c].isna().any():
            raise AssertionError(f"Missing values remain in {c}")
    for c in schema.targets:
        if c in schema.categorical and not completed[c].isin(original[c].dropna().unique()).all():
            raise AssertionError(f"Invalid generated category in {c}")
    return True


def aggregate_imputations(original, imputations, schema, numeric_reducer="median"):
    """Pool stochastic MICE draws using numeric median (MAE) or mean; mode for categories.

    Median is the point-estimate choice under absolute-error loss. Mean is retained
    as a legitimate sensitivity analysis, calculated from the same imputations.
    """
    if numeric_reducer not in ("mean", "median"):
        raise ValueError("numeric_reducer must be mean or median")
    out = original.copy()
    for c in schema.targets:
        missing = original[c].isna()
        if not missing.any():
            continue
        stack = pd.concat([d.loc[missing, c].reset_index(drop=True) for d in imputations], axis=1)
        values = stack.mode(axis=1)[0] if c in schema.categorical else (stack.median(axis=1) if numeric_reducer == "median" else stack.mean(axis=1))
        # Numeric pooled estimates may be fractional even for integer-valued PMM donors.
        if c not in schema.categorical:
            out[c] = out[c].astype(float)
        out.loc[missing, c] = values.to_numpy()
    out = apply_derived(out, schema)
    verify_completion(original, out, schema)
    return out


def eda_table(frame, schema):
    rows = []
    for c in frame:
        seen = frame[c].dropna()
        numeric = c not in schema.categorical and pd.api.types.is_numeric_dtype(frame[c])
        rows.append(dict(attribute=c, dtype=str(frame[c].dtype), missing=int(frame[c].isna().sum()),
                         missing_percent=100*frame[c].isna().mean(), unique=seen.nunique(),
                         target=c in schema.targets, derived=c in schema.derived,
                         excluded=c in schema.exclude, constant=seen.nunique() <= 1,
                         minimum=seen.min() if numeric and len(seen) else np.nan,
                         maximum=seen.max() if numeric and len(seen) else np.nan,
                         median=seen.median() if numeric and len(seen) else np.nan,
                         most_frequent=str(seen.mode().iloc[0]) if len(seen) else None,
                         dominant_share=seen.value_counts(normalize=True).iloc[0] if len(seen) else np.nan))
    return pd.DataFrame(rows)


def association_table(frame, schema):
    """Pairwise observed-data associations; no truth or initialized values used."""
    columns = schema.model_columns(frame)
    rows = []
    for i, a in enumerate(columns):
        for b in columns[i+1:]:
            pair = frame[[a, b]].dropna()
            ac, bc = a in schema.categorical, b in schema.categorical
            score = np.nan
            if not ac and not bc:
                kind = 'Spearman rho (signed)'
                if pair[a].nunique()>1 and pair[b].nunique()>1:
                    score = pair[a].corr(pair[b], method='spearman')
            elif ac and bc:
                kind = 'Cramers V (bias corrected)'
                ct = pd.crosstab(pair[a], pair[b])
                n = ct.to_numpy().sum()
                r, k = ct.shape
                if min(r, k)>1 and n>1:
                    phi2 = chi2_contingency(ct, correction=False)[0]/n
                    den = min(r-(r-1)**2/(n-1), k-(k-1)**2/(n-1))-1
                    score = np.sqrt(max(0, phi2-(r-1)*(k-1)/(n-1))/den) if den>0 else np.nan
            else:
                kind = 'Correlation ratio eta'
                cat, num = (a, b) if ac else (b, a)
                values = pair[num].astype(float)
                total = ((values-values.mean())**2).sum()
                if total>0:
                    groups = pair.groupby(cat, observed=True)[num].agg(['count', 'mean'])
                    score = np.sqrt((groups['count']*(groups['mean']-values.mean())**2).sum()/total)
            rows.append(dict(attribute_1=a, attribute_2=b, measure=kind,
                             association=score, paired_observations=len(pair)))
    return pd.DataFrame(rows, columns=['attribute_1', 'attribute_2', 'measure',
                                        'association', 'paired_observations'])


def measurement_audit(measured, truth, schema):
    """Diagnostic only: quantify noise in visible values without correcting it."""
    rows = []
    for c in schema.targets:
        seen = measured[c].notna()
        a, b = measured.loc[seen, c], truth.loc[seen, c]
        if b.isna().any():
            raise ValueError(f"Ground truth is incomplete for {c}")
        changed = (a.astype(str) != b.astype(str)) if c in schema.categorical else ~np.isclose(
            a.astype(float), b.astype(float), rtol=0, atol=1e-10)
        rows.append(dict(target=c, visible_count=int(seen.sum()),
                         changed_visible_share=float(np.mean(changed)) if len(a) else np.nan,
                         visible_MAE=float(np.mean(np.abs(a-b))) if c not in schema.categorical else np.nan))
    return pd.DataFrame(rows)


def benchmark(measured, truth, completions, schema, scenario):
    """Score ONLY originally missing target cells, per method and per attribute."""
    rows = []
    for c in schema.targets + list(schema.derived):
        missing = measured[c].isna()
        if not missing.any():
            continue
        if truth.loc[missing, c].isna().any():
            raise ValueError(f"Truth is missing evaluation values for {c}")
        category = c in schema.categorical
        y = truth.loc[missing, c]
        for method, complete in completions.items():
            p = complete.loc[missing, c]
            row = dict(scenario=scenario, target=c, method=method, n=int(missing.sum()),
                       kind='categorical' if category else 'numeric', derived=c in schema.derived)
            if category:
                yt, yp = y.astype(str), p.astype(str)
                labels = sorted(set(yt) | set(yp))
                matrix = confusion_matrix(yt, yp, labels=labels)
                support = matrix.sum(axis=1)
                recall = np.divide(matrix.diagonal(), support, out=np.zeros(len(labels)), where=support>0)
                row.update(accuracy=accuracy_score(yt, yp),
                           balanced_accuracy=float(recall[support>0].mean()),
                           macro_F1=f1_score(yt, yp, labels=labels, average='macro', zero_division=0),
                           error=1-accuracy_score(yt, yp))
            else:
                yt, yp = y.to_numpy(float), p.to_numpy(float)
                residual = yp-yt
                variance = np.sum((yt-yt.mean())**2)
                row.update(MAE=float(np.abs(residual).mean()), RMSE=float(np.sqrt(np.mean(residual**2))),
                           R2=float(1-np.sum(residual**2)/variance) if variance>0 else np.nan,
                           bias=float(residual.mean()),
                           spread_ratio=float(np.std(yp)/np.std(yt)) if np.std(yt)>0 else np.nan,
                           wasserstein=float(wasserstein_distance(yt, yp)), error=float(np.abs(residual).mean()))
            rows.append(row)
    scores = pd.DataFrame(rows)
    if scores.empty:
        return scores
    base = scores.loc[scores.method=='Median/mode', ['target', 'error']].set_index('target')['error']
    scores['baseline_error'] = scores.target.map(base)
    scores['skill'] = np.where(scores.baseline_error>0, 1-scores.error/scores.baseline_error, np.nan)
    return scores


def uncertainty_table(measured, imputations, schema, truth=None, interval=(0.1, 0.9)):
    """Cell-level empirical spread/agreement; these are not calibrated confidence intervals."""
    rows = []
    for c in schema.targets:
        indices = measured.index[measured[c].isna()]
        if not len(indices):
            continue
        values = np.column_stack([d.loc[indices, c].to_numpy() for d in imputations])
        if c in schema.categorical:
            for index, draws in zip(indices, values):
                counts = Counter(draws)
                label, count = counts.most_common(1)[0]
                row = dict(row_index=int(index), target=c, modal_value=label,
                           agreement=count/len(draws), distinct_draws=len(counts))
                if truth is not None:
                    row['truth_in_draws'] = truth.loc[index, c] in counts
                rows.append(row)
        else:
            array = values.astype(float)
            low, high = np.quantile(array, interval, axis=1)
            for j, index in enumerate(indices):
                row = dict(row_index=int(index), target=c, mean=float(array[j].mean()),
                           draw_std=float(array[j].std(ddof=1)) if len(imputations)>1 else np.nan,
                           lower=float(low[j]), upper=float(high[j]))
                if truth is not None:
                    row['truth_in_draw_range'] = bool(low[j]<=truth.loc[index, c]<=high[j])
                rows.append(row)
    return pd.DataFrame(rows)


def paired_bootstrap(measured, truth, baseline, point, schema, seed=219, draws=500):
    """Descriptive paired row bootstrap, conditional on already fitted imputations."""
    rng = np.random.default_rng(seed)
    rows = []
    for c in schema.targets:
        missing = measured[c].isna()
        if not missing.any():
            continue
        y, b, p = truth.loc[missing, c], baseline.loc[missing, c], point.loc[missing, c]
        if c in schema.categorical:
            delta = (p.astype(str)!=y.astype(str)).to_numpy(float)-(b.astype(str)!=y.astype(str)).to_numpy(float)
        else:
            delta = np.abs(p.to_numpy(float)-y.to_numpy(float))-np.abs(b.to_numpy(float)-y.to_numpy(float))
        estimates = [rng.choice(delta, size=len(delta), replace=True).mean() for _ in range(draws)]
        low, high = np.quantile(estimates, [0.025, 0.975])
        rows.append(dict(target=c, loss='error rate' if c in schema.categorical else 'absolute error',
                         point_minus_baseline=float(delta.mean()), lower_95=low, upper_95=high,
                         interpretation='lower loss' if high<0 else 'higher loss' if low>0 else 'not separated'))
    return pd.DataFrame(rows)


def observed_holdout(frame, schema, fraction=0.1, seed=219):
    """Hide additional observed targets BEFORE fitting; report against these observed labels.

    At nonzero noise this tests reconstruction of noisy measurements, not clean truth.
    """
    if not 0<fraction<0.5:
        raise ValueError("Use a holdout fraction between 0 and 0.5.")
    rng = np.random.default_rng(seed)
    hidden = frame.copy()
    masks = pd.DataFrame(False, index=frame.index, columns=schema.targets)
    for c in schema.targets:
        indices = frame.index[frame[c].notna()].to_numpy()
        n = min(max(1, int(len(indices)*fraction)), max(0, len(indices)-2))
        chosen = rng.choice(indices, size=n, replace=False)
        masks.loc[chosen, c] = True
        hidden.loc[chosen, c] = np.nan
    for dest, rule in schema.derived.items():
        if rule['source'] in masks:
            hidden.loc[masks[rule['source']], dest] = np.nan
    return hidden, masks


def score_holdout(reference, holdout_masks, predictions, schema):
    # A score-only proxy mask prevents scoring pre-existing missing cells.
    proxy = reference.copy()
    for c in schema.targets:
        # Existing missing entries are outside this holdout experiment. Fill only the
        # scoring proxy, never the actual fitting table or the reference labels.
        proxy[c] = proxy[c].fillna(reference[c].dropna().iloc[0])
        proxy.loc[holdout_masks[c], c] = np.nan
    score_schema = Schema(schema.targets, schema.categorical, schema.exclude)
    return benchmark(proxy, reference, predictions, score_schema, 'observed-label holdout')


def environment_manifest():
    from importlib.metadata import version
    packages = ['numpy', 'pandas', 'scipy', 'scikit-learn', 'matplotlib', 'nbformat', 'nbclient', 'ipykernel']
    return {'python': platform.python_version(), 'packages': {p: version(p) for p in packages}}


def save_manifest(path, config, input_audit, schemas):
    payload = {'environment': environment_manifest(), 'config': config,
               'inputs': input_audit.to_dict('records'),
               'schemas': {name: vars(schema) for name, schema in schemas.items()}}
    Path(path).write_text(json.dumps(payload, indent=2, default=str), encoding='utf-8')
