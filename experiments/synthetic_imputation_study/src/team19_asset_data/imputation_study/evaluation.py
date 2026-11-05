"""Scoring against truth. This module runs after fitting and feeds nothing back to models."""

from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from .columns import (
    ABSOLUTE, BY_NAME, CATEGORY_TARGET, COLUMNS, COMPANIONS, CONTROL, GAS_READINGS, NUMERIC_TARGETS, TARGETS,
)
from .models import NUMERIC_METHODS
from .runner import ALL_NUMERIC, ITERATIVE, PROBABILITY_COLUMNS

KEYS = ["scenario", "experiment", "variant", "seed", "target", "method"]
BASELINES = ("Median", "Mode")


def attach_truth(predictions, truths):
    """Join truth to prediction rows. Each row is one originally missing cell."""
    parts = []
    for scenario, block in predictions.groupby("scenario", sort=False):
        block = block.copy()
        block["truth"] = np.nan
        for target, index in block.groupby("target").groups.items():
            ids = block.loc[index, "row_id"].to_numpy()
            block.loc[index, "truth"] = truths[scenario].loc[ids, target].to_numpy()
        parts.append(block)
    result = pd.concat(parts, ignore_index=True)
    assert not result.duplicated(KEYS + ["row_id"]).any(), "a cell was predicted twice"
    return result


def numeric_scores(name, truth, prediction):
    """Error metrics for one numeric column. Failed predictions count as misses."""
    column = BY_NAME[name]
    y, p = np.asarray(truth, float), np.asarray(prediction, float)
    valid = np.isfinite(p)
    error = p[valid] - y[valid]
    absolute = np.abs(error)
    variance = float(np.sum((y[valid] - y[valid].mean()) ** 2)) if valid.any() else 0.0
    if column.tolerance_kind == ABSOLUTE:
        eligible = np.ones(len(y), dtype=bool)
        scaled = np.abs(p - y)
    else:
        # A percentage of a true zero is undefined, so those cells leave the denominator.
        eligible = y != 0
        scaled = np.abs(p - y) / np.where(eligible, np.abs(y), 1.0)
    tight, loose = column.tolerance
    scores = {
        "n": len(y), "coverage": valid.mean() if len(y) else np.nan,
        "MAE": absolute.mean() if valid.any() else np.nan,
        "RMSE": np.sqrt(np.mean(error**2)) if valid.any() else np.nan,
        "bias": error.mean() if valid.any() else np.nan,
        "R2": 1 - np.sum(error**2) / variance if variance > 0 else np.nan,
        "median error": np.median(absolute) if valid.any() else np.nan,
        "p90 error": np.quantile(absolute, 0.9) if valid.any() else np.nan,
        "within tight": np.mean(scaled[eligible] <= tight) if eligible.any() else np.nan,
        "within loose": np.mean(scaled[eligible] <= loose) if eligible.any() else np.nan,
        "tolerance cells": int(eligible.sum()),
    }
    if column.integer:
        scores["exact"] = float(np.mean(p == y))
    return scores


def class_scores(truth, prediction):
    y, p = np.asarray(truth, float), np.asarray(prediction, float)
    classes = np.unique(y)
    recall = [np.mean(p[y == c] == c) for c in classes]
    f1 = []
    for c in classes:
        hits = np.sum((p == c) & (y == c))
        denominator = np.sum(y == c) + np.sum(p == c)
        f1.append(2 * hits / denominator if denominator else 0.0)
    return {
        "n": len(y), "coverage": float(np.isfinite(p).mean()), "accuracy": float(np.mean(p == y)),
        "macro F1": float(np.mean(f1)), "balanced accuracy": float(np.mean(recall)),
    }


def score_tables(scored):
    """Per-experiment scores. Skill is the share of the median's error removed."""
    numeric, category = [], []
    for key, block in scored.groupby(KEYS, sort=False):
        entry = dict(zip(KEYS, key))
        if entry["target"] == CATEGORY_TARGET:
            category.append({**entry, **class_scores(block.truth, block.prediction)})
        else:
            numeric.append({**entry, **numeric_scores(entry["target"], block.truth, block.prediction)})
    numeric, category = pd.DataFrame(numeric), pd.DataFrame(category)
    baseline = numeric[numeric.method.eq("Median") & numeric.variant.eq("all predictors")]
    baseline = baseline[["scenario", "seed", "target", "MAE"]].rename(columns={"MAE": "median MAE"})
    numeric = numeric.merge(baseline, on=["scenario", "seed", "target"], how="left")
    numeric["skill"] = 1 - numeric.MAE / numeric["median MAE"]
    return numeric, category


def add_scenario_levels(table, scenarios):
    table = table.copy()
    table["completion"] = table.scenario.map(lambda s: scenarios[s].completion)
    table["noise"] = table.scenario.map(lambda s: scenarios[s].noise)
    return table


def _bootstrap(values, draws, seed):
    values = np.asarray(values, float)
    if len(values) < 2:
        return np.nan, np.nan
    generator = np.random.default_rng(seed)
    means = np.concatenate([
        values[generator.integers(0, len(values), size=(min(100, draws - start), len(values)))].mean(axis=1)
        for start in range(0, draws, 100)
    ])
    return tuple(np.quantile(means, [0.025, 0.975]))


def _cell_loss(block):
    """Per-cell loss: absolute error, or a miss indicator for the nominal target."""
    if block.target.iloc[0] == CATEGORY_TARGET:
        return (block.prediction != block.truth).astype(float)
    return (block.prediction - block.truth).abs()


def paired_intervals(scored, config):
    """Row bootstrap of each method against the baseline, and of the two best methods.

    The interval is conditional on the fitted models. It does not include
    refitting or the random generation of the dataset.
    """
    rows = []
    main = scored[scored.experiment.eq("main")]
    for (scenario, target), block in main.groupby(["scenario", "target"], sort=False):
        categorical = target == CATEGORY_TARGET
        loss = block.assign(loss=_cell_loss(block)).pivot(index="row_id", columns="method", values="loss")
        baseline = "Mode" if categorical else "Median"
        ranked = sorted((m for m in loss if m != baseline), key=lambda m: loss[m].mean())
        pairs = [(baseline, m, "versus baseline") for m in ranked] + [(ranked[0], ranked[1], "best two")]
        for first, second, kind in pairs:
            difference = (loss[second] - loss[first]).to_numpy()
            low, high = _bootstrap(difference, config.bootstrap_draws, config.seed)
            rows.append({
                "scenario": scenario, "target": target, "first": first, "second": second, "comparison": kind,
                "first error": loss[first].mean(), "second error": loss[second].mean(),
                "difference": difference.mean(), "low": low, "high": high, "n": len(loss),
                "metric": "error rate" if categorical else "MAE",
            })
    return pd.DataFrame(rows)


def scenario_effects(scored, scenarios, config):
    """Effect of each control on the same cells.

    All scenarios share one truth and nested masks, so a change between two
    scenarios is measured on identical cells and comes from that control alone.
    For completeness, the cells are those hidden at both levels.
    """
    main = scored[scored.experiment.eq("main") & ~scored.method.isin(BASELINES)]
    items = list(scenarios.values())
    comparisons = []
    for a in items:
        for b in items:
            if a.completion == b.completion and a.noise < b.noise:
                comparisons.append(("noise", a, b))
            if a.noise == b.noise and a.completion < b.completion:
                comparisons.append(("completeness", a, b))
    rows = []
    tables = {name: block for name, block in main.groupby("scenario", sort=False)}
    for control, a, b in comparisons:
        if a.name not in tables or b.name not in tables:
            continue
        first = tables[a.name].assign(loss=_loss_any)[["target", "method", "row_id", "loss"]]
        second = tables[b.name].assign(loss=_loss_any)[["target", "method", "row_id", "loss"]]
        joined = first.merge(second, on=["target", "method", "row_id"], suffixes=("_a", "_b"))
        for (target, method), block in joined.groupby(["target", "method"], sort=False):
            difference = (block.loss_b - block.loss_a).to_numpy()
            low, high = _bootstrap(difference, config.bootstrap_draws, config.seed)
            before = block.loss_a.mean()
            rows.append({
                "control": control, "held completion": a.completion if control == "noise" else np.nan,
                "held noise": a.noise if control == "completeness" else np.nan,
                "from": a.noise if control == "noise" else a.completion,
                "to": b.noise if control == "noise" else b.completion,
                "target": target, "method": method, "n": len(block), "error from": before,
                "error to": block.loss_b.mean(), "relative change": difference.mean() / before if before else np.nan,
                "difference": difference.mean(), "low": low, "high": high,
            })
    return pd.DataFrame(rows)


def effect_matrix(effects, control, held_column, held_value):
    """Relative MAE change from the lowest to the highest level, with clear effects marked."""
    block = effects[effects.control.eq(control) & effects[held_column].eq(held_value)
                    & effects.target.ne(CATEGORY_TARGET)]
    block = block[block["from"].eq(block["from"].min()) & block.to.eq(block.to.max())]
    methods = [m for m in NUMERIC_METHODS if m in set(block.method)]
    matrix = block.pivot(index="target", columns="method", values="relative change")
    matrix = matrix.reindex(list(NUMERIC_TARGETS))[methods]
    clear = block.assign(clear=~((block.low <= 0) & (block.high >= 0)))
    marks = clear.pivot(index="target", columns="method", values="clear").reindex(list(NUMERIC_TARGETS))[methods]
    return matrix, marks.fillna(False).to_numpy(dtype=bool)


def _loss_any(frame):
    """Row-wise loss for a table that mixes numeric targets and model_id."""
    miss = (frame.prediction != frame.truth).astype(float)
    return (frame.prediction - frame.truth).abs().where(frame.target.ne(CATEGORY_TARGET), miss)


def visible_group(target, frame):
    """Label each cell by which of the target's generator companions could be seen."""
    companions = COMPANIONS[target]
    if not companions:
        return np.repeat("all cells", len(frame))
    parts = [np.where(frame[c].notna(), f"{c} seen", f"{c} missing") for c in companions]
    return np.array([" · ".join(items) for items in zip(*parts)])


def group_table(scored, measured, config):
    """Error by which inputs a cell could see, using measured data only to form groups."""
    rows = []
    selected = scored[scored.experiment.eq("main") & scored.scenario.isin(config.noise_series)]
    for (scenario, target, method), block in selected.groupby(["scenario", "target", "method"], sort=False):
        labels = visible_group(target, measured[scenario].loc[block.row_id])
        for group, values in _cell_loss(block).groupby(labels):
            rows.append({
                "scenario": scenario, "target": target, "method": method, "group": group,
                "n": len(values), "error": values.mean(), "small group": len(values) < config.small_group,
            })
    table = pd.DataFrame(rows)
    baseline = table[table.method.isin(BASELINES)][["scenario", "target", "group", "error"]]
    table = table.merge(baseline.rename(columns={"error": "baseline error"}), on=["scenario", "target", "group"])
    table["skill"] = 1 - table.error / table["baseline error"]
    return table


def tail_table(scored, config):
    """Error by true value, after the fact: this explains failures and sets no rule."""
    rows = []
    selected = scored[scored.experiment.eq("main") & scored.scenario.eq(config.representative)
                      & scored.target.ne(CATEGORY_TARGET)]
    for (target, method), block in selected.groupby(["target", "method"], sort=False):
        y = block.truth
        if BY_NAME[target].zero_inflated:
            middle = y[y > 0].median()
            bands = {"true zero": y.eq(0), "positive, lower half": y.gt(0) & y.le(middle),
                     "positive, upper half": y.gt(middle)}
        else:
            low, high = y.quantile([0.1, 0.9])
            bands = {"lowest 10%": y.le(low), "middle 80%": y.gt(low) & y.lt(high), "highest 10%": y.ge(high)}
        for band, mask in bands.items():
            part = block[mask]
            rows.append({
                "target": target, "method": method, "band": band, "n": len(part),
                "MAE": (part.prediction - part.truth).abs().mean(), "bias": (part.prediction - part.truth).mean(),
            })
    return pd.DataFrame(rows)


def _distribution(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return {
        "Wasserstein": wasserstein_distance(y, p), "mean shift": p.mean() - y.mean(),
        "SD ratio": p.std() / y.std() if y.std() > 0 else np.nan,
        "q10 shift": np.quantile(p, 0.1) - np.quantile(y, 0.1),
        "q50 shift": np.quantile(p, 0.5) - np.quantile(y, 0.5),
        "q90 shift": np.quantile(p, 0.9) - np.quantile(y, 0.9),
    }


def distribution_table(scored, draws, measured, truth, config):
    """Whether imputed values keep the shape of the truth.

    Repeated imputations are judged one completed dataset at a time and then
    averaged. Averaging the draws first would remove the spread they exist to keep.
    """
    rows = []

    def add(target, method, row_ids, values):
        results = {"missing cells": _distribution(truth.loc[row_ids, target], values)}
        filled = measured[target].copy()
        filled.loc[row_ids] = values
        results["completed column"] = _distribution(truth[target], filled)
        return {scope: {"target": target, "method": method, "scope": scope, **result}
                for scope, result in results.items()}

    point = scored[scored.scenario.eq(config.representative) & scored.experiment.eq("main")
                   & scored.target.ne(CATEGORY_TARGET)]
    for (target, method), block in point.groupby(["target", "method"], sort=False):
        rows.extend(add(target, method, block.row_id.to_numpy(), block.prediction.to_numpy()).values())
    numeric_draws = draws[draws.target.ne(CATEGORY_TARGET)]
    for (target, method), block in numeric_draws.groupby(["target", "method"], sort=False):
        per_draw = [add(target, method, part.row_id.to_numpy(), part.value.to_numpy())
                    for _, part in block.groupby("draw")]
        for scope in ("missing cells", "completed column"):
            frame = pd.DataFrame([item[scope] for item in per_draw])
            rows.append({"target": target, "method": f"{method} (per draw)", "scope": scope,
                         **frame.drop(columns=["target", "method", "scope"]).mean().to_dict()})
    return pd.DataFrame(rows)


def class_share_table(scored, config):
    """Predicted class shares against the truth on missing model_id cells."""
    rows = []
    selected = scored[scored.scenario.eq(config.representative) & scored.experiment.isin(["main", "joint"])
                      & scored.target.eq(CATEGORY_TARGET)]
    for method, block in selected.groupby("method", sort=False):
        truth = block.truth.value_counts(normalize=True)
        predicted = block.prediction.value_counts(normalize=True)
        levels = truth.index.union(predicted.index)
        gap = predicted.reindex(levels, fill_value=0) - truth.reindex(levels, fill_value=0)
        rows.append({"method": method, "total variation": 0.5 * gap.abs().sum(),
                     **{f"share gap {int(k)}": v for k, v in gap.sort_index().items()}})
    return pd.DataFrame(rows)


def range_table(scored):
    """How often raw predictions fell outside a column's valid range."""
    rows = []
    selected = scored[scored.experiment.eq("main") & scored.target.ne(CATEGORY_TARGET)]
    for (target, method), block in selected.groupby(["target", "method"], sort=False):
        column = BY_NAME[target]
        low = -np.inf if column.lower is None else column.lower
        high = np.inf if column.upper is None else column.upper
        rows.append({"target": target, "method": method, "n": len(block),
                     "outside valid range": float(((block.raw < low) | (block.raw > high)).mean())})
    return pd.DataFrame(rows)


def _complete(measured, cells):
    frame = measured.copy()
    for target, block in cells.groupby("target"):
        frame.loc[block.row_id.to_numpy(), target] = block.value.to_numpy()
    return frame


def coherence_table(scored, draws, measured, config):
    """Whether completed rows still obey the generator's definitions.

    Single-target methods fill each cell on its own, so a completed row can
    contradict itself even when every cell looks accurate. The reference column
    is the residual in rows with nothing imputed, which is what noise alone gives.
    """
    sources = {}
    selected = scored[scored.scenario.eq(config.representative) & scored.target.ne(CATEGORY_TARGET)]
    point = selected[selected.experiment.eq("main")]
    for method in NUMERIC_METHODS:
        block = point[point.method.eq(method)]
        sources[method] = block[["row_id", "target"]].assign(value=block.prediction.to_numpy())
    joint = selected[selected.experiment.eq("joint")]
    if len(joint):
        sources["Joint miceforest PMM (mean of draws)"] = joint[["row_id", "target"]].assign(value=joint.prediction.to_numpy())
    if len(draws):
        first = draws[draws.method.eq("Joint miceforest PMM") & draws.draw.eq(0) & draws.target.ne(CATEGORY_TARGET)]
        sources["Joint miceforest PMM (one draw)"] = first[["row_id", "target", "value"]]
    relations = {
        "altitude = x + y + 100": (("x", "y", "altitude"), lambda f: (f.altitude - f.x - f.y - 100).abs()),
        "overgrowth = max(2 humid - salt, 0)": (
            ("measured_humid", "measured_salt", "measured_overgrowth"),
            lambda f: (f.measured_overgrowth - (2 * f.measured_humid - f.measured_salt).clip(lower=0)).abs()),
        "total_gas = gas a + b + c": (
            GAS_READINGS + ("total_gas",),
            lambda f: (f.total_gas - f.measured_gas_a - f.measured_gas_b - f.measured_gas_c).abs()),
    }
    rows = []
    for relation, (fields, residual) in relations.items():
        involved = measured[list(fields)].isna().any(axis=1)
        reference = residual(measured[list(fields)].dropna())
        for method, cells in sources.items():
            frame = _complete(measured, cells)
            values = residual(frame.loc[involved, list(fields)].dropna())
            rows.append({"relation": relation, "method": method, "rows": len(values),
                         "mean residual": values.mean(), "residual in fully observed rows": reference.mean()})
    return pd.DataFrame(rows)


def draw_table(scored, draws, config):
    """Spread of repeated imputations and whether the truth falls inside it."""
    rows = []
    selected = scored[scored.experiment.isin(["pmm", "joint"])]
    for (target, method), block in selected.groupby(["target", "method"], sort=False):
        part = draws[draws.target.eq(target) & draws.method.eq(method)]
        wide = part.pivot(index="row_id", columns="draw", values="value").reindex(block.row_id)
        y = block.set_index("row_id").truth
        entry = {"target": target, "method": method, "n": len(block), "draws": wide.shape[1],
                 "distinct draws per cell": wide.nunique(axis=1).mean()}
        if target == CATEGORY_TARGET:
            entry["truth among draws"] = wide.eq(y, axis=0).any(axis=1).mean()
        else:
            low, high = wide.quantile(config.interval[0], axis=1), wide.quantile(config.interval[1], axis=1)
            entry["truth in 80% band"] = (y.ge(low) & y.le(high)).mean()
            entry["mean band width"] = (high - low).mean()
        rows.append(entry)
    return pd.DataFrame(rows)


def probability_table(scored, config):
    """Quality of model_id probabilities, including the calibrated forest."""
    rows = []
    selected = scored[scored.target.eq(CATEGORY_TARGET) & scored.scenario.eq(config.representative)
                      & scored.seed.eq(config.seed) & scored.prob_0.notna()
                      & scored.experiment.isin(["main", "calibration", "ablation"])]
    selected = selected[selected.variant.isin(["all predictors", "without gas columns"])]
    for (variant, method), block in selected.groupby(["variant", "method"], sort=False):
        probabilities = block[PROBABILITY_COLUMNS].to_numpy()
        truth = block.truth.to_numpy().astype(int)
        chosen = probabilities[np.arange(len(block)), truth]
        confidence = probabilities.max(axis=1)
        correct = probabilities.argmax(axis=1) == truth
        onehot = np.eye(probabilities.shape[1])[truth]
        bins = np.minimum((confidence * 10).astype(int), 9)
        calibration_gap = sum(
            (bins == b).mean() * abs(correct[bins == b].mean() - confidence[bins == b].mean())
            for b in np.unique(bins)
        )
        ordered = np.argsort(-confidence, kind="stable")
        rows.append({
            "predictors": variant, "method": method, "n": len(block), "accuracy": correct.mean(),
            "log loss": -np.log(np.maximum(chosen, 1e-12)).mean(),
            "Brier": ((probabilities - onehot) ** 2).sum(axis=1).mean(),
            "calibration gap": calibration_gap,
            "accuracy of most confident 50%": correct[ordered[: len(block) // 2]].mean(),
        })
    return pd.DataFrame(rows).sort_values(["predictors", "log loss"])


def budget_table(numeric, log, config):
    """IterativeImputer MAE at low iteration limits against the converged main run."""
    limit = config.parameters[ITERATIVE]["max_iter"]
    main = numeric[numeric.experiment.eq("main") & numeric.method.eq(ITERATIVE)
                   & numeric.scenario.eq(config.representative)].assign(limit=limit)
    low = numeric[numeric.experiment.eq("budget")].assign(limit=lambda f: f.variant.astype(int))
    table = pd.concat([low, main]).pivot(index="target", columns="limit", values="MAE")
    table = table.reindex(list(NUMERIC_TARGETS))
    table.columns = [f"MAE at limit {c}" for c in table.columns]
    runs = log[log.method.eq(ITERATIVE) & log.experiment.eq("main")]
    used = runs.groupby("scenario").agg(
        **{"fits": ("task_id", "size"), "fewest iterations": ("n_iter", "min"), "most iterations": ("n_iter", "max"),
           "fits stopped at the limit": ("n_iter", lambda s: int((s >= limit).sum()))}
    ).reset_index()
    return table, used


def default_comparison(numeric, category, config):
    """MAE or error rate with the earlier settings against the tuned settings, on the same cells."""
    keys = ["target", "method"]
    selected = numeric[numeric.scenario.eq(config.representative) & numeric.seed.eq(config.seed)]
    tuned = selected[selected.experiment.eq("main")][keys + ["MAE"]].rename(columns={"MAE": "tuned"})
    default = selected[selected.experiment.eq("defaults")][keys + ["MAE"]].rename(columns={"MAE": "default"})
    table = default.merge(tuned, on=keys)
    labels = category[category.scenario.eq(config.representative) & category.seed.eq(config.seed)]
    tuned = labels[labels.experiment.eq("main")].assign(tuned=lambda f: 1 - f.accuracy)[keys + ["tuned"]]
    default = labels[labels.experiment.eq("defaults")].assign(default=lambda f: 1 - f.accuracy)[keys + ["default"]]
    table = pd.concat([table, default.merge(tuned, on=keys)], ignore_index=True)
    table["relative change"] = np.where(table.default > 0, table.tuned / table.default.where(table.default > 0) - 1, 0.0)
    return table


def stability_table(numeric, category, config):
    """Variation across fold seeds. The same rows are reused, so these are not new samples."""
    selected = numeric[numeric.scenario.eq(config.representative) & numeric.experiment.isin(["main", "stability"])].copy()
    selected["rank"] = selected.groupby(["target", "seed"]).MAE.rank(method="min")
    table = selected.groupby(["target", "method"], sort=False).agg(
        **{"MAE low": ("MAE", "min"), "MAE high": ("MAE", "max"),
           "best rank": ("rank", "min"), "worst rank": ("rank", "max")}).reset_index()
    labels = category[category.scenario.eq(config.representative) & category.experiment.isin(["main", "stability"])].copy()
    labels["rank"] = labels.groupby("seed").accuracy.rank(ascending=False, method="min")
    classes = labels.groupby("method", sort=False).agg(
        **{"accuracy low": ("accuracy", "min"), "accuracy high": ("accuracy", "max"),
           "best rank": ("rank", "min"), "worst rank": ("rank", "max")}).reset_index()
    return table, classes


def ablation_table(numeric, category):
    """Change in error when a predictor block is removed, per scenario, target and method."""
    keys = ["scenario", "target", "method"]
    base = numeric[numeric.experiment.eq("main")][keys + ["MAE"]].rename(columns={"MAE": "full"})
    table = numeric[numeric.experiment.eq("ablation")].merge(base, on=keys)
    table["relative change"] = table.MAE / table.full - 1
    table = table[keys + ["variant", "full", "MAE", "relative change"]].rename(columns={"MAE": "without"})
    base = category[category.experiment.eq("main")][keys + ["accuracy"]].rename(columns={"accuracy": "full"})
    labels = category[category.experiment.eq("ablation")].merge(base, on=keys)
    labels = labels[keys + ["variant", "full", "accuracy"]].rename(columns={"accuracy": "without"})
    return table, labels


def export_completions(scored, measured, scenario, folder):
    """Write one completed table per method, with no truth columns.

    Every filled value is out-of-fold. model_id is filled by the classifier of the
    same family; IterativeImputer has no classifier, so its model_id stays missing.
    """
    partner = {"Median": "Mode", "KNN imputer": "KNN"}
    block = scored[scored.experiment.eq("main") & scored.scenario.eq(scenario)]
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for method in NUMERIC_METHODS:
        chosen = block[(block.method.eq(method) & block.target.ne(CATEGORY_TARGET))
                       | (block.method.eq(partner.get(method, method)) & block.target.eq(CATEGORY_TARGET))]
        filled = _complete(measured, chosen.assign(value=chosen.prediction))
        name = method.replace(" ", "_") + ".csv"
        filled.to_csv(folder / name)
        rows.append({"method": method, "file": name, "cells still missing": int(filled.isna().sum().sum())})
    return pd.DataFrame(rows)


def cost_table(log):
    """Fit and predict time per method. Parallel workers share the machine, so times are indicative."""
    selected = log[log.status.eq("success")].copy()
    # One IterativeImputer task covers every numeric target, so its time is shared between them.
    selected["targets covered"] = np.where(selected.target.eq(ALL_NUMERIC), len(NUMERIC_TARGETS), 1)
    table = selected.groupby(["experiment", "method"], sort=False).agg(
        tasks=("task_id", "size"), targets=("targets covered", "sum"),
        **{"fit seconds": ("fit_seconds", "sum"), "predict seconds": ("predict_seconds", "sum")}
    ).reset_index()
    table["seconds per target and fold"] = (table["fit seconds"] + table["predict seconds"]) / table.targets
    return table.drop(columns="targets")


def importance_matrix(importance, scenario, method):
    block = importance[importance.scenario.eq(scenario) & importance.method.eq(method)]
    return block.pivot(index="target", columns="predictor", values="importance").reindex(list(TARGETS))


def run_summary(log, plan):
    counts = log.status.value_counts()
    return pd.DataFrame([{
        "planned tasks": len(plan), "finished tasks": len(log),
        "succeeded": int(counts.get("success", 0)), "failed": int(counts.get("failed", 0)),
        "tasks with warnings": int(log.warnings.gt(0).sum()),
    }])


def warning_table(log):
    """Each kind of warning once, with an example message."""
    raised = log[log.warnings.gt(0)]
    if raised.empty:
        return pd.DataFrame(columns=["method", "warning_types", "tasks", "warning_example"])
    return raised.groupby(["method", "warning_types"], sort=False).agg(
        tasks=("task_id", "size"), warning_example=("warning_example", "first")).reset_index()


def _effect_text(effects, control, target, method, config):
    """One clause on how the same cells changed when a control moved end to end."""
    part = effects[effects.control.eq(control) & effects.target.eq(target) & effects.method.eq(method)]
    if control == "noise":
        part = part[part["held completion"].eq(_representative_level(config, "completion"))]
    else:
        part = part[part["held noise"].eq(_representative_level(config, "noise"))]
    if part.empty:
        return "not available"
    row = part.loc[(part.to - part["from"]).idxmax()]
    clear = not (row.low <= 0 <= row.high)
    return f"{row['relative change']:+.0%}{'' if clear else ' (not clear)'}"


def _representative_level(config, kind):
    match = re.match(r"completion([\d.]+)-noise([\d.]+)", config.representative)
    return float(match[1] if kind == "completion" else match[2])


def _describe_target(target, main, pairs, effects, groups, config):
    rep = main[main.scenario.eq(config.representative) & main.target.eq(target) & main.method.ne("Median")]
    ranked = rep.sort_values("MAE")
    best, second = ranked.iloc[0], ranked.iloc[1]
    gap = second.MAE / best.MAE - 1
    pair = pairs[pairs.scenario.eq(config.representative) & pairs.target.eq(target) & pairs.comparison.eq("best two")].iloc[0]
    separated = not (pair.low <= 0 <= pair.high)
    if not separated:
        lead = f"not clearly separated from {second.method}, whose MAE is {gap:.1%} higher"
    elif gap < config.material_gap:
        lead = f"practically tied with {second.method}, whose MAE is {gap:.1%} higher"
    else:
        lead = f"clearly ahead of {second.method}, whose MAE is {gap:.0%} higher"
    text = (f"- **{target}:** {best.method}, skill {best.skill:.0%}, {lead}. "
            f"MAE change on the same cells: noise 0 to 0.1 {_effect_text(effects, 'noise', target, best.method, config)}, "
            f"completeness 0.7 to 0.9 {_effect_text(effects, 'completeness', target, best.method, config)}.")
    visible = groups[groups.scenario.eq(config.representative) & groups.target.eq(target)
                     & groups.method.eq(best.method) & ~groups["small group"]]
    if len(visible) > 1:
        high, low = visible.loc[visible.skill.idxmax()], visible.loc[visible.skill.idxmin()]
        text += f" Skill runs from {low.skill:.0%} ({low.group}) to {high.skill:.0%} ({high.group})."
    return text


def findings(numeric, category, pairs, effects, groups, config):
    """A summary written from this run's tables, grouped by what the columns are."""
    main = numeric[numeric.experiment.eq("main")]
    sections = []
    families = {}
    for column in COLUMNS:
        if column.name != CATEGORY_TARGET and column.family != CONTROL:
            families.setdefault(column.family, []).append(column.name)
    for family, targets in families.items():
        lines = [_describe_target(t, main, pairs, effects, groups, config) for t in targets]
        sections.append(f"**{family.capitalize()}**\n\n" + "\n".join(lines))
    labels = category[category.experiment.eq("main") & category.scenario.eq(config.representative)]
    best = labels[labels.method.ne("Mode")].sort_values("accuracy", ascending=False).iloc[0]
    base = labels[labels.method.eq("Mode")].iloc[0]
    lowest = category[category.experiment.eq("main") & category.method.eq(best.method)].accuracy.min()
    sections.append(f"**model_id**\n\n- {best.method} reaches {best.accuracy:.1%} accuracy against {base.accuracy:.1%} "
                    f"for the mode, and at least {lowest:.1%} in every scenario.")
    control = pairs[pairs.target.eq("random_placebo") & pairs.comparison.eq("versus baseline")]
    better, worse = int(control.high.lt(0).sum()), int(control.low.gt(0).sum())
    largest = main[main.target.eq("random_placebo") & main.method.ne("Median")].skill.max()
    sections.append(
        f"**Negative control**\n\n- {better} of {len(control)} comparisons with the median favour a method and "
        f"{worse} favour the median. The largest skill reached is {largest:.1%}. With this many comparisons, about "
        f"{0.025 * len(control):.0f} would favour a method by chance alone at the 95% level."
    )
    return "\n\n".join(sections)
