"""Scoring against truth. This module runs after fitting and feeds nothing back to models."""

from pathlib import Path
import re

import numpy as np
import pandas as pd

from .columns import ABSOLUTE, BY_NAME, CATEGORY_TARGET, NUMERIC_TARGETS, TARGETS
from .data import RANDOM
from .models import NUMERIC_METHODS
from .runner import ALL_COLUMNS, ALL_NUMERIC, ITERATIVE, PROBABILITY_COLUMNS

KEYS = ["scenario", "experiment", "variant", "seed", "target", "method"]
BASELINES = ("Median", "Mode")
# Experiments that fit the baseline themselves. The others (ablation, structure, budget,
# calibration) reuse the baseline of the main experiment, which has the same folds.
BASELINE_EXPERIMENTS = ("main", "stability", "mechanism", "spatial")


def fold_design(experiments):
    """Whole-block folds for the spatial experiment, random row folds for every other."""
    return pd.Series(experiments).eq("spatial").map({True: "blocks", False: "rows"}).to_numpy()


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
    # The baseline comes from the same scenario, seed and kind of fold split, on the same cells.
    numeric["split"] = fold_design(numeric.experiment)
    baseline = numeric[numeric.method.eq("Median") & numeric.variant.eq("all predictors")
                       & numeric.experiment.isin(BASELINE_EXPERIMENTS)]
    baseline = baseline[["scenario", "seed", "split", "target", "MAE"]].rename(columns={"MAE": "median MAE"})
    numeric = numeric.merge(baseline, on=["scenario", "seed", "split", "target"], how="left").drop(columns="split")
    numeric["skill"] = 1 - numeric.MAE / numeric["median MAE"]
    return numeric, category


def add_scenario_levels(table, scenarios):
    table = table.copy()
    table["completion"] = table.scenario.map(lambda s: scenarios[s].completion)
    table["noise"] = table.scenario.map(lambda s: scenarios[s].noise)
    table["mechanism"] = table.scenario.map(lambda s: scenarios[s].mechanism)
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
    items = [s for s in scenarios.values() if s.mechanism == RANDOM]
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
            # Rows are put in a fixed order first: the bootstrap resamples positions, and tasks
            # finish in a different order in every run.
            block = block.sort_values("row_id")
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


def complete_table(measured, cells):
    frame = measured.copy()
    for target, block in cells.groupby("target"):
        frame.loc[block.row_id.to_numpy(), target] = block.value.to_numpy()
    return frame


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
    """IterativeImputer MAE at other iteration limits against the main run."""
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


def export_completions(scored, measured, scenario, folder):
    """Write one completed table per method, with no truth columns.

    Every filled value is out-of-fold. model_id is filled by the classifier of the
    same model; IterativeImputer reports no model_id, so its model_id stays missing.
    """
    partner = {"Median": "Mode"}
    block = scored[scored.experiment.eq("main") & scored.scenario.eq(scenario)]
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for method in NUMERIC_METHODS:
        chosen = block[(block.method.eq(method) & block.target.ne(CATEGORY_TARGET))
                       | (block.method.eq(partner.get(method, method)) & block.target.eq(CATEGORY_TARGET))]
        filled = complete_table(measured, chosen.assign(value=chosen.prediction))
        name = method.replace(" ", "_") + ".csv"
        filled.to_csv(folder / name)
        rows.append({"method": method, "file": name, "cells still missing": int(filled.isna().sum().sum())})
    return pd.DataFrame(rows)


def cost_table(log):
    """Fit and predict time per method. Parallel workers share the machine, so times are indicative."""
    selected = log[log.status.eq("success")].copy()
    # One chained task covers many targets, so its time is shared between them.
    selected["targets covered"] = selected.target.map(
        {ALL_NUMERIC: len(NUMERIC_TARGETS), ALL_COLUMNS: len(TARGETS)}).fillna(1)
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


def target_table(numeric, pairs, effects, config):
    """For each numeric target: the best method, how clear its lead is, and the effect of each control."""
    main = numeric[numeric.experiment.eq("main") & numeric.seed.eq(config.seed)]
    rows = []
    for target in [t for t in NUMERIC_TARGETS if t != "random_placebo"]:
        rep = main[main.scenario.eq(config.representative) & main.target.eq(target) & main.method.ne("Median")]
        ranked = rep.sort_values("MAE")
        best, second = ranked.iloc[0], ranked.iloc[1]
        pair = pairs[pairs.scenario.eq(config.representative) & pairs.target.eq(target) & pairs.comparison.eq("best two")].iloc[0]
        gap = second.MAE / best.MAE - 1
        separated = not (pair.low <= 0 <= pair.high)
        lead = "clear" if separated and gap >= config.material_gap else "small" if separated else "not separated"
        rows.append({
            "target": target, "best": best.method, "skill": best.skill, "next": second.method, "gap to next": gap,
            "lead": lead, "noise 0 to 0.1": _effect_text(effects, "noise", target, best.method, config),
            "completion 0.7 to 0.9": _effect_text(effects, "completeness", target, best.method, config),
        })
    return pd.DataFrame(rows)


