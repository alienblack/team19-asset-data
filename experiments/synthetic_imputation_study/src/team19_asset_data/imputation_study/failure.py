"""When, how and why imputation fails: the tables behind chapters 5 to 9 of the notebook.

Everything here runs after fitting, on saved predictions. Every comparison keeps
the median (or the mode for ``model_id``) as its reference: skill is the share of
the baseline's error that a model removes on the same cells.
"""

from itertools import combinations

import numpy as np
import pandas as pd

from .columns import (
    ABSOLUTE, BY_NAME, CATEGORY_TARGET, CO_DEFINING, GENERATOR_LINKS, NAMES, NUMERIC_TARGETS, PARENTS, TARGETS,
)
from .data import RANDOM
from .evaluation import complete_table
from .runner import NO_BASE, PROBABILITY_COLUMNS

# The models followed in detail differ in how they correct the shared ridge base. Two
# reference models separate that base from the rest: one is the ridge alone, and one is
# a boosted chain with no ridge.
MAIN_METHODS = ("CatBoost", "Random Forest", "KNN imputer")
REFERENCE_METHODS = {"IterativeImputer": "ridge only", "miceforest direct": "boosted chain without the ridge"}
SHOWN_METHODS = MAIN_METHODS + tuple(REFERENCE_METHODS)
LEAD_METHOD = MAIN_METHODS[0]
BASELINE_OF = {True: "Mode", False: "Median"}

# Skill below FAILED means that a model removes less than a tenth of the baseline's
# error, so it adds almost nothing. Below WEAK it removes less than half.
FAILED, WEAK = 0.10, 0.50
SMALL = 30  # groups with fewer cells are not reported
BANDS = ("lowest 10%", "10% to 30%", "middle 40%", "70% to 90%", "highest 10%")
SIGNAL_TARGETS = tuple(t for t in TARGETS if t != "random_placebo")

# How the generator builds each column, in words, to group columns that should behave alike.
KINDS = {
    "near copy of another column": ("x", "install_date"),
    "exact sum": ("altitude", "y", "total_gas"),
    "threshold": ("measured_overgrowth",),
    "random draw scaled by another column": ("measured_humid", "measured_salt"),
    "decline that depends on the model": ("measured_gas_a", "measured_gas_b", "measured_gas_c"),
    "product that depends on the model": ("life",),
    "category drawn from a moving window": ("model_id",),
    "independent draw, seen only through its effects": ("leakage",),
    "no relationship": ("random_placebo",),
}
KIND_OF = {target: kind for kind, targets in KINDS.items() for target in targets}


def state(skill):
    """The label a skill value gets under the failure standard."""
    if not np.isfinite(skill):
        return "not measured"
    return "fails" if skill < FAILED else "weak" if skill < WEAK else "works"


def cell_table(scored, measured, truth):
    """One row per predicted cell with its error, the baseline's error on the same cell and its conditions.

    ``seen`` counts the directly linked columns that were visible in the row, using measured
    data only. ``band`` places the true value in its column's range, so it is known only afterwards.
    """
    keep = ["scenario", "experiment", "variant", "seed", "fold", "method", "target", "row_id", "raw", "prediction", "truth"]
    table = scored[keep + [c for c in PROBABILITY_COLUMNS if c in scored]].copy()
    categorical = table.target.eq(CATEGORY_TARGET)
    table["signed"] = (table.prediction - table.truth).where(~categorical)
    table["error"] = (table.prediction - table.truth).abs().where(~categorical, table.prediction.ne(table.truth).astype(float))
    baseline = table[table.method.isin(BASELINE_OF.values()) & table.variant.eq("all predictors")]
    baseline = baseline[["scenario", "seed", "target", "row_id", "error"]].rename(columns={"error": "base_error"})
    table = table.merge(baseline, on=["scenario", "seed", "target", "row_id"], how="left")
    table["seen"], table["links"], table["rank"], table["within"] = 0, 0, np.nan, np.nan
    for (scenario, target), index in table.groupby(["scenario", "target"], sort=False).groups.items():
        rows = table.loc[index, "row_id"].to_numpy()
        links = list(GENERATOR_LINKS[target])
        table.loc[index, "links"] = len(links)
        if links:
            table.loc[index, "seen"] = measured[scenario][links].notna().sum(axis=1).reindex(rows).to_numpy()
        column = BY_NAME[target]
        if target != CATEGORY_TARGET:
            table.loc[index, "rank"] = truth[scenario][target].rank(pct=True).reindex(rows).to_numpy()
            y, error = table.loc[index, "truth"].to_numpy(), table.loc[index, "error"].to_numpy()
            if column.tolerance_kind == ABSOLUTE:
                inside = error <= column.tolerance[1]
            else:
                inside = np.where(y != 0, error / np.where(y != 0, np.abs(y), 1.0) <= column.tolerance[1], np.nan)
            table.loc[index, "within"] = np.asarray(inside, dtype=float)
    table["seen_share"] = (table.seen / table.links.where(table.links > 0)).round(3)
    table["band"] = pd.cut(table["rank"], [0, 0.1, 0.3, 0.7, 0.9, 1.0], labels=list(BANDS)).astype(object)
    return table


def skill_by(cells, keys, minimum=SMALL):
    """Error, baseline error and skill for each group of cells."""
    table = cells.groupby(keys, sort=False, observed=True).agg(
        n=("error", "size"), error=("error", "mean"), baseline=("base_error", "mean"),
        bias=("signed", "mean"), within=("within", "mean")).reset_index()
    table["skill"] = 1 - table.error / table.baseline.where(table.baseline > 0)
    return table[table.n >= minimum].reset_index(drop=True)


def main_cells(cells, config, scenario=None):
    block = cells[cells.experiment.eq("main") & cells.seed.eq(config.seed)]
    return block if scenario is None else block[block.scenario.eq(scenario)]


# ----------------------------------------------------------------- overview


def overall_table(cells, config):
    """Skill of every model on every target in the representative scenario, with the baseline's error."""
    table = skill_by(main_cells(cells, config, config.representative), ["target", "method"])
    wide = table.pivot(index="target", columns="method", values="skill").reindex(list(TARGETS))
    order = [m for m in SHOWN_METHODS if m in wide] + [m for m in wide if m not in SHOWN_METHODS + ("Median", "Mode")]
    baseline = table[table.method.isin(BASELINE_OF.values())].set_index("target").error
    wide = wide[order]
    wide.insert(0, "baseline error", baseline.reindex(wide.index))
    wide.insert(0, "kind of relationship", [KIND_OF[t] for t in wide.index])
    return wide


def scenario_table(cells, config):
    """Skill in each of the nine generator scenarios."""
    table = skill_by(main_cells(cells, config), ["scenario", "target", "method"])
    return table[~table.method.isin(BASELINE_OF.values())]


def visibility_table(cells, config, scenarios=None):
    """Skill by the number of directly linked columns visible in the row."""
    block = main_cells(cells, config)
    block = block[block.scenario.isin(scenarios or config.noise_series) & block.links.gt(0)]
    table = skill_by(block, ["scenario", "target", "method", "links", "seen"])
    table["share seen"] = table.seen / table.links
    return table[~table.method.isin(BASELINE_OF.values())]


def low_visibility(cells):
    """Cells in which at most a third of the directly linked columns were visible."""
    return cells[cells.links.gt(0) & (cells.seen / cells.links.where(cells.links > 0) <= 1 / 3 + 1e-9)]


def failure_map(cells, ablation, config, method=LEAD_METHOD):
    """Skill of one model on every target under each condition studied.

    The first column is the reference scenario; the others change one thing at a time.
    """
    representative = config.representative
    match = dict(zip(("completion", "noise"), _levels(representative)))
    names = {_levels(s): s for s in set(cells.scenario) if s.count("-") == 1}
    noisy = names.get((match["completion"], max(n for c, n in names if c == match["completion"])))
    sparse = names.get((min(c for c, n in names if n == match["noise"]), match["noise"]))
    both = names.get((min(c for c, _ in names), max(n for _, n in names)))
    block = cells[cells.method.eq(method) & cells.seed.eq(config.seed)]
    main = block[block.experiment.eq("main")]
    columns = {
        "reference scenario": main[main.scenario.eq(representative)],
        "highest noise": main[main.scenario.eq(noisy)],
        "lowest completion": main[main.scenario.eq(sparse)],
        "both together": main[main.scenario.eq(both)],
        "few linked columns visible": low_visibility(main[main.scenario.eq(representative)]),
    }
    for scenario in sorted(set(block[block.experiment.eq("mechanism")].scenario)):
        mechanism = scenario.rsplit("-", 1)[1]
        label = {"age": "older assets hidden more", "value": "higher values hidden more"}.get(mechanism, mechanism)
        columns[label] = block[block.scenario.eq(scenario)]
    result = pd.DataFrame({label: skill_by(part, ["target"]).set_index("target").skill for label, part in columns.items()})
    removed = ablation[ablation.method.eq(method) & ablation.scenario.eq(representative)]
    if len(removed):
        worst = removed.assign(skill=1 - removed.without / removed["baseline"]).groupby("target").skill.min()
        result["most harmful block removed"] = worst
    return result.reindex(list(TARGETS))


def _levels(name):
    completion, noise = name.split("-")[:2]
    return float(completion.replace("completion", "")), float(noise.replace("noise", ""))


def ablation_scores(cells, config):
    """Error with and without each predictor block, on the same cells, with the baseline's error."""
    block = cells[cells.seed.eq(config.seed)]
    full = skill_by(block[block.experiment.eq("main")], ["scenario", "target", "method"])
    full = full[["scenario", "target", "method", "error", "baseline"]].rename(columns={"error": "full"})
    without = skill_by(block[block.experiment.eq("ablation")], ["scenario", "target", "method", "variant"])
    without = without[["scenario", "target", "method", "variant", "error"]].rename(columns={"error": "without"})
    table = without.merge(full, on=["scenario", "target", "method"])
    table["relative change"] = table.without / table.full - 1
    table["skill without"] = 1 - table.without / table.baseline
    return table


# ------------------------------------------------------------------- when


def band_table(cells, config):
    """Error and bias by where the true value lies in its column, in units of the column's spread."""
    block = main_cells(cells, config, config.representative)
    block = block[block.target.ne(CATEGORY_TARGET) & block.band.notna()]
    table = skill_by(block, ["target", "method", "band"])
    spread = block.groupby("target").truth.std()
    table["error in SD"] = table.error / table.target.map(spread)
    table["bias in SD"] = table.bias / table.target.map(spread)
    return table


def mechanism_table(cells, scenarios, config):
    """Skill and bias when cells are hidden at random, by age or by value, at the same overall share."""
    names = [config.representative] + [n for n, s in scenarios.items() if s.mechanism != RANDOM]
    block = cells[cells.scenario.isin(names) & cells.experiment.isin(["main", "mechanism"]) & cells.seed.eq(config.seed)]
    table = skill_by(block, ["scenario", "target", "method"])
    spread = block[block.target.ne(CATEGORY_TARGET)].groupby("target").truth.std()
    table["bias in SD"] = table.bias / table.target.map(spread)
    table["mechanism"] = table.scenario.map(lambda n: scenarios[n].mechanism)
    return table


def mechanism_groups(cells, truth, scenarios, config):
    """Skill by age third, to see whether the assets that lose more cells are also filled worse."""
    names = [config.representative] + [n for n, s in scenarios.items() if s.mechanism != RANDOM]
    block = cells[cells.scenario.isin(names) & cells.experiment.isin(["main", "mechanism"]) & cells.seed.eq(config.seed)].copy()
    age = truth[config.representative].install_date.rank(pct=True)
    block["age third"] = pd.cut(age.reindex(block.row_id).to_numpy(), [0, 1 / 3, 2 / 3, 1],
                                labels=["oldest third", "middle third", "newest third"]).astype(object)
    table = skill_by(block, ["scenario", "method", "age third", "target"])
    table = table[table.target.isin(SIGNAL_TARGETS)]
    result = table.groupby(["scenario", "method", "age third"], sort=False).skill.mean().reset_index()
    result["mechanism"] = result.scenario.map(lambda n: scenarios[n].mechanism)
    return result


def kind_table(cells, config, method=LEAD_METHOD):
    """Skill of one model by the kind of relationship that defines the column, under each condition."""
    block = main_cells(cells, config)
    block = block[block.method.eq(method)]
    table = skill_by(block, ["scenario", "target"])
    levels = table.scenario.map(_levels)
    table["completion"], table["noise"] = [c for c, _ in levels], [n for _, n in levels]
    reference = table[table.scenario.eq(config.representative)].set_index("target").skill
    hidden = skill_by(low_visibility(block[block.scenario.eq(config.representative)]), ["target"]).set_index("target").skill
    rows = []
    for kind, targets in KINDS.items():
        part = table[table.target.isin(targets)]
        rows.append({
            "kind of relationship": kind, "columns": ", ".join(targets),
            "reference": reference.reindex(targets).mean(),
            "best scenario": part.groupby("scenario").skill.mean().max(),
            "worst scenario": part.groupby("scenario").skill.mean().min(),
            "few linked columns visible": hidden.reindex(targets).mean(),
        })
    return pd.DataFrame(rows)


# -------------------------------------------------------------------- how


def _slope(block):
    y, p = block.truth.to_numpy(float), block.prediction.to_numpy(float)
    spread = np.var(y)
    return np.cov(p, y, bias=True)[0, 1] / spread if spread > 0 else np.nan


def shrinkage_table(cells, config):
    """How far predictions follow the truth: slope 1 follows it fully, slope 0 is a constant guess.

    The spread ratio compares the spread of imputed values with the spread of the truth on the same cells.
    """
    block = main_cells(cells, config)
    block = block[block.scenario.isin(config.noise_series) & block.target.ne(CATEGORY_TARGET)
                  & block.target.ne("random_placebo") & block.links.gt(0)]
    rows = []
    for keys, part in block.groupby(["scenario", "target", "method", "links", "seen"], sort=False):
        if len(part) >= SMALL:
            rows.append({**dict(zip(["scenario", "target", "method", "links", "seen"], keys)), "n": len(part),
                         "slope": _slope(part), "spread ratio": part.prediction.std() / part.truth.std()})
    table = pd.DataFrame(rows)
    table["share seen"] = table.seen / table.links
    return table


def overall_shrinkage(cells, config):
    """Slope and spread ratio over all hidden cells of a column, in each scenario."""
    block = main_cells(cells, config)
    block = block[block.target.ne(CATEGORY_TARGET)]
    rows = []
    for keys, part in block.groupby(["scenario", "target", "method"], sort=False):
        rows.append({**dict(zip(["scenario", "target", "method"], keys)), "slope": _slope(part),
                     "spread ratio": part.prediction.std() / part.truth.std() if part.truth.std() > 0 else np.nan})
    return pd.DataFrame(rows)


def range_table(cells, config):
    """Share of raw predictions outside the column's valid range, before they are limited to it."""
    block = main_cells(cells, config)
    block = block[block.target.ne(CATEGORY_TARGET) & ~block.method.isin(BASELINE_OF.values())]
    rows = []
    for (scenario, target, method), part in block.groupby(["scenario", "target", "method"], sort=False):
        column = BY_NAME[target]
        low = -np.inf if column.lower is None else column.lower
        high = np.inf if column.upper is None else column.upper
        rows.append({"scenario": scenario, "target": target, "method": method,
                     "outside valid range": float(((part.raw < low) | (part.raw > high)).mean())})
    return pd.DataFrame(rows)


def decline_table(scenario_scores, visibility, config, method=LEAD_METHOD):
    """Whether skill is lost early or late as noise rises and as linked columns are hidden.

    A share near 1 means that most of the loss comes at the first step; near 0.5 means an even decline.
    """
    completion = _levels(config.representative)[0]
    block = scenario_scores[scenario_scores.method.eq(method)].copy()
    levels = block.scenario.map(_levels)
    block["completion"], block["noise"] = [c for c, _ in levels], [n for _, n in levels]
    noise = block[block.completion.eq(completion)].pivot(index="target", columns="noise", values="skill")
    steps = sorted(noise.columns)
    rows = []
    for target in noise.index:
        first, total = noise.loc[target, steps[0]] - noise.loc[target, steps[1]], noise.loc[target, steps[0]] - noise.loc[target, steps[-1]]
        seen = visibility[visibility.method.eq(method) & visibility.target.eq(target)
                          & visibility.scenario.eq(config.representative)].sort_values("seen")
        entry = {"target": target, "skill lost to noise": total,
                 "share lost at the first noise step": first / total if abs(total) >= 0.02 else np.nan}
        if len(seen) >= 2:
            entry["skill with every linked column"] = seen.skill.iloc[-1]
            entry["skill with the fewest"] = seen.skill.iloc[0]
            if len(seen) >= 3 and seen.skill.iloc[-1] - seen.skill.iloc[0] >= 0.02:
                entry["share lost when the first linked column is hidden"] = (
                    (seen.skill.iloc[-1] - seen.skill.iloc[-2]) / (seen.skill.iloc[-1] - seen.skill.iloc[0]))
        rows.append(entry)
    return pd.DataFrame(rows).set_index("target").reindex(list(TARGETS)).reset_index()


def _pair_correlations(frame, rows_by_pair):
    values = []
    for (a, b), rows in rows_by_pair.items():
        values.append(np.corrcoef(frame[a].to_numpy()[rows], frame[b].to_numpy()[rows])[0, 1])
    return np.array(values)


def correlation_table(cells, measured, truth, config):
    """Gap between correlations in completed data and in the truth, in every scenario.

    All 91 pairs of numeric columns are compared on the rows that hold an imputed cell of the pair,
    because fully observed rows would hide any change.
    """
    columns = list(NUMERIC_TARGETS)
    block = cells[cells.experiment.isin(["main", "mechanism"]) & cells.seed.eq(config.seed) & cells.target.ne(CATEGORY_TARGET)]
    rows = []
    for scenario, part in block.groupby("scenario", sort=False):
        missing = measured[scenario][columns].isna()
        rows_by_pair = {(a, b): (missing[a] | missing[b]).to_numpy() for a, b in combinations(columns, 2)}
        labels = [" ~ ".join(pair) for pair in rows_by_pair]
        reference = _pair_correlations(truth[scenario], rows_by_pair)
        for method, values in part.groupby("method", sort=False):
            filled = complete_table(measured[scenario], values[["row_id", "target"]].assign(value=values.prediction.to_numpy()))
            gap = np.abs(_pair_correlations(filled, rows_by_pair) - reference)
            rows.append({"scenario": scenario, "method": method, "mean gap": gap.mean(), "largest gap": gap.max(),
                         "pair with the largest gap": labels[int(gap.argmax())]})
    return pd.DataFrame(rows)


def consistency_table(cells, measured, config):
    """Whether completed rows still obey three of the generator's definitions, at each noise level.

    The reference is the residual in rows with nothing imputed, which is what measurement noise alone gives.
    """
    relations = {
        "altitude = x + y + 100": (("x", "y", "altitude"), lambda f: (f.altitude - f.x - f.y - 100).abs()),
        "overgrowth = max(2 humid - salt, 0)": (
            ("measured_humid", "measured_salt", "measured_overgrowth"),
            lambda f: (f.measured_overgrowth - (2 * f.measured_humid - f.measured_salt).clip(lower=0)).abs()),
        "total_gas = gas a + b + c": (
            ("measured_gas_a", "measured_gas_b", "measured_gas_c", "total_gas"),
            lambda f: (f.total_gas - f.measured_gas_a - f.measured_gas_b - f.measured_gas_c).abs()),
    }
    block = main_cells(cells, config)
    block = block[block.scenario.isin(config.noise_series) & block.target.ne(CATEGORY_TARGET)]
    rows = []
    for (scenario, method), part in block.groupby(["scenario", "method"], sort=False):
        frame = complete_table(measured[scenario], part[["row_id", "target"]].assign(value=part.prediction.to_numpy()))
        for relation, (fields, residual) in relations.items():
            touched = measured[scenario][list(fields)].isna().any(axis=1)
            rows.append({
                "scenario": scenario, "relation": relation, "method": method,
                "mean residual in completed rows": residual(frame.loc[touched, list(fields)].dropna()).mean(),
                "mean residual in fully observed rows": residual(measured[scenario][list(fields)].dropna()).mean(),
            })
    return pd.DataFrame(rows)


def confidence_table(cells, config, by="seen"):
    """Whether the probability a model gives its model_id answer falls when the answer is more often wrong."""
    block = main_cells(cells, config)
    block = block[block.target.eq(CATEGORY_TARGET) & block.prob_0.notna() & ~block.method.isin(BASELINE_OF.values())]
    if by == "seen":
        block = block[block.scenario.eq(config.representative)]
    confidence = block[PROBABILITY_COLUMNS].max(axis=1)
    block = block.assign(confidence=confidence, hit=1 - block.error,
                         sure_and_wrong=((confidence >= 0.9) & block.error.eq(1)).astype(float))
    keys = ["method", "seen"] if by == "seen" else ["method", "scenario"]
    table = block.groupby(keys, sort=False).agg(
        n=("hit", "size"), accuracy=("hit", "mean"), confidence=("confidence", "mean"),
        **{"sure and wrong": ("sure_and_wrong", "mean")}).reset_index()
    table["confidence minus accuracy"] = table.confidence - table.accuracy
    return table[table.n >= SMALL]


def structure_table(cells, config):
    """Skill with the ridge base, without it, with the ridge alone and with a boosted chain that has no ridge."""
    block = cells[cells.scenario.eq(config.representative) & cells.seed.eq(config.seed)]
    main = skill_by(block[block.experiment.eq("main")], ["target", "method"])
    plain = skill_by(block[block.experiment.eq("structure") & block.variant.eq(NO_BASE)], ["target", "method"])
    wide = main.pivot(index="target", columns="method", values="skill")
    result = pd.DataFrame(index=list(TARGETS))
    for method in sorted(set(plain.method), key=lambda m: list(SHOWN_METHODS).index(m) if m in SHOWN_METHODS else 99):
        result[f"{method}: no ridge"] = plain[plain.method.eq(method)].set_index("target").skill
        result[f"{method}: with ridge"] = wide[method]
    for method, label in REFERENCE_METHODS.items():
        if method in wide:
            result[f"{method} ({label})"] = wide[method]
    return result


def error_overlap(cells, config):
    """Correlation between the cell errors of two models: 1 means they fail on the same cells."""
    block = main_cells(cells, config, config.representative)
    block = block[block.target.ne(CATEGORY_TARGET) & block.target.ne("random_placebo") & ~block.method.isin(BASELINE_OF.values())]
    matrices = []
    for _, part in block.groupby("target", sort=False):
        matrices.append(part.pivot(index="row_id", columns="method", values="error").corr(method="spearman"))
    order = [m for m in SHOWN_METHODS if m in matrices[0]] + [m for m in matrices[0] if m not in SHOWN_METHODS]
    return (sum(matrices) / len(matrices)).loc[order, order]


# -------------------------------------------------------------------- why


def information_table(cells, config, method=LEAD_METHOD):
    """Splits each column's remaining error into what hidden linked columns cost and what remains with all of them visible."""
    block = main_cells(cells, config, config.representative)
    block = block[block.method.eq(method)]
    overall = skill_by(block, ["target"]).set_index("target").skill
    complete = skill_by(block[block.links.gt(0) & block.seen.eq(block.links)], ["target"]).set_index("target")
    table = pd.DataFrame({"skill over all hidden cells": overall, "skill with every linked column visible": complete.skill,
                          "cells with every linked column visible": complete.n})
    table["lost to hidden linked columns"] = table["skill with every linked column visible"] - table["skill over all hidden cells"]
    table["not reached even with every linked column"] = 1 - table["skill with every linked column visible"]
    return table.reindex([t for t in TARGETS if t != "random_placebo"])


def noise_table(cells, measured, truth, scenario_scores, config, method=LEAD_METHOD):
    """Noise in a column and in the columns linked to it, against the error the model gains from noise.

    Noise is the mean measurement error of visible cells at the highest noise level, in units of
    the column's spread. The error change compares the same cells at the lowest and highest noise.
    """
    completion = _levels(config.representative)[0]
    names = sorted((s for s in config.noise_series), key=lambda s: _levels(s)[1])
    low, high = names[0], names[-1]
    seen = measured[high].notna()
    noise = ((measured[high] - truth[high]).abs().where(seen).mean() / truth[high].std()).drop(CATEGORY_TARGET)
    scores = scenario_scores[scenario_scores.method.eq(method)].set_index(["scenario", "target"])
    rows = []
    for target in SIGNAL_TARGETS:
        links = [c for c in GENERATOR_LINKS[target] if c != CATEGORY_TARGET]
        before, after = scores.loc[(low, target)], scores.loc[(high, target)]
        rows.append({
            "target": target, "noise in the column itself": noise.get(target, 0.0),
            "mean noise in its linked columns": float(noise.reindex(links).mean()) if links else 0.0,
            "largest noise in a linked column": float(noise.reindex(links).max()) if links else 0.0,
            "error change from noise": after.error / before.error - 1,
            "skill at the lowest noise": before.skill, "skill at the highest noise": after.skill,
        })
    return pd.DataFrame(rows), completion


DIRECT, TOGETHER, THROUGH, STAND_IN, NO_LINK = (
    "direct", "defines a column together with the target", "linked through another column",
    "stands in for a near copy of a linked column", "no link")
LINK_CLASSES = (DIRECT, TOGETHER, THROUGH, STAND_IN, NO_LINK)
# x and install_date are near copies, because assets were installed in grid order.
NEAR_COPY = {"x": "install_date", "install_date": "x"}


def _ancestors(name):
    found, queue = set(), list(PARENTS[name])
    while queue:
        item = queue.pop()
        if item not in found:
            found.add(item)
            queue.extend(PARENTS[item])
    return found


def _link(target, predictor):
    """Where a predictor stands in the generator's definition of a target."""
    if predictor in GENERATOR_LINKS[target]:
        return DIRECT
    if predictor in CO_DEFINING[target]:
        return TOGETHER
    # One is computed from the other through intermediate columns, or both are computed from a shared column.
    if (_ancestors(target) | {target}) & (_ancestors(predictor) | {predictor}):
        return THROUGH
    return NO_LINK


def link_class(target, predictor):
    label = _link(target, predictor)
    copy = NEAR_COPY.get(predictor)
    if label == NO_LINK and copy and copy != target and _link(target, copy) != NO_LINK:
        return STAND_IN
    return label


def classify_links(importance):
    """Mean importance over folds, with each predictor labelled by its place in the generator's definition."""
    table = importance.groupby(["method", "scenario", "target", "predictor"], sort=False).importance.mean().reset_index()
    table = table[table.target.ne("random_placebo")]
    table["link"] = [link_class(t, p) for t, p in zip(table.target, table.predictor)]
    return table


def link_table(importance, threshold=0.02):
    """How far the columns a model relies on match the generator's definition."""
    table = classify_links(importance)
    table["relied on"] = table.importance > threshold
    rows = []
    for (method, scenario), block in table.groupby(["method", "scenario"], sort=False):
        relied, direct = block[block["relied on"]], block[block.link.eq(DIRECT)]
        share = relied.link.value_counts(normalize=True)
        rows.append({"method": method, "scenario": scenario, "direct links": len(direct),
                     "direct links relied on": direct["relied on"].mean(), "columns relied on": len(relied),
                     **{f"of those: {label}": share.get(label, 0.0) for label in LINK_CLASSES}})
    return pd.DataFrame(rows)


def unexplained_links(importance, scenario, method, threshold=0.02, count=10):
    """The strongest columns a model relied on that are not in the target's own definition."""
    table = classify_links(importance)
    block = table[table.scenario.eq(scenario) & table.method.eq(method) & ~table.link.isin([DIRECT, TOGETHER])
                  & table.importance.gt(threshold)]
    return block.sort_values("importance", ascending=False)[["target", "predictor", "importance", "link"]].head(count).reset_index(drop=True)


def integrity_table(cells, measured, config):
    """Checks that rule out leakage through the evaluation itself."""
    from .runner import fold_ids
    block = main_cells(cells, config)
    predicted_observed, wrong_fold = 0, 0
    for (scenario, target), part in block.groupby(["scenario", "target"], sort=False):
        predicted_observed += int(measured[scenario][target].reindex(part.row_id).notna().sum())
    for scenario, part in block.groupby("scenario", sort=False):
        frame = measured[scenario]
        assignment = pd.Series(fold_ids(len(frame), config.folds, config.seed), index=frame.index)
        wrong_fold += int((assignment.reindex(part.row_id).to_numpy() != part.fold.to_numpy()).sum())
    counts = block.groupby(["scenario", "method", "target"]).row_id.agg(["size", "nunique"]).reset_index()
    hidden = pd.concat({s: measured[s].isna().sum() for s in set(block.scenario)}).rename("hidden")
    hidden.index.names = ["scenario", "target"]
    counts = counts.merge(hidden.reset_index(), on=["scenario", "target"])
    return pd.DataFrame([
        {"check": "Predictions for cells that were visible", "count": predicted_observed, "expected": 0},
        {"check": "Predicted rows outside the held-out fold", "count": wrong_fold, "expected": 0},
        {"check": "Cells predicted more than once by a method", "count": int((counts["size"] != counts["nunique"]).sum()), "expected": 0},
        {"check": "Method and target groups that do not cover every hidden cell", "count": int((counts["size"] != counts.hidden).sum()), "expected": 0},
    ])


# ------------------------------------------------------------------- value


def complete_rows_table(measured, scenarios):
    """Rows that an analysis could use if it kept only rows with no missing cell."""
    rows = []
    for name, frame in measured.items():
        rows.append({"scenario": name, "mechanism": scenarios[name].mechanism, "rows": len(frame),
                     "rows with no missing cell": int(frame.notna().all(axis=1).sum()),
                     "share of rows": float(frame.notna().all(axis=1).mean()),
                     "share of cells visible": float(frame.notna().to_numpy().mean())})
    return pd.DataFrame(rows)


def _quantities(frame, threshold):
    """Summary values an asset analysis might take from the table."""
    values = {}
    for model in range(4):
        values[f"mean life, model {model}"] = frame.life[frame.model_id.eq(model)].mean()
    values["share of assets with life below the lowest tenth"] = (frame.life < threshold).mean()
    pair = frame[["install_date", "life"]].dropna()
    values["life lost per day installed"] = -np.polyfit(pair.install_date, pair.life, 1)[0]
    values["90th percentile of overgrowth"] = frame.measured_overgrowth.quantile(0.9)
    values["spread of total gas (SD)"] = frame.total_gas.std()
    return values


def downstream_table(cells, measured, truth, config, scenario=None):
    """Summary values computed from each completed table, against the same values from the truth.

    ``rows with no missing cell`` keeps only complete rows, which is what an analysis without
    imputation would use. ``measured values with true values in the gaps`` shows what a perfect
    imputation would give, so its error comes from measurement noise alone.
    """
    scenario = scenario or config.representative
    frame, reference = measured[scenario], truth[scenario]
    threshold = reference.life.quantile(0.1)
    sources = {"rows with no missing cell": frame.dropna(),
               "measured values with true values in the gaps": frame.fillna(reference)}
    block = cells[cells.scenario.eq(scenario) & cells.experiment.isin(["main", "mechanism"]) & cells.seed.eq(config.seed)]
    labels = block[block.target.eq(CATEGORY_TARGET)]
    for method, part in block[block.target.ne(CATEGORY_TARGET)].groupby("method", sort=False):
        partner = "Mode" if method == "Median" else method
        classes = labels[labels.method.eq(partner)]
        if classes.empty:
            continue  # a model that reports no model_id cannot give the values by model
        chosen = pd.concat([part, classes])
        name = "Median and Mode" if method == "Median" else method
        sources[name] = complete_table(frame, chosen[["row_id", "target"]].assign(value=chosen.prediction.to_numpy()))
    target = _quantities(reference, threshold)
    rows = []
    for source, table in sources.items():
        for quantity, value in _quantities(table, threshold).items():
            rows.append({"source": source, "quantity": quantity, "estimate": value, "truth": target[quantity],
                         "relative error": value / target[quantity] - 1, "rows used": len(table)})
    return pd.DataFrame(rows)
