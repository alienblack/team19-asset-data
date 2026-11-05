"""Evaluation of the relationships between columns.

The generator's definitions appear here and are used only after fitting, to judge
what the models learned. Nothing in this module feeds back to a model.
"""

from itertools import combinations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_predict

from .columns import CATEGORY_TARGET, GAS_READINGS, GENERATOR_LINKS, NUMERIC_TARGETS, TARGETS, finalise
from .evaluation import complete_table
from .models import NUMERIC_METHODS

LINK_THRESHOLD = 0.02  # importance above this counts as the model relying on a column
SIGNAL_TARGETS = [t for t in NUMERIC_TARGETS if t != "random_placebo"]
FORMULA = "Formula"
V1_SCORES = Path("outputs/imputation-study/baselines/v1/run_tables/numeric_scores.csv")
SELECTION = Path("outputs/imputation-study/optimisation/clean_selection")
SELECTION_FILES = {"KNN imputer": "knn", "Random Forest": "forest", "HistGradientBoosting": "hgb", "CatBoost": "cat"}


def _life(frame):
    """The generator's definition of remaining life."""
    hazard = np.select(
        [frame.model_id.eq(0), frame.model_id.eq(1), frame.model_id.eq(2)],
        [frame.measured_humid + frame.measured_salt + frame.measured_overgrowth,
         frame.measured_humid + frame.measured_overgrowth, frame.measured_salt + frame.measured_overgrowth],
        default=frame.measured_overgrowth,
    )
    return 100_000 - frame.install_date * hazard - 25_000 * frame.total_gas / 300_000


def _leakage(frame):
    """Middle of the leakage values the generator allows, given exact gases, model and date."""
    day = frame.install_date.to_numpy(float)
    starts = {0: (10000, 10000, 10000), 1: (12000, 9000, 9000), 2: (9000, 12000, 9000), 3: (9000, 9000, 12000)}
    start = np.array([starts[int(m)] for m in frame.model_id])
    weights = np.column_stack((day, 0.0005 * day**2, day + 0.00025 * day**2))
    gases = frame[list(GAS_READINGS)].to_numpy(float)
    low = np.maximum(0.8, ((0.975 * start - gases) / weights).max(axis=1))
    high = np.minimum(1.2, ((1.025 * start - gases) / weights).min(axis=1))
    return (low + high) / 2


# Each entry: the target, the inputs that define it, and the definition applied to those inputs.
RELATIONS = {
    "altitude from x and y": ("altitude", ("x", "y"), lambda f: f.x + f.y + 100),
    "x from altitude and y": ("x", ("altitude", "y"), lambda f: f.altitude - f.y - 100),
    "y from altitude and x": ("y", ("altitude", "x"), lambda f: f.altitude - f.x - 100),
    "total_gas from gas readings": ("total_gas", GAS_READINGS, lambda f: f[list(GAS_READINGS)].sum(axis=1)),
    "gas a from total and other readings": (
        "measured_gas_a", ("total_gas", "measured_gas_b", "measured_gas_c"),
        lambda f: f.total_gas - f.measured_gas_b - f.measured_gas_c),
    "overgrowth from humidity and salt": (
        "measured_overgrowth", ("measured_humid", "measured_salt"),
        lambda f: (2 * f.measured_humid - f.measured_salt).clip(lower=0)),
    "life from all its inputs": (
        "life", ("install_date", "model_id", "measured_humid", "measured_salt", "measured_overgrowth", "total_gas"), _life),
}
LEAKAGE_RELATION = "leakage from gases, date and model"
LEAKAGE_INPUTS = ("install_date", "model_id") + GAS_READINGS


def _relation_rows(label, target, inputs, formula, block, frame, scenario):
    visible = frame.index[frame[list(inputs)].notna().all(axis=1) & frame[target].isna()]
    part = block[block.target.eq(target) & block.row_id.isin(visible)]
    if part.empty:
        return []
    errors = part.assign(error=(part.prediction - part.truth).abs()).groupby("method").error.mean()
    rows = [{"relation": label, "scenario": scenario, "n": len(visible), "method": m, "MAE": e} for m, e in errors.items()]
    if formula is not None:
        cells = part.drop_duplicates("row_id")
        estimate = np.asarray(formula(frame.loc[cells.row_id]), dtype=float)
        error = np.abs(finalise(target, estimate) - cells.truth.to_numpy()).mean()
        rows.append({"relation": label, "scenario": scenario, "n": len(visible), "method": FORMULA, "MAE": error})
    return rows


def relation_table(scored, measured, config):
    """Error on cells whose defining inputs are all visible, against the definition applied to those inputs."""
    rows = []
    main = scored[scored.experiment.eq("main") & scored.seed.eq(config.seed)]
    for scenario in config.noise_series:
        block, frame = main[main.scenario.eq(scenario)], measured[scenario]
        for label, (target, inputs, formula) in RELATIONS.items():
            rows.extend(_relation_rows(label, target, inputs, formula, block, frame, scenario))
        # The allowed range of leakage is valid only when the gases are exact.
        exact = scenario == config.noise_series[0]
        rows.extend(_relation_rows(LEAKAGE_RELATION, "leakage", LEAKAGE_INPUTS, _leakage if exact else None,
                                   block, frame, scenario))
    return pd.DataFrame(rows)


def relation_matrix(relations, scenario):
    """One scenario of ``relation_table`` as relations by method."""
    block = relations[relations.scenario.eq(scenario)]
    matrix = block.pivot(index=["relation", "n"], columns="method", values="MAE")
    order = [m for m in ("Median", *NUMERIC_METHODS[1:], FORMULA) if m in matrix]
    return matrix.reindex(sorted(matrix.index, key=lambda key: list(block.relation.unique()).index(key[0])))[order]


def _two_steps(target):
    direct = set(GENERATOR_LINKS.get(target, ()))
    return {q for p in direct for q in GENERATOR_LINKS.get(p, ())} - direct - {target}


def _area_under_curve(score, positive):
    """Chance that a linked pair outranks an unlinked one."""
    ranks = rankdata(score)
    linked, other = positive.sum(), (~positive).sum()
    return (ranks[positive].sum() - linked * (linked + 1) / 2) / (linked * other) if linked and other else np.nan


def classify_links(importance):
    """Label each target and predictor pair as direct, two steps apart, or unrelated."""
    table = importance[importance.target.ne("random_placebo")].copy()
    table["direct"] = [p in GENERATOR_LINKS.get(t, ()) for t, p in zip(table.target, table.predictor)]
    table["two steps"] = [(not d) and p in _two_steps(t) for t, p, d in zip(table.target, table.predictor, table.direct)]
    table["unrelated"] = ~table.direct & ~table["two steps"]
    table["relied on"] = table.importance > LINK_THRESHOLD
    return table


def link_table(importance):
    """How far permutation importance matches the links in the generator's definition."""
    rows = []
    table = classify_links(importance)
    for (method, scenario), block in table.groupby(["method", "scenario"], sort=False):
        relied = block["relied on"].to_numpy()
        direct = block.direct.to_numpy()
        share = max(relied.sum(), 1)
        rows.append({
            "method": method, "scenario": scenario, "direct links": int(direct.sum()),
            "ranking score": _area_under_curve(block.importance.to_numpy(), direct),
            "direct links found": (relied & direct).sum() / max(direct.sum(), 1),
            "relied on: direct": (relied & direct).sum() / share,
            "relied on: two steps": (relied & block["two steps"].to_numpy()).sum() / share,
            "relied on: unrelated": (relied & block.unrelated.to_numpy()).sum() / share,
        })
    return pd.DataFrame(rows)


def missed_links(importance, scenario, method):
    """Direct links below the threshold, strongest first."""
    table = classify_links(importance)
    block = table[table.scenario.eq(scenario) & table.method.eq(method) & table.direct & ~table["relied on"]]
    return block.sort_values("importance", ascending=False)[["target", "predictor", "importance"]].reset_index(drop=True)


def _pair_correlations(frame, rows_by_pair):
    pearson, spearman = [], []
    for (a, b), rows in rows_by_pair.items():
        x, y = frame[a].to_numpy()[rows], frame[b].to_numpy()[rows]
        pearson.append(np.corrcoef(x, y)[0, 1])
        spearman.append(np.corrcoef(rankdata(x), rankdata(y))[0, 1])
    return np.array(pearson), np.array(spearman)


def correlation_table(scored, draws, measured, truth, config, draw_count=5):
    """Whether correlations survive on the rows that contain an imputed cell of each pair.

    Rows with both cells observed are left out, because they would hide any change.
    Repeated imputations are judged one completed dataset at a time and then averaged.
    """
    columns = list(NUMERIC_TARGETS)
    missing = measured[columns].isna()
    rows_by_pair = {(a, b): (missing[a] | missing[b]).to_numpy() for a, b in combinations(columns, 2)}
    labels = [" ~ ".join(pair) for pair in rows_by_pair]
    true_pearson, true_spearman = _pair_correlations(truth, rows_by_pair)

    def gaps(frame):
        pearson, spearman = _pair_correlations(frame, rows_by_pair)
        return np.abs(pearson - true_pearson), np.abs(spearman - true_spearman)

    rows = []
    selected = scored[scored.scenario.eq(config.representative) & scored.experiment.eq("main")
                      & scored.target.ne(CATEGORY_TARGET) & scored.seed.eq(config.seed)]
    for method, block in selected.groupby("method", sort=False):
        pearson, spearman = gaps(complete_table(measured, block[["row_id", "target"]].assign(value=block.prediction.to_numpy())))
        rows.append({"method": method, "mean Pearson gap": pearson.mean(), "largest Pearson gap": pearson.max(),
                     "mean Spearman gap": spearman.mean(), "largest gap pair": labels[int(pearson.argmax())]})
    numeric_draws = draws[draws.target.isin(columns) & draws.method.eq("Joint miceforest PMM")] if len(draws) else draws
    if len(numeric_draws):
        results = [gaps(complete_table(measured, part)) for draw, part in numeric_draws.groupby("draw") if draw < draw_count]
        pearson = np.mean([r[0] for r in results], axis=0)
        spearman = np.mean([r[1] for r in results], axis=0)
        rows.append({"method": "Joint miceforest PMM (per draw)", "mean Pearson gap": pearson.mean(),
                     "largest Pearson gap": pearson.max(), "mean Spearman gap": spearman.mean(),
                     "largest gap pair": labels[int(pearson.argmax())]})
    return pd.DataFrame(rows).sort_values("mean Pearson gap").reset_index(drop=True)


def visibility_table(scored, measured, config):
    """Skill by the share of a target's linked columns that were visible in the row."""
    block = scored[scored.experiment.eq("main") & scored.scenario.eq(config.representative) & scored.seed.eq(config.seed)
                   & scored.target.isin(SIGNAL_TARGETS)]
    rows = []
    for target, part in block.groupby("target", sort=False):
        links = list(GENERATOR_LINKS[target])
        share = measured.loc[part.row_id, links].notna().mean(axis=1).to_numpy()
        level = np.select([share == 1, share >= 0.6], ["all linked columns seen", "most seen"], default="few seen")
        table = part.assign(level=level, error=(part.prediction - part.truth).abs())
        errors = table.groupby(["method", "level"]).error.agg(["mean", "size"]).reset_index()
        baseline = errors[errors.method.eq("Median")].set_index("level")["mean"]
        errors["skill"] = 1 - errors["mean"] / errors.level.map(baseline)
        rows.append(errors[errors.method.ne("Median")].assign(target=target).rename(columns={"size": "n"}))
    return pd.concat(rows, ignore_index=True)[["target", "method", "level", "n", "skill"]]


def recoverable_table(truth, numeric, config):
    """How much of each column the other true columns could explain, against what was reached.

    The first column uses the complete truth, which no model sees, so it is a reference
    for what the data allows. Rows are shuffled across folds because they are stored in grid order.
    """
    main = numeric[numeric.experiment.eq("main") & numeric.seed.eq(config.seed) & numeric.method.ne("Median")]
    easiest = main.groupby("scenario").skill.mean().idxmax()
    splitter = KFold(5, shuffle=True, random_state=config.seed)
    rows = []
    for target in NUMERIC_TARGETS:
        model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.1, random_state=config.seed)
        predicted = cross_val_predict(model, truth.drop(columns=target), truth[target], cv=splitter)
        residual = np.sum((truth[target] - predicted) ** 2)
        reachable = 1 - residual / np.sum((truth[target] - truth[target].mean()) ** 2)
        reached = main[main.target.eq(target)]
        rows.append({
            "target": target, "R2 from all other true columns": reachable,
            "best R2, representative scenario": reached[reached.scenario.eq(config.representative)].R2.max(),
            "best R2, easiest scenario": reached[reached.scenario.eq(easiest)].R2.max(),
        })
    return pd.DataFrame(rows)


def progress_table(numeric, root, config):
    """Mean skill of the earlier study against this run, on the same cells."""
    path = Path(root) / V1_SCORES
    if config.rows is not None or not path.exists():
        return None

    def mean_skill(table):
        block = table[table.scenario.eq(config.representative) & table.seed.eq(config.seed)
                      & table.experiment.isin(["main", "pmm", "joint"]) & table.target.isin(SIGNAL_TARGETS)]
        return block.groupby("method").skill.mean()

    table = pd.DataFrame({"earlier study": mean_skill(pd.read_csv(path)), "this run": mean_skill(numeric)}).dropna()
    table = table.drop(index="Median", errors="ignore")
    table["change"] = table["this run"] - table["earlier study"]
    return table.sort_values("this run", ascending=False)


def base_table(root):
    """Selection score of each model with no base, a straight-line base and a second-order base.

    Scores are error relative to the median on selection cells, so lower is better.
    Only candidates that differ in nothing but the base are compared.
    """
    rows = []
    for method, name in SELECTION_FILES.items():
        path = Path(root) / SELECTION / f"{name}.csv"
        if not path.exists():
            continue
        table = pd.read_csv(path)
        scores = table.assign(score=table[SIGNAL_TARGETS].mean(axis=1)).groupby("settings").score.mean()
        settings = {text: json.loads(text) for text in scores.index}
        plain = next(text for text, values in settings.items() if values["base"] == "none")

        def match(base):
            wanted = {**settings[plain], "base": base}
            return next(scores[t] for t, values in settings.items() if values == wanted)

        rows.append({"method": method, "no base": match("none"), "straight-line base": match("linear"),
                     "second-order base": match("block")})
    return pd.DataFrame(rows)
