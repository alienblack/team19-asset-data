"""Evaluation of the relationships between columns.

The generator's definitions appear here and are used only after fitting, to judge
what the models learned. Nothing in this module feeds back to a model.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_predict

from .columns import GAS_READINGS, NUMERIC_TARGETS, finalise
from .names import NUMERIC_METHODS

FORMULA = "Formula"


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


