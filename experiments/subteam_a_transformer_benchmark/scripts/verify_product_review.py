"""Independently verify the saved product-review demo without fitting a model."""
from argparse import ArgumentParser
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from subteam_a.data import TRANSFORMER
from subteam_a.evaluate import relative_skill, tolerance_limit


def read_table(folder, name):
    path = folder / f"{name}.csv"
    if not path.exists():
        path = folder / f"{name}.csv.gz"
    return pd.read_csv(path, float_precision="round_trip")


def close(actual, expected, message):
    if not np.allclose(actual, expected, rtol=1e-9, atol=1e-8, equal_nan=True):
        raise ValueError(message)


def verify(folder):
    folder = Path(folder)
    measured = read_table(folder, "measured_test")
    completed = read_table(folder, "completed_test")
    ledger = read_table(folder, "hidden_predictions")
    metrics = read_table(folder, "field_metrics")
    summary = read_table(folder, "comparison")
    partitions = read_table(folder, "partitions")
    if ledger.duplicated(["condition", "method", "asset_id", "attribute"]).any():
        raise ValueError("Duplicate hidden-cell predictions")
    if metrics.duplicated(["condition", "method", "attribute"]).any():
        raise ValueError("Duplicate field metric rows")
    if measured.duplicated(["condition", "asset_id"]).any():
        raise ValueError("Duplicate measured records")
    block_counts = partitions.groupby("spatial_block").partition.nunique()
    if (block_counts > 1).any() or partitions.asset_id.duplicated().any():
        raise ValueError("Spatial partitions overlap")

    for (condition, method), frame in completed.groupby(["condition", "method"]):
        raw = measured.loc[measured.condition.eq(condition)].set_index("asset_id")
        if frame.asset_id.duplicated().any() or set(frame.asset_id) != set(raw.index):
            raise ValueError("Completed records do not match the measured asset IDs")
        filled = frame.set_index("asset_id").loc[raw.index, list(TRANSFORMER.names)]
        values = raw[list(TRANSFORMER.names)].to_numpy()
        points = filled.to_numpy()
        visible = np.isfinite(values)
        if not np.array_equal(points[visible], values[visible]):
            raise ValueError("A recorded observation changed in the completed export")
        if not np.isfinite(points).all():
            raise ValueError("The completed export contains a missing or nonfinite value")
        hidden = raw[list(TRANSFORMER.names)].isna().stack()
        expected = hidden[hidden].index
        cells = ledger.loc[ledger.condition.eq(condition) & ledger.method.eq(method)]
        indexed = cells.set_index(["asset_id", "attribute"])
        if set(indexed.index) != set(expected):
            raise ValueError("The ledger does not match the actual hidden cells")
        row_numbers = filled.index.get_indexer(cells.asset_id)
        columns = np.array([TRANSFORMER.i(name) for name in cells.attribute])
        close(points[row_numbers, columns], cells["fill"], "Completed fills disagree with the ledger")

    for row in metrics.itertuples():
        cells = ledger.loc[ledger.condition.eq(row.condition) & ledger.method.eq(row.method)
                           & ledger.attribute.eq(row.attribute)]
        if len(cells) != row.n:
            raise ValueError("The hidden-cell count disagrees with a field metric")
        if row.type == "categorical":
            errors = cells["fill"].ne(cells.truth).to_numpy()
            floor_errors = cells.floor.ne(cells.truth).to_numpy()
            within = ~errors
            probabilities = cells[[f"p_code_{code}" for code in range(4)]].to_numpy()
            close(probabilities.sum(axis=1), 1.0, "Category probabilities are not normalised")
            if (probabilities < 0).any():
                raise ValueError("Negative categorical probability")
        else:
            errors = (cells["fill"] - cells.truth).abs().to_numpy()
            floor_errors = (cells.floor - cells.truth).abs().to_numpy()
            within = errors <= tolerance_limit(row.attribute, cells.truth.to_numpy(), 0, tight=False) + 1e-10
            coverage = ((cells.truth >= cells.lower) & (cells.truth <= cells.upper)).mean()
            close(coverage, row.coverage, "Numeric coverage disagrees with the ledger")
        close(errors.mean(), row.error, "Field MAE or misclassification error disagrees with the ledger")
        close(floor_errors.mean(), row.floor_error, "Floor error disagrees with the ledger")
        close(within.mean(), row.tolerance, "Loose tolerance agreement disagrees with the ledger")
        score = relative_skill(errors.mean(), floor_errors.mean(), row.type == "categorical")
        close(score, row.skill, "Field skill disagrees with the ledger")

    numeric = metrics.loc[metrics.type.eq("numeric") & metrics.attribute.ne("random_placebo")].copy()
    numeric["breaks"] = numeric.skill.lt(0.10) | numeric.tolerance.lt(0.50)
    recalculated = numeric.groupby(["condition", "method"]).agg(
        mean_numeric_skill=("skill", "mean"), fields_breaking=("breaks", "sum"),
        mean_within_loose=("tolerance", "mean"), mean_numeric_coverage=("coverage", "mean"),
    ).sort_index()
    supplied = summary.set_index(["condition", "method"]).sort_index()
    if not recalculated.index.equals(supplied.index):
        raise ValueError("Comparison rows do not match the scored conditions and methods")
    close(recalculated.to_numpy(), supplied[recalculated.columns].to_numpy(),
          "The comparison summary disagrees with the field metrics")
    components = read_table(folder, "gmm_components")
    close(components.posterior_weight.sum(), 1.0, "GMM component weights do not sum to one")
    close(components.posterior_weight * components.conditional_mean_days,
          components.weighted_mean_contribution, "GMM component contributions disagree")
    example = read_table(folder, "gmm_example_record")
    date = example.loc[example.attribute.eq("install_date"), "gmm_completed_value"].iloc[0]
    close(max(0, components.weighted_mean_contribution.sum()), date,
          "GMM component contributions do not reproduce the example fill")
    return {"status": "PASS", "field_metrics_checked": len(metrics),
            "hidden_predictions_checked": len(ledger),
            "completed_model_conditions": int(completed.groupby(["condition", "method"]).ngroups),
            "observed_values_preserved": True, "spatial_partitions_disjoint": True}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.output_directory), indent=2))


if __name__ == "__main__":
    main()
