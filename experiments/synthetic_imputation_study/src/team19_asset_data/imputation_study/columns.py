"""What each dataset column means, and what that meaning implies for modelling.

Every decision here follows from the column definition in
``data/datasets/generator/SyntheticDataset.ipynb``: its type, measurement scale,
valid range and whether it carries measurement noise. Nothing here passes a
relationship between columns to a model. ``COMPANIONS`` and ``GENERATOR_LINKS``
are used only after fitting, to organise the evaluation.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

GRID_MAX = 99
LIFE_CEILING = 100_000  # the life formula starts here and only subtracts

ABSOLUTE, RELATIVE = "absolute", "relative"
INTERVAL, RATIO, NOMINAL = "interval", "ratio", "nominal"
LOCATION, ASSET, READING, CONTROL = "location and date", "asset condition", "field reading", "control"


@dataclass(frozen=True)
class Column:
    name: str
    meaning: str
    family: str
    unit: str
    scale: str
    noisy: bool  # the measured value carries multiplicative noise
    treatment: str
    integer: bool = False
    lower: float | None = None
    upper: float | None = None
    tolerance: tuple = ()
    tolerance_kind: str | None = None
    zero_inflated: bool = False


COLUMNS = (
    Column(
        "x", "Grid row of the asset location", LOCATION, "grid cells", INTERVAL, False,
        "Numeric coordinate, not a category. Predictions stay on the grid (rounded, 0-99). "
        "The zero point is arbitrary, so tolerance is in cells rather than percent.",
        integer=True, lower=0, upper=GRID_MAX, tolerance=(1, 5), tolerance_kind=ABSOLUTE,
    ),
    Column(
        "y", "Grid column of the asset location", LOCATION, "grid cells", INTERVAL, False,
        "Same treatment as x.",
        integer=True, lower=0, upper=GRID_MAX, tolerance=(1, 5), tolerance_kind=ABSOLUTE,
    ),
    Column(
        "install_date", "Installation day, counted from the start of the sequence", LOCATION, "days",
        INTERVAL, False,
        "Whole-day counter with no calendar origin, so it stays a number and is not converted "
        "to age. The origin is arbitrary, so tolerance is in days rather than percent.",
        integer=True, lower=0, tolerance=(7, 30), tolerance_kind=ABSOLUTE,
    ),
    Column(
        "model_id", "Transformer model", ASSET, "category", NOMINAL, False,
        "Nominal label: the numbers 0-3 have no order. One-hot encoded as a predictor, with "
        "missing as its own level, and classified as a target. Never regressed or rounded.",
    ),
    Column(
        "altitude", "Ground altitude at the location", LOCATION, "synthetic units", INTERVAL, False,
        "Whole-number field limited to the range the grid can produce (100-298). Its zero point "
        "does not compare assets, so tolerance is in units rather than percent.",
        integer=True, lower=100, upper=2 * GRID_MAX + 100, tolerance=(1, 5), tolerance_kind=ABSOLUTE,
    ),
    Column(
        "measured_humid", "Humidity exposure reading", READING, "index", RATIO, True,
        "Noisy reading between 0 and 1 with a true zero, so percent tolerance is valid. Predictions "
        "are limited to that range. Scored against the noise-free truth.",
        lower=0, upper=1, tolerance=(0.10, 0.15), tolerance_kind=RELATIVE,
    ),
    Column(
        "measured_salt", "Salt exposure reading", READING, "index", RATIO, True,
        "Same treatment as measured_humid.",
        lower=0, upper=1, tolerance=(0.10, 0.15), tolerance_kind=RELATIVE,
    ),
    Column(
        "measured_overgrowth", "Vegetation overgrowth reading", READING, "index", RATIO, True,
        "Floored at zero by definition, so exact zeros are real readings, not missing data. Percent "
        "tolerance cannot apply to a true zero; zero and positive cells are evaluated separately.",
        lower=0, upper=2, tolerance=(0.10, 0.15), tolerance_kind=RELATIVE, zero_inflated=True,
    ),
    Column(
        "life", "Remaining-life indicator", ASSET, "synthetic units", RATIO, False,
        "Exact asset outcome. It sits in a narrow band just under its ceiling, where ±10% would "
        "accept almost any value, so tolerance is in units. Predictions cannot exceed the ceiling.",
        lower=0, upper=LIFE_CEILING, tolerance=(100, 500), tolerance_kind=ABSOLUTE,
    ),
    Column(
        "random_placebo", "Random number unrelated to any other column", CONTROL, "index", RATIO, True,
        "Negative control, limited to 0-1. It is imputed and scored like any target, but any real "
        "gain over the median would be a false finding.",
        lower=0, upper=1, tolerance=(0.10, 0.15), tolerance_kind=RELATIVE,
    ),
    Column(
        "measured_gas_a", "Dissolved gas A reading", ASSET, "synthetic units", RATIO, True,
        "Noisy positive concentration. The values span a narrow range, so no log transform is "
        "needed. Tolerance is tighter than for field readings because the range is narrow.",
        lower=0, tolerance=(0.05, 0.10), tolerance_kind=RELATIVE,
    ),
    Column(
        "measured_gas_b", "Dissolved gas B reading", ASSET, "synthetic units", RATIO, True,
        "Same treatment as measured_gas_a.",
        lower=0, tolerance=(0.05, 0.10), tolerance_kind=RELATIVE,
    ),
    Column(
        "measured_gas_c", "Dissolved gas C reading", ASSET, "synthetic units", RATIO, True,
        "Same treatment as measured_gas_a.",
        lower=0, tolerance=(0.05, 0.10), tolerance_kind=RELATIVE,
    ),
    Column(
        "total_gas", "Total of the three true gas levels", ASSET, "synthetic units", RATIO, False,
        "Exact by design, so at non-zero noise it differs from the sum of the gas readings. "
        "It varies by only a few percent, so tolerance is ±1% and ±5%.",
        lower=0, tolerance=(0.01, 0.05), tolerance_kind=RELATIVE,
    ),
    Column(
        "leakage", "Leakage rate of the asset", ASSET, "ratio", RATIO, False,
        "Exact by design and limited to its defined range of 0.8-1.2.",
        lower=0.8, upper=1.2, tolerance=(0.05, 0.10), tolerance_kind=RELATIVE,
    ),
)

BY_NAME = {column.name: column for column in COLUMNS}
NAMES = tuple(column.name for column in COLUMNS)
CATEGORY_TARGET = "model_id"
NUMERIC_TARGETS = tuple(c.name for c in COLUMNS if c.scale != NOMINAL)
TARGETS = NUMERIC_TARGETS + (CATEGORY_TARGET,)
GAS_READINGS = ("measured_gas_a", "measured_gas_b", "measured_gas_c")

# Columns that appear in the generator's definition of each target. Evaluation
# groups cells by which of them were visible; no model receives this table.
COMPANIONS = {
    "x": ("install_date", "altitude"),
    "y": ("x", "altitude"),
    "install_date": ("x", "y"),
    "altitude": ("x", "y"),
    "measured_humid": ("measured_salt", "measured_overgrowth"),
    "measured_salt": ("measured_humid", "measured_overgrowth"),
    "measured_overgrowth": ("measured_humid", "measured_salt"),
    "model_id": ("install_date", "measured_gas_a"),
    "life": ("model_id", "measured_overgrowth"),
    "measured_gas_a": ("model_id", "install_date"),
    "measured_gas_b": ("model_id", "install_date"),
    "measured_gas_c": ("model_id", "install_date"),
    "total_gas": ("install_date", "leakage"),
    "leakage": ("total_gas", "install_date"),
    "random_placebo": (),
}

# Columns linked to each target by the generator's definition, shown beside
# learned importance. These links are read from the definition, not estimated.
GENERATOR_LINKS = {
    "x": ("altitude", "install_date"),
    "y": ("altitude", "install_date", "measured_salt"),
    "install_date": ("x", "y", "model_id", "life", "total_gas", "leakage") + GAS_READINGS,
    "altitude": ("x", "y", "measured_humid"),
    "measured_humid": ("altitude", "measured_overgrowth", "life"),
    "measured_salt": ("y", "measured_overgrowth", "life"),
    "measured_overgrowth": ("measured_humid", "measured_salt", "life"),
    "model_id": ("install_date", "life") + GAS_READINGS,
    "life": ("install_date", "model_id", "measured_humid", "measured_salt", "measured_overgrowth", "total_gas"),
    "measured_gas_a": ("model_id", "install_date", "leakage", "total_gas"),
    "measured_gas_b": ("model_id", "install_date", "leakage", "total_gas"),
    "measured_gas_c": ("model_id", "install_date", "leakage", "total_gas"),
    "total_gas": ("install_date", "leakage", "life") + GAS_READINGS,
    "leakage": ("install_date", "total_gas") + GAS_READINGS,
    "random_placebo": (),
}


def column_table():
    """One row per column, for display in the notebook."""
    rows = []
    for c in COLUMNS:
        if c.tolerance_kind == ABSOLUTE:
            tolerance = f"±{c.tolerance[0]:g} and ±{c.tolerance[1]:g} {c.unit}"
        elif c.tolerance_kind == RELATIVE:
            tolerance = f"±{c.tolerance[0]:.0%} and ±{c.tolerance[1]:.0%}"
        else:
            tolerance = "exact class"
        low = "-" if c.lower is None else f"{c.lower:g}"
        high = "-" if c.upper is None else f"{c.upper:g}"
        rows.append({
            "column": c.name, "meaning": c.meaning, "family": c.family, "unit": c.unit, "scale": c.scale,
            "noisy": "yes" if c.noisy else "no", "valid range": f"{low} to {high}",
            "tolerance": tolerance, "treatment": c.treatment,
        })
    return pd.DataFrame(rows)


def finalise(name, values):
    """Apply the valid range and whole-number rule that follow from a column's meaning."""
    column = BY_NAME[name]
    result = np.asarray(values, dtype=float).copy()
    if column.lower is not None or column.upper is not None:
        result = np.clip(result, column.lower, column.upper)
    return np.round(result) if column.integer else result
