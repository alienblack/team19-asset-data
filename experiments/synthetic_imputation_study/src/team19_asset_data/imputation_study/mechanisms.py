"""Scenarios in which cells are not hidden completely at random.

The generator hides every cell with the same chance. These files hide the same share of
cells overall, but the chance depends on the asset's age or on the value itself, so the
effect of the missingness mechanism can be measured with everything else unchanged.
They are built from the generator's own files and are written again identically each time.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from .columns import CATEGORY_TARGET, NAMES
from .data import discover

AGE, VALUE = "age", "value"
DESCRIPTIONS = {
    AGE: "Older assets lose more cells: the hidden share falls from 30% for the oldest row to 10% for the newest.",
    VALUE: "Higher values are hidden more often: the hidden share rises from 10% for the lowest value of a "
           "column to 30% for the highest. model_id is nominal and is hidden at random.",
}


def _chance(truth, mechanism, extra):
    """Chance that a cell visible in the base file is hidden as well; the mean is ``extra``."""
    if mechanism == AGE:
        age_rank = 1 - truth.install_date.rank(pct=True).to_numpy()
        return np.repeat((2 * extra * age_rank)[:, None], truth.shape[1], axis=1)
    chance = 2 * extra * truth.rank(pct=True).to_numpy()
    chance[:, list(truth.columns).index(CATEGORY_TARGET)] = extra
    return chance


def write_mechanism_files(root, config):
    """Write one measured/truth pair per mechanism and return their scenarios."""
    base_scenarios = discover(Path(root) / config.data_directory)
    representative = base_scenarios[config.representative]
    base = max((s for s in base_scenarios.values() if s.noise == representative.noise), key=lambda s: s.completion)
    if base.completion <= representative.completion:
        raise ValueError("A file with more visible cells than the representative scenario is needed")
    # Exact parsing, so that visible cells are written back with the digits the generator wrote.
    measured = pd.read_csv(base.measured, float_precision="round_trip")
    truth = pd.read_csv(base.truth, float_precision="round_trip")
    if tuple(measured.columns) != NAMES:
        raise ValueError(f"Unexpected columns in {base.measured}")
    # Hiding this share of the visible cells brings the file to the representative completion.
    extra = (base.completion - representative.completion) / base.completion
    folder = Path(root) / config.mechanism_directory
    folder.mkdir(parents=True, exist_ok=True)
    for index, mechanism in enumerate(config.mechanisms):
        generator = np.random.default_rng(config.seed + index)
        hidden = generator.random(measured.shape) < _chance(truth, mechanism, extra)
        name = f"{representative.name}-{mechanism}"
        contents = {"measured": measured.mask(hidden).to_csv(index=False, lineterminator="\n").encode("utf-8"),
                    "truth": Path(base.truth).read_bytes()}  # the truth is the generator's file, unchanged
        for kind, content in contents.items():
            path = folder / f"{name}-{kind}.csv"
            if not path.exists() or path.read_bytes() != content:
                path.write_bytes(content)
    found = discover(folder)
    return {name: found[name] for name in (f"{representative.name}-{m}" for m in config.mechanisms)}
