"""Scenarios that the generator's nine files do not cover, built from those files.

Every scenario keeps the generator's truth file unchanged and differs from a generator scenario
in one respect only, so its effect can be measured with everything else held fixed:

- **age**, **value** and **group** hide the same share of cells as the representative scenario,
  but choose them differently.
- **errors** keeps the representative scenario's mask and makes some visible values wrong.
- **sparse** hides more cells at random than any generator file, at the representative noise.

The files are written again identically each time they are built.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from .columns import CATEGORY_TARGET, NAMES
from .data import discover

AGE, VALUE, GROUP, ERRORS, SPARSE = "age", "value", "group", "errors", "sparse"
# Columns that go missing together in the group scenario. These follow how the records are
# collected (a site survey, a gas test, an environment survey, the asset nameplate), and are
# the groups used by Subteam A. life, leakage and the placebo keep their own draws.
GROUPS = {
    "location": ("x", "y", "altitude"),
    "gases": ("measured_gas_a", "measured_gas_b", "measured_gas_c", "total_gas"),
    "environment": ("measured_humid", "measured_salt", "measured_overgrowth"),
    "date and model": ("install_date", "model_id"),
}
# Recording errors in the errors scenario, as in Subteam A's structural-error test.
DATE_ERROR_RATE, DATE_ERROR_DAYS, MODEL_ERROR_RATE = 0.05, 60, 0.02
DESCRIPTIONS = {
    AGE: "Older assets lose more cells: the hidden share falls from 30% for the oldest row to 10% for the newest.",
    VALUE: "Higher values are hidden more often: the hidden share rises from 10% for the lowest value of a "
           "column to 30% for the highest. model_id is nominal and is hidden at random.",
    GROUP: "Related columns go missing together: location, gas readings, environment readings, and date with "
           "model are each hidden as a block. life, leakage and the placebo are hidden on their own.",
    ERRORS: f"The representative mask, with {DATE_ERROR_RATE:.0%} of visible installation days shifted by "
            f"{DATE_ERROR_DAYS} days and {MODEL_ERROR_RATE:.0%} of visible model_id values replaced by another model.",
    SPARSE: "More cells hidden completely at random than in any generator file, at the representative noise level.",
}


def _chance(truth, mechanism, extra):
    """Chance that a cell visible in the base file is hidden as well; the mean is ``extra``."""
    if mechanism == AGE:
        age_rank = 1 - truth.install_date.rank(pct=True).to_numpy()
        return np.repeat((2 * extra * age_rank)[:, None], truth.shape[1], axis=1)
    chance = 2 * extra * truth.rank(pct=True).to_numpy()
    chance[:, list(truth.columns).index(CATEGORY_TARGET)] = extra
    return chance


def _grouped_draws(generator, shape):
    """Uniform draws in which every column of a group takes its group's first draw."""
    draws = generator.random(shape)
    for members in GROUPS.values():
        columns = [NAMES.index(c) for c in members]
        draws[:, columns] = draws[:, columns[:1]]
    return draws


def _with_errors(measured, generator):
    """Shift some visible installation days and swap some visible model_id values."""
    wrong = measured.copy()
    date = wrong.install_date
    shifted = date.notna() & (generator.random(len(wrong)) < DATE_ERROR_RATE)
    sign = np.where(generator.random(len(wrong)) < 0.5, -1, 1)
    sign = np.where(date.to_numpy() < DATE_ERROR_DAYS, 1, sign)  # a day count cannot fall below zero
    wrong.loc[shifted, "install_date"] = date[shifted] + sign[shifted.to_numpy()] * DATE_ERROR_DAYS
    model = wrong[CATEGORY_TARGET]
    swapped = model.notna() & (generator.random(len(wrong)) < MODEL_ERROR_RATE)
    offset = generator.integers(1, 4, len(wrong))
    wrong.loc[swapped, CATEGORY_TARGET] = (model[swapped] + offset[swapped.to_numpy()]) % 4
    return wrong


def _write(folder, name, measured, truth_path):
    contents = {"measured": measured.to_csv(index=False, lineterminator="\n").encode("utf-8"),
                "truth": Path(truth_path).read_bytes()}  # the truth is the generator's file, unchanged
    for kind, content in contents.items():
        path = folder / f"{name}-{kind}.csv"
        if not path.exists() or path.read_bytes() != content:
            path.write_bytes(content)


def _read(path):
    # Exact parsing, so that visible cells are written back with the digits the generator wrote.
    frame = pd.read_csv(path, float_precision="round_trip")
    if tuple(frame.columns) != NAMES:
        raise ValueError(f"Unexpected columns in {path}")
    return frame


def scenario_names(config):
    names = [f"{config.representative}-{m}" for m in config.mechanisms]
    noise = config.representative.split("-noise")[1]
    names += [f"completion{c:g}-noise{noise}-{SPARSE}" for c in config.sparse_completions]
    return names


def write_mechanism_files(root, config):
    """Write one measured/truth pair per added scenario and return their scenarios."""
    base_scenarios = discover(Path(root) / config.data_directory)
    representative = base_scenarios[config.representative]
    same_noise = [s for s in base_scenarios.values() if s.noise == representative.noise]
    base = max(same_noise, key=lambda s: s.completion)
    if base.completion <= representative.completion:
        raise ValueError("A file with more visible cells than the representative scenario is needed")
    measured, truth = _read(base.measured), _read(base.truth)
    # Hiding this share of the visible cells brings the file to the representative completion.
    extra = (base.completion - representative.completion) / base.completion
    folder = Path(root) / config.mechanism_directory
    folder.mkdir(parents=True, exist_ok=True)
    for index, mechanism in enumerate(config.mechanisms):
        generator = np.random.default_rng(config.seed + index)
        name = f"{representative.name}-{mechanism}"
        if mechanism in (AGE, VALUE):
            hidden = generator.random(measured.shape) < _chance(truth, mechanism, extra)
            _write(folder, name, measured.mask(hidden), base.truth)
        elif mechanism == GROUP:
            hidden = _grouped_draws(generator, measured.shape) < extra
            _write(folder, name, measured.mask(hidden), base.truth)
        elif mechanism == ERRORS:
            _write(folder, name, _with_errors(_read(representative.measured), generator), representative.truth)
        else:
            raise ValueError(f"Unknown mechanism: {mechanism}")
    # Heavier random missingness, nested inside the sparsest generator file.
    sparsest = min(same_noise, key=lambda s: s.completion)
    start = _read(sparsest.measured)
    draws = np.random.default_rng(config.seed + 100).random(start.shape)
    for completion in sorted(config.sparse_completions, reverse=True):
        if completion >= sparsest.completion:
            raise ValueError("Sparse completions must be below every generator completion")
        hidden = draws < (sparsest.completion - completion) / sparsest.completion
        name = f"completion{completion:g}-noise{config.representative.split('-noise')[1]}-{SPARSE}"
        _write(folder, name, start.mask(hidden), sparsest.truth)
    found = discover(folder)
    return {name: found[name] for name in scenario_names(config)}
