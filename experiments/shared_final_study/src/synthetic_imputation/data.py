"""Reading the measured/truth file pairs and auditing them against the generator's design."""

from dataclasses import dataclass
from pathlib import Path
import hashlib
import re

import numpy as np
import pandas as pd

from .columns import BY_NAME, NAMES

FILE_PATTERN = re.compile(
    r"^(completion(?P<completion>[\d.]+)-noise(?P<noise>[\d.]+?)(?:-(?P<mechanism>[a-z]+))?)"
    r"-(?P<kind>measured|truth)\.csv$"
)
RANDOM = "random"  # cells hidden completely at random, as the generator writes them


@dataclass(frozen=True)
class Scenario:
    name: str
    completion: float
    noise: float
    measured: Path
    truth: Path
    mechanism: str = RANDOM  # how cells were chosen for hiding


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def discover(*directories):
    """Pair measured and truth files by name; unpaired files are an error. Missing folders are skipped."""
    found = {}
    for directory in directories:
        for path in sorted(Path(directory).glob("*.csv")):
            match = FILE_PATTERN.match(path.name)
            if match is None:
                raise ValueError(f"Unrecognised file name: {path.name}")
            entry = found.setdefault(match[1], {"completion": match["completion"], "noise": match["noise"],
                                                "mechanism": match["mechanism"] or RANDOM})
            entry[match["kind"]] = path
    if not found:
        raise FileNotFoundError(f"No scenario files in {directories}")
    result = {}
    for name, entry in found.items():
        if not {"measured", "truth"} <= entry.keys():
            raise ValueError(f"Unpaired scenario: {name}")
        result[name] = Scenario(
            name, float(entry["completion"]), float(entry["noise"]), entry["measured"], entry["truth"],
            entry["mechanism"],
        )
    return result


def _read(path):
    frame = pd.read_csv(path)
    if tuple(frame.columns) != NAMES:
        raise ValueError(f"Unexpected columns in {path}: {list(frame.columns)}")
    frame = frame.astype(float)
    if np.isinf(frame.to_numpy()).any():
        raise ValueError(f"Infinite values in {path}")
    frame.index.name = "row_id"
    return frame


def read_measured(scenario):
    """The table a model sees."""
    return _read(scenario.measured)


def read_truth(scenario):
    truth = _read(scenario.truth)
    if truth.isna().any().any():
        raise ValueError(f"Truth has missing cells: {scenario.truth}")
    return truth


def audit_files(scenarios):
    rows = []
    for scenario in scenarios.values():
        measured, truth = read_measured(scenario), read_truth(scenario)
        if measured.shape != truth.shape:
            raise ValueError(f"Shape mismatch in {scenario.name}")
        rows.append({
            "scenario": scenario.name, "completion": scenario.completion, "noise": scenario.noise,
            "rows": len(measured), "columns": measured.shape[1],
            "missing share": measured.isna().to_numpy().mean(),
            "columns with missing cells": int(measured.isna().any().sum()),
        })
    return pd.DataFrame(rows)


def audit_fields(scenarios):
    """Compare each column with its definition: masking rate and the noise bound."""
    rows = []
    for scenario in scenarios.values():
        measured, truth = read_measured(scenario), read_truth(scenario)
        for name in NAMES:
            seen = measured[name].notna()
            a, b = measured.loc[seen, name], truth.loc[seen, name]
            positive = b.abs() > 0
            relative = ((a - b).abs() / b.abs())[positive]
            rows.append({
                "scenario": scenario.name, "completion": scenario.completion, "noise": scenario.noise,
                "column": name, "observed share": seen.mean(),
                "changed share": float((a != b).mean()),
                "largest relative change": float(relative.max()) if len(relative) else 0.0,
                "measurement MAE": float((a - b).abs().mean()),
                "defined as noisy": BY_NAME[name].noisy,
            })
    return pd.DataFrame(rows)


def audit_design(scenarios):
    """Check the properties that make scenarios comparable cell by cell."""
    items = list(scenarios.values())
    truths = [read_truth(s) for s in items]
    masks = {(s.completion, s.noise): read_measured(s).isna() for s in items}
    completions = sorted({s.completion for s in items})
    noises = sorted({s.noise for s in items})
    same_across_noise = all(
        masks[(c, n)].equals(masks[(c, noises[0])]) for c in completions for n in noises
    )
    nested = all(
        (masks[(high, noises[0])] <= masks[(low, noises[0])]).all().all()
        for low, high in zip(completions, completions[1:])
    )
    return pd.DataFrame([
        {"property": "Every file has the same truth", "holds": all(t.equals(truths[0]) for t in truths[1:])},
        {"property": "A completeness level hides the same cells at every noise level", "holds": same_across_noise},
        {"property": "Cells hidden at higher completeness are also hidden at lower completeness", "holds": nested},
    ])


def relationship_residuals(truth):
    """Largest deviation from the generator's definitions in the truth table."""
    t = truth
    gas = 25000 * (t.total_gas / 300000)
    day = t.install_date
    hazards = np.column_stack([
        day * (t.measured_humid + t.measured_salt + t.measured_overgrowth),
        day * (t.measured_humid + t.measured_overgrowth),
        day * (t.measured_salt + t.measured_overgrowth),
        day * t.measured_overgrowth,
    ])
    expected = 100000 - gas - hazards[np.arange(len(t)), t.model_id.astype(int)]
    checks = {
        "altitude = x + y + 100": t.altitude - t.x - t.y - 100,
        "overgrowth = max(2 humid - salt, 0)":
            t.measured_overgrowth - (2 * t.measured_humid - t.measured_salt).clip(lower=0),
        "total_gas = gas a + b + c": t.total_gas - t.measured_gas_a - t.measured_gas_b - t.measured_gas_c,
        "life follows the formula of the row's model": t.life - expected,
    }
    return pd.DataFrame([{"definition": k, "largest residual": float(v.abs().max())} for k, v in checks.items()])
