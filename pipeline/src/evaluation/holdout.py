"""Partitions and hide-and-score masks for evaluation.

Two tools live here:

* **Partitions** split records into named groups (train / validation /
  calibration / test) by identifier, optionally keeping every record of a
  group (e.g. a feeder) in one partition. The assignment is written to a
  manifest so every experiment scores the same records.
* **Holdout masks** hide a fraction of CURRENTLY OBSERVED cells so that a
  method's output for those cells can be compared with the values that were
  hidden. Pre-existing missing values are never scored. Categorical holdouts
  can be stratified so that every class with enough observed cells is
  represented.

Nothing here fits or scores anything.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

PARTITIONS = ("train", "validation", "calibration", "test")


# ---------------------------------------------------------------------------
# Partitions
# ---------------------------------------------------------------------------


def _ids(frame: pd.DataFrame, id_column: str) -> pd.Series:
    if id_column not in frame.columns:
        raise ValueError(f"'{id_column}' is required for traceable partitions.")
    ids = frame[id_column].astype(str)
    if frame[id_column].isna().any() or not ids.is_unique or (ids.str.strip() == "").any():
        raise ValueError(f"'{id_column}' must contain unique, non-missing identifiers.")
    return ids


def _allocated_counts(n: int, fractions: Mapping[str, float]) -> dict[str, int]:
    names = list(fractions)
    targets = np.array([fractions[p] for p in names], dtype=float) * n
    counts = np.floor(targets).astype(int)
    remainder = n - int(counts.sum())
    for i in np.argsort(-(targets - counts), kind="stable")[:remainder]:
        counts[i] += 1
    return dict(zip(names, counts.tolist()))


def make_partitions(
    frame: pd.DataFrame,
    fractions: Mapping[str, float],
    seed: int,
    id_column: str,
    strategy: str = "row",
    group_column: str | None = None,
) -> pd.DataFrame:
    """Return a manifest ``[id_column, partition]`` for ``frame``.

    ``fractions`` maps partition names to shares that sum to one. ``row``
    draws records independently; ``group`` keeps every record with the same
    ``group_column`` value in one partition (greedy fill of the partition that
    is furthest below its target).
    """
    if not fractions or any(not isinstance(v, (int, float)) or v < 0 for v in fractions.values()):
        raise ValueError("fractions must map partition names to non-negative shares.")
    if not np.isclose(sum(fractions.values()), 1.0):
        raise ValueError("fractions must sum to one.")
    if strategy not in ("row", "group"):
        raise ValueError("strategy must be 'row' or 'group'.")
    ids = _ids(frame, id_column)
    rng = np.random.default_rng(seed)
    counts = _allocated_counts(len(frame), fractions)
    names = list(fractions)
    chosen = np.empty(len(frame), dtype=object)
    if strategy == "row":
        order = rng.permutation(len(frame))
        at = 0
        for name in names:
            chosen[order[at : at + counts[name]]] = name
            at += counts[name]
    else:
        if not group_column or group_column not in frame.columns:
            raise ValueError(f"Group-aware split requires group column '{group_column}'.")
        if frame[group_column].isna().any():
            raise ValueError(f"Group column '{group_column}' contains missing values.")
        sizes = frame.groupby(group_column, sort=False).size().to_dict()
        groups = list(sizes)
        rng.shuffle(groups)
        active = [name for name in names if fractions[name] > 0]
        if len(groups) < len(active):
            raise ValueError("Not enough groups for the requested partitions.")
        assigned: dict[object, str] = {}
        actual = {name: 0 for name in names}
        for name, group in zip(active, groups[: len(active)]):
            assigned[group] = name
            actual[name] += sizes[group]
        for group in groups[len(active) :]:
            name = max(active, key=lambda p: (counts[p] - actual[p], -actual[p]))
            assigned[group] = name
            actual[name] += sizes[group]
        chosen = frame[group_column].map(assigned).to_numpy()
    return pd.DataFrame({id_column: ids.to_numpy(), "partition": chosen})


def split_by_manifest(frame: pd.DataFrame, manifest: pd.DataFrame, id_column: str) -> dict[str, pd.DataFrame]:
    """Return ``{partition: sub-frame}`` using the manifest's assignment."""
    ids = _ids(frame, id_column)
    if not {id_column, "partition"}.issubset(manifest.columns):
        raise ValueError("Manifest needs the identifier column and a 'partition' column.")
    lookup = manifest.set_index(manifest[id_column].astype(str))["partition"]
    if lookup.index.has_duplicates:
        raise ValueError("Manifest has duplicate identifiers.")
    missing = sorted(set(ids) - set(lookup.index))
    if missing:
        raise ValueError(f"{len(missing)} input records are not in the manifest (e.g. {missing[:3]}).")
    assignment = ids.map(lookup).to_numpy()
    names = list(dict.fromkeys(manifest["partition"]))
    return {name: frame.iloc[np.flatnonzero(assignment == name)].copy() for name in names}


def load_or_create_manifest(
    frame: pd.DataFrame,
    manifest_path: str | Path,
    fractions: Mapping[str, float],
    seed: int,
    id_column: str,
    strategy: str = "row",
    group_column: str | None = None,
) -> pd.DataFrame:
    """Reuse an existing manifest for these identifiers or create and save one."""
    manifest_path = Path(manifest_path)
    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path, dtype={id_column: str})
        ids = set(_ids(frame, id_column))
        if set(manifest[id_column].astype(str)) != ids:
            raise ValueError("Saved manifest does not cover exactly the input identifiers.")
        if strategy == "group":
            if group_column not in frame.columns:
                raise ValueError(f"Missing group column '{group_column}'.")
            check = pd.DataFrame(
                {
                    "group": frame[group_column].to_numpy(),
                    "partition": frame[id_column].astype(str).map(manifest.set_index(manifest[id_column].astype(str))["partition"]).to_numpy(),
                }
            )
            if check.groupby("group")["partition"].nunique().gt(1).any():
                raise ValueError("Saved manifest splits a group across partitions.")
        return manifest
    manifest = make_partitions(frame, fractions, seed, id_column, strategy, group_column)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)
    return manifest


# ---------------------------------------------------------------------------
# Hide-and-score masks
# ---------------------------------------------------------------------------


def hide_cells(frame: pd.DataFrame, mask: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``frame`` with masked cells set to missing."""
    hidden = frame.copy(deep=True)
    for column in mask.columns:
        if column in hidden.columns and mask[column].any():
            positions = mask[column].to_numpy(dtype=bool)
            if pd.api.types.is_numeric_dtype(hidden[column]):
                hidden.loc[positions, column] = np.nan
            else:
                values = hidden[column].astype(object).to_numpy(copy=True)
                values[positions] = np.nan
                hidden[column] = pd.Series(values, index=hidden.index, dtype=object)
    return hidden


def _validate_rate_seed(rate: float, seed: int) -> None:
    if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not np.isfinite(rate) or not 0 < rate < 1:
        raise ValueError("Holdout rate must be strictly between zero and one.")
    if type(seed) is not int or seed < 0:
        raise ValueError("Holdout seed must be a non-negative integer.")


def prepare_holdout(
    frame: pd.DataFrame,
    columns: Sequence[str],
    rate: float = 0.10,
    seed: int = 319,
    *,
    stratify_columns: Sequence[str] = (),
    linked_columns: Mapping[str, Sequence[str]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Hide ``rate`` of the CURRENTLY OBSERVED cells of each column.

    Returns ``(masked, truth, mask)`` where ``mask`` is True only at newly
    hidden positions, ``truth`` is a copy of the input, and ``masked`` is the
    input with those cells set to missing. At least one observed cell per
    column is always left in place.

    Columns listed in ``stratify_columns`` are hidden class by class: each
    class with at least two observed cells contributes ``round(rate * n_class)``
    hidden cells (minimum one), so minority classes are represented in the
    holdout instead of being drawn at random.

    ``linked_columns`` maps a column to the derived columns computed from it
    (``{"installation_year": ["asset_age"]}``): whenever a cell of the source is
    hidden, the linked cells of the same record are hidden too, so a derived
    value can never reveal a hidden truth. Linked cells are marked in the
    returned mask but are not scored unless they are also listed in ``columns``.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a DataFrame.")
    if not frame.index.is_unique:
        raise ValueError("Record index must uniquely identify rows.")
    _validate_rate_seed(rate, seed)
    targets = tuple(columns)
    if not targets or len(targets) != len(set(targets)):
        raise ValueError("Holdout columns must be non-empty and unique.")
    unknown = sorted(set(targets) - set(frame.columns))
    if unknown:
        raise ValueError(f"Unknown holdout columns: {unknown}.")
    stratified = set(stratify_columns)

    rng = np.random.default_rng(seed)
    mask = pd.DataFrame(False, index=frame.index, columns=frame.columns)
    for column in targets:
        observed = frame[column].notna().to_numpy()
        observed_positions = np.flatnonzero(observed)
        if len(observed_positions) < 2:
            raise ValueError(f"Cannot hold out '{column}': at least two observed values are needed.")
        selected: list[int] = []
        if column in stratified:
            classes = frame[column].astype(object).to_numpy()
            for value in pd.unique(classes[observed]):
                positions = observed_positions[classes[observed_positions] == value]
                if len(positions) < 2:
                    continue
                n_hide = max(1, int(round(rate * len(positions))))
                n_hide = min(n_hide, len(positions) - 1)
                selected.extend(rng.choice(positions, size=n_hide, replace=False).tolist())
            if not selected:
                n_hide = min(max(1, int(round(rate * len(observed_positions)))), len(observed_positions) - 1)
                selected = rng.choice(observed_positions, size=n_hide, replace=False).tolist()
        else:
            n_hide = max(1, int(round(rate * len(observed_positions))))
            n_hide = min(n_hide, len(observed_positions) - 1)
            selected = rng.choice(observed_positions, size=n_hide, replace=False).tolist()
        mask.iloc[sorted(selected), mask.columns.get_loc(column)] = True
        for linked in (linked_columns or {}).get(column, ()):
            if linked in mask.columns:
                positions = np.asarray(sorted(selected))
                observed_linked = frame[linked].notna().to_numpy()[positions]
                mask.iloc[positions[observed_linked], mask.columns.get_loc(linked)] = True

    truth = frame.copy(deep=True)
    masked = hide_cells(frame, mask)
    return masked, truth, mask


def repeated_holdouts(
    frame: pd.DataFrame,
    columns: Sequence[str],
    rate: float,
    seeds: Sequence[int],
    *,
    stratify_columns: Sequence[str] = (),
    linked_columns: Mapping[str, Sequence[str]] | None = None,
) -> list[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """Independent holdouts, one per seed, for averaging noisy scores."""
    if not seeds:
        raise ValueError("At least one seed is required.")
    return [prepare_holdout(frame, columns, rate, int(seed), stratify_columns=stratify_columns, linked_columns=linked_columns) for seed in seeds]


def validate_alignment(
    masked: pd.DataFrame,
    truth: pd.DataFrame,
    mask: pd.DataFrame,
    columns: Sequence[str],
) -> None:
    """Reject positional misalignment and accidentally exposed hidden truth."""
    for name, other in (("truth", truth), ("mask", mask)):
        if not isinstance(other, pd.DataFrame):
            raise TypeError(f"{name} must be a DataFrame.")
        if not masked.index.equals(other.index):
            raise ValueError(f"{name} index is not aligned with the masked frame.")
        if not masked.columns.equals(other.columns):
            raise ValueError(f"{name} columns are not aligned with the masked frame.")
    if not masked.index.is_unique:
        raise ValueError("Record index must uniquely identify rows.")
    for column in columns:
        if not pd.api.types.is_bool_dtype(mask[column]):
            raise TypeError(f"mask['{column}'] must be boolean.")
        positions = mask[column].to_numpy(dtype=bool)
        if positions.any():
            if not masked[column].isna().to_numpy()[positions].all():
                raise ValueError(f"Hidden cells are still visible in masked['{column}'].")
            if not truth[column].notna().to_numpy()[positions].all():
                raise ValueError(f"Hidden cells lack truth in '{column}'.")
