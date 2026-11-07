"""Evaluation mode: score imputation against known reference values.

Operational runs (``oc-pipeline all / impute / apply``) never see ground
truth. Evaluation mode takes records whose true values are known, keeps a
record-disjoint test partition aside, fills the gaps with each method and
scores only the cells that were hidden *and* have a known truth.

Inputs are distinguished explicitly:

* **truth** - the complete reference (every cell known);
* **masked input** - either a file produced by the masking engine (its
  missing cells are the ones to score; the masking configuration travels with
  it) or, when only the reference is given, a mask drawn here at a configured
  rate with a recorded seed, independently inside each partition;
* **partitions** - ``fit`` / ``validation`` / ``test`` assigned by identifier,
  grouped by ``feeder_id`` when the schema declares a group column, written
  to ``partition_manifest.csv`` and reused when a manifest is supplied. The
  validation partition is reserved for hyperparameter tuning and is not
  scored here; the test partition is scored once per run.

Two protocols are reported:

``unseen_records``
    persistable methods are fitted on the fit partition and applied to the
    test partition - the fit-once/apply-later use;
``transductive``
    every method fills the table formed by the fit and test partitions
    together and the test cells are scored - the whole-table use, in which
    Original and Predictive MissForest can be compared like for like.

Nothing here tunes hyperparameters: they come from ``imputation.yaml``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ..imputation.config import ImputationConfig
from ..imputation.registry import build_imputer
from ..ingestion.schema import DataSchema
from ..preprocessing.prepare import prepare_for_imputation
from .holdout import hide_cells, load_or_create_manifest, make_partitions, prepare_holdout, split_by_manifest
from .metrics import evaluate_imputation, per_class_report

PROTOCOLS = ("unseen_records", "transductive")
PARTITIONS = ("fit", "validation", "test")


@dataclass(frozen=True)
class EvaluationSettings:
    fractions: dict[str, float] = field(default_factory=lambda: {"fit": 0.60, "validation": 0.15, "test": 0.25})
    split_seed: int = 219
    strategy: str = "group"
    group_column: str | None = "feeder_id"
    mask_rate: float = 0.20
    mask_seed: int = 419
    repeats: int = 1
    protocols: tuple[str, ...] = PROTOCOLS
    methods: tuple[str, ...] | None = None
    recover: bool = True


@dataclass
class EvaluationInputs:
    masked: pd.DataFrame
    truth: pd.DataFrame
    manifest: pd.DataFrame
    parts: dict[str, pd.DataFrame]
    masks: list[pd.DataFrame]
    mask_source: str
    recovery_summary: dict[str, Any]


@dataclass
class EvaluationOutput:
    metrics: pd.DataFrame
    per_class: pd.DataFrame
    summary: pd.DataFrame
    scored_cells: pd.DataFrame
    metadata: dict[str, Any]


def _align_truth(masked: pd.DataFrame, truth: pd.DataFrame, id_column: str, alignment_columns: Sequence[str] = ()) -> pd.DataFrame:
    ids = masked[id_column].astype(str)
    truth_ids = truth[id_column].astype(str)
    if not truth_ids.is_unique:
        raise ValueError("truth identifiers must be unique.")
    missing = sorted(set(ids) - set(truth_ids))
    if missing:
        raise ValueError(f"{len(missing)} input identifiers are absent from the truth file (e.g. {missing[:3]}).")
    aligned = truth.set_index(truth_ids).loc[ids]
    aligned = aligned.reset_index(drop=True)
    aligned.index = masked.index
    for column in alignment_columns:
        if column not in masked or column not in aligned:
            raise ValueError(f"Declared row-alignment column '{column}' is missing from the paired files.")
        observed = masked[column].notna() & aligned[column].notna()
        if observed.any() and not masked.loc[observed, column].equals(aligned.loc[observed, column]):
            raise ValueError(
                f"Measured/truth row alignment failed for '{column}'. "
                "Paired files must preserve row order or carry a genuine stable ID."
            )
    for column in masked.columns:
        if column not in aligned.columns:
            aligned[column] = masked[column]
    return aligned.loc[:, list(masked.columns)]


def observed_difference_summary(measured: pd.DataFrame, truth: pd.DataFrame, schema: DataSchema) -> pd.DataFrame:
    """Describe observed measurement differences, distinct from missing-cell imputation error."""
    rows: list[dict[str, Any]] = []
    for column in schema.column_names:
        if column == schema.id_column or column not in measured or column not in truth:
            continue
        actual, reference = measured[column], truth[column]
        comparable = actual.notna() & reference.notna()
        spec = schema.column(column)
        mae: float | None = None
        if spec.is_numeric:
            left = pd.to_numeric(actual.loc[comparable], errors="coerce").to_numpy(dtype=float)
            right = pd.to_numeric(reference.loc[comparable], errors="coerce").to_numpy(dtype=float)
            difference = ~np.isclose(left, right, rtol=1e-10, atol=1e-9, equal_nan=True)
            if len(left):
                mae = float(np.abs(left - right).mean())
        else:
            difference = actual.loc[comparable].astype(str).to_numpy() != reference.loc[comparable].astype(str).to_numpy()
        n = int(comparable.sum())
        changed = int(np.asarray(difference).sum())
        rows.append({
            "column": column, "role": spec.role,
            "missing_measured": int(actual.isna().sum()),
            "observed_compared": n, "observed_different": changed,
            "difference_rate": changed / n if n else np.nan,
            "observed_mae": mae,
        })
    return pd.DataFrame(rows)


def build_evaluation_inputs(
    schema: DataSchema,
    config: ImputationConfig,
    settings: EvaluationSettings,
    *,
    truth: pd.DataFrame,
    masked: pd.DataFrame | None = None,
    manifest_path: Path | None = None,
) -> EvaluationInputs:
    """Align truth and masked input, partition the records and fix the masks to score.

    ``truth`` and ``masked`` are typed frames (schema types applied). When
    ``masked`` is None the reference is masked here, independently inside each
    partition, ``settings.repeats`` times with seeds ``mask_seed + r``.
    """
    id_column = schema.id_column
    model_columns = list(config.numeric_columns) + list(config.categorical_columns)
    if masked is not None:
        truth = _align_truth(masked, truth, id_column, schema.id_alignment_columns)
        base = masked
        mask_source = "masked_input"
    else:
        base = truth.copy(deep=True)
        mask_source = "drawn_in_evaluation"

    strategy = settings.strategy if (settings.strategy == "group" and settings.group_column and settings.group_column in base.columns) else "row"
    if manifest_path is not None:
        manifest = load_or_create_manifest(base, manifest_path, settings.fractions, settings.split_seed, id_column, strategy, settings.group_column if strategy == "group" else None)
    else:
        manifest = make_partitions(base, settings.fractions, settings.split_seed, id_column, strategy, settings.group_column if strategy == "group" else None)
    manifest = manifest.rename(columns={"partition": "partition"})
    parts_raw = split_by_manifest(base, manifest, id_column)

    masks: list[pd.DataFrame] = []
    if masked is not None:
        mask = pd.DataFrame(False, index=base.index, columns=base.columns)
        for column in model_columns:
            mask[column] = base[column].isna().to_numpy() & truth[column].notna().to_numpy()
        masks.append(mask)
        working = base
    else:
        stratify = tuple(config.categorical_columns)
        links = schema.linked_columns()
        for repeat in range(settings.repeats):
            mask = pd.DataFrame(False, index=base.index, columns=base.columns)
            for index, name in enumerate(PARTITIONS):
                part = parts_raw.get(name)
                if part is None or part.empty:
                    continue
                eligible = [c for c in model_columns if part[c].notna().sum() >= 2]
                _, _, part_mask = prepare_holdout(
                    part, eligible, rate=settings.mask_rate, seed=settings.mask_seed + 10 * repeat + index,
                    stratify_columns=stratify, linked_columns=links,
                )
                mask.loc[part.index, part_mask.columns] = part_mask.to_numpy()
            masks.append(mask)
        working = hide_cells(base, masks[0])

    # Deterministic recovery is part of the operational flow; cells it restores are
    # no longer gaps and are excluded from scoring (they are exact by construction).
    prepared = prepare_for_imputation(working, schema, config, recover=settings.recover)
    recovered = prepared.recovered_cells
    for mask in masks:
        for column in model_columns:
            if column in recovered.columns:
                mask[column] = mask[column].to_numpy() & ~recovered[column].to_numpy()
    parts = split_by_manifest(prepared.frame, manifest, id_column)
    return EvaluationInputs(
        masked=prepared.frame, truth=truth, manifest=manifest, parts=parts, masks=masks,
        mask_source=mask_source, recovery_summary=prepared.recovery.summary,
    )


def _score(
    truth: pd.DataFrame, filled: pd.DataFrame, mask: pd.DataFrame, numeric: Sequence[str], categorical: Sequence[str],
    *, protocol: str, method: str, repeat: int, seconds: float, id_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    table = evaluate_imputation(truth, filled, mask, numeric, categorical)
    table.insert(0, "protocol", protocol)
    table.insert(1, "method", method)
    table.insert(2, "repeat", repeat)
    table["fit_seconds"] = seconds
    classes = []
    for column in categorical:
        report = per_class_report(truth, filled, mask, column)
        report.insert(0, "protocol", protocol)
        report.insert(1, "method", method)
        report.insert(2, "repeat", repeat)
        classes.append(report)
    cells = []
    ids = truth[id_column].astype(str).to_numpy()
    for column in list(numeric) + list(categorical):
        positions = np.flatnonzero(mask[column].to_numpy(dtype=bool))
        if not len(positions):
            continue
        cells.append(pd.DataFrame({
            "protocol": protocol, "method": method, "repeat": repeat, "record_id": ids[positions], "column": column,
            "truth": truth[column].to_numpy()[positions], "imputed": filled[column].to_numpy()[positions],
        }))
    return table, (pd.concat(classes, ignore_index=True) if classes else pd.DataFrame()), (pd.concat(cells, ignore_index=True) if cells else pd.DataFrame())


def run_protocols(
    inputs: EvaluationInputs,
    schema: DataSchema,
    config: ImputationConfig,
    settings: EvaluationSettings,
    *,
    methods: Sequence[str],
    logger: logging.Logger | None = None,
) -> EvaluationOutput:
    logger = logger or logging.getLogger("overhead_conductors")
    numeric, categorical = tuple(config.numeric_columns), tuple(config.categorical_columns)
    id_column = schema.id_column
    fit_idx = inputs.parts["fit"].index
    test_idx = inputs.parts["test"].index
    if len(test_idx) == 0 or len(fit_idx) == 0:
        raise ValueError("Evaluation needs non-empty fit and test partitions.")
    metric_rows, class_rows, cell_rows = [], [], []
    runtimes: dict[str, float] = {}

    for repeat, mask in enumerate(inputs.masks):
        masked_all = inputs.masked if inputs.mask_source == "masked_input" or repeat == 0 else hide_cells(inputs.truth, mask)
        if inputs.mask_source != "masked_input" and repeat > 0:
            masked_all = prepare_for_imputation(masked_all, schema, config, recover=settings.recover).frame
        test_mask = mask.loc[test_idx]
        test_truth = inputs.truth.loc[test_idx]
        test_masked = masked_all.loc[test_idx]

        if "unseen_records" in settings.protocols:
            for name in methods:
                imputer = build_imputer(name, config)
                if not imputer.supports_persistence:
                    continue
                started = perf_counter()
                imputer.fit(masked_all.loc[fit_idx])
                filled = imputer.transform(test_masked).imputed
                seconds = perf_counter() - started
                table, classes, cells = _score(test_truth, filled, test_mask, numeric, categorical, protocol="unseen_records", method=name, repeat=repeat, seconds=seconds, id_column=id_column)
                metric_rows.append(table); class_rows.append(classes); cell_rows.append(cells)
                runtimes[f"unseen_records/{name}/{repeat}"] = seconds
                logger.info("evaluation: unseen_records %s repeat %d (%.1fs)", name, repeat, seconds)

        if "transductive" in settings.protocols:
            both_idx = fit_idx.append(test_idx)
            table_masked = masked_all.loc[both_idx]
            for name in methods:
                started = perf_counter()
                filled_all = build_imputer(name, config).fit_transform(table_masked).imputed
                filled = filled_all.loc[test_idx]
                seconds = perf_counter() - started
                table, classes, cells = _score(test_truth, filled, test_mask, numeric, categorical, protocol="transductive", method=name, repeat=repeat, seconds=seconds, id_column=id_column)
                metric_rows.append(table); class_rows.append(classes); cell_rows.append(cells)
                runtimes[f"transductive/{name}/{repeat}"] = seconds
                logger.info("evaluation: transductive %s repeat %d (%.1fs)", name, repeat, seconds)

    metrics = pd.concat(metric_rows, ignore_index=True) if metric_rows else pd.DataFrame()
    per_class = pd.concat([c for c in class_rows if len(c)], ignore_index=True) if any(len(c) for c in class_rows) else pd.DataFrame()
    scored = pd.concat([c for c in cell_rows if len(c)], ignore_index=True) if any(len(c) for c in cell_rows) else pd.DataFrame()
    value_columns = ["n_masked", "mae", "rmse", "median_absolute_error", "nrmse", "signed_bias", "accuracy", "balanced_accuracy", "macro_f1", "fit_seconds"]
    if len(metrics):
        summary = metrics.groupby(["protocol", "method", "variable", "variable_type"], sort=False)[value_columns].mean().reset_index()
        stds = metrics.groupby(["protocol", "method", "variable"], sort=False)[["mae", "macro_f1"]].std(ddof=1).add_suffix("_std").reset_index()
        summary = summary.merge(stds, on=["protocol", "method", "variable"], how="left")
        summary["n_masked"] = summary["n_masked"].round().astype(int)
        summary["n_repeats"] = len(inputs.masks)
        summary["within_5_years"] = np.nan
        if len(scored):
            year_cells = scored[(scored["column"] == "installation_year")]
            if len(year_cells):
                tolerance = (pd.to_numeric(year_cells["imputed"], errors="coerce") - pd.to_numeric(year_cells["truth"], errors="coerce")).abs() <= 5
                share = year_cells.assign(ok=tolerance).groupby(["protocol", "method"])["ok"].mean()
                for (protocol, method), value in share.items():
                    summary.loc[(summary["protocol"] == protocol) & (summary["method"] == method) & (summary["variable"] == "installation_year"), "within_5_years"] = float(value)
    else:
        summary = pd.DataFrame()

    metadata = {
        "mask_source": inputs.mask_source,
        "protocols": list(settings.protocols),
        "methods": list(methods),
        "partitions": {name: int(len(part)) for name, part in inputs.parts.items()},
        "partition_strategy": "group" if settings.group_column and settings.group_column in inputs.masked.columns and settings.strategy == "group" else "row",
        "split_seed": settings.split_seed,
        "fractions": dict(settings.fractions),
        "mask_rate": settings.mask_rate if inputs.mask_source != "masked_input" else None,
        "mask_seed": settings.mask_seed if inputs.mask_source != "masked_input" else None,
        "repeats": len(inputs.masks),
        "scored_cells_test": {column: int(inputs.masks[0].loc[test_idx, column].sum()) for column in list(numeric) + list(categorical)},
        "recovery": inputs.recovery_summary,
        "runtimes_seconds": runtimes,
        "note": "Scores are on cells hidden in the test partition whose true value is known. The validation partition is reserved for tuning and not scored.",
    }
    return EvaluationOutput(metrics=metrics, per_class=per_class, summary=summary, scored_cells=scored, metadata=metadata)
