"""Rule-driven data cleaning with an audit log.

The cleaner applies the corrections declared in ``cleaning_rules.yaml`` and
records every changed cell in ``cleaning_log.csv`` (record identifier, column,
old value, new value, rule). It never fills missing values and never changes a
value that no rule covers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml

from ..ingestion.schema import DataSchema

LOG_COLUMNS = ("row", "record_id", "column", "old_value", "new_value", "rule")
_WS = re.compile(r"\s+")


class CleaningError(ValueError):
    """Raised when a rule set to ``error`` fires or the rules file is invalid."""


# Rule families whose effect on observed values needs the data owner's sign-off.
# ``approval.<family>`` in cleaning_rules.yaml says whether that sign-off exists
# for the dataset at hand; an unapproved family is not applied.
APPROVAL_FAMILIES = ("code_mappings", "sentinel_values", "out_of_range", "unknown_category", "non_integer", "non_numeric")


@dataclass(frozen=True)
class FamilyApproval:
    requires_domain_approval: bool = False
    approved: bool = True
    approved_by: str | None = None
    note: str = ""

    @property
    def active(self) -> bool:
        return self.approved or not self.requires_domain_approval


@dataclass(frozen=True)
class CleaningRules:
    strip_whitespace: bool = True
    collapse_internal_whitespace: bool = True
    unknown_category_action: str = "keep"
    #: Spelling variants: other spellings, cases or abbreviations of the allowed labels themselves.
    synonyms: Mapping[str, Mapping[str, str]] = field(default_factory=dict)
    #: Code mappings: codes or product designations interpreted as a label (domain decision).
    code_mappings: Mapping[str, Mapping[str, str]] = field(default_factory=dict)
    non_numeric_action: str = "missing"
    out_of_range_action: str = "missing"
    sentinel_action: str = "missing"
    non_integer_action: str = "round"
    exact_duplicate_action: str = "drop"
    duplicate_id_action: str = "error"
    approval: Mapping[str, FamilyApproval] = field(default_factory=dict)

    def family_active(self, family: str) -> bool:
        """True when the family may change data (approved, or needing no approval)."""
        return self.approval.get(family, FamilyApproval()).active

    def inactive_families(self) -> tuple[str, ...]:
        return tuple(f for f in APPROVAL_FAMILIES if not self.family_active(f))

    def __post_init__(self) -> None:
        choices = {
            "unknown_category_action": ("keep", "missing"),
            "non_numeric_action": ("missing", "error"),
            "out_of_range_action": ("missing", "keep", "error"),
            "sentinel_action": ("missing", "keep"),
            "non_integer_action": ("round", "missing", "error"),
            "exact_duplicate_action": ("drop", "keep", "error"),
            "duplicate_id_action": ("error", "keep_first"),
        }
        for name, allowed in choices.items():
            if getattr(self, name) not in allowed:
                raise CleaningError(f"{name} must be one of {allowed}.")
        for label, table in (("spelling_variants", self.synonyms), ("code_mappings", self.code_mappings)):
            for column, mapping in table.items():
                if not isinstance(mapping, Mapping) or not all(
                    isinstance(k, str) and isinstance(v, str) for k, v in mapping.items()
                ):
                    raise CleaningError(f"{label}.{column} must map strings to strings.")
        for family in self.approval:
            if family not in APPROVAL_FAMILIES:
                raise CleaningError(f"approval.{family}: unknown rule family (expected one of {APPROVAL_FAMILIES}).")


def parse_cleaning_rules(raw: Mapping[str, Any] | None) -> CleaningRules:
    raw = raw or {}
    if not isinstance(raw, Mapping):
        raise CleaningError("Cleaning rules must be a YAML mapping.")
    text = raw.get("text") or {}
    categorical = raw.get("categorical") or {}
    numeric = raw.get("numeric") or {}
    duplicates = raw.get("duplicates") or {}
    for section_name, section in (("text", text), ("categorical", categorical), ("numeric", numeric), ("duplicates", duplicates)):
        if not isinstance(section, Mapping):
            raise CleaningError(f"Cleaning rules section '{section_name}' must be a mapping.")
    def _lower_table(section: Mapping[str, Any] | None) -> dict[str, dict[str, str]]:
        return {
            column: {str(k).strip().lower(): str(v) for k, v in (mapping or {}).items()}
            for column, mapping in (section or {}).items()
        }

    if "synonyms" in categorical and "spelling_variants" in categorical:
        raise CleaningError("Use either categorical.spelling_variants or its older name categorical.synonyms, not both.")
    synonyms = _lower_table(categorical.get("spelling_variants", categorical.get("synonyms")))
    code_mappings = _lower_table(categorical.get("code_mappings"))
    approval_raw = raw.get("approval") or {}
    if not isinstance(approval_raw, Mapping):
        raise CleaningError("Cleaning rules section 'approval' must be a mapping.")
    approval: dict[str, FamilyApproval] = {}
    for family, entry in approval_raw.items():
        if not isinstance(entry, Mapping):
            raise CleaningError(f"approval.{family} must be a mapping.")
        requires = entry.get("requires_domain_approval", False)
        approved = entry.get("approved", not requires)
        if type(requires) is not bool or type(approved) is not bool:
            raise CleaningError(f"approval.{family}: requires_domain_approval and approved must be booleans.")
        approved_by = entry.get("approved_by")
        if approved and requires and not (isinstance(approved_by, str) and approved_by.strip()):
            raise CleaningError(f"approval.{family}: an approved family must name approved_by.")
        approval[str(family)] = FamilyApproval(
            requires_domain_approval=requires,
            approved=approved,
            approved_by=str(approved_by) if approved_by else None,
            note=str(entry.get("note", "")),
        )
    if code_mappings and "code_mappings" not in approval:
        # Code mappings are never applied without an explicit approval entry.
        approval["code_mappings"] = FamilyApproval(requires_domain_approval=True, approved=False)
    return CleaningRules(
        strip_whitespace=bool(text.get("strip_whitespace", True)),
        collapse_internal_whitespace=bool(text.get("collapse_internal_whitespace", True)),
        unknown_category_action=str(categorical.get("unknown_category_action", "keep")),
        synonyms=synonyms,
        code_mappings=code_mappings,
        approval=approval,
        non_numeric_action=str(numeric.get("non_numeric_action", "missing")),
        out_of_range_action=str(numeric.get("out_of_range_action", "missing")),
        sentinel_action=str(numeric.get("sentinel_action", "missing")),
        non_integer_action=str(numeric.get("non_integer_action", "round")),
        exact_duplicate_action=str(duplicates.get("exact_duplicate_action", "drop")),
        duplicate_id_action=str(duplicates.get("duplicate_id_action", "error")),
    )


def load_cleaning_rules(path: str | Path) -> CleaningRules:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Cleaning rules not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return parse_cleaning_rules(yaml.safe_load(handle))


@dataclass
class CleaningResult:
    cleaned: pd.DataFrame
    log: pd.DataFrame
    dropped_rows: pd.DataFrame
    summary: dict[str, Any]


class _Logger:
    def __init__(self, ids: pd.Series) -> None:
        self.ids = ids
        self.rows: list[dict[str, Any]] = []

    def record(self, positions: np.ndarray, column: str, old: pd.Series, new: pd.Series, rule: str) -> None:
        for pos in positions:
            self.rows.append(
                {
                    "row": int(pos),
                    "record_id": self.ids.iloc[pos] if pos < len(self.ids) else "",
                    "column": column,
                    "old_value": _display(old.iloc[pos]),
                    "new_value": _display(new.iloc[pos]),
                    "rule": rule,
                }
            )

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows, columns=list(LOG_COLUMNS))


def _display(value: Any) -> Any:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return value


def _is_missing(series: pd.Series) -> pd.Series:
    return series.isna()


def _normalise_text(series: pd.Series, rules: CleaningRules) -> pd.Series:
    def fix(value: Any) -> Any:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return np.nan
        text = str(value)
        if rules.strip_whitespace:
            text = text.strip()
        if rules.collapse_internal_whitespace:
            text = _WS.sub(" ", text)
        return text

    return series.map(fix).astype(object)


def clean_records(frame: pd.DataFrame, schema: DataSchema, rules: CleaningRules) -> CleaningResult:
    """Apply the cleaning rules to a text frame and return typed, logged output."""
    working = frame.copy(deep=True).astype(object)
    working.index = pd.RangeIndex(len(working))
    id_column = schema.id_column
    ids = _normalise_text(working[id_column], rules) if id_column in working.columns else pd.Series([""] * len(working), dtype=object)
    log = _Logger(ids)
    tokens = schema.missing_token_set()
    pending_counts: dict[str, int] = {}

    # 1. exact duplicate rows -------------------------------------------------
    dropped = working.iloc[0:0]
    exact_dup = working.duplicated(keep="first").to_numpy()
    if exact_dup.any():
        if rules.exact_duplicate_action == "error":
            raise CleaningError(f"{int(exact_dup.sum())} exact duplicate rows found; rules forbid dropping them.")
        if rules.exact_duplicate_action == "drop":
            positions = np.flatnonzero(exact_dup)
            dropped = working.iloc[positions].copy()
            marker = pd.Series(["dropped"] * len(working), dtype=object)
            log.record(positions, "*", ids, marker, "exact_duplicate_row")
            working = working.iloc[~exact_dup].copy()
            ids = ids.iloc[~exact_dup].reset_index(drop=True)
            working.index = pd.RangeIndex(len(working))
            log.ids = ids

    # 2. per-column rules -------------------------------------------------------
    for name, spec in schema.columns.items():
        if name not in working.columns:
            continue
        original = working[name].astype(object)
        current = _normalise_text(original, rules)

        changed = (current.fillna("\0") != original.fillna("\0")).to_numpy()
        changed = changed & ~(_is_missing(current) & _is_missing(original)).to_numpy()
        if changed.any():
            log.record(np.flatnonzero(changed), name, original, current, "whitespace")

        token_mask = current.notna() & current.map(lambda v: str(v).lower() in tokens).astype(bool)
        if token_mask.any():
            new = current.mask(token_mask, np.nan)
            log.record(np.flatnonzero(token_mask.to_numpy()), name, current, new, "missing_token")
            current = new

        if spec.is_categorical:
            synonyms = rules.synonyms.get(name, {})
            codes = rules.code_mappings.get(name, {}) if rules.family_active("code_mappings") else {}
            pending_codes = rules.code_mappings.get(name, {}) if not rules.family_active("code_mappings") else {}
            folded = {label.lower(): label for label in (spec.allowed or ())}

            def canonical(value: Any) -> Any:
                if value is None or (isinstance(value, float) and np.isnan(value)):
                    return np.nan
                key = str(value).lower()
                if str(value) in (spec.allowed or ()):
                    return value
                if key in folded:
                    return folded[key]
                if key in synonyms:
                    return synonyms[key]
                if key in codes:
                    return codes[key]
                return value

            mapped = current.map(canonical).astype(object)
            syn_positions, code_positions, case_positions = [], [], []
            for pos, (old, new) in enumerate(zip(current, mapped)):
                if isinstance(old, str) and old != new:
                    key = old.lower()
                    if key in folded:
                        case_positions.append(pos)
                    elif key in synonyms:
                        syn_positions.append(pos)
                    else:
                        code_positions.append(pos)
            if case_positions:
                log.record(np.asarray(case_positions), name, current, mapped, "label_case")
            if syn_positions:
                log.record(np.asarray(syn_positions), name, current, mapped, "spelling_variant")
            if code_positions:
                log.record(np.asarray(code_positions), name, current, mapped, "code_mapping")
            current = mapped
            if pending_codes:
                pending = current.map(lambda v: isinstance(v, str) and v.lower() in pending_codes).astype(bool)
                if pending.any():
                    pending_counts[name] = int(pending.sum())
            if spec.allowed:
                unknown = current.notna() & ~current.isin(spec.allowed)
                if unknown.any() and rules.unknown_category_action == "missing" and rules.family_active("unknown_category"):
                    new = current.mask(unknown, np.nan)
                    log.record(np.flatnonzero(unknown.to_numpy()), name, current, new, "unknown_category")
                    current = new

        elif spec.is_numeric:
            parsed = pd.to_numeric(current, errors="coerce")
            non_numeric = current.notna() & parsed.isna()
            if non_numeric.any():
                if rules.non_numeric_action == "error":
                    raise CleaningError(f"Column '{name}' has {int(non_numeric.sum())} non-numeric values.")
                if not rules.family_active("non_numeric"):
                    raise CleaningError(
                        f"Column '{name}' has {int(non_numeric.sum())} non-numeric values and the "
                        "non_numeric family is not approved; a numeric column cannot keep text."
                    )
                log.record(np.flatnonzero(non_numeric.to_numpy()), name, current, parsed.astype(object), "non_numeric")
            values = parsed.astype(float)

            if spec.sentinel_values and rules.sentinel_action == "missing" and rules.family_active("sentinel_values"):
                sentinel = values.isin(spec.sentinel_values)
                if sentinel.any():
                    new = values.mask(sentinel, np.nan)
                    log.record(np.flatnonzero(sentinel.to_numpy()), name, values.astype(object), new.astype(object), "sentinel_value")
                    values = new

            out_of_range = pd.Series(False, index=values.index)
            if spec.min is not None:
                out_of_range |= values < spec.min
            if spec.max is not None:
                out_of_range |= values > spec.max
            if out_of_range.any():
                if rules.out_of_range_action == "error":
                    raise CleaningError(f"Column '{name}' has {int(out_of_range.sum())} out-of-range values.")
                if rules.out_of_range_action == "missing" and rules.family_active("out_of_range"):
                    new = values.mask(out_of_range, np.nan)
                    log.record(np.flatnonzero(out_of_range.to_numpy()), name, values.astype(object), new.astype(object), "out_of_range")
                    values = new

            if spec.dtype == "integer":
                non_int = values.notna() & (values != np.round(values))
                if non_int.any() and rules.family_active("non_integer"):
                    if rules.non_integer_action == "error":
                        raise CleaningError(f"Column '{name}' has {int(non_int.sum())} non-integer values.")
                    if rules.non_integer_action == "round":
                        new = values.where(~non_int, np.round(values))
                        log.record(np.flatnonzero(non_int.to_numpy()), name, values.astype(object), new.astype(object), "non_integer_rounded")
                    else:
                        new = values.mask(non_int, np.nan)
                        log.record(np.flatnonzero(non_int.to_numpy()), name, values.astype(object), new.astype(object), "non_integer_missing")
                    values = new
            current = values.astype(float)

        working[name] = current if spec.is_numeric else current.astype(object)

    # 3. duplicate identifiers after cleaning -------------------------------------
    if id_column in working.columns:
        id_values = working[id_column]
        dup = id_values.notna() & id_values.duplicated(keep="first")
        if dup.any():
            if rules.duplicate_id_action == "error":
                examples = ", ".join(map(str, id_values[dup].unique()[:5]))
                raise CleaningError(
                    f"{int(dup.sum())} records share an identifier with an earlier, different record "
                    f"(e.g. {examples}). Resolve in the source or set duplicates.duplicate_id_action: keep_first."
                )
            positions = np.flatnonzero(dup.to_numpy())
            marker = pd.Series(["dropped"] * len(working), dtype=object)
            log.record(positions, "*", id_values.astype(object), marker, "duplicate_identifier")
            dropped = pd.concat([dropped, working.iloc[positions]], axis=0)
            working = working.iloc[~dup.to_numpy()].copy()
            working.index = pd.RangeIndex(len(working))

    log_frame = log.frame()
    summary = {
        "rows_in": int(len(frame)),
        "rows_out": int(len(working)),
        "rows_dropped": int(len(dropped)),
        "cells_changed": int((log_frame["column"] != "*").sum()),
        "changes_by_rule": {k: int(v) for k, v in log_frame["rule"].value_counts().sort_index().items()},
        "changes_by_column": {k: int(v) for k, v in log_frame["column"].value_counts().sort_index().items()},
        "inactive_families": list(rules.inactive_families()),
        "pending_code_mappings": pending_counts,
    }
    return CleaningResult(cleaned=working, log=log_frame, dropped_rows=dropped, summary=summary)


def plan_changes(frame: pd.DataFrame, schema: DataSchema, rules: CleaningRules) -> CleaningResult:
    """Dry run: the changes :func:`clean_records` would make, without keeping the result.

    The input is never modified by either function; the only difference is
    intent. Use the returned ``log`` and ``summary`` to review planned changes
    (and families skipped for lack of approval) before applying them.
    """
    return clean_records(frame, schema, rules)
