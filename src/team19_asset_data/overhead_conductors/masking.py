"""
Reusable masking engine for controlled missing-data experiments.

The masking engine is intentionally independent from the synthetic-data generator.

Typical flow:
    complete reference dataset
        -> choose one or more rule IDs from YAML
        -> apply masking
        -> save masked dataset + metadata
        -> impute
        -> compare imputed values with *_true columns

Important:
- Original values are preserved in <column>_true columns before masking.
- <column>__masked boolean flags identify intentionally masked cells.
- Linked columns are masked on the same rows when required.
  Example: installation_year -> asset_age_2014.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml


class MaskingEngine:
    """Apply reproducible masking rules defined in a YAML configuration file."""

    def __init__(self, config: Dict[str, Any]):
        self.config = deepcopy(config)
        self.rules = self.config.get("rules", {})
        self.settings = self.config.get("settings", {})
        self.maskable_attributes = self.config.get("maskable_attributes", [])
        self.protected_attributes = set(self.config.get("protected_attributes", []))
        self.linked_columns = self.config.get("linked_columns", {})

        if not self.rules:
            raise ValueError("Masking configuration does not contain any rules.")

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "MaskingEngine":
        """Load masking configuration from YAML."""
        config_path = Path(config_path)

        with config_path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

        if not isinstance(config, dict):
            raise ValueError(f"Invalid YAML configuration: {config_path}")

        return cls(config)

    def apply_rule(
        self,
        df: pd.DataFrame,
        rule_id: str,
        seed_override: int | None = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Apply one configured masking rule."""
        if rule_id not in self.rules:
            available = ", ".join(sorted(self.rules))
            raise KeyError(
                f"Unknown rule '{rule_id}'. Available rules: {available}"
            )

        rule = deepcopy(self.rules[rule_id])
        if seed_override is not None:
            rule["seed"] = int(seed_override)
            
        strategy = rule.get("strategy")

        handlers = {
            "random_attribute": self._apply_random_attribute,
            "target_random": self._apply_target_random,
            "conditional": self._apply_conditional,
            "group": self._apply_group,
            "multi_column": self._apply_multi_column,
        }

        if strategy not in handlers:
            raise ValueError(
                f"Rule '{rule_id}' uses unsupported strategy '{strategy}'."
            )

        result = df.copy(deep=True)
        masked_df, details = handlers[strategy](result, rule)

        metadata = {
            "rule_id": rule_id,
            "name": rule.get("name", rule_id),
            "description": rule.get("description", ""),
            "strategy": strategy,
            "seed": int(rule.get("seed", self.settings.get("default_seed", 220))),
            "rows_in_dataset": int(len(df)),
            **details,
        }

        return masked_df, metadata

    def apply_rules(
        self,
        df: pd.DataFrame,
        rule_ids: Sequence[str],
    ) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """Apply several rules sequentially to the same dataset."""
        result = df.copy(deep=True)
        all_metadata: List[Dict[str, Any]] = []

        for rule_id in rule_ids:
            result, metadata = self.apply_rule(result, rule_id)
            all_metadata.append(metadata)

        return result, all_metadata

    # ------------------------------------------------------------------
    # Strategy implementations
    # ------------------------------------------------------------------

    def _apply_random_attribute(
        self,
        df: pd.DataFrame,
        rule: Dict[str, Any],
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Randomly choose one or more eligible attributes, then mask a fraction
        of each selected attribute.

        selection_mode:
        - independent: each selected attribute gets its own random row sample.
        - shared_rows: the same random rows are masked across all attributes.
        """
        rng = self._rng(rule)
        rate = self._validate_rate(rule["rate"])
        number_of_attributes = int(rule.get("number_of_attributes", 1))
        selection_mode = rule.get("selection_mode", "independent")

        eligible = [
            column
            for column in self.maskable_attributes
            if column in df.columns and column not in self.protected_attributes
        ]

        if number_of_attributes < 1:
            raise ValueError("number_of_attributes must be at least 1.")

        if number_of_attributes > len(eligible):
            raise ValueError(
                f"Rule requests {number_of_attributes} attributes, "
                f"but only {len(eligible)} are eligible in this dataset."
            )

        selected_attributes = [
            str(x)
            for x in rng.choice(
                eligible,
                size=number_of_attributes,
                replace=False,
            )
        ]

        masked_counts: Dict[str, int] = {}
        linked_counts: Dict[str, int] = {}

        if selection_mode == "shared_rows":
            # Shared rows are selected from rows where every selected attribute
            # currently has a known value.
            available_mask = df[selected_attributes].notna().all(axis=1)
            available_indices = df.index[available_mask]
            selected_rows = self._sample_indices(rng, available_indices, rate)

            for column in selected_attributes:
                self._mask_column(df, column, selected_rows)
                masked_counts[column] = int(len(selected_rows))
                self._mask_linked_columns(
                    df,
                    source_column=column,
                    indices=selected_rows,
                    linked_counts=linked_counts,
                )

        elif selection_mode == "independent":
            for column in selected_attributes:
                available_indices = df.index[df[column].notna()]
                selected_rows = self._sample_indices(rng, available_indices, rate)

                self._mask_column(df, column, selected_rows)
                masked_counts[column] = int(len(selected_rows))
                self._mask_linked_columns(
                    df,
                    source_column=column,
                    indices=selected_rows,
                    linked_counts=linked_counts,
                )

        else:
            raise ValueError(
                "selection_mode must be 'independent' or 'shared_rows'."
            )

        return df, {
            "requested_rate": rate,
            "number_of_attributes": number_of_attributes,
            "selection_mode": selection_mode,
            "selected_attributes": selected_attributes,
            "masked_counts": masked_counts,
            "linked_masked_counts": linked_counts,
        }

    def _apply_target_random(
        self,
        df: pd.DataFrame,
        rule: Dict[str, Any],
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Mask a fixed proportion of one explicitly configured target."""
        rng = self._rng(rule)
        target = self._require_column(df, rule["target"])
        rate = self._validate_rate(rule["rate"])

        available_indices = df.index[df[target].notna()]
        selected_rows = self._sample_indices(rng, available_indices, rate)

        self._mask_column(df, target, selected_rows)

        linked_counts: Dict[str, int] = {}
        self._mask_linked_columns(
            df,
            source_column=target,
            indices=selected_rows,
            linked_counts=linked_counts,
        )

        return df, {
            "target": target,
            "requested_rate": rate,
            "masked_counts": {target: int(len(selected_rows))},
            "linked_masked_counts": linked_counts,
        }

    def _apply_conditional(
        self,
        df: pd.DataFrame,
        rule: Dict[str, Any],
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Apply different masking rates to a target based on numeric ranges in
        another column.
        """
        rng = self._rng(rule)
        target = self._require_column(df, rule["target"])
        condition_column = self._require_column(df, rule["condition_column"])

        total_masked = 0
        condition_details = []
        linked_counts: Dict[str, int] = {}

        for condition in rule.get("conditions", []):
            low = condition["min"]
            high = condition["max"]
            rate = self._validate_rate(condition["rate"])

            eligible_mask = (
                df[condition_column].between(low, high, inclusive="both")
                & df[target].notna()
            )
            eligible_indices = df.index[eligible_mask]
            selected_rows = self._sample_indices(rng, eligible_indices, rate)

            self._mask_column(df, target, selected_rows)
            self._mask_linked_columns(
                df,
                source_column=target,
                indices=selected_rows,
                linked_counts=linked_counts,
            )

            count = int(len(selected_rows))
            total_masked += count
            condition_details.append(
                {
                    "min": low,
                    "max": high,
                    "rate": rate,
                    "eligible_rows": int(len(eligible_indices)),
                    "masked_rows": count,
                }
            )

        return df, {
            "target": target,
            "condition_column": condition_column,
            "masked_counts": {target: total_masked},
            "linked_masked_counts": linked_counts,
            "conditions": condition_details,
        }

    def _apply_group(
        self,
        df: pd.DataFrame,
        rule: Dict[str, Any],
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Apply different masking rates to selected groups."""
        rng = self._rng(rule)
        target = self._require_column(df, rule["target"])
        group_column = self._require_column(df, rule["group_column"])

        default_rate = self._validate_rate(rule["default_rate"])
        selected_group_rate = self._validate_rate(rule["selected_group_rate"])
        selected_groups = set(rule.get("selected_groups", []))

        total_masked = 0
        group_details = []
        linked_counts: Dict[str, int] = {}

        groups = sorted(df[group_column].dropna().astype(str).unique())

        for group in groups:
            eligible_mask = (
                df[group_column].astype(str).eq(group)
                & df[target].notna()
            )
            eligible_indices = df.index[eligible_mask]

            rate = (
                selected_group_rate
                if group in selected_groups
                else default_rate
            )

            selected_rows = self._sample_indices(rng, eligible_indices, rate)

            self._mask_column(df, target, selected_rows)
            self._mask_linked_columns(
                df,
                source_column=target,
                indices=selected_rows,
                linked_counts=linked_counts,
            )

            count = int(len(selected_rows))
            total_masked += count

            group_details.append(
                {
                    "group": group,
                    "rate": rate,
                    "eligible_rows": int(len(eligible_indices)),
                    "masked_rows": count,
                    "high_missingness_group": group in selected_groups,
                }
            )

        return df, {
            "target": target,
            "group_column": group_column,
            "default_rate": default_rate,
            "selected_group_rate": selected_group_rate,
            "selected_groups": sorted(selected_groups),
            "masked_counts": {target: total_masked},
            "linked_masked_counts": linked_counts,
            "groups": group_details,
        }

    def _apply_multi_column(
        self,
        df: pd.DataFrame,
        rule: Dict[str, Any],
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Mask the primary target and then mask additional predictor columns for
        controlled subsets of the same target-masked rows.
        """
        rng = self._rng(rule)
        target = self._require_column(df, rule["target"])
        rate = self._validate_rate(rule["rate"])
        scope = rule.get("additional_mask_scope", "target_rows")

        available_indices = df.index[df[target].notna()]
        target_rows = self._sample_indices(rng, available_indices, rate)

        self._mask_column(df, target, target_rows)

        masked_counts: Dict[str, int] = {target: int(len(target_rows))}
        linked_counts: Dict[str, int] = {}

        self._mask_linked_columns(
            df,
            source_column=target,
            indices=target_rows,
            linked_counts=linked_counts,
        )

        if scope != "target_rows":
            raise ValueError(
                "Currently supported additional_mask_scope is 'target_rows'."
            )

        for column, additional_rate in rule.get("additional_masks", {}).items():
            column = self._require_column(df, column)
            additional_rate = self._validate_rate(additional_rate)

            # Additional masks are chosen only from the rows where target was
            # already masked. This creates the intended harder stress test.
            candidate_indices = pd.Index(
                [idx for idx in target_rows if pd.notna(df.at[idx, column])]
            )
            selected_rows = self._sample_indices(
                rng,
                candidate_indices,
                additional_rate,
            )

            self._mask_column(df, column, selected_rows)
            masked_counts[column] = int(len(selected_rows))

            self._mask_linked_columns(
                df,
                source_column=column,
                indices=selected_rows,
                linked_counts=linked_counts,
            )

        return df, {
            "target": target,
            "requested_rate": rate,
            "additional_mask_scope": scope,
            "masked_counts": masked_counts,
            "linked_masked_counts": linked_counts,
        }

    # ------------------------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------------------------

    def _rng(self, rule: Dict[str, Any]) -> np.random.Generator:
        seed = int(
            rule.get(
                "seed",
                self.settings.get("default_seed", 220),
            )
        )
        return np.random.default_rng(seed)

    @staticmethod
    def _validate_rate(rate: Any) -> float:
        rate = float(rate)
        if not 0 <= rate <= 1:
            raise ValueError(f"Masking rate must be between 0 and 1, got {rate}.")
        return rate

    @staticmethod
    def _require_column(df: pd.DataFrame, column: str) -> str:
        if column not in df.columns:
            raise KeyError(f"Required column '{column}' is missing from dataset.")
        return column

    @staticmethod
    def _sample_indices(
        rng: np.random.Generator,
        available_indices: Iterable,
        rate: float,
    ) -> pd.Index:
        available_indices = pd.Index(list(available_indices))

        if len(available_indices) == 0 or rate == 0:
            return pd.Index([])

        n_mask = int(round(len(available_indices) * rate))
        n_mask = min(n_mask, len(available_indices))

        if n_mask == 0:
            return pd.Index([])

        selected = rng.choice(
            available_indices.to_numpy(),
            size=n_mask,
            replace=False,
        )
        return pd.Index(selected)

    @staticmethod
    def _ensure_truth_column(df: pd.DataFrame, column: str) -> str:
        """
        Preserve the original value once. If several rules are applied
        sequentially, an existing *_true column is never overwritten.
        """
        truth_column = f"{column}_true"

        if truth_column not in df.columns:
            df[truth_column] = df[column]

        return truth_column

    @staticmethod
    def _ensure_mask_flag(df: pd.DataFrame, column: str) -> str:
        flag_column = f"{column}__masked"

        if flag_column not in df.columns:
            df[flag_column] = False

        return flag_column

    def _mask_column(
        self,
        df: pd.DataFrame,
        column: str,
        indices: pd.Index,
    ) -> None:
        """Preserve truth, flag masked cells, then replace values with NaN."""
        self._require_column(df, column)

        if len(indices) == 0:
            return

        self._ensure_truth_column(df, column)
        flag_column = self._ensure_mask_flag(df, column)

        df.loc[indices, flag_column] = True
        df.loc[indices, column] = np.nan

    def _mask_linked_columns(
        self,
        df: pd.DataFrame,
        source_column: str,
        indices: pd.Index,
        linked_counts: Dict[str, int],
    ) -> None:
        """Mask columns that would otherwise leak the selected target."""
        for linked_column in self.linked_columns.get(source_column, []):
            if linked_column not in df.columns:
                raise KeyError(
                    f"Linked column '{linked_column}' configured for "
                    f"'{source_column}' is missing from the dataset."
                )

            self._mask_column(df, linked_column, indices)
            linked_counts[linked_column] = (
                linked_counts.get(linked_column, 0) + int(len(indices))
            )
