"""Preparation of a cleaned, typed frame for an imputation method.

Two steps, in this order:

1. deterministic recovery of missing sources from their observed derived
   columns (``derived_fields.recover_sources``), so those cells are observed
   for the imputer and carry provenance ``derived``;
2. feature selection and the leakage guard (``feature_selection.select_model_frame``).

Algorithm-specific encoding and initial fills stay inside the methods.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from ..ingestion.schema import DataSchema
from .derived_fields import DerivationResult, recover_sources
from .feature_selection import ModelFrame, select_model_frame


@dataclass
class PreparedFrame:
    #: The frame handed to the imputer, in the caller's column order.
    frame: pd.DataFrame
    recovery: DerivationResult
    model_frame: ModelFrame

    @property
    def recovered_cells(self) -> pd.DataFrame:
        return self.recovery.cells if self.recovery.cells is not None else pd.DataFrame(False, index=self.frame.index, columns=self.frame.columns)


def prepare_for_imputation(frame: pd.DataFrame, schema: DataSchema, imputation_config: Any, *, recover: bool = True) -> PreparedFrame:
    """Recover what can be derived, then select and type the model attributes."""
    recovery = recover_sources(frame, schema) if recover else DerivationResult(frame=frame.copy(deep=True), log=pd.DataFrame(), summary={}, cells=pd.DataFrame(False, index=frame.index, columns=frame.columns))
    model_frame = select_model_frame(recovery.frame, schema, imputation_config)
    ordered = model_frame.frame.loc[:, list(frame.columns)]
    return PreparedFrame(frame=ordered, recovery=recovery, model_frame=model_frame)
