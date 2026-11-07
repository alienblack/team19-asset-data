"""Postprocessing: derived fields after imputation, output typing, cell provenance, integrity.

The imputation methods return the completed model attributes; this package
turns that into the delivered dataset:

1. derived columns are filled or recomputed from their (possibly imputed)
   sources (:func:`derive_fields`, kept in ``preprocessing.derived_fields`` for
   import compatibility);
2. integer columns are exported as integers (:func:`finalise_output_types`);
3. every non-observed cell is classified as derived / imputed / unresolved
   (:func:`build_cell_provenance`);
4. the integrity checks confirm that observed values were not altered and
   the constraints hold (:func:`check_integrity`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import pandas as pd

from ..evaluation.integrity_checks import IntegrityResult, check_integrity
from ..ingestion.loader import finalise_output_types
from ..ingestion.schema import DataSchema
from ..preprocessing.derived_fields import DerivationResult, derive_fields
from .provenance import PROVENANCE_COLUMNS, STATUSES, ProvenanceResult, build_cell_provenance


@dataclass
class FinalOutput:
    frame: pd.DataFrame
    derivation: DerivationResult
    provenance: ProvenanceResult
    integrity: IntegrityResult


def finalise_output(
    before: pd.DataFrame,
    imputed: pd.DataFrame,
    schema: DataSchema,
    *,
    model_columns: Sequence[str],
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
    constraints: dict[str, dict[str, Any]] | None,
    method: str,
    recovered_cells: pd.DataFrame | None = None,
    diagnostics: pd.DataFrame | None = None,
    prepared: pd.DataFrame | None = None,
) -> FinalOutput:
    """Derive, type, classify and check the completed dataset.

    ``before`` is the cleaned frame before recovery and imputation (the
    reference for provenance); ``prepared`` is the frame the imputer actually
    received (after recovery), used for the integrity comparison so recovered
    cells count as observed inputs there. ``imputed`` is the method's output.
    """
    prepared = prepared if prepared is not None else before
    derivation = derive_fields(imputed, schema, original=prepared)
    final = finalise_output_types(derivation.frame, schema)
    provenance = build_cell_provenance(
        before,
        final,
        schema,
        model_columns=model_columns,
        method=method,
        recovered_cells=recovered_cells,
        derived_cells=derivation.cells,
        diagnostics=diagnostics,
    )
    integrity = check_integrity(prepared, final, schema, numeric_columns, categorical_columns, constraints)
    return FinalOutput(frame=final, derivation=derivation, provenance=provenance, integrity=integrity)


__all__ = [
    "PROVENANCE_COLUMNS",
    "STATUSES",
    "DerivationResult",
    "FinalOutput",
    "IntegrityResult",
    "ProvenanceResult",
    "build_cell_provenance",
    "check_integrity",
    "derive_fields",
    "finalise_output",
    "finalise_output_types",
]
