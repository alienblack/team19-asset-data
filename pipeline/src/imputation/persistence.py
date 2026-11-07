"""Saving and loading fitted imputers, with fitting-record provenance.

A saved model is a joblib file plus a ``<model>.provenance.json`` sidecar that
lists a SHA-256 fingerprint of every record identifier the model was fitted
on. Before a saved model is evaluated on a file, :func:`check_provenance`
refuses any record that was part of fitting, so "independent evaluation"
is enforced mechanically rather than by convention. Fingerprints are hashes,
not identifiers, so the sidecar does not disclose asset identifiers.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import sklearn

from .. import __version__ as package_version
from .registry import BaseImputer

FORMAT_VERSION = 1


class ProvenanceError(ValueError):
    """Raised when evaluation records overlap with the model's fitting records."""


def _provenance_path(model_path: Path) -> Path:
    return model_path.with_name(model_path.name + ".provenance.json")


def fingerprint_ids(frame: pd.DataFrame, id_column: str) -> set[str]:
    if id_column not in frame.columns:
        raise ValueError(f"Identifier column '{id_column}' not found.")
    ids = frame[id_column].astype(str)
    if ids.isna().any() or not ids.is_unique:
        raise ValueError(f"'{id_column}' must contain unique identifiers.")
    return {hashlib.sha256(value.encode("utf-8")).hexdigest() for value in ids}


def model_fingerprint(imputer: BaseImputer, fitted_frame: pd.DataFrame | None = None) -> dict[str, Any]:
    """What a saved model expects of its input: columns, constraints, vocabularies.

    Stored in the provenance sidecar and compared by :func:`check_compatibility`
    before a saved model is applied to another file.
    """
    vocabularies: dict[str, list[str]] = {}
    model = getattr(imputer, "model", None)
    if model is not None and getattr(model, "vocabularies_", None):
        vocabularies = {column: [str(v) for v in values] for column, values in model.vocabularies_.items()}
    elif fitted_frame is not None:
        for column in imputer.categorical_columns:
            if column in fitted_frame.columns:
                vocabularies[column] = sorted(str(v) for v in fitted_frame[column].dropna().unique())
    return {
        "method": imputer.name,
        "numeric_columns": list(imputer.numeric_columns),
        "categorical_columns": list(imputer.categorical_columns),
        "constraints": {k: dict(v) for k, v in imputer.constraints.items()},
        "vocabularies": vocabularies,
        "hyperparameters": imputer.describe().get("config", {}),
    }


class CompatibilityError(ValueError):
    """Raised when a saved model does not fit the configuration or data it is applied to."""


def check_compatibility(
    path: str | Path,
    imputer: BaseImputer,
    *,
    numeric_columns: list[str] | tuple[str, ...],
    categorical_columns: list[str] | tuple[str, ...],
    constraints: dict[str, dict[str, Any]] | None = None,
    frame: pd.DataFrame | None = None,
) -> dict[str, list[str]]:
    """Compare a loaded model with the current configuration and input.

    Returns ``{"errors": [...], "warnings": [...]}``. Errors: model columns
    differ from the configuration, or the input lacks model columns.
    Warnings: constraints differ, or the input holds category labels the
    model never saw (they carry no information for prediction and are
    reported per column).
    """
    errors: list[str] = []
    warnings_: list[str] = []
    if tuple(imputer.numeric_columns) != tuple(numeric_columns) or tuple(imputer.categorical_columns) != tuple(categorical_columns):
        errors.append(
            f"model columns {list(imputer.numeric_columns) + list(imputer.categorical_columns)} differ from the "
            f"configuration {list(numeric_columns) + list(categorical_columns)}"
        )
    sidecar = _provenance_path(Path(path))
    fingerprint = {}
    if sidecar.is_file():
        fingerprint = json.loads(sidecar.read_text(encoding="utf-8")).get("fingerprint", {})
    if constraints is not None and fingerprint.get("constraints") not in (None, {}):
        saved = {k: {kk: (float(vv) if isinstance(vv, (int, float)) and not isinstance(vv, bool) else vv) for kk, vv in v.items()} for k, v in fingerprint["constraints"].items()}
        current = {k: {kk: (float(vv) if isinstance(vv, (int, float)) and not isinstance(vv, bool) else vv) for kk, vv in v.items()} for k, v in constraints.items()}
        if saved != current:
            warnings_.append(f"constraints at fit time {saved} differ from the current schema {current}; imputed values follow the model's constraints")
    if frame is not None:
        missing = [c for c in list(numeric_columns) + list(categorical_columns) if c not in frame.columns]
        if missing:
            errors.append(f"input lacks model columns {missing}")
        for column, vocabulary in fingerprint.get("vocabularies", {}).items():
            if column in frame.columns:
                seen = set(vocabulary)
                unseen = sorted(set(str(v) for v in frame[column].dropna().unique()) - seen)
                if unseen:
                    warnings_.append(f"{column}: {len(unseen)} label(s) never seen at fit time {unseen[:5]}; they carry no information for prediction")
    return {"errors": errors, "warnings": warnings_}


def save_imputer(
    imputer: BaseImputer,
    path: str | Path,
    fitted_frame: pd.DataFrame,
    id_column: str,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Persist a fitted imputer and write its provenance sidecar (fitting-id fingerprints + model fingerprint)."""
    if not imputer.supports_persistence:
        raise ValueError(f"Method '{imputer.name}' is transductive and cannot be saved for later application.")
    if not imputer.is_fitted:
        raise ValueError("Fit the imputer before saving it.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    created = datetime.now(timezone.utc).isoformat(timespec="seconds")
    payload = {
        "format_version": FORMAT_VERSION,
        "package_version": package_version,
        "sklearn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        "method": imputer.name,
        "created_utc": created,
        "object": imputer,
    }
    joblib.dump(payload, path, compress=3)
    provenance = {
        "format_version": FORMAT_VERSION,
        "method": imputer.name,
        "created_utc": created,
        "id_column": id_column,
        "n_fitting_rows": int(len(fitted_frame)),
        "fitting_id_sha256": sorted(fingerprint_ids(fitted_frame, id_column)),
        "fingerprint": model_fingerprint(imputer, fitted_frame),
        **(extra or {}),
    }
    _provenance_path(path).write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return path


def load_imputer(path: str | Path) -> BaseImputer:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Model file not found: {path}")
    payload = joblib.load(path)
    if not isinstance(payload, dict) or "object" not in payload:
        raise ValueError("File is not a saved overhead_conductors imputer.")
    if payload.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"Unsupported model format version {payload.get('format_version')}.")
    if payload.get("sklearn_version") != sklearn.__version__:
        warnings.warn(
            f"Model saved with scikit-learn {payload.get('sklearn_version')} but loaded with "
            f"{sklearn.__version__}; verify outputs before relying on them.",
            stacklevel=2,
        )
    imputer = payload["object"]
    if not isinstance(imputer, BaseImputer):
        raise ValueError("Saved object is not an imputer.")
    return imputer


def read_provenance(path: str | Path) -> dict[str, Any]:
    sidecar = _provenance_path(Path(path))
    if not sidecar.is_file():
        raise ProvenanceError(
            f"No provenance sidecar next to '{path}'. Save models with save_imputer so independent evaluation can be verified."
        )
    return json.loads(sidecar.read_text(encoding="utf-8"))


def check_provenance(path: str | Path, frame: pd.DataFrame, id_column: str) -> dict[str, Any]:
    """Raise :class:`ProvenanceError` if any record in ``frame`` was used to fit the model."""
    info = read_provenance(path)
    if info.get("id_column") != id_column:
        raise ProvenanceError(
            f"Model was fitted with identifier column '{info.get('id_column')}' but '{id_column}' was given."
        )
    overlap = fingerprint_ids(frame, id_column) & set(info.get("fitting_id_sha256", []))
    if overlap:
        raise ProvenanceError(
            f"{len(overlap)} of {len(frame)} records were used to fit this model; they cannot be used for independent evaluation."
        )
    return info
