"""Method registry contract and model persistence with provenance."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.imputation import (
    METHODS,
    ImputationOutput,
    ProvenanceError,
    available_methods,
    build_imputer,
    check_provenance,
    load_imputation_config,
    load_imputer,
    read_provenance,
    resolve_hyperparameters,
    save_imputer,
)
from src.imputation.registry import DIAGNOSTIC_COLUMNS
from src.ingestion import load_schema
from src.preprocessing import select_model_frame
from sample_frames import make_masked_frame

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = load_schema(ROOT / "config" / "overhead_conductors" / "data_schema.yaml")
CONFIG = load_imputation_config(ROOT / "config" / "overhead_conductors" / "imputation.yaml", SCHEMA)
FAST = {"missforest": {"n_estimators": 8, "max_iter": 2}, "predictive_missforest": {"n_estimators": 8, "n_passes": 2}, "median_mode": None}


def _frame(n: int = 160, seed: int = 11) -> pd.DataFrame:
    return select_model_frame(make_masked_frame(n, seed), SCHEMA, CONFIG).frame


@pytest.mark.parametrize("method", list(METHODS))
def test_every_method_honours_the_output_contract(method):
    X = _frame()
    imputer = build_imputer(method, CONFIG, FAST[method])
    output = imputer.fit_transform(X)
    assert isinstance(output, ImputationOutput)
    imputed, diagnostics = output.imputed, output.diagnostics
    assert list(imputed.columns) == list(X.columns) and imputed.index.equals(X.index)
    model_columns = list(CONFIG.numeric_columns) + list(CONFIG.categorical_columns)
    assert not imputed[model_columns].isna().any().any()
    # observed cells untouched, non-model columns untouched
    for column in X.columns:
        observed = X[column].notna()
        if pd.api.types.is_numeric_dtype(X[column]):
            assert np.allclose(X.loc[observed, column].astype(float), imputed.loc[observed, column].astype(float))
        else:
            assert (X.loc[observed, column].astype(str) == imputed.loc[observed, column].astype(str)).all()
    assert list(diagnostics.columns) == list(DIAGNOSTIC_COLUMNS)
    assert len(diagnostics) == int(X[model_columns].isna().sum().sum())
    assert set(diagnostics["column"]) <= set(model_columns)
    assert output.metadata["method"] == method and "runtime_seconds" in output.metadata
    # constraints applied to imputed cells
    filled_years = imputed.loc[X["installation_year"].isna(), "installation_year"]
    assert (filled_years == filled_years.round()).all() and filled_years.between(1900, 2026).all()


def test_predictive_diagnostics_carry_confidence_fields():
    X = _frame()
    output = build_imputer("predictive_missforest", CONFIG, FAST["predictive_missforest"]).fit_transform(X)
    cat = output.diagnostics[output.diagnostics["kind"] == "categorical"]
    num = output.diagnostics[output.diagnostics["kind"] == "numeric"]
    assert cat["max_proba"].between(0, 1).all() and cat["margin"].between(0, 1).all()
    assert (num["tree_std"] >= 0).all() and (num["tree_p10"] <= num["tree_p90"]).all()
    assert (output.diagnostics["prediction_source"] == "random_forest").all()


def test_resolve_hyperparameters_layers_defaults_selected_and_overrides():
    params = resolve_hyperparameters(CONFIG, "predictive_missforest")
    assert params["max_depth"] == CONFIG.selected_hyperparameters["predictive_missforest"]["max_depth"]
    assert params["n_estimators"] == CONFIG.predictive_missforest_config["n_estimators"]
    overridden = resolve_hyperparameters(CONFIG, "predictive_missforest", {"max_depth": 3, "n_estimators": 5})
    assert overridden["max_depth"] == 3 and overridden["n_estimators"] == 5
    assert resolve_hyperparameters(CONFIG, "median_mode")["numeric_strategy"] == "median"
    with pytest.raises(ValueError, match="Unknown method"):
        build_imputer("knn", CONFIG)
    assert available_methods() == ("median_mode", "missforest", "predictive_missforest")


def test_selected_hyperparameters_block_is_validated(tmp_path):
    import yaml

    raw = yaml.safe_load((ROOT / "config" / "overhead_conductors" / "imputation.yaml").read_text(encoding="utf-8"))
    raw["selected_hyperparameters"]["predictive_missforest"]["random_state"] = 1
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="may only override"):
        load_imputation_config(path)
    raw["selected_hyperparameters"] = {"knn": {"k": 3}}
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="non-tunable"):
        load_imputation_config(path)


def test_save_load_and_provenance(tmp_path):
    X = _frame(120, seed=21)
    imputer = build_imputer("predictive_missforest", CONFIG, FAST["predictive_missforest"]).fit(X)
    path = save_imputer(imputer, tmp_path / "model.joblib", X, "segment_id", extra={"note": "test"})
    assert path.is_file() and (tmp_path / "model.joblib.provenance.json").is_file()
    info = read_provenance(path)
    assert info["n_fitting_rows"] == len(X) and info["note"] == "test" and len(info["fitting_id_sha256"]) == len(X)
    assert all(len(h) == 64 for h in info["fitting_id_sha256"])  # hashes, not identifiers

    reloaded = load_imputer(path)
    assert reloaded.name == "predictive_missforest" and reloaded.is_fitted
    assert reloaded.transform(X).imputed.equals(imputer.transform(X).imputed)

    with pytest.raises(ProvenanceError, match="used to fit"):
        check_provenance(path, X.iloc[:3], "segment_id")
    unseen = X.iloc[:3].copy()
    unseen["segment_id"] = ["NEW_1", "NEW_2", "NEW_3"]
    assert check_provenance(path, unseen, "segment_id")["method"] == "predictive_missforest"
    with pytest.raises(ProvenanceError, match="identifier column"):
        check_provenance(path, unseen.rename(columns={"segment_id": "asset_id"}), "asset_id")


def test_transductive_and_unfitted_models_cannot_be_saved(tmp_path):
    X = _frame(80, seed=5)
    legacy = build_imputer("missforest", CONFIG, FAST["missforest"]).fit(X)
    with pytest.raises(ValueError, match="transductive"):
        save_imputer(legacy, tmp_path / "legacy.joblib", X, "segment_id")
    with pytest.raises(ValueError, match="Fit the imputer"):
        save_imputer(build_imputer("median_mode", CONFIG), tmp_path / "mm.joblib", X, "segment_id")
    median = build_imputer("median_mode", CONFIG).fit(X)
    save_imputer(median, tmp_path / "mm.joblib", X, "segment_id")
    assert load_imputer(tmp_path / "mm.joblib").statistics_ == median.statistics_


def test_compatibility_check_catches_column_and_vocabulary_mismatches(tmp_path):
    from src.imputation import check_compatibility, read_provenance

    X = _frame(100, seed=33)
    imputer = build_imputer("predictive_missforest", CONFIG, FAST["predictive_missforest"]).fit(X)
    path = save_imputer(imputer, tmp_path / "m.joblib", X, "segment_id")
    fingerprint = read_provenance(path)["fingerprint"]
    assert fingerprint["numeric_columns"] == list(CONFIG.numeric_columns) and "material" in fingerprint["vocabularies"]
    ok = check_compatibility(path, load_imputer(path), numeric_columns=CONFIG.numeric_columns, categorical_columns=CONFIG.categorical_columns, constraints=CONFIG.constraints, frame=X)
    assert ok == {"errors": [], "warnings": []}
    new = X.copy()
    new.loc[new.index[0], "material"] = "Unobtainium"
    report = check_compatibility(path, load_imputer(path), numeric_columns=CONFIG.numeric_columns, categorical_columns=CONFIG.categorical_columns, constraints=CONFIG.constraints, frame=new)
    assert report["errors"] == [] and any("Unobtainium" in w for w in report["warnings"])
    bad = check_compatibility(path, load_imputer(path), numeric_columns=("installation_year",), categorical_columns=CONFIG.categorical_columns, frame=X.drop(columns=["covering"]))
    assert len(bad["errors"]) == 2
