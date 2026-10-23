# Common overhead-conductor imputation pilot

Version 1.0 | 22 October 2026 | PGS219-190

This stage applies two methods to the **same frozen synthetic observations** prepared by PGS219-189. The seven model fields are `installation_year`, `distance_to_sea_km`, `voltage_category`, `location_group`, `corrosion_zone`, `material`, and `covering`. `segment_id` is used only to track rows. `conductor_type`, `asset_age`, `length_km`, and evaluator-only truth are not model inputs.

## Dependency and run

The median/mode baseline and the input-preparation contract are in PR #18. The Predictive MissForest implementation belongs to Zisan's separate PR #17. This runner imports that model when it is available; it does not copy or modify the model implementation. Merge and review both dependencies before calling the combined workflow complete on `main`.

From a checkout containing both dependencies, install the repository requirements, prepare an input pilot as described in `input_preparation_runbook_v1.md`, then run:

```sh
PYTHONPATH=src python -m team19_asset_data.overhead_conductors.run_common_imputation \
  --pilot-dir outputs/overhead_input_pilot_v1 \
  --output outputs/overhead_common_imputation_v1 \
  --estimators 24 --passes 2 --seed 219
```

The output path must be new or empty. All model-facing inputs come from `observed_inputs/`, plus the split and mask manifests under `audit/`. The runner never opens `evaluator_only/`. It verifies partition IDs, row order, masked cells, the seven-field schema, and that saved baseline statistics reproduce from the fitting observations.

## Fit, replay, and outputs

The median/mode baseline uses only the fitting-observation medians and modes. Predictive MissForest fits once on `fitting_N00.csv`, is saved to `predictive_forest.joblib`, then is loaded and applied to `test_N00.csv` and `test_N11.csv` without refitting. N00 has no added predictor error; N11 has controlled predictor error **before** the same test cells are masked. Both methods see exactly the same observed rows and hidden cells within each condition.

The runner writes `baseline_test_N00.csv`, `baseline_test_N11.csv`, `predictive_forest_test_N00.csv`, and `predictive_forest_test_N11.csv`. Originally present values are preserved, valid fill domains are checked, and `asset_age` is derived as `2014 - installation_year` only after completion. `candidate_cells.csv` contains one row per hidden cell, method, and condition. It records the candidate, source/fallback, and method-specific diagnostics. `forest_diagnostics.csv` retains the forest's more detailed per-cell evidence. `run_manifest.json` records the input protocol/truth hashes, exact observed-input and source-file hashes, library versions, run settings, row and candidate counts, and fitted-model hash. The truth hash is copied from the input audit; the truth file itself is never opened.

Forest vote share and tree spread are **uncalibrated diagnostics**, not probabilities of correctness or accepted-fill decisions. They describe the forest's raw prediction; the saved candidate may also have undergone range or integer constraints. The median/mode baseline has no probability score. This stage does not compare candidates with clean answers, correct plausible-but-wrong observed values, calculate reconstruction metrics, or claim an effect on SAPN's PoF model. Those questions belong to PGS219-191 and later client-approved downstream validation.

## Integration check on the open PRs

In a temporary checkout containing PR #18 plus only the predictive-model module from PR #17, the 10,000-row pilot completed with 6,000 fitting and 2,000 test rows. Each method produced 4,200 hidden-cell candidates per test condition, with no missing candidates or duplicate cell keys. All four completed views retained the test IDs and observed cells, contained no missing model fields, and derived age consistently. A second run reproduced all completed, candidate and diagnostic CSV files byte-for-byte. The serialized model bytes and its hash differed, so binary identity is not claimed. This is a synthetic integration check, not a quality or SAPN-data result. It must be repeated on reviewed `main` after the dependencies merge.
