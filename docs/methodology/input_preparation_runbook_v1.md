# Overhead-conductor pilot input preparation

Version 1.0 | 22 October 2026 | Subtask 1 implementation

This stage produces a frozen input contract for a **synthetic-only** pilot. It does not fit an imputation model, repair present-but-wrong values, calibrate confidence, or evaluate Probability of Failure. Its runnable settings are in `overhead_conductor_pilot_protocol_v1.yaml`; the wider comparative methodology is maintained under a separate Jira task.

## Run from a fresh checkout

The input-preparation stage needs Python 3.11+, NumPy, pandas, and PyYAML. From the repository root:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install 'numpy>=1.26,<3' 'pandas>=2.2,<3' 'PyYAML>=6,<7' 'pytest>=8,<9'
PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_input_preparation.py
PYTHONPATH=src .venv/bin/python -m team19_asset_data.overhead_conductors.run_input_preparation \
  --protocol docs/methodology/overhead_conductor_pilot_protocol_v1.yaml \
  --output outputs/overhead_input_pilot_v1
```

The CLI generates the 10,000-row strength-50 reference with seed 901 using the repository generator and route-length calibration. Alternatively pass `--input` with a **complete synthetic reference only**. The output path must be new or empty; the command refuses to overwrite an existing run. `outputs/` is Git-ignored. Never use this command on authorised SAPN records.

## Output boundary

| Path | Contents | Permitted user |
| --- | --- | --- |
| `evaluator_only/clean_truth.csv` | Complete synthetic answers | Simulator/evaluator only |
| `evaluator_only/cell_change_ledger.csv` | Clean, pre-mask and observed cell values; corruption and mask flags | Simulator/evaluator only |
| `observed_inputs/fitting_N00.csv` | Seven model fields plus segment ID; 20% fitting masks | Fitting code |
| `observed_inputs/tuning_N00.csv`, `calibration_N00.csv` | Frozen 30% masks | Selection / later calibration code |
| `observed_inputs/test_N00.csv`, `test_N11.csv` | Same test IDs and masks; N11 has predictor error before masking | Frozen application code |
| `observed_inputs/frozen_preprocessor.json` | Fitting-observation medians, means, standard deviations, modes and category vocabularies | Frozen transforms |
| `audit/partitions.csv`, `mask_manifest.csv`, `tracking.csv` | Row and field allocations; non-model tracking fields | Run audit |
| `audit/schema_audit.json`, `unknown_categories.csv`, `run_manifest.json` | Validation, unseen-category flags, hashes, seeds, realized counts | Run audit |

The model-facing CSVs never contain clean answers, corruption labels, `conductor_type`, `asset_age`, `length_km`, feeder ID, or other shortcut fields. The evaluator ledger must not be joined into a fitting or prediction frame. Each partition is row-random within one synthetic population; it is **not** a future-year or held-out-feeder test.

## Checks and preprocessing

The audit rejects missing/duplicate IDs, missing complete-reference cells, nonnumeric or out-of-range numeric fields, inconsistent age/year, unknown declared labels, and non-synthetic records. It reports invalid input instead of silently deleting rare but valid assets. Range/unit checks cannot identify a plausible but incorrect value; N11 deliberately contains such predictor errors and records them only in the evaluator ledger.

`FrozenPreprocessor.fit()` accepts only a seven-field observed fitting view. Baseline fills use fitting-observed medians and modes in original units. Forest predictor preparation uses original numeric units with fitting means for missing predictors and one-hot categories learned on fitting observations. Neural input preparation (for the **planned** neural method) standardises numeric fields using fitting-observed mean/standard deviation, one-hot encodes fitting categories, and adds one availability bit per field. Any application category not seen during fitting is flagged in `unknown_categories.csv`; it never expands a fitted vocabulary. The target is excluded from each forest predictor matrix.

For the 10,000-row pilot, the expected partition counts are 6,000 / 1,000 / 1,000 / 2,000. Material and year each have 1,200 fitting masks and 600 test masks. N11 selects 400 test cells each for sea-distance error and voltage, location and corrosion replacement. Both test conditions reuse exactly the same 30% hidden-cell masks. The manifest reports **actual** changed counts separately from selected cells.

Two runs with identical code and dependencies should produce identical files. Compare the output directories with `diff -qr` or check the hashes in `audit/run_manifest.json`. Before publishing results, freeze the code revision and do not change the protocol after inspecting test outcomes.
