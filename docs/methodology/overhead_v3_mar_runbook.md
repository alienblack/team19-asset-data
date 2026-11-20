# Reproduce the PGS219-192 MAR comparison

The frozen Phase 1 output must exist; reproduce it with `overhead_v3_phase1_runbook.md` if needed. The MAR runner refuses to overwrite a nonempty output directory. It checks the clean reference hash and 1930-2014 year support before fitting. Run from the repository root:

```bash
PYTHONPATH=src python -m team19_asset_data.overhead_conductors.run_v3_mar \
  --pilot-output outputs/v3_phase1_final_20261025 \
  --output outputs/v3_mar_final_20261025
```

For a smoke run, pass `--arms voltage_mar material_mar` and a new output path. The four arm definitions and probabilities are in `overhead_v3_mar_design.md` and the output `manifest.json`. Subsetting arms does not change their random streams.

`population_age_metrics.csv` compares expected-share and complete-case errors. `hidden_year_cell_metrics.csv` has point-year MAE and signed bias. `mask_counts.csv` gives subgroup denominators, assigned probabilities and realised mask rates for every partition. `test_group_age_metrics.csv` gives subgroup age-share errors and counts. Each `arm/seed_XXX/` directory holds observed inputs, year masks, a saved forest, candidate point years and a compressed per-cell test PMF. The generator reference and actual calibration outcomes appear only in the evaluator's result tables, not model inputs. `integrity_checks.json` and `manifest.json` record run assumptions and upstream hashes.

```bash
PYTHONPATH=src python -m pytest -q tests/test_v3_mar.py tests/test_v3_age_distributions.py
```

The present output is synthetic and has no SAPN record-level truth or PoF result. Do not treat one test arm's probabilities as estimated SAPN missingness rates.
