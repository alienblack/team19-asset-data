# Reproduce the v3 age-distribution comparison

From the repository root, install `requirements.txt` and run the frozen Phase 1 pilot and paired noise comparison using their own runbooks if their ignored output directories are absent. Do not use the truth files as model input.

```bash
PYTHONPATH=src python -m team19_asset_data.overhead_conductors.run_v3_age_distributions \
  --pilot-output outputs/v3_phase1_final_20261025 \
  --noise-output outputs/v3_noise_client_20261025 \
  --output outputs/v3_age_distribution_checked_paired_20261025
```

The output directory must be absent or empty; the runner will not overwrite existing evidence. Omit `--noise-output` for a clean-only run. The source pilot has generator seed 219, strength 100, five split/mask seeds 101-105, a 60/10/10/20 split and 30% hidden test and calibration years. Noise settings come from the saved paired comparison and are not retuned here.

`population_age_metrics.csv` records test errors for expected aged-50-or-older share, five age bands and year Wasserstein distance, plus observed and expected-hidden age-50 counts for decomposition. The `condition/seed_XXX/predicted_year_pmfs_test.csv.gz` files contain one 1930-2014 PMF per hidden cell and method, without ground truth. `evaluator_only/calibration_cells.csv` and `calibration_bins.csv` contain outcome-based probability checks. `integrity_checks.json` and `manifest.json` record mask counts, methods, upstream manifest hashes and the clean-truth year-support check. The generator reference appears only for clean MCAR evaluation.

```bash
PYTHONPATH=src python -m pytest -q tests/test_v3_age_distributions.py tests/test_v3_pilot.py tests/test_v3_noise_comparison.py
```

Do not label the binary probabilities calibrated from these five bins; the sparse extreme bins require further validation. Do not present the noise arms as correction of wrong observed values.
