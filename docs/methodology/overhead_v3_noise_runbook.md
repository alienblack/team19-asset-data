# Reproduce the PGS219-192 clean/noisy comparison

This run uses only the repository's controlled synthetic data. Run from the repository root with its Python dependencies installed. Use new output directories; neither runner overwrites an existing result.

```sh
PYTHONPATH=src python -m pytest -q tests/test_v3_pilot.py tests/test_v3_noise_comparison.py
PYTHONPATH=src python -m team19_asset_data.overhead_conductors.run_v3_pilot \
  --output outputs/v3_phase1_final_20261025
PYTHONPATH=src python -m team19_asset_data.overhead_conductors.run_v3_noise_comparison \
  --pilot-output outputs/v3_phase1_final_20261025 \
  --output outputs/v3_noise_client_20261025
```

The first runner freezes five seeds, partitions, target masks and clean tuning choices. The second reads those saved artefacts, applies each error mechanism to a separate complete copy, overlays it on the frozen observed views, and refits on noisy fitting rows without retuning. It keeps clean truth outside the model-facing files. If the directories above already exist, choose fresh names and use the corresponding `--pilot-output` path.

The noisy output contains `observed_changes.csv` (actual changed *available* cells and denominators), compact `observed_change_summary.csv` and `primary_paired_summary.csv`, per-arm/per-seed `observed_test.csv` and `candidate_cells.csv`, `cell_metrics.csv`, `population_metrics.csv`, `paired_cell_metrics.csv`, `paired_population_metrics.csv`, `material_confusion.csv`, `material_by_decade.csv`, plots, run manifest and integrity checks. `evaluator_only/review_cases.csv` contains clean answers and must not be supplied to a model. Forest models and error-step metadata are saved under each arm/seed. The clean generator reference is **not** reused to score noisy arms.

The noise arms are exploratory scenarios, not estimates of SAPN's data quality. A plausible but wrong observed value is not automatically corrected. Uncertainty diagnostics are not calibrated confidence or acceptance probabilities. The entire run is synthetic and does not calculate PoF.
