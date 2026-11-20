# Predictive MissForest — Experimental Evaluation

This runbook was reconciled with the YAML workflow when PGS219-186 was integrated
with the newer main branch. It describes the current commands. Historical E5
results remain unchanged and are a separate, in-batch experiment.

## 1. Purpose and input

Compare a training-fitted Median/Mode floor and Predictive MissForest on the same
held-out synthetic cells. The input can already contain missing values. An
additional evaluation mask hides only currently observed cells. Naturally missing
values have no known truth and are excluded from reconstruction scoring.

Use `--truth-csv` to supply a separate clean synthetic reference when the input is
noisy. Otherwise the temporarily hidden recorded input values are the reference.
Models do not receive the clean truth as a fitting input.

The seven model fields are installation year, distance to sea, voltage, location,
corrosion zone, material and covering. Identifiers and derived/leakage fields
remain excluded from model targets and predictors as configured.

## 2. Fitting and evaluation

The YAML `data_split` creates 60% train, 10% validation, 10% reserved calibration
and 20% test partitions. Tuning, when enabled, uses train and validation only.
A separate evaluation forest is fitted on train only and scores the held-out test
cells. The Median/Mode statistics also come from train only.

After held-out scoring, a separate deployment forest is fitted on all input rows.
This final full-data model must not be used for independent evaluation on those
same rows. Fitting-ID provenance is saved beside both models. For subsequent
`evaluate_only` runs, the runner rejects fitting/evaluation ID overlap.

The passes are fixed, not a convergence-based stopping rule. Forest disagreement
and percentile bands are exploratory diagnostics, not calibrated confidence.

## 3. Smoke test

Run from the repository root in an environment with the project dependencies:

```bash
PYTHONPATH=src python -m experiments.run_predictive_imputation_experiment \
  --input-csv data/synthetic/overhead_conductors/strength_100/material_model_ready_data.csv \
  --config config/imputation/overhead_conductors_prototype_v2.yaml \
  --output-dir outputs/predictive_missforest_evaluation_smoke \
  --model-path outputs/models/predictive_missforest_smoke.joblib \
  --sample-rows 350 --n-estimators 20 --n-passes 2
```

The prototype YAML enables evaluation and disables tuning. Its YAML values control
mask rates, mask seeds and partition seeds. The old `--seed`, `--mask-rate`,
`--tune` and `--grid` flags are not part of the current CLI.

## 4. Full evaluation and tuning

For a full evaluation, remove the three development-size overrides above. For a
parameter search, use `config/imputation/overhead_conductors_tuning_v2.yaml`.
That file enables tuning with the all-attribute objective, four depth choices and
three leaf-size choices. Change the YAML for a smaller grid or a different mask
rate. A selected configuration is not guaranteed to reproduce an earlier E3
configuration under another protocol.

```bash
PYTHONPATH=src python -m experiments.run_predictive_imputation_experiment \
  --input-csv data/synthetic/overhead_conductors/strength_100/material_model_ready_data.csv \
  --config config/imputation/overhead_conductors_tuning_v2.yaml \
  --output-dir outputs/predictive_missforest_tuned_evaluation
```

## 5. Important options

| Option | Use |
|---|---|
| `--input-csv` | Complete or already-incomplete observed input |
| `--truth-csv` | Optional separate clean synthetic reference for scoring |
| `--config` | YAML methods, workflow, split and evaluation settings |
| `--output-dir` | Run-specific metrics and exports |
| `--model-path` | Override deployment model destination |
| `--mode` | fit_only, fit_transform, transform_only or evaluate_only |
| `--partition-manifest` | Reuse the same ID-based partitions across scenarios |
| `--evaluation-mask-csv` | Reuse an additional evaluation mask keyed by segment ID |
| `--sample-rows`, `--n-estimators`, `--n-passes` | Development-only overrides |

See `PYTHONPATH=src python -m experiments.run_predictive_imputation_experiment --help`.

## 6. Outputs and interpretation

| File | Meaning |
|---|---|
| `metrics_test.csv` | Test-only masked-cell scores for both methods |
| `partition_manifest.csv` | ID membership: train, validation, calibration and test |
| `masked_test.csv`, `test_hidden_mask.csv` | Exact test inputs and additional hidden positions |
| `predictive_imputed_test.csv`, `baseline_imputed_test.csv` | Held-out completions |
| `evaluation_model.joblib` | Forest fitted only on the training partition |
| Configured `model_path` | Separate deployment forest fitted on all input rows |
| `*.provenance.json` | Fitting-ID hashes used to detect later evaluation overlap |
| `predictive_imputed_full.csv` | Full-input completion in fit_transform mode |
| `predictive_diagnostics_test.csv` | Uncalibrated forest diagnostics |
| `run_metadata.json` | Actual workflow, partition sizes and fitted settings |
| `tuning_validation.csv` | Candidate scores when tuning is enabled |

Lower numeric MAE is better. Higher categorical macro-F1 gives equal importance
to each class. Scores concern only additional hidden test cells. They do not
measure accuracy on naturally missing fields or validate real SAPN performance.

## 7. Noise integration

PGS219-186 adds nested same-seed numeric/categorical noise levels. The integrated
implementation also retains main's explicit mixed-noise support for material and
installation year, with linked asset age recomputed. Material stays protected in
legacy categorical-only scenarios. Numeric noise levels control both selected
fraction and maximum relative error, while installation years use `year_step`.

Existing saved E5 results are not rerun or replaced by this merge. Expanded mixed
scenarios on main include year/material corruption. Check saved scenario metadata
before comparing them with a historical predictor-only experiment.

## 8. Relationship to the E5 noise robustness experiment

**Do not confuse these evaluations.**

| Held-out runner documented above | E5 mixed-noise comparison |
|---|---|
| Starts with a **complete** synthetic reference | Creates or uses **0% / 30% / 50% noisy** versions of a synthetic reference |
| Splits records **before** fitting and scoring | E5 results fitted and scored on the **same 10,000-record masked batch** |
| Compares **training-fitted Median/Mode** and Predictive MissForest | Compares **Simple Mean/Mode**, Original MissForest and tuned Predictive E3 |
| Uses a random per-partition evaluation mask | Uses the pre-established **MF01** mask (2,000 values per modeled field) |
| Measures masked-cell reconstruction on **held-out synthetic records** | Measures in-batch masked-cell reconstruction under increasing synthetic corruption |

E5 uses the `N06_mixed_30` and `N06_mixed_50` scenarios with noise seed `329`, a common clean reference, and fixed E3 parameters. In the existing E5 report, two headline attributes are distance-to-sea MAE and corrosion-zone Macro-F1. E5 showed deterioration with noise, but it does **not** independently establish performance on unseen rows or real SAPN assets.

The three-method 0/30/50 comparison requires the team's **separate E5 orchestration/noise pipeline**, not `experiments.run_predictive_imputation_experiment`. This README intentionally does not invent CLI commands for that separate workflow. The consolidated outputs, if present in your repository, can be inspected at:

```text
outputs/noise_robustness/comparison_0_30_50/all_metrics.csv
outputs/noise_robustness/comparison_0_30_50/numerical_comparison.csv
outputs/noise_robustness/comparison_0_30_50/categorical_comparison.csv
```

## 9. Tests

```bash
PYTHONPATH=src python -m pytest -q tests/
```

Regression tests verify nested selections, material protection in legacy noise,
explicit mixed material/year errors, observed-cell preservation, train/test
separation and exclusion of naturally missing values from evaluation.

## 10. Limits

These are synthetic records with designed errors and missingness. A single split
does not establish stability across seeds. The reserved calibration partition has
no calibration operation in this runner. No downstream SAPN reliability model is
validated here. The separate transformer benchmark has its own calibration and
stress-test pipeline.
