# Predictive MissForest — Experimental Evaluation

This document explains how to run the **Predictive MissForest evaluation experiment** for Team 19's SA Power Networks overhead-conductor capstone project.

The evaluation runner compares a **training-fitted Median/Mode baseline** against the **Predictive MissForest** implementation on the **same held-out synthetic test cells**. It also verifies that a saved model can be reloaded and can reproduce its predictions.

> **Scope:** This is a development benchmark using synthetic data. It is not a performance assessment on real SA Power Networks asset records. The separate E5 0% / 30% / 50% mixed-noise experiment is described below, but **is not run by this command**.

## 1. Prerequisites

Run all commands from the root of the `team19-asset-data` repository.

- Python environment with `numpy`, `pandas`, `scikit-learn`, `joblib`, `PyYAML` and `pytest` installed.
- The `src/team19_asset_data` package and `experiments/run_predictive_imputation_experiment.py` present in this branch.
- An **unmasked, complete synthetic reference CSV** containing all seven configured model fields.
- An imputation YAML configuration defining those model fields and the Predictive MissForest settings.

Check the required files:

```bash
pwd
ls experiments/run_predictive_imputation_experiment.py
ls config/imputation/overhead_conductors_all_attributes_v2.yaml
ls data/synthetic/overhead_conductors/strength_100/material_model_ready_data.csv
```

> The runner **rejects incomplete ground truth**. Do **not** pass `data/synthetic/overhead_conductors/strength_100/masked/MF01/masked_MF01.csv` to this evaluation runner. It generates its **own evaluation masks after splitting** the complete reference data.

## 2. How the evaluation works

The runner follows a reproducible, row-disjoint protocol:

| Partition | Proportion | Purpose |
|---|---:|---|
| Fit | 60% | Train the baseline's statistics and Predictive MissForest forests |
| Tune | 10% | Select hyperparameters when `--tune` is provided |
| Calibration | 10% | Reserved; not used for fitting, tuning or scoring in this runner |
| Test | 20% | Evaluate methods on intentionally hidden values, using identical test masks |

After splitting, the runner hides a reproducible fraction of cells in the fit, tune and test partitions. It then:

1. Fits Median/Mode statistics and Predictive MissForest **only on the fitting partition**.
2. Optionally selects model hyperparameters using the tuning partition (`--tune`).
3. Saves the fitted Predictive MissForest model to disk.
4. Applies both methods to the **same masked held-out test records**.
5. Calculates errors **only at deliberately masked test positions**, against the original complete synthetic values.
6. Reloads the saved forest model and asserts that its transformed output matches the previously generated test output.

The seven model fields are `installation_year`, `distance_to_sea_km`, `voltage_category`, `location_group`, `corrosion_zone`, `material`, and `covering`. Non-model fields (such as identifiers and known leakage/derived fields) must not be treated as independent target labels.

## 3. Quick smoke test

Use a small sample and fewer trees/passes to verify that the full pipeline runs. **These are development settings, not the final tuned E3 results.**

```bash
PYTHONPATH=src python -m experiments.run_predictive_imputation_experiment \
  --input-csv data/synthetic/overhead_conductors/strength_100/material_model_ready_data.csv \
  --config config/imputation/overhead_conductors_all_attributes_v2.yaml \
  --output-dir outputs/predictive_missforest_evaluation_smoke \
  --sample-rows 350 \
  --n-estimators 20 \
  --n-passes 2 \
  --seed 219 \
  --mask-rate 0.30
```

`--sample-rows 350` takes the first 350 complete rows **before** the seeded random partition. Omit it for the full 10,000-row run.

## 4. Full held-out evaluation

Run with the complete synthetic reference and the configured model settings:

```bash
PYTHONPATH=src python -m experiments.run_predictive_imputation_experiment \
  --input-csv data/synthetic/overhead_conductors/strength_100/material_model_ready_data.csv \
  --config config/imputation/overhead_conductors_all_attributes_v2.yaml \
  --output-dir outputs/predictive_missforest_evaluation \
  --seed 219 \
  --mask-rate 0.30
```

**Important:** Without `--tune`, model parameters come from the YAML configuration. Do not describe this run as “tuned E3” unless its `methods.predictive_missforest` section actually specifies the E3 settings. For the previously selected E3 configuration, these were **100 trees, five fixed passes, `max_depth=8`, and `min_samples_leaf=2`**. The runner's CLI exposes overrides for tree/pass counts, but not depth or minimum leaf size; set the latter in the YAML, or use tuning.

The **five passes are fixed**. They are not a convergence-based stopping criterion; Original MissForest uses a different, convergence-based workflow.

## 5. Optional hyperparameter search

The runner supports two tuning grids. Tuning uses only the training and tuning partitions; final scores are calculated on the held-out test partition.

**Quick tuning smoke test:**

```bash
PYTHONPATH=src python -m experiments.run_predictive_imputation_experiment \
  --input-csv data/synthetic/overhead_conductors/strength_100/material_model_ready_data.csv \
  --config config/imputation/overhead_conductors_all_attributes_v2.yaml \
  --output-dir outputs/predictive_missforest_tuning_smoke \
  --sample-rows 350 \
  --n-estimators 20 \
  --n-passes 2 \
  --tune \
  --grid quick
```

**Full parameter search:**

```bash
PYTHONPATH=src python -m experiments.run_predictive_imputation_experiment \
  --input-csv data/synthetic/overhead_conductors/strength_100/material_model_ready_data.csv \
  --config config/imputation/overhead_conductors_all_attributes_v2.yaml \
  --output-dir outputs/predictive_missforest_tuned_evaluation \
  --seed 219 \
  --mask-rate 0.30 \
  --n-estimators 100 \
  --n-passes 5 \
  --tune \
  --grid full
```

- `quick` checks two candidate combinations (`max_depth` 4 or unlimited, `min_samples_leaf` 1).
- `full` checks twelve combinations: `max_depth` in `{4, 8, 16, None}` and `min_samples_leaf` in `{1, 2, 10}`.
- The winner is selected using the tuning subset. Its actual configuration is saved in `run_metadata.json`.
- The winner **is not guaranteed** to match the previously chosen E3 combination for every dataset or evaluation protocol.

## 6. Main CLI options

| Argument | Purpose | Default |
|---|---|---|
| `--input-csv` | Path to **complete synthetic reference data** | Required |
| `--config` | Imputation YAML file | Supply explicitly (recommended) |
| `--output-dir` | Folder for models, metrics and artefacts | Required |
| `--sample-rows` | Limit the number of initial reference rows | All rows |
| `--seed` | Reproducible partition and masking seed | `219` |
| `--mask-rate` | Approximate fraction of observed cells hidden within each evaluated partition | `0.30` |
| `--n-estimators` | Development override for number of trees per forest | From YAML |
| `--n-passes` | Development override for fixed sequential passes | From YAML |
| `--tune` | Enable training/tuning split hyperparameter search | Disabled |
| `--grid` | Tuning grid: `quick` or `full` (only relevant with `--tune`) | `full` |

For the complete argument list:

```bash
PYTHONPATH=src python -m experiments.run_predictive_imputation_experiment --help
```

## 7. Output files

The selected `--output-dir` contains:

| File | Description |
|---|---|
| `metrics_test.csv` | Baseline and Predictive MissForest scores for hidden held-out test cells |
| `partition_manifest.csv` | Original row/segment identifiers and fit/tune/calibration/test membership |
| `masked_test.csv` | Incomplete test records given to the imputers |
| `test_hidden_mask.csv` | Boolean mask identifying which test cells were deliberately hidden |
| `predictive_imputed_test.csv` | Completed held-out test data from Predictive MissForest |
| `baseline_imputed_test.csv` | Completed held-out test data from training-fitted Median/Mode |
| `predictive_missforest.joblib` | Saved fitted Predictive MissForest pipelines |
| `predictive_diagnostics_test.csv` | Exploratory per-cell forest diagnostics; **not calibrated confidence** |
| `convergence_training.csv` | Per-pass training diagnostics; five fixed passes do **not** imply convergence |
| `run_metadata.json` | Seed, partition sizes, masked-cell counts, fitted settings and fit report |
| `tuning_validation.csv` | Candidate validation scores, **only when `--tune` is used** |

On successful completion, the runner prints partition counts, the number of trained forest pipelines, selected metrics for `installation_year` and `material`, and the output folder path.

### Inspect the held-out results

```bash
python - <<'PY'
import pandas as pd
from pathlib import Path
out = Path('outputs/predictive_missforest_evaluation')
metrics = pd.read_csv(out / 'metrics_test.csv')
print(metrics[['method', 'variable', 'n_masked', 'mae', 'macro_f1']].to_string(index=False))
PY
```

**Metric interpretation:** Lower **MAE** is better for numeric fields; higher **Macro-F1** is better for categorical fields (particularly where categories are imbalanced). Blank MAE values for categories and blank Macro-F1 values for numbers are expected.

### Inspect what was saved

```bash
python - <<'PY'
import json
from pathlib import Path
out = Path('outputs/predictive_missforest_evaluation')
meta = json.loads((out / 'run_metadata.json').read_text())
print('Partitions:', meta['partitions'])
print('Tuned:', meta['tuned'])
print('Model settings:', meta['predictive_config'])
print('Trained models:', meta['fit_report']['n_models'])
PY
```

For the standard 10,000-record reference, the expected partition sizes are **6,000 fit / 1,000 tune / 1,000 reserved calibration / 2,000 test**. The precise count of hidden cells is reported in `run_metadata.json` and is not necessarily exactly 30% because evaluation masks are randomly drawn.

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

## 9. Tests and reproducibility checks

To run the new merge-request tests (assuming they have been added to `tests/`):

```bash
PYTHONPATH=src python -m pytest -q \
  tests/test_predictive_missforest_review.py \
  tests/test_predictive_experiment_runner_review.py
```

Run the repository's full test suite:

```bash
PYTHONPATH=src python -m pytest -q tests/
```

Before sharing experiment outputs, confirm that the saved metadata records the correct input path, seed, mask rate and hyperparameters. Do not reuse numbers from the live demonstration as held-out evaluation results.

## 10. Common issues

| Problem | Resolution |
|---|---|
| `No module named experiments.run_predictive_imputation_experiment` | Run from repository root; confirm the experiment file exists at `experiments/run_predictive_imputation_experiment.py`. |
| `No module named team19_asset_data` | Run with `PYTHONPATH=src`, or install the project package in your environment. |
| Error that complete truth is required | You passed an already-masked CSV. Use the complete `material_model_ready_data.csv` file for this runner. |
| `--tune` takes a long time | Try the 350-row smoke run with `--grid quick`, fewer trees and fewer passes before the full search. |
| Different results from the E5 report | The held-out runner uses a **different protocol and baseline**. Direct numeric comparison is not appropriate. |
| Unexpected hyperparameters | Check `methods.predictive_missforest` in the YAML and the `predictive_config` recorded in `run_metadata.json`. |
| `joblib` model compatibility or trust warning | Reload models in a compatible scikit-learn environment and only load model files produced by trusted sources. |

## 11. Limitations

- These experiments use synthetic conductor records and generated missingness, not measured SAPN data-quality rates.
- Holding out rows provides stronger evidence about **synthetic generalisation** than same-batch replay, but it does not validate behaviour on real assets or missing-not-at-random records.
- A single mask/partition seed does not estimate variability across random seeds; additional repeated experiments are needed.
- The runner does not test the impact of imputed data on SAPN's downstream reliability/PoF model.
- Tree probabilities, disagreement scores and percentile bands are diagnostic outputs, **not calibrated probabilities or prediction intervals**.

**Recommended merge-request description:** "Adds a reproducible row-disjoint evaluation runner for Predictive MissForest, compares against a training-fitted Median/Mode baseline on identical masked held-out test cells, verifies saved-model replay and documents outputs and limitations. The separate E5 noise-robustness benchmark is not reproduced by this runner."
