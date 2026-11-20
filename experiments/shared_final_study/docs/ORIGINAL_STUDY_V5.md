> Historical version-5 study documentation. Use README.md and START_HERE.md for the updated execution and report workflow.

# Synthetic imputation study

Version 5, 2026-11-15.

This study fills the missing cells of a synthetic asset table with eleven imputation models and a baseline. It scores
every estimate against the known true value, and measures when, how and why imputation goes wrong. The notebook runs
every experiment and explains how to read each result. This file describes the data, the models, the experiments and how
to run them; it does not discuss results.

## Contents

| Path | Content |
|---|---|
| `README.md` | This file |
| `notebooks/imputation.ipynb` | The study notebook, saved with the outputs of run `20261115-121714` |
| `src/synthetic_imputation/` | The study code (package `synthetic_imputation`) |
| `tests/` | 62 tests |
| `data/datasets/` | The nine generator scenarios (`<scenario>-measured.csv`, `<scenario>-truth.csv`) and the generator notebook |
| `data/datasets_mechanisms/` | The six added scenarios. The code writes them again from the generator files on every run |
| `outputs/imputation-study/runs/` | Saved runs. Each run is a folder named by its start time |
| `pyproject.toml`, `requirements.txt` | Package definition and exact package versions |

The results of run `20261115-121714` are shared as separate packages: `02_result_tables_and_figures` (result tables,
filled tables, settings selection and figures; its README says what every file contains) and
`03_raw_predictions_only_for_rerunning` (every single prediction, needed only to repeat the evaluation).

## Data

One synthetic table of 10,000 assets and 15 columns, made by the generator notebook in `data/datasets/generator`.
Each scenario has two files with the same rows and columns: `measured` (with hidden cells and noise) and `truth`.

| Column | Meaning | Kind |
|---|---|---|
| `x`, `y` | Grid position of the asset (0–99) | Location |
| `install_date` | Installation day, counted from the start of the sequence | Location and date |
| `model_id` | Equipment model, 4 classes with no order | Category |
| `altitude` | Ground altitude (100–298) | Location |
| `measured_humid`, `measured_salt` | Humidity and salt exposure readings (0–1), noisy | Field reading |
| `measured_overgrowth` | Vegetation overgrowth reading (≥ 0, many zeros), noisy | Field reading |
| `life` | Remaining-life indicator (≤ 100,000) | Asset condition |
| `measured_gas_a`, `_b`, `_c` | Dissolved gas readings, noisy | Asset condition |
| `total_gas` | Total of the three true gas levels | Asset condition |
| `leakage` | Leakage rate (0.8–1.2) | Asset condition |
| `random_placebo` | Random number unrelated to every other column, noisy | Control |

`columns.py` records the meaning, valid range, noise and scoring tolerance of every column. The generator's formulas are
never given to a model; they are used only after fitting, to organise and explain the evaluation.

| Scenario | Cells kept | Noise | How cells are hidden or changed |
|---|---|---|---|
| `completion{0.9,0.8,0.7}-noise{0.0,0.05,0.1}` | 90%, 80%, 70% | 0, 5%, 10% | Completely at random (generator files) |
| `completion0.8-noise0.05-age` | 80% | 5% | Older assets lose more cells |
| `completion0.8-noise0.05-value` | 80% | 5% | Higher values are hidden more often |
| `completion0.8-noise0.05-group` | 80% | 5% | Related columns are hidden together as blocks |
| `completion0.8-noise0.05-errors` | 80% | 5% | Reference mask; 5% of visible installation days shifted by 60 days, 2% of visible `model_id` swapped |
| `completion{0.5,0.3}-noise0.05-sparse` | 50%, 30% | 5% | More cells hidden at random |

The reference scenario is `completion0.8-noise0.05`. Each added scenario keeps the same truth table and changes one
thing from it. `mechanisms.py` describes how each is built.

## Evaluation

- **Folds.** The rows are split into 5 folds with seed 219. Each model is fitted on the measured cells of 4 folds and
  fills every hidden cell of the 5th, so every scored cell is filled by a model that never saw its row. One experiment
  uses folds made of whole 10 × 10 blocks of the grid instead.
- **Truth.** Truth files are read only after a model's predictions are saved, and only for scoring.
- **Baseline.** Median for numeric columns and most common class for `model_id`, from the training rows.
- **Skill.** 1 − (model's mean absolute error ÷ baseline's mean absolute error) on the same cells. 0 means no better
  than the baseline and 1 means perfect. `model_id` is scored by accuracy, macro F1 and balanced accuracy.
- **Placebo.** `random_placebo` is filled and scored like every column; skill above 0 there would mean a model invents
  relationships.
- **Comparisons.** Bootstrap intervals (1,000 draws) for error differences on the same cells; a relative difference
  below 5% is reported as practically tied.

## Models

All models receive the same measured training rows and return a numeric estimate for every hidden cell and a class for
`model_id`.

| Model | Method | Main settings |
|---|---|---|
| Median and Mode | Baseline | – |
| KNN imputer | Chained: ridge base, then the average of the most similar rows | 6 passes, 80 neighbours, distance weights |
| IterativeImputer | Chained equations with one ridge on each column's inputs, squares and products | 10 passes, median start |
| Random Forest | Chained: ridge base, then a random forest on what is left | 4 passes, 100 trees, leaf ≥ 2, half the rows per tree |
| HistGradientBoosting | Chained: ridge base, then boosted trees | 6 passes, ≤ 400 rounds with early stopping, rate 0.05 |
| CatBoost | Chained: ridge base, then boosted symmetric trees | 6 passes, ≤ 800 rounds with early stopping, depth 4 |
| miceforest direct | Chained LightGBM in one miceforest kernel, no donor matching | 3 passes, 200 rounds, rate 0.05 |
| GMM (`Gaussian EM` in the code) | Gaussian mixture fitted by EM on incomplete rows | 6 groups, ≤ 8 EM steps, penalty 0.001 |
| Masked MLP | Neural network trained to rebuild observed cells hidden on purpose | Width 256, 60 epochs |
| LightGBM chained | Median start, then chained LightGBM | 3 passes, 100 trees, rate 0.07 |
| Predictive MissForest | Mean start, then chained random forests | 5 passes, 100 trees, leaf ≥ 10 |
| MICE-PMM | Chained equations with predictive mean matching | 5 chains, 10 sweeps, 5 donors |

- **Ridge base.** From the second pass on, a ridge regression on the other columns and their squares and products takes
  the smooth part of each column, and the local model learns the rest.
- **Range limit.** The chained models with a ridge keep every estimate inside the column's training range.
- **Shared code.** Models taken as written from the project's packages are in `src/synthetic_imputation/vendor`;
  `models_external.py` adapts them to the same inputs and outputs. Their features that encode how the data was generated
  (gas-decay formulas, exact formulas, placebo blanking) are switched off.

All settings are in `runner.py` (`default_parameters`). LightGBM chained, Predictive MissForest and MICE-PMM use the
settings their own documentation records. The other models' settings were chosen by one fixed rule, written before any
candidate was scored and applied without the truth:

1. Use the three scenarios at 70% completion and hide a further 10% of the visible cells in each test fold.
2. Score each candidate by its error relative to the median on those cells, averaged over the 13 numeric columns.
3. Drop candidates more than 1% worse than the median on `random_placebo`.
4. Keep the candidates within 0.002 of the best score, and choose the fastest.

The candidates and scores are in `outputs/imputation-study/optimisation`. In the shared packages they are in
`tables/7_settings_selection` of `02_result_tables_and_figures`.

## Experiments

| Experiment | What it does | Tasks |
|---|---|---:|
| `main` | Every model in the nine generator scenarios | 1,170 |
| `mechanism` | Every model in the six added scenarios | 780 |
| `ablation` | Models refitted without one block of columns | 750 |
| `importance` | Permutation importance from single-column Random Forest and CatBoost | 450 |
| `stability` | Reference scenario with fold seeds 220 and 221 | 260 |
| `spatial` | Reference scenario with folds of whole grid blocks | 130 |
| `structure` | The four chained models with a ridge base, without it | 20 |
| `budget` | IterativeImputer with 5 and 30 passes | 10 |
| `calibration` | Random Forest with calibrated `model_id` probabilities | 10 |

A task is one model fitted on one fold of one scenario in one experiment: 3,580 tasks in total.

## Notebook

| Chapter | Content |
|---|---|
| 1 | Setup and run mode |
| 2 | The dataset: columns, file checks, scenarios |
| 3 | Experimental design: folds, scoring, leakage checks |
| 4 | Each model: settings, scores and checks |
| 5 | Overview across models |
| 6 | Conditions under which results degrade |
| 7 | Form of the failure |
| 8 | Causes |
| 9 | What imputation gives a later analysis |
| 10 | Limits of the experiment |

## Environment

Python 3.12. Create the environment and the notebook kernel **Python (imputation)**:

```
conda create -n imputation python=3.12
conda activate imputation
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python -m ipykernel install --user --name imputation --display-name "Python (imputation)"
```

`lightgbm` must stay at 4.5.0 and `miceforest` at 6.0.5; `miceforest_compat.py` patches that version only.

## Running

Open `notebooks/imputation.ipynb` with the kernel **Python (imputation)** and use Restart and Run All. The setup cell
chooses the mode:

| Setting in the setup cell | Effect | Time |
|---|---|---|
| `EVALUATION_RUN_ID = "20261115-121714"` | Loads the saved predictions of that run and repeats the evaluation, tables and figures. Nothing is fitted | About 2 minutes |
| `EVALUATION_RUN_ID = None` | Fits every model again and writes a new run folder | About 2 hours 20 minutes with 12 workers |
| `QUICK = True` | Every stage on 600 rows with small models, to check that the notebook works. Not findings | A few minutes |
| `RESUME_RUN_ID = "<run folder>"` | Continues an interrupted run | |

The notebook is saved in the first mode. To use it, the folder `20261115-121714` from `03_raw_predictions_only_for_rerunning` must be at
`outputs/imputation-study/runs/20261115-121714`. Each run records a contract (`contract.json`) with the settings and the
hashes of the data and the prediction code. A saved run is loaded only while these match, so it stays valid while
`columns.py`, `data.py`, `models.py`, `runner.py`, `models_external.py`, `miceforest_compat.py`, `vendor/`, the data files
and the settings are unchanged. The other modules can change freely.

Tests: run `pytest` in this folder.

## Code

| File | Content |
|---|---|
| `columns.py` | Meaning, valid range and tolerance of each column, and the generator's definition as a graph |
| `data.py` | Reads and audits the measured and truth files |
| `mechanisms.py` | Builds the six added scenarios from the generator files |
| `models.py` | The baseline and the models implemented in this study |
| `models_external.py` | Adapters for the models in `vendor` |
| `vendor/` | Model code from the project's packages, used as written |
| `runner.py` | Settings, task plan, parallel execution, resuming and run contracts |
| `study.py` | Runs each model and builds the tables for the notebook |
| `evaluation.py` | Scores, paired intervals, stability and cost |
| `failure.py` | Tables on when, how and why imputation fails |
| `relationships.py` | Checks against the generator's formulas |
| `figures.py` | Charts |
| `miceforest_compat.py` | Fixes for miceforest 6.0.5 |

## Design limits

The data is synthetic, with one generated truth table. Numeric estimates are single values; no ranges are scored. Four
models share the ridge base. Models from the project's packages run under this study's protocol with their
generator-specific features off, so their results differ from their own reports. Chapter 10 of the notebook lists every
limit.
