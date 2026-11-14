# Imputation study

The code behind `notebooks/imputation.ipynb`. The notebook fills missing cells in the synthetic transformer dataset with several models and measures when, how and why they fail. It runs the experiments and explains how to read each result. Conclusions belong in the report, and this file gives only a short summary.

## Running

Open `notebooks/imputation.ipynb` with the kernel **Python (imputation)** and use Restart and Run All.

| Setting in the setup cell | Effect |
|---|---|
| `EVALUATION_RUN_ID = "<run folder>"` | Loads saved predictions and repeats the evaluation only, in about 2 minutes. The notebook is saved this way. |
| `EVALUATION_RUN_ID = None` | Fits every model: 2,230 tasks, about 1 hour 45 minutes with 12 workers. |
| `QUICK = True` | Runs every stage on 600 rows with small models, to check that the notebook works. |
| `RESUME_RUN_ID = "<run folder>"` | Continues an interrupted run. |

A saved run can be reused while `columns.py`, `data.py`, `models.py`, `runner.py`, the data files and the configuration are unchanged. The other modules can change freely.

Tests: `pytest tests/test_imputation_study.py` (52 tests).

## Files

| File | Content |
|---|---|
| `columns.py` | Meaning, valid range and tolerance of each column, and the generator's definition as a graph |
| `data.py` | Reads and audits the measured and truth files |
| `mechanisms.py` | Builds two scenarios in which cells are not hidden at random: by asset age and by value |
| `models.py` | The imputation models |
| `runner.py` | Settings, task plan, parallel execution and resuming |
| `study.py` | Runs each model and builds the tables for the notebook |
| `evaluation.py` | Scores, paired intervals, stability and cost |
| `failure.py` | Tables on when, how and why imputation fails |
| `relationships.py` | Checks against the generator's formulas |
| `figures.py` | Charts |

## Experiments

Median and mode are the baseline of every comparison. Skill is the share of the baseline's error that a model removes on the same cells.

| Experiment | Purpose |
|---|---|
| main | Every model in the nine generator scenarios (3 completion levels by 3 noise levels) |
| mechanism | Every model when older assets, or higher values, are hidden more often |
| structure | Chained models refitted with no ridge base |
| ablation | Refits without a block of predictors |
| importance | Permutation importance from single-target models, on five folds |
| stability, budget, calibration | Other fold seeds, IterativeImputer pass limits, calibrated `model_id` probabilities |

## Summary of run 20261114-114941

All 2,230 tasks succeeded. The numbers below are for CatBoost in the scenario with 0.8 completion and 0.05 noise, unless stated.

- **Level.** Mean skill is 0.81 for CatBoost and Random Forest and 0.79 for KNN. No model beats the baseline on the placebo column.
- **When.** Skill falls most when few of a column's linked columns are visible in the same row: by 0.24 on average, and from 0.82 to 0.35 for `y`. Lower completion costs 0.06, and the highest noise 0.01. Hiding more cells for older assets or for higher values changes mean skill by about 0.01.
- **How.** Estimates move towards the centre of a column: the lowest tenth of values is overestimated and the highest tenth underestimated in 12 of 13 columns. Correlations between columns are kept, with a mean gap of 0.010 against 0.214 for the baseline. When higher values are hidden more often, `leakage` is underestimated by 0.24 standard deviations.
- **Why.** The main cause is missing information in the row. Some error cannot be removed with the data available: 79% of the baseline's error on `leakage` remains even with every linked column visible. Noise is carried from linked columns into exact ones, such as `total_gas`. The shared ridge base adds most to KNN.
- **Use.** Only 348 of 10,000 rows have no missing cell. Eight summary values are off by 0.2% on average after imputation, against 3.3% from complete rows alone and 24% from complete rows when gaps depend on age.

## Limits

The data is synthetic, with one generated truth table. Four models share the ridge base. Numeric estimates are single values with no range. The notebook lists the limits in full in chapter 10.
