> [!IMPORTANT]
> **Final version of the synthetic imputation study (PGS219-285).** This branch holds the study code that goes with
> the imputation methods and results report. Its prediction code is identical to version 5, so it reads run
> `20261115-121714` without refitting; versions 1 to 5 are archived in the Git history of this folder (PGS219-274).
> The branch is not merged into `main`, which holds the team's final product. Data and saved runs are kept outside
> the repository, so paths to `data/` and `outputs/imputation-study/runs/` below do not resolve here; the
> notebook's saved outputs are the record of the run.

# Imputation of Missing Values in a Synthetic Asset Dataset

Team 19 · University of Adelaide · SA Power Networks · Final version, November 2026

Asset registers are rarely complete: attributes such as location, installation date, equipment model and condition
readings are often missing or recorded with error. This project evaluates how well a missing attribute can be recovered
from the other attributes of the same record, and identifies the conditions under which recovery degrades, the form the
degradation takes, and its causes.

The study uses a synthetic register of 10,000 assets in which measurement noise, missingness and the relationships
between attributes are set in advance, so every imputed value is scored against its true value. Eleven imputation models
are evaluated under one protocol in fifteen scenarios. Two main models are compared with the baseline throughout:

| Role | Model | Approach |
|---|---|---|
| Baseline | Median and mode | One value per column from the training data; uses no other attribute |
| Main model, tree-based | Random Forest | Chained imputation: each attribute is predicted from the others by a random forest on a ridge base, over several passes |
| Main model, not tree-based | Gaussian mixture model (GMM) | Joint model: every record is drawn from a mixture of multivariate normal distributions fitted by expectation–maximisation |

## Documentation

| Document | Content |
|---|---|
| `notebooks/imputation.ipynb` | The study: data, design, models, results and limitations, saved with the outputs of the final run |
| `docs/DESIGN.md` | Design document: workflow, code structure, data model, libraries, design decisions and test cases |
| `README.md` | This file: installation, running and project layout |

## Quick start

Python 3.12. Create the environment and the notebook kernel **Python (imputation)** once:

```
conda create -n imputation python=3.12
conda activate imputation
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python -m ipykernel install --user --name imputation --display-name "Python (imputation)"
```

Open `notebooks/imputation.ipynb`, select the kernel **Python (imputation)** and use **Restart and Run All**. The
notebook is set to the final run, `20261115-121714`: every model section finds its predictions saved, the evaluation
tables are read from the run folder, and the whole notebook completes in about 30 seconds.

Run the tests from this folder:

```
pytest
```

## Running modes

The setup cell of the notebook selects the mode.

| Setting | Effect | Time |
|---|---|---|
| `RUN_ID = "20261115-121714"` (saved) | Every task is already saved, so no model is fitted; saved evaluation tables are reused | About 30 seconds |
| `RUN_ID = "<run folder>"` | Continues an interrupted run: saved tasks are skipped and the rest are fitted | Depends on the remaining tasks |
| `RUN_ID = None` | Starts a new run folder and fits every model | About 2 hours 20 minutes with 12 workers |
| `EVALUATION_RUN_ID = "<run folder>"` | Evaluates a finished run without entering the task lists of the model sections | About 30 seconds |
| `QUICK = True` | Every stage on 600 rows with small models, to check the code; its numbers are not results | A few minutes |

Saved predictions are reused only while the run contract (`contract.json`) matches: the same settings, data files and
prediction code (`columns.py`, `data.py`, `models.py`, `models_external.py`, `runner.py`, `miceforest_compat.py` and
`vendor/`). Saved evaluation tables are rebuilt automatically, in about 3 minutes, whenever the evaluation code or the
set of finished tasks changes. `docs/DESIGN.md` section 5 describes both rules.

## Project layout

| Path | Content |
|---|---|
| `notebooks/imputation.ipynb` | The study notebook |
| `src/synthetic_imputation/` | The study package (`docs/DESIGN.md` section 4 describes every module) |
| `tests/` | Automated tests |
| `data/datasets/` | The nine generator scenarios (`<scenario>-measured.csv`, `<scenario>-truth.csv`) and the generator notebook |
| `data/datasets_mechanisms/` | The six added scenarios, rewritten from the generator's files on every run |
| `outputs/imputation-study/runs/20261115-121714/` | The final run: predictions, run contract, evaluation tables (CSV), figures and completed tables |
| `outputs/imputation-study/optimisation/` | Records of the settings selection: rules, candidates, scores and scripts |
| `outputs/imputation-study/tools/` | Scripts that build and run the notebook and fit the models outside it |
| `docs/` | The design document |

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
| `random_placebo` | Random number unrelated to every other column, noisy | Negative control |

### Generation

The generator notebook (seed 219) walks a 100 × 100 grid row by row and creates one asset per position.
`U(a, b)` is a uniform random draw; `day` is the installation day.

- **Installation day.** Starts at day 2 and moves to the next day after every 5 to 15 installations; on reaching a day
  divisible by 7 it skips two days. It therefore rises along the grid order, so `x` and `install_date` are near copies.
- **Equipment model.** Drawn from the models available on that day: 0 below day 200; 0–1 to 399; 0–2 to 699; 1–3 to
  999; 2–3 to 1,199; 3 from 1,200.
- **Formulas.**

  ```
  altitude   = x + y + 100
  humid      = U(0, 1) × altitude / 300
  salt       = U(0, 1) × (y + 200) / 300
  overgrowth = max(2 × humid − salt, 0)
  leakage    = U(0.8, 1.2)
  placebo    = U(0, 1)
  gas_a      = max(f_a × start_a − day × leakage, 0)
  gas_b      = max(f_b × start_b − 0.0005 × day² × leakage, 0)
  gas_c      = max(f_c × start_c − day × leakage − 0.00025 × day² × leakage, 0)      f = U(0.975, 1.025)
  total_gas  = gas_a + gas_b + gas_c
  life       = 100000 − hazards − 25000 × total_gas / 300000
  ```

  Starting gas levels (a, b, c): model 0 10,000 / 10,000 / 10,000; model 1 12,000 / 9,000 / 9,000; model 2
  9,000 / 12,000 / 9,000; model 3 9,000 / 9,000 / 12,000. Hazards: model 0 `day × (humid + salt + overgrowth)`;
  model 1 `day × (humid + overgrowth)`; model 2 `day × (salt + overgrowth)`; model 3 `day × overgrowth`.
- **Measured files.** The truth table is generated once and shared by all scenarios. Each cell gets one noise draw
  `u = U(−1, 1)` and one mask draw `U(0, 1)`, shared by all scenarios. The seven readings (humidity, salt, overgrowth,
  placebo, three gases) become `true × (1 + u × noise)`; all other columns stay exact. A cell is hidden when its mask
  draw is above the completion level, so masks are nested across completion levels and values are missing completely
  at random.

The generator's formulas are never given to a model; they are used only after fitting, to organise and explain the
evaluation.

### Scenarios

| Scenario | Cells kept | Noise | How cells are hidden or changed |
|---|---|---|---|
| `completion{0.9,0.8,0.7}-noise{0.0,0.05,0.1}` | 90%, 80%, 70% | 0, 5%, 10% | Completely at random (generator files) |
| `completion0.8-noise0.05-age` | 80% | 5% | Older assets lose more cells |
| `completion0.8-noise0.05-value` | 80% | 5% | Higher values are hidden more often |
| `completion0.8-noise0.05-group` | 80% | 5% | Related columns are hidden together as blocks |
| `completion0.8-noise0.05-errors` | 80% | 5% | Representative mask; 5% of visible installation days shifted by 60 days, 2% of visible `model_id` replaced |
| `completion{0.5,0.3}-noise0.05-sparse` | 50%, 30% | 5% | More cells hidden at random |

The representative scenario is `completion0.8-noise0.05`.

## Evaluation

- **Folds.** Five folds with seed 219, shared by all models. Each hidden cell is filled by a model fitted without its
  row. One experiment uses folds of whole 10 × 10 grid blocks instead.
- **Truth.** Truth files are read only after a model's predictions are saved, and only for scoring.
- **Skill.** 1 − (model MAE ÷ baseline MAE) on the same cells: 0 equals the baseline, 1 is perfect. Skill of 0.5 or more
  works, 0.1 to 0.5 is weak, below 0.1 fails. `model_id` is also scored by accuracy, macro F1 and balanced accuracy.
- **Negative control.** Skill above 0 on `random_placebo` would mean that a model invents relationships.
- **Uncertainty.** Paired row-bootstrap intervals (1,000 draws) on the same cells.

## Models

| Model | Group | Main settings |
|---|---|---|
| Median and mode | Baseline | – |
| KNN imputer | Chained, selected settings | 6 passes, ridge base, 80 neighbours, distance weights |
| IterativeImputer | Chained, selected settings | 10 passes, ridge on linear and second-order terms |
| **Random Forest** | Chained, selected settings | 4 passes, ridge base, 100 trees, leaf ≥ 2, half the rows per tree |
| HistGradientBoosting | Chained, selected settings | 6 passes, ridge base, ≤ 400 rounds with early stopping, rate 0.05 |
| CatBoost | Chained, selected settings | 6 passes, ridge base, ≤ 800 rounds with early stopping, depth 4 |
| miceforest direct | Chained, selected settings | 3 passes, 200 LightGBM rounds, rate 0.05 |
| **GMM** | Joint models | 6 components, ≤ 8 EM iterations, covariance penalty 0.001 |
| Masked MLP | Joint models | Layers of 256 and 128 units, 60 epochs, 40% of cells hidden per pass |
| LightGBM chained | Chained, documented settings | 3 passes, 100 trees, rate 0.07 |
| Predictive MissForest | Chained, documented settings | 5 passes, 100 trees, leaf ≥ 10 |
| MICE-PMM | Chained, documented settings | 5 chains, 10 sweeps, 5 donors |

Settings marked "selected" were chosen by one fixed rule, written before any candidate was scored and applied without
the truth (notebook section 3.3). The records are in `outputs/imputation-study/optimisation`. GMM, the masked MLP,
LightGBM chained, Predictive MissForest and MICE-PMM run their original source code (`src/synthetic_imputation/vendor`)
through an adapter, with features that encode the generator switched off.

## Experiments

| Experiment | What it does | Tasks |
|---|---|---:|
| `main` | Every model in the nine generator scenarios | 1,170 |
| `mechanism` | Every model in the six added scenarios | 780 |
| `ablation` | Models refitted without one block of columns | 750 |
| `importance` | Permutation importance from single-target Random Forest and CatBoost | 450 |
| `stability` | Representative scenario with fold seeds 220 and 221 | 260 |
| `spatial` | Representative scenario with folds of whole grid blocks | 130 |
| `structure` | The four chained models with a ridge base, without it | 20 |
| `budget` | IterativeImputer with 5 and 30 passes | 10 |
| `calibration` | Random Forest with calibrated `model_id` probabilities | 10 |
| **Total** | | **3,580** |

## Limitations

The data is synthetic, with one generated truth table, and the results do not establish performance on real SA Power
Networks assets. Numeric estimates are single values. Four chained models share the ridge base. Models run from their
original source code have their generator-encoding features switched off, so their results differ from those reported
with that code. Notebook chapter 10 lists every limitation.
