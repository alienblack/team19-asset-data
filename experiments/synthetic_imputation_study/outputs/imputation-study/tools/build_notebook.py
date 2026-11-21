"""Builds the study notebook, ``notebooks/imputation.ipynb``, from the text and code written in this script.

Every cell is written here; ``base_notebook.ipynb`` supplies only the notebook metadata (kernel and language).
The notebook is written without outputs. Run from the project folder, then execute the notebook to save its outputs:

    python outputs/imputation-study/tools/build_notebook.py

Conventions of the notebook:

- Tables and figures are numbered by section ("Table 6.3a", "Figure 6.3a") with ``fg.table`` and ``fg.caption``.
  A table's caption is above it and a figure's caption below it; the text cites them by number.
- Formulas are written in LaTeX; code, column and file names in backticks.
- Every code cell starts with a comment that states what it shows.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # the project folder
SOURCE = Path(__file__).with_name("base_notebook.ipynb")
path = ROOT / "notebooks/imputation.ipynb"
old = json.loads(SOURCE.read_text(encoding="utf-8"))


def md(text):
    return ("markdown", text.strip("\n") + "\n")


def new(text):
    return ("code", text.strip("\n") + "\n")


def run_model(family, section):
    """The code cell of one model section."""
    return new(f"""
# Fit the model's tasks, or read them when saved, then show its results in the representative scenario.
STUDY.run_model("{family}")
STUDY.show_model("{family}", "{section}")
""")


cells = []
add = cells.append

# ---------------------------------------------------------------- title and setup
add(md("""
# Imputation of Missing Values in a Synthetic Asset Dataset

**Team 19** · University of Adelaide · in partnership with SA Power Networks<br>
Final version · November 2026

---

**Abstract.** Asset registers are rarely complete: location, installation date, equipment model and condition readings are often missing or recorded with error. This study evaluates how well a missing attribute can be recovered from the other attributes of the same record, and identifies the conditions under which recovery degrades, the form the degradation takes, and its causes. A synthetic register of 10,000 assets is used, in which measurement noise, missingness and the relationships between columns are set in advance, so every imputed value can be scored against its true value. Eleven imputation models are evaluated under one protocol in fifteen scenarios, and two main models are compared with the baseline throughout.

**Models compared**

| Role | Model | Type |
|:---|:---|:---|
| Baseline | Median and mode | One value per column |
| Main model | **Random Forest** | Tree-based, chained |
| Main model | **GMM** (Gaussian mixture model) | Probabilistic, joint |

**Protocol**

- **Scenarios:** 15 in total. Nine combine completion 0.9, 0.8 and 0.7 with noise 0, 5% and 10%; six add other missingness mechanisms, recording errors and heavier missingness.
- **Validation:** five folds; every hidden cell is predicted by a model that has not seen its record.
- **Information:** models receive measured data only; the truth is used for scoring only.
- **Measure:** skill, the share of the baseline's error removed on the same cells (section 3.4).

**Contents**

1. Setup
2. Data
3. Experimental design
4. Models, one section per model
5. Overview of results for all eleven models
6. Conditions of degradation
7. Characteristics of failure
8. Causes of failure
9. Value for downstream analysis
10. Limitations

Appendix: pairwise comparisons, stability, runtime, exported tables and references.

Chapters 6 to 9 compare Random Forest and GMM with the baseline.

**Conventions.** Tables and figures are numbered by section (Table 6.3a, Figure 6.3a); table captions stand above the table and figure captions below it. The design document, `docs/DESIGN.md`, describes the workflow, the code, the libraries, the design decisions and the tests.

**Scope.** The notebook documents the experiments and explains how to read each result; their interpretation is reported separately. The results describe this synthetic dataset and do not establish performance on real SA Power Networks assets.
"""))
add(md("""
## 1. Setup

**Environment.** Create it once, select the kernel **Python (imputation)** and use Restart and Run All.

```
conda create -n imputation python=3.12
conda activate imputation
pip install -r requirements.txt
python -m ipykernel install --user --name imputation --display-name "Python (imputation)"
```

**Run options.** The next cell loads the study code and sets the run options.

| Option | Effect |
|---|---|
| `RUN_ID` | Run folder in `outputs/imputation-study/runs`. Tasks already saved there are skipped and the rest are fitted. `None` starts a new folder and fits every model. |
| `EVALUATION_RUN_ID` | Evaluates a finished run without entering the task lists of the model sections. |
| `QUICK` | Runs every stage on a small sample with small models to check the code. Its numbers are not results. |

**Saved results.** Results are reused at two levels. The predictions of each task are saved when the task finishes, and the evaluation tables built from them are saved in the run folder. Saved predictions are reused only while the data, the settings and the prediction code are unchanged; saved tables are rebuilt whenever the evaluation code or the set of finished tasks changes. The notebook is set to the final run, `20261115-121714`; Run All reads its saved results, refits no model and completes in under a minute.
"""))
add(new('''
# Study code, run options and the study object that holds the run folder.
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from IPython.display import Markdown, display

ROOT = Path.cwd().resolve()
if not (ROOT / "src/synthetic_imputation").is_dir():
    ROOT = ROOT.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
# A kernel left open keeps earlier versions of these modules, so they are always reloaded.
for name in [m for m in sys.modules if m.startswith("synthetic_imputation")]:
    del sys.modules[name]

from synthetic_imputation import evaluation as ev
from synthetic_imputation import failure as fl
from synthetic_imputation import figures as fg
from synthetic_imputation import relationships as rel
from synthetic_imputation.columns import NUMERIC_TARGETS, TARGETS, column_table
from synthetic_imputation.data import (
    audit_design, audit_fields, audit_files, discover, read_truth, relationship_residuals,
)
from synthetic_imputation.mechanisms import DESCRIPTIONS
from synthetic_imputation.names import CATEGORY_METHODS, NUMERIC_METHODS
from synthetic_imputation.runner import (
    StudyConfig, default_workers, plan_table, quick_config,
)
from synthetic_imputation.study import Study
from synthetic_imputation.miceforest_compat import require_supported_version

QUICK = False                # True: small sample and small models, to check the code only
RUN_ID = "20261115-121714"   # saved tasks in this run are skipped; None starts a new run
EVALUATION_RUN_ID = None     # a finished run to evaluate without entering the model sections
WORKERS = default_workers()

CONFIG = quick_config() if QUICK else StudyConfig()
ALL_SCENARIOS = discover(ROOT / CONFIG.data_directory)
SCENARIOS = {name: ALL_SCENARIOS[name] for name in (CONFIG.scenarios or sorted(ALL_SCENARIOS))}
REPRESENTATIVE = CONFIG.representative
NOISE_FREE, HIGH_NOISE = CONFIG.noise_series[0], CONFIG.noise_series[-1]

require_supported_version()
fg.apply_style()
pd.set_option("display.max_columns", None, "display.max_colwidth", 140, "display.width", 200,
              "display.max_rows", 200)
print("Mode:", "QUICK CHECK, not findings" if QUICK else "full study")
print("Scenarios:", len(SCENARIOS), "| worker processes:", WORKERS)

STUDY = Study(ROOT, CONFIG, SCENARIOS, WORKERS, RUN_ID, EVALUATION_RUN_ID)
print("Run folder:", STUDY.directory)
print("Stage:", "evaluation of a finished run" if EVALUATION_RUN_ID
      else "saved tasks are skipped, missing tasks are fitted" if RUN_ID else "every model is fitted")
'''))

# ---------------------------------------------------------------- 2. data
add(md(r"""
## 2. Data

### 2.1 Generation

The dataset describes 10,000 assets on a 100 × 100 grid, generated row by row in grid order, so installation days increase along the grid. Each asset has an equipment model, environmental exposures, three gas readings, a leakage rate and a remaining-life value. Every scenario consists of two tables with the same rows and columns:

- **truth**: the values before measurement error and before any cell is hidden;
- **measured**: the table the models see, with measurement error and hidden cells.

Two controls are set in the file name. `completion`, $c$, is the probability that a cell remains visible: a cell is hidden when its mask draw $m \sim \mathcal{U}(0, 1)$ exceeds $c$, independently of every value, so values are missing completely at random (Rubin, 1976). `noise`, $\eta$, bounds a uniform multiplicative error on the measured columns,

$$\tilde{v} = v\,(1 + \eta\,u), \qquad u \sim \mathcal{U}(-1, 1),$$

so `noise0.05` allows a reading to deviate by up to ±5% of its true value $v$. The nine scenarios share the same assets, the same draws $u$ and $m$, and therefore nested masks, so a difference between two scenarios is attributable to the changed control and can be measured on identical cells.
"""))
add(md("""
### 2.2 Column definitions

Each column's meaning, type, valid range and error tolerance follow from what the column represents, not from patterns in the data (Table 2.2).
"""))
add(new("""
# Meaning, type, valid range and tolerance of every column, fixed before any data was read.
with pd.option_context("display.max_colwidth", None):
    fg.table("Table 2.2", "Column definitions: meaning, type, valid range and error tolerance.", column_table())
"""))
add(md("""
Tolerances differ by column:

- **Coordinates, days and altitude** have an arbitrary zero point, so absolute limits are used.
- **`life`** lies in a narrow band below its ceiling, where ±10% would accept almost any value, so absolute limits are used.
- **Humidity, salt, overgrowth and the placebo** have a true zero and a wide relative range, so percentage limits are used.
- **Gas columns and `leakage`** vary less and have tighter percentage limits; `total_gas` has the tightest.
"""))
add(md("""
### 2.3 File audit

The next three cells check each file pair against its definition. Table 2.3a and Figure 2.3 give the shape and the observed share of each column; `measurement MAE` is the error already present in observed cells. Table 2.3b gives the measurement error by column and noise level, and Table 2.3c tests it against the bound in the file name.
"""))
add(new("""
# Shape, observed share and measurement error of every file pair.
FILE_AUDIT = audit_files(ALL_SCENARIOS)
FIELD_AUDIT = audit_fields(ALL_SCENARIOS)
fg.table("Table 2.3a", "Shape, observed share and measurement error of each scenario's file pair.", FILE_AUDIT.round(4))
fg.plot_missingness(FIELD_AUDIT, None)
fg.caption("Figure 2.3", "Observed share of each column at the three completion levels of the generator's files.")
"""))
add(new("""
# Measurement error by column and noise level, and three checks against the file definitions.
noise_check = FIELD_AUDIT.groupby(["column", "noise"], sort=False).agg(
    changed=("changed share", "mean"),
    largest=("largest relative change", "max"),
    measurement_mae=("measurement MAE", "mean"),
).unstack("noise")
fg.table("Table 2.3b", "Share of observed cells changed by noise, largest relative change and measurement MAE, "
         "by column and noise level.", noise_check.round(4))

noise_free = FIELD_AUDIT[FIELD_AUDIT.noise.eq(0)]
inside_bound = (FIELD_AUDIT["largest relative change"] <= FIELD_AUDIT.noise + 1e-9).all()  # tolerance for rounding
never_noisy = FIELD_AUDIT[~FIELD_AUDIT["defined as noisy"]]["changed share"].eq(0).all()
checks = pd.DataFrame({
    "check": ["Noise-free files equal the truth in every observed cell",
              "Every measured value lies inside the bound in its file name",
              "Columns defined without noise are unchanged in every file"],
    "passed": [bool(noise_free["changed share"].eq(0).all()), bool(inside_bound), bool(never_noisy)],
})
fg.table("Table 2.3c", "Checks of the measured files against their definitions.", checks)
"""))
add(md("Table 2.3d confirms that scenarios can be compared cell by cell, and Table 2.3e that the truth satisfies the generator's definitions."))
add(new("""
# Properties that make scenarios comparable, and the generator's definitions checked on the truth.
with pd.option_context("display.max_colwidth", None):
    fg.table("Table 2.3d", "Properties that make the scenarios comparable cell by cell.",
             audit_design(ALL_SCENARIOS))
fg.table("Table 2.3e", f"Largest deviation of the truth from the generator's definitions, `{REPRESENTATIVE}`.",
         relationship_residuals(read_truth(ALL_SCENARIOS[REPRESENTATIVE])))
"""))
add(md("""
### 2.4 Generator revisions

The dataset was regenerated after review with the generator's author; the first version is kept in `data/datasets_v1`.

- **Masking** covers all 15 columns; the first version covered only the first eight.
- **Noise** applies only to humidity, salt, overgrowth, the placebo and the three gas readings. `life`, `total_gas` and `leakage` are exact by design, so some columns stay reliable when others degrade.
- **Gas and `life`:** more gas reduces `life` for every equipment model. The scale `total_gas/300000` is kept from the first version, so this term is small.
- **Model 0** uses `day_count` in its overgrowth term, as the other models do; the first version used an undefined variable.
- **Missingness** in the generator's files is completely at random only.
"""))
add(md("""
### 2.5 Additional scenarios

The generator hides every cell with the same probability and stops at 0.7 completion. Six scenarios are derived from its files to cover other conditions. Each differs from a generator scenario in one respect, uses the generator's truth file, and keeps every visible value unchanged unless the scenario concerns wrong values.

| Scenario | Source file | Change |
|---|---|---|
| age | 0.9, reduced to 0.8 completion | Older assets lose more cells |
| value | 0.9, reduced to 0.8 completion | Higher values are hidden more often |
| group | 0.9, reduced to 0.8 completion | Related columns are hidden together: location, gas readings, environmental readings, and date with model |
| errors | 0.8, same mask | 5% of visible installation days shift by 60 days; 2% of visible `model_id` values become another model |
| completion 0.5 and 0.3 | 0.7 | Further cells hidden at random, nested within the 0.7 mask |

Table 2.5a reports, for each scenario, the hidden share overall, by age third and in the lowest and highest fifth of `life`; the share of rows in which `x`, `y` and `altitude` are all hidden; and the number of visible values that differ from the representative scenario. Table 2.5b states how each scenario is constructed.
"""))
add(new("""
# Hidden share by age and by value of `life`, and changed visible values, in each added scenario.
truth = read_truth(ALL_SCENARIOS[REPRESENTATIVE])
age_third = pd.qcut(truth.install_date.rank(method="first"), 3, labels=["oldest third", "middle third", "newest third"])
life = truth.life
reference = STUDY.measured[REPRESENTATIVE]
rows = []
for name, frame in {REPRESENTATIVE: reference, **{n: STUDY.measured[n] for n in STUDY.mechanisms}}.items():
    hidden = frame.isna()
    by_age = hidden.mean(axis=1).groupby(age_third.reindex(frame.index), observed=True).mean()
    # A visible value counts as changed only where both files show the cell.
    changed = (frame != reference) & frame.notna() & reference.reindex(frame.index).notna()
    rows.append({
        "scenario": name, "mechanism": STUDY.scenarios[name].mechanism, "hidden share": hidden.to_numpy().mean(),
        **by_age.to_dict(),
        "life: lowest fifth": hidden.life[life.reindex(frame.index) <= life.quantile(0.2)].mean(),
        "life: highest fifth": hidden.life[life.reindex(frame.index) >= life.quantile(0.8)].mean(),
        "x, y and altitude all hidden": (hidden.x & hidden.y & hidden.altitude).mean(),
        "visible values changed": int(changed.to_numpy().sum()),
    })
fg.table("Table 2.5a", "Hidden share overall, by age third and by fifth of `life`, and visible values changed "
         "relative to the representative scenario.", pd.DataFrame(rows).round(3))
construction = pd.Series(DESCRIPTIONS, name="construction").rename_axis("scenario").to_frame()
with pd.option_context("display.max_colwidth", None):
    fg.table("Table 2.5b", "Construction of each added scenario.", construction)
"""))

# ---------------------------------------------------------------- 3. design
add(md(r"""
## 3. Experimental design

### 3.1 Targets and predictors

All 15 columns contain hidden cells, so all are imputation targets: 14 numeric columns and the categorical `model_id`. For each target, every other column is a predictor. `random_placebo` is unrelated to every other column and serves as a negative control: no model should predict it better than the median.

### 3.2 Preprocessing

All fitted quantities, including means, scales and category levels, come from the training fold.

| Step | Decision | Reason |
|---|---|---|
| Missing predictors | Left missing at the start | Each method handles gaps natively, so no fixed value is invented; chained models then replace gaps with their own estimates |
| Missing indicators | Not used | Masks in the generator's files are random, so an indicator carries no information about the value |
| `model_id` as a predictor | One-hot for distance and linear models; a category or number for trees | The column is nominal, so distances and coefficients must not treat its codes as an order |
| Row position | Never a feature | It would reveal the grid position and is not a property of an asset |
| Predictions | Limited to the valid range; whole-number columns rounded | This follows from the column definitions; the raw value is kept, and section 7.2 reports how often the limit applied |

### 3.3 Hyperparameter selection

Settings were fixed before the evaluation, from a fixed list of candidates, by a rule set before any candidate was scored. The same settings apply in every scenario, so differences between scenarios are not caused by different settings.

- **Data:** the three scenarios at 0.7 completion. Masks are nested, so these cells are visible in every scenario; no cell hidden in any scenario and no truth file is read.
- **Score:** a further 10% of observed cells is hidden, and the error on those cells is compared with the median's error, excluding `random_placebo`.
- **Choice:** candidates more than 1% worse than the median on `random_placebo` are dropped; of the rest, those within 0.002 of the best score are kept, and the fastest is chosen.

The candidates included each chained model without a ridge base and with a linear base, and every chained model was offered the same range of passes. The second-order base scored best for every model; section 8.5 measures its effect again against the truth. The rule and all scores are in `outputs/imputation-study/optimisation/clean_selection`.

Five models run from the project's shared source code in `src/synthetic_imputation/vendor`. The source code of GMM and the masked MLP records default settings and a search space but no selected settings, so both were chosen with the same rule from the defaults and seven candidates drawn from that space. Two further rounds, each declared before it was scored, extended the candidates beyond the edges of the space, after which selection stopped. LightGBM chained, Predictive MissForest and MICE-PMM have fully documented settings and are used as documented. The rule, candidates and scores are in `outputs/imputation-study/optimisation/merge_selection`.

### 3.4 Evaluation metrics

Five folds are shared by all methods within a seed. Each hidden cell is predicted by a model that has not seen its row and is scored against the truth. For a model $f$ and a set of hidden cells $H$ of one column, with true values $v_i$ and estimates $\hat{v}_i$:

$$\mathrm{MAE}_f = \frac{1}{|H|} \sum_{i \in H} \lvert \hat{v}_i - v_i \rvert, \qquad \mathrm{skill}_f = 1 - \frac{\mathrm{MAE}_f}{\mathrm{MAE}_{\text{baseline}}},$$

where both errors are computed on the same cells $H$. Skill is 0 for the baseline, 1 for exact recovery and below 0 for a model worse than the baseline.

| Measure | Use |
|---|---|
| MAE, RMSE, R², bias | Size and direction of error, in the column's unit |
| Skill | Share of the baseline's error removed on the same cells |
| Relative change in error | $\mathrm{MAE}_{\text{after}} / \mathrm{MAE}_{\text{before}} - 1$ on the same cells, for a change of condition |
| Error in SD | Mean signed or absolute error divided by the column's standard deviation in the truth, to compare columns |
| Tolerance rate | Share within the column's tight and loose limits (section 2.2) |
| Exact rate | Share of whole-number columns recovered exactly |
| Accuracy, macro F1, balanced accuracy | `model_id`; log loss, Brier score and calibration gap for its probabilities |

MAE is never averaged across columns, because their units differ. Skill is unit-free and is therefore used to compare columns and conditions; section 5.1 defines the failure criterion in terms of skill.

### 3.5 Experiment plan

Table 3.5 lists the planned tasks; the experiments are described below it.
"""))
add(new("""
# Planned tasks of every experiment.
fg.table("Table 3.5", f"Planned tasks by experiment ({len(STUDY.plan):,} in total).", plan_table(STUDY.plan))
"""))
add(md("""
- **main:** every model in each of the nine generator scenarios. A chained model is one task per fold, covering all its columns.
- **stability:** the representative scenario with two further fold seeds.
- **budget:** IterativeImputer at other pass limits.
- **structure:** the chained models refitted without the ridge base, in the representative scenario.
- **calibration:** the same forest with calibrated `model_id` probabilities, with all predictors and without the gas columns.
- **ablation:** refits without one block of predictors, at the three noise levels at 0.8 completion. Targets inside the removed block are skipped.
- **importance:** permutation importance on held-out observed labels, from single-target random forest and CatBoost models on all five folds.
- **mechanism:** every model in the six scenarios of section 2.5.
- **spatial:** every model in the representative scenario, with folds made of whole 10 × 10 blocks of the grid.
"""))

# ---------------------------------------------------------------- 4. models
add(md("""
## 4. Models

Each section describes a model and its settings, then runs it and shows its results in the representative scenario, `completion0.8-noise0.05`, unless stated otherwise. The baseline runs first, because skill is measured against it. Each task is saved as soon as it finishes, and model code never reads truth.

Every model section shows the same outputs, lettered in order: error and skill on each numeric column (Table 4.*n*a), results on `model_id` (Table 4.*n*b, where the model predicts it), skill in each generator scenario (Figure 4.*n*), accuracy on `model_id` in each scenario, and fitting time.

**Model groups.** The models fall into three groups by structure and by the source of their settings. The main models, Random Forest and GMM, come from the first two groups: a tree-based chained model and a joint probabilistic model.

| Group | Models | Settings |
|---|---|---|
| Chained, selected settings | KNN, IterativeImputer, Random Forest, HistGradientBoosting, CatBoost, miceforest direct | Selection rule (section 3.3) |
| Joint models | GMM, Masked MLP | Selection rule (section 3.3) |
| Chained, documented settings | LightGBM chained, Predictive MissForest, MICE-PMM | As documented with the source code |

**Chained imputation with a ridge base.** KNN, Random Forest, HistGradientBoosting and CatBoost share one structure:

1. The first pass predicts each column in turn from the others, starting from the original gaps; each prediction is filled in immediately, so the next column can use it.
2. Later passes repeat this on complete rows, using the latest estimates.
3. In later passes, a ridge regression (Hoerl & Kennard, 1970) on the other columns and their second-order terms (squares and pairwise products) fits the smooth part of each column.
4. The local model, trees or neighbours, fits what the ridge leaves; the prediction is the sum of the two.
5. Every estimate is limited to its column's observed training range. Without this limit the ridge base can extrapolate at low completion, feed on its own estimates and diverge.

The ridge represents sums, which trees and neighbours approximate only with many small steps; the local model represents thresholds and interactions. Two models serve as references for this structure: IterativeImputer (section 4.3) is the ridge alone, and miceforest direct (section 4.7) is a boosted chain without the ridge. Section 8.5 compares them.

The models of the last two groups run their source code unchanged through an adapter, `models_external.py`, that gives them the same folds, inputs and outputs as every other model. Parts of that code that encode the generator's relationships are switched off, as each section states.

Where a model predicts `model_id`, the class comes from a separate classifier trained on the original gaps, because a filled-in estimate would otherwise be treated as a measurement.
"""))
add(md("""
### 4.1 Baseline: median and mode

The baseline fills each hidden cell with one value from the training fold: the median for numeric columns and the most frequent class for `model_id`. It uses no other column and therefore learns no relationship.

**Reading the results.** A model must beat the baseline to show that it has learned a relationship. On `random_placebo`, the median is the correct answer.
"""))
add(run_model("Median and Mode", "4.1"))
add(md("""
### 4.2 K-nearest neighbours (KNN)

KNN averages the values of the rows most similar to the row being filled (Troyanskaya et al., 2001). It assumes no functional form but requires similar rows to exist.

**Settings.** Six passes with the ridge base and 80 neighbours weighted by distance. Columns are scaled so that no unit dominates the distance. The first pass compares only the cells two rows share; later passes compare complete rows. `model_id` enters as one-hot columns.

**Reading the results.** Every column counts equally in the distance, including unrelated ones such as `random_placebo`. Section 8.5 separates the contribution of the ridge base from that of the neighbours.
"""))
add(run_model("KNN", "4.2"))
add(md("""
### 4.3 IterativeImputer

IterativeImputer (Pedregosa et al., 2011) predicts each numeric column from the others and repeats. Here it uses the ridge alone, with no local model.

**Settings.** Ten passes on scaled columns. Linear terms have a penalty of 30 and second-order terms a penalty of 1,000, because the many second-order terms need stronger shrinkage; without a penalty, near-duplicate columns such as `altitude` and `x + y + 100` make the chain unstable. `model_id` enters as one-hot columns that remain empty when it is missing, so the chain estimates them as well. Estimates are limited to the observed training range.

**Reading the results.** The penalties trade exact recovery of sums for stability. Second-order terms allow the model to represent `life` and the overgrowth threshold. The model does not predict `model_id`, so Table 4.3b gives its fitting time.
"""))
add(run_model("IterativeImputer", "4.3"))
add(md("""
**Pass limit.** Table 4.3c compares MAE on the same cells at other pass limits; Table 4.3d gives the passes used in each scenario. A fit that stops at the limit has not converged. Ten passes gave the best score during selection.
"""))
add(new("""
# MAE on the same cells at other pass limits, and the passes each fit used before stopping.
passes, iterations = STUDY.tables["IterativeImputer"]["passes"]
fg.table("Table 4.3c", f"IterativeImputer: MAE at each pass limit on the same cells, `{REPRESENTATIVE}`.", passes.round(4))
fg.table("Table 4.3d", "IterativeImputer: passes used before stopping, by scenario.", iterations)
"""))
add(md("""
### 4.4 Random Forest

A random forest averages many decision trees, each grown on a sample of the rows (Breiman, 2001). Averaging reduces the variance of single trees, and the trees capture thresholds and interactions without being told about them. Run as a chain, the model follows the MissForest approach (Stekhoven & Bühlmann, 2012).

**Settings.** Four passes with the ridge base and 100 trees per column, each grown on half of the rows, with at least two rows per leaf. `model_id` enters as a number, which the trees can split into its four levels.

**Reading the results.** A forest alone cannot predict outside the range of its training labels; the ridge base can, which matters for sums such as `altitude`.
"""))
add(run_model("Random Forest", "4.4"))
add(md("""
**Calibrated probabilities for `model_id`.** A prediction made with probability $p$ should be correct with frequency $p$; the calibration gap measures the difference. Table 4.4e repeats the comparison without the gas columns, because with them the task is too easy to separate the methods. The calibrated model uses the same forest settings and trains on inner folds, so each fit sees fewer rows.
"""))
add(new("""
# Probability quality on `model_id`, with and without calibration, with all predictors and without the gas columns.
fg.table("Table 4.4e", "Random Forest: accuracy, log loss, Brier score and calibration gap on `model_id`, "
         "with and without calibrated probabilities.", STUDY.tables["Random Forest"]["probability"].round(4))
"""))
add(md("""
### 4.5 HistGradientBoosting

Gradient boosting adds trees sequentially, each fitted to the error left by the previous ones (Friedman, 2001). HistGradientBoosting bins each column first, which makes it fast at this data size.

**Settings.** Six passes with the ridge base and trees of up to 31 leaves. In the first pass each split learns a direction for missing values. Each regressor stops adding trees when the error on 15% held-out rows stops falling, so a column without signal receives few trees. `model_id` is a native categorical input.

**Reading the results.** Boosting can fit fine detail and therefore also noise; the result on `random_placebo` shows how far this goes.
"""))
add(run_model("HistGradientBoosting", "4.5"))
add(md("""
### 4.6 CatBoost

CatBoost (Prokhorenkova et al., 2018) is gradient boosting on symmetric trees, in which all nodes at the same depth share one split. This acts as built-in regularisation.

**Settings.** Six passes with the ridge base and trees of depth four. Each regressor stops when the error on 15% held-out rows has not improved for 30 rounds. Gaps in the first pass are handled natively; `model_id` enters as a number.

**Reading the results.** Comparison with section 4.5 shows the effect of stronger regularisation on noisy columns.
"""))
add(run_model("CatBoost", "4.6"))
add(md("""
### 4.7 miceforest direct

miceforest runs chained equations with LightGBM gradient boosting (Ke et al., 2017). This section uses its predictions directly, without donor matching and without the ridge base. An audited compatibility patch for miceforest 6.0.5 is applied.

**Settings.** Three passes of 200 boosting rounds, each tree using 70% of the columns. Two defaults are changed: the chain starts from the gaps instead of random values, which would feed the first models with noise, and it runs true gradient boosting instead of a depth-limited random forest that ignores the learning rate.

**Reading the results.** The model shares nothing with the ridge base, so it shows what a boosted chain reaches without it.
"""))
add(run_model("miceforest direct", "4.7"))
add(md(r"""
### 4.8 Gaussian mixture model (GMM)

The GMM treats each row as drawn from one of $K$ multivariate normal components, each with its own mean $\mu_k$ and all sharing one covariance $\Sigma$. It is fitted by expectation–maximisation on incomplete rows (Dempster et al., 1977; Ghahramani & Jordan, 1994), using only the cells each row shows. With $o$ the visible and $h$ the hidden cells of a row, the estimate is the conditional mean averaged over components by their posterior probabilities $\pi_k(x_o)$:

$$\hat{x}_h = \sum_{k=1}^{K} \pi_k(x_o)\left[\mu_{k,h} + \Sigma_{ho}\,\Sigma_{oo}^{-1}\,(x_o - \mu_{k,o})\right].$$

`model_id` enters as one-hot columns, and its class probabilities follow from the component weights.

**Settings.** Six components, eight EM iterations, a covariance penalty of 0.001 and a weight of 1.8 on the `model_id` columns (section 3.3). Numeric columns are centred on the training median and scaled by the training spread. The source code's gas-decay feature is switched off, because it is built from the generator's gas formula.

**Reading the results.** Within one component the estimate is linear in the visible cells, so thresholds and products are approximated by the mixture of components.
"""))
add(run_model("GMM", "4.8"))
add(md("""
### 4.9 Masked MLP

A neural network learns to reconstruct cells. In each training pass a share of the visible training cells is hidden, and the network learns to predict them from the rest, following the ReMasker design (Du et al., 2024). Columns enter standardised, with gaps set to the training median and a missing-value indicator beside each; `model_id` enters as one-hot columns. The network returns a value and a scale for each numeric column and class probabilities for `model_id`.

**Settings.** Hidden layers of 256 and 128 units, 60 training passes, a learning rate of 0.0015, 40% of visible cells hidden per pass and no dropout (section 3.3). The source code's exact-formula step and its blanking of the placebo column are switched off, because both use knowledge of the generator.

**Reading the results.** The selection score was still improving when selection stopped, so these settings are not necessarily the network's best.
"""))
add(run_model("Masked MLP", "4.9"))
add(md("""
### 4.10 LightGBM chained

The chain first fills gaps with the training median and mode. In each of three passes it fits one LightGBM model (Ke et al., 2017) per column on the rows where that column is observed and replaces the column's gaps with its predictions. The stored models are replayed on new rows in the same order.

**Settings.** 100 trees per model, a learning rate of 0.07 and 31 leaves, as documented. `model_id` is a categorical input.

**Reading the results.** Section 4.7 runs another LightGBM chain; this one starts from the median and uses fewer rounds. Neither has the ridge base.
"""))
add(run_model("LightGBM chained", "4.10"))
add(md("""
### 4.11 Predictive MissForest

A chain of random forests in the MissForest style (Stekhoven & Bühlmann, 2012). Gaps start from the training mean and mode. In each of five passes one forest per column is fitted on the rows where that column is observed, and its predictions replace the gaps. Every forest is stored, so the passes can be replayed on new rows without refitting.

**Settings.** 100 trees, at least 10 rows per leaf and the square root of the predictors at each split, as documented.

**Reading the results.** Section 4.4 is also a forest chain, with the ridge base and other forest settings; section 8.5 includes that chain without the ridge.
"""))
add(run_model("Predictive MissForest", "4.11"))
add(md("""
### 4.12 MICE with predictive mean matching (MICE-PMM)

Multivariate imputation by chained equations (van Buuren & Groothuis-Oudshoorn, 2011). Gaps start from values drawn at random from the same column, and in each sweep every column is fitted in turn on the current values of the others. Numeric columns use predictive mean matching (Little, 1988): a lightly penalised Bayesian linear model predicts each gap, and the gap takes the observed value of one of the five rows with the closest predictions. `model_id` uses a logistic regression fitted on a bootstrap sample of the observed rows, and its label is drawn from the predicted probabilities. Five independent chains run ten sweeps each.

**Settings.** Five chains, ten sweeps, five donors, a penalty of 0.0001 on the linear model, and logistic regression with C = 1 and up to 500 iterations, as documented. The source code completes a whole table, so its models would see the visible cells of the rows being scored; here every model, scaler and encoder is fitted on training rows only, and donors come from training rows only. The fitting and drawing steps are unchanged. The estimate is the mean of the five chains for numeric columns and the share of chains choosing each class for `model_id`.

**Reading the results.** Each chain copies an observed value, so a single chain keeps the spread of a column, while the mean of five chains narrows it. The model's own documentation reports it below the median on the placebo in every scenario.
"""))
add(run_model("MICE-PMM", "4.12"))

# ---------------------------------------------------------------- 5. overview
add(md("""
## 5. Overview of results

Chapters 5 to 9 and the appendix use saved predictions only and fit no model. The next cell combines the results of all models and builds the evaluation tables, or reads them from the run folder when they were built before from the same code and tasks. This chapter compares all models with the baseline; chapters 6 to 9 compare the two main models, Random Forest and GMM, with the baseline, and views of a single model are shown for each main model, labelled a for Random Forest and b for GMM.
"""))
add(new("""
# Combine every model's saved results; the evaluation tables are read from the run folder, or built once.
STUDY.finish()
NUMERIC, CATEGORY, PAIRS, FIGURES = STUDY.numeric, STUDY.category, STUDY.pairs, STUDY.figures
MAIN, SHOWN = list(STUDY.main), list(fl.SHOWN_METHODS)
COMPARED = ["Median"] + MAIN                                          # the baseline and the main models
FILE = {method: method.lower().replace(" ", "_") for method in MAIN}  # figure file names
LETTER = dict(zip(MAIN, "ab"))                                        # Random Forest: a, GMM: b
print("Main models compared with the baseline:", ", ".join(MAIN))
"""))
add(md("""
### 5.1 Failure criterion

A model is assessed by its skill (section 3.4): the share of the baseline's error it removes on the same cells.

| State | Skill | Meaning |
|---|---|---|
| works | 0.5 or more | Removes at least half of the baseline's error |
| weak | 0.1 to 0.5 | Removes less than half |
| fails | below 0.1 | Adds almost nothing to the baseline |

Tolerance limits (section 2.2) are not used as the criterion, because the share within a limit depends on a column's range, and on a narrow column the baseline can already meet the limit. Table 5.1 reports both measures for the main models and the baseline.
"""))
add(new("""
# Share within the loose tolerance and skill on each numeric column, for the baseline and the main models.
standard = STUDY.standard
table = standard.pivot(index="target", columns="method", values="within").reindex(list(NUMERIC_TARGETS))[COMPARED]
table.columns = [f"within loose tolerance: {c}" for c in table.columns]
for method in MAIN:
    table[f"skill: {method}"] = standard[standard.method.eq(method)].set_index("target").skill
fg.table("Table 5.1", f"Share of hidden cells within the loose tolerance, and skill, by numeric column, `{REPRESENTATIVE}`.",
         table.round(3))
"""))
add(md("""
### 5.2 Overall performance

Figure 5.2 shows the skill of every model on every column in the representative scenario; the shaded band marks the failing range and the vertical line the baseline. Table 5.2 adds the kind of relationship that defines each column and the baseline's error in the column's unit.
"""))
add(new("""
# Skill of every model on every column in the representative scenario.
fg.plot_overall(STUDY.overall, SHOWN, FIGURES)
fg.caption("Figure 5.2", f"Skill of every model on every column, `{REPRESENTATIVE}`. The shaded band is the failing range "
           "(skill below 0.1); the vertical line is the baseline.")
with pd.option_context("display.max_colwidth", 60):
    fg.table("Table 5.2", "Skill of every model on every column, with the kind of relationship and the baseline's MAE.",
             STUDY.overall.round(3))
"""))
add(md("""
### 5.3 Condition map: main models

Each column of the map changes one condition relative to the representative scenario. Figures 5.3a and 5.3b give the skill of each main model, with bold values below 0.5; Tables 5.3a and 5.3b classify every value by the criterion of section 5.1.

- **highest noise**, **lowest completion**, **both together:** other generator scenarios.
- **few linked columns visible:** cells of the representative scenario whose row shows at most a third of the directly linked columns.
- **older assets hidden more**, **higher values hidden more**, **related columns hidden together**, **wrong recorded values**, **completion 0.5**, **completion 0.3:** the scenarios of section 2.5.
- **held out by location:** the representative scenario with folds of whole blocks of the grid.
- **most harmful block removed:** the lowest skill after one block of predictors is removed and the model refitted.

Chapters 6 to 8 examine these conditions in turn.
"""))
add(new("""
# Skill of each main model under each condition, and its state under the failure criterion.
for method in MAIN:
    label = f"5.3{LETTER[method]}"
    fg.plot_failure_map(STUDY.failure_maps[method], f"{method}: skill under each condition", FIGURES,
                        f"failure_map_{FILE[method]}", fl.WEAK)
    fg.caption(f"Figure {label}", f"{method}: skill on each column under each condition; bold values are below 0.5.")
    fg.table(f"Table {label}", f"{method}: state of each value of Figure {label} (works, weak or fails).",
             STUDY.failure_maps[method].apply(lambda column: column.map(fl.state)))
"""))
add(md("""
### 5.4 Condition map: all models

Figure 5.4 repeats section 5.3 for every model. Each value is the mean skill over the columns with a relationship, so one cell summarises one model under one condition; bold values are below 0.5. Table 5.4 adds each model's group. The removed-block column is empty for miceforest direct, which is not refitted without predictor blocks.
"""))
add(new("""
# Mean skill of every model under each condition, with the model's group.
view = STUDY.model_groups.drop(columns="group")
fg.plot_failure_map(view, "Mean skill over the columns with a relationship, by model and condition", FIGURES,
                    "models_by_condition", fl.WEAK)
fg.caption("Figure 5.4", "Mean skill over the columns with a relationship, by model and condition; bold values are below 0.5.")
fg.table("Table 5.4", "Values of Figure 5.4 with the group of each model.", STUDY.model_groups.round(3))
"""))

# ---------------------------------------------------------------- 6. conditions
add(md("""
## 6. Conditions of degradation

This chapter proceeds from conditions that affect the whole table to those that affect a single cell: the generator's two controls, heavier missingness, the missingness mechanism, wrong recorded values and spatial hold-out, then the visible columns of a row, the kind of relationship and the position of the true value.

### 6.1 Noise and completion

In Figure 6.1a each panel shows one column, with skill against the noise bound and one line style per completion level. Vertical axes follow each panel's data, so panels do not share a scale and the baseline at 0 often lies below the visible range. The nine scenarios share one truth and nested masks, so differences between points reflect the controls alone.

Figures 6.1b and 6.1c measure each control on identical cells. Values are the relative change in error from the lowest to the highest level (section 3.4), and bold values have a 95% row-bootstrap interval that excludes zero.
"""))
add(new("""
# Skill against the noise bound, then the effect of each control measured on identical cells.
main_scores = STUDY.by_scenario[STUDY.by_scenario.method.isin(MAIN)]
fg.plot_lines(main_scores, "noise", "Skill as noise rises, at each completion level", FIGURES, "skill_by_noise",
              "Noise bound in the file name", MAIN, style="completion", style_label="completion")
fg.caption("Figure 6.1a", "Skill of the main models against the noise bound, one panel per column; "
           "line style marks the completion level.")
completion_level, noise_level = STUDY.scenarios[REPRESENTATIVE].completion, STUDY.scenarios[REPRESENTATIVE].noise
for letter, (control, held, level, label) in zip("bc", (("noise", "held completion", completion_level, "noise"),
                                                        ("completeness", "held noise", noise_level, "completion"))):
    matrix, marks = ev.effect_matrix(STUDY.effects, control, held, level)
    keep = [i for i, method in enumerate(matrix.columns) if method in COMPARED]  # the baseline and the main models
    matrix, marks = matrix.iloc[:, keep], marks[:, keep]
    fg.plot_heatmap(matrix, f"Change in error from the lowest to the highest {label}, on the same cells", FIGURES,
                    f"{control}_effect", "Relative change in error", center=0.0, limit=1.0, marks=marks, fmt="{:+.0%}")
    fg.caption(f"Figure 6.1{letter}", f"Relative change in MAE from the lowest to the highest {label} level on identical "
               f"cells, with {held.removeprefix('held ')} held at {level}; bold values have a 95% interval that excludes zero.")
"""))
add(md("""
### 6.2 Heavier random missingness

The generator's files stop at 0.7 completion. Two added scenarios hide further cells at random, down to 0.5 and 0.3 completion at noise 0.05, nested within the 0.7 mask (section 2.5). Figure 6.2 follows skill from 0.9 to 0.3 completion for the main models; Table 6.2 gives their mean skill over the columns with a relationship.
"""))
add(new("""
# Skill as completion falls from 0.9 to 0.3 at noise 0.05.
completion = STUDY.completion
fg.plot_lines(completion[completion.method.isin(MAIN)], "completion", "Skill as completion falls, at noise 0.05", FIGURES,
              "skill_by_completion", "Completion (share of cells visible)", MAIN)
fg.caption("Figure 6.2", "Skill of the main models as completion falls from 0.9 to 0.3 at noise 0.05, one panel per column.")
fg.table("Table 6.2", "Mean skill over the columns with a relationship, by completion level.",
         completion[completion.target.isin(fl.SIGNAL_TARGETS)].groupby(["method", "completion"]).skill.mean()
         .unstack().reindex(MAIN).round(3))
"""))
add(md("""
### 6.3 Missingness mechanism

Four scenarios hide the same share of cells: at random, more often for older assets, more often for higher values, and in blocks of related columns. Figures 6.3a and 6.3b show the skill of each main model in each scenario; Table 6.3a gives the mean skill over the columns with a relationship.

Skill can conceal a systematic shift. When higher values are hidden more often, the visible values of a column are lower than the hidden ones, and an estimate anchored to the visible range is biased downwards. Table 6.3b therefore gives the mean signed error in units of the column's standard deviation, for the main models and the baseline. Table 6.3c gives mean skill by age third, to check whether assets with more gaps are also filled less accurately.
"""))
add(new("""
# Skill, signed error and skill by age under four missingness mechanisms with the same hidden share.
scores = STUDY.mechanism_scores
order = ["random", "age", "value", "group"]
for method in MAIN:
    skill = scores[scores.method.eq(method)].pivot(index="target", columns="mechanism", values="skill").reindex(list(TARGETS))
    fg.plot_bars(skill[order], f"{method}: skill by how cells were chosen for hiding", FIGURES,
                 f"skill_by_mechanism_{FILE[method]}", "Error removed vs baseline (%)")
    fg.caption(f"Figure 6.3{LETTER[method]}", f"{method}: skill on each column under each missingness mechanism.")
signal = scores[scores.target.isin(fl.SIGNAL_TARGETS) & scores.mechanism.isin(order)]
fg.table("Table 6.3a", "Mean skill over the columns with a relationship, by missingness mechanism.",
         signal.groupby(["method", "mechanism"]).skill.mean().unstack()[order].reindex(COMPARED).round(3))
bias = scores[scores.method.isin(COMPARED) & scores.target.ne("model_id") & scores.mechanism.isin(order)]
fg.table("Table 6.3b", "Mean signed error in units of the column's standard deviation, by mechanism.",
         bias.pivot(index="target", columns=["method", "mechanism"], values="bias in SD").reindex(list(NUMERIC_TARGETS)).round(3))
groups = STUDY.mechanism_groups
groups = groups[groups.method.isin(MAIN) & groups.mechanism.isin(order)]
fg.table("Table 6.3c", "Mean skill by age third of the asset, by mechanism.",
         groups.pivot(index=["method", "mechanism"], columns="age third", values="skill").round(3))
"""))
add(md("""
### 6.4 Wrong recorded values

The errors scenario keeps the representative mask and corrupts some visible values: 5% of visible installation days shift by 60 days, and 2% of visible `model_id` values become another model. Because the mask is unchanged, every hidden cell is compared with itself, once with correct and once with wrong visible values. Rows without a wrong value show the effect that reaches them through the fitted model alone.

Table 6.4 gives the mean skill over the columns with a relationship, with correct and with wrong visible values. Figure 6.4 gives the relative change in error by column in rows that hold a wrong value; a cell is empty where the error with correct values was zero, which applies to most `model_id` cells.
"""))
add(new("""
# The same hidden cells filled from correct and from partly wrong visible values.
wrong = STUDY.wrong_values
summary = wrong[wrong.target.isin(fl.SIGNAL_TARGETS)].groupby(["method", "row"])[["skill", "skill with wrong values"]].mean()
fg.table("Table 6.4", "Mean skill over the columns with a relationship, with correct and with wrong visible values, "
         "in rows with and without a wrong value.", summary.unstack("row").reindex(MAIN).round(3))
change = wrong[wrong.row.eq("row with a wrong value")].pivot(index="target", columns="method", values="relative change in error")
change = change.reindex(list(TARGETS))[[m for m in MAIN if m in change]]
if change.notna().any().any():  # a small sample can leave too few rows with a wrong value
    fg.plot_heatmap(change, "Change in error on the same hidden cells when the row holds a wrong value", FIGURES, "wrong_values",
                    "Relative change in error", center=0.0, limit=1.0, fmt="{:+.0%}")
    fg.caption("Figure 6.4", "Relative change in MAE on the same hidden cells when the row holds a wrong value.")
"""))
add(md("""
### 6.5 Spatial hold-out

Random folds leave training rows next to every held-out row. Here the representative scenario is split into folds of whole 10 × 10 blocks of the grid, so a held-out asset has no training asset nearby (Roberts et al., 2017), as when records are filled for an area without training records. Blocks are defined by grid position, which no model receives. Table 6.5 gives the mean skill over the columns with a relationship under both splits; Figure 6.5 gives the change in skill by column.
"""))
add(new("""
# Skill with random folds and with folds of whole grid blocks.
splits = STUDY.splits
fg.table("Table 6.5", "Mean skill over the columns with a relationship, with random folds and with folds of whole grid blocks.",
         splits[splits.target.isin(fl.SIGNAL_TARGETS)].groupby("method")[["random rows", "whole blocks", "change"]].mean()
         .reindex(MAIN).round(3))
change = splits.pivot(index="target", columns="method", values="change").reindex(list(TARGETS))
fg.plot_heatmap(change[[m for m in MAIN if m in change]], "Change in skill when folds are whole blocks of the grid", FIGURES,
                "fold_design", "Change in skill", center=0.0, limit=0.5, fmt="{:+.2f}")
fg.caption("Figure 6.5", "Change in skill on each column when folds are whole 10 × 10 blocks of the grid.")
"""))
add(md("""
### 6.6 Visible linked columns

A column's directly linked columns are those it is computed from and those computed from it, by the generator's definition. For each hidden cell, the number of linked columns visible in the same row is counted from the measured data. In Figure 6.6 each panel shows skill against the visible share, with one line style per noise level; groups with fewer than 30 cells are omitted. Table 6.6 gives the values of the main models in the representative scenario.
"""))
add(new("""
# Skill by the number of directly linked columns visible in the row of each hidden cell.
seen = ev.add_scenario_levels(STUDY.visibility, STUDY.scenarios)
fg.plot_lines(seen[seen.method.isin(MAIN)], "share seen", "Skill by the share of directly linked columns visible in the row",
              FIGURES, "skill_by_visible_links", "Linked columns visible", MAIN, style="noise", style_label="noise", share_seen=True)
fg.caption("Figure 6.6", "Skill of the main models by the share of directly linked columns visible in the row; "
           "line style marks the noise level.")
main_seen = seen[seen.method.isin(MAIN) & seen.scenario.eq(REPRESENTATIVE)].copy()
main_seen["visible"] = main_seen.seen.astype(int).astype(str) + " of " + main_seen.links.astype(int).astype(str)
main_seen["order"] = main_seen.target.map(list(TARGETS).index)  # keeps the column order of the definitions
fg.table("Table 6.6", f"Skill by the number of linked columns visible, `{REPRESENTATIVE}`.",
         main_seen.pivot_table(index=["order", "target", "visible"], columns="method", values="skill")
         .droplevel("order")[MAIN].round(3))
"""))
add(md("""
### 6.7 Kind of relationship

Columns are grouped by how the generator builds them. Tables 6.7a and 6.7b give the mean skill of each main model for each kind: in the representative scenario, in its best and worst generator scenario, and when few linked columns are visible.
"""))
add(new("""
# Mean skill by kind of relationship, for each main model.
with pd.option_context("display.max_colwidth", 70):
    for method in MAIN:
        fg.table(f"Table 6.7{LETTER[method]}", f"{method}: mean skill by kind of relationship in the representative, "
                 "best and worst scenario and with few linked columns visible.", STUDY.kinds[method].round(3))
"""))
add(md("""
### 6.8 Position of the true value

Cells are grouped by the position of their true value within the column: the lowest tenth, the next fifth, the middle 40%, and the same bands on the high side. The grouping is known only after the fact, so it explains errors and implies no filling rule.

Figure 6.8 shows the mean signed error in units of the column's standard deviation, from the lowest band on the left to the highest on the right. A line that falls from left to right means that low values are overestimated and high values underestimated; the dashed line is the baseline. Tables 6.8a and 6.8b give the size of the error of each main model.
"""))
add(new("""
# Signed and absolute error by band of the true value.
fg.plot_bands(STUDY.bands, COMPARED, list(fl.BANDS), FIGURES)
fg.caption("Figure 6.8", "Mean signed error in units of the column's standard deviation, by band of the true value; "
           "the dashed line is the baseline.")
for method in MAIN:
    fg.table(f"Table 6.8{LETTER[method]}", f"{method}: mean absolute error in units of the column's standard deviation, "
             "by band of the true value.",
             STUDY.bands[STUDY.bands.method.eq(method)].pivot(index="target", columns="band", values="error in SD")
             .reindex(index=list(NUMERIC_TARGETS), columns=list(fl.BANDS)).round(3))
"""))

# ---------------------------------------------------------------- 7. characteristics
add(md(r"""
## 7. Characteristics of failure

Chapter 6 located the conditions under which skill falls. This chapter describes the estimates under those conditions.

### 7.1 Shrinkage towards the centre

The slope $b$ of the regression of estimates on true values, $\hat{v} = a + b\,v$, is 1 when estimates follow the truth and 0 when they are constant. Figure 7.1 plots $b$ against the share of linked columns visible; a slope that falls with the share means that, with less information, a model returns values closer to the centre of the column.

Table 7.1 gives the spread ratio $\sigma(\hat{v}) / \sigma(v)$, the standard deviation of the imputed values relative to that of the truth, over all hidden cells of the representative scenario. Values below 1 mean that completed columns are narrower than the real ones.
"""))
add(new("""
# Slope of estimates on the truth by visible linked columns, and the spread of imputed values.
slopes = ev.add_scenario_levels(STUDY.shrinkage, STUDY.scenarios)
fg.plot_lines(slopes[slopes.method.isin(MAIN)], "share seen", "Slope of predictions against the truth", FIGURES, "slope_by_visible_links",
              "Linked columns visible", MAIN, style="noise", style_label="noise", share_seen=True, y="slope",
              ylabel="Slope (1 follows the truth, 0 is a constant)", targets=list(NUMERIC_TARGETS))
fg.caption("Figure 7.1", "Slope of estimates on true values by the share of linked columns visible; "
           "line style marks the noise level.")
spread = STUDY.spread
fg.table("Table 7.1", f"Spread ratio of imputed to true values over all hidden cells, `{REPRESENTATIVE}`.",
         spread[spread.scenario.eq(REPRESENTATIVE)].pivot(index="target", columns="method", values="spread ratio")
         .reindex(list(NUMERIC_TARGETS))[COMPARED].round(3))
"""))
add(md("""
### 7.2 Out-of-range predictions

Table 7.2 gives the share of raw predictions outside a column's valid range, before they are limited to it; scores use the limited values. Columns with a floor or a ceiling, such as `measured_overgrowth`, are where averaging models can leave the range.
"""))
add(new("""
# Share of raw estimates outside the valid range, before the range limit.
ranges = STUDY.ranges
fg.table("Table 7.2", f"Share of raw estimates outside the column's valid range, `{REPRESENTATIVE}`.",
         ranges[ranges.scenario.eq(REPRESENTATIVE)].pivot(index="target", columns="method", values="outside valid range")
         .reindex(list(NUMERIC_TARGETS))[MAIN].round(4))
"""))
add(md("""
### 7.3 Shape of the decline

Tables 7.3a and 7.3b describe the two declines of each main model.

- **Noise:** the skill lost from the lowest to the highest noise, and the share of that loss incurred at the first step. A share near 1 means that the first increment of noise causes most of the damage.
- **Hidden linked columns:** skill with every linked column visible and with the fewest, and the share of the loss incurred when the first column is hidden. A low share means that one hidden column is tolerated and several are not.

Shares are left empty where the loss is below 0.02.
"""))
add(new("""
# Skill lost with noise and with hidden linked columns, and the share of each loss at the first step.
for method in MAIN:
    fg.table(f"Table 7.3{LETTER[method]}", f"{method}: skill lost with noise and with hidden linked columns, "
             "and the share of each loss incurred at the first step.", STUDY.decline[method].round(3))
"""))
add(md("""
### 7.4 Relationships between columns

Accurate cells do not guarantee that relationships between columns are preserved. Table 7.4a compares correlations in the completed data with those in the truth across all 91 pairs of numeric columns, on the rows that hold an imputed cell of the pair; a gap of 0 means that the correlation is unchanged.

Table 7.4b checks three generator definitions on completed rows. The reference is the residual in rows with nothing imputed, which reflects measurement noise alone.
"""))
add(new("""
# Correlations and generator definitions in the completed tables, against the truth.
gaps = STUDY.correlations[STUDY.correlations.method.isin(COMPARED)]
fg.table("Table 7.4a", "Mean gap between correlations in the completed data and in the truth, "
         "over the 91 pairs of numeric columns, by scenario.",
         gaps.pivot(index="method", columns="scenario", values="mean gap").reindex(COMPARED).round(4))
consistency = STUDY.consistency[STUDY.consistency.method.isin(COMPARED)]
wide = consistency.pivot(index=["relation", "scenario"], columns="method", values="mean residual in completed rows")[COMPARED]
wide["fully observed rows"] = consistency.groupby(["relation", "scenario"])["mean residual in fully observed rows"].first()
fg.table("Table 7.4b", "Mean residual of three generator definitions in completed rows and in fully observed rows.",
         wide.round(4))
"""))
add(md("""
### 7.5 Confidence of `model_id` predictions

Models report a probability with each `model_id` prediction. Tables 7.5a and 7.5b compare confidence with accuracy, first by the number of linked columns visible and then by scenario. `sure and wrong` is the share of cells predicted wrongly with a probability of 0.9 or more. Confidence that falls together with accuracy means that errors can be identified without the truth. Numeric estimates are single values and carry no such signal.
"""))
add(new("""
# Mean confidence against accuracy on `model_id`.
by_seen = STUDY.confidence_by_seen
fg.table("Table 7.5a", "Confidence and accuracy on `model_id` by the number of linked columns visible.",
         by_seen[by_seen.method.isin(MAIN)].sort_values(["method", "seen"]).round(3))
by_scenario = STUDY.confidence_by_scenario
fg.table("Table 7.5b", "Confidence and accuracy on `model_id` by scenario.",
         by_scenario[by_scenario.method.isin(MAIN)].sort_values(["method", "scenario"]).round(3))
"""))
add(md("""
### 7.6 Shared errors between models

Table 7.6 gives, for every pair of models, the rank correlation between their cell errors, averaged over columns. A value near 1 means that two models are wrong on the same cells, which points to a limit in the data rather than in either model.
"""))
add(new("""
# Rank correlation of cell errors between every pair of models.
fg.table("Table 7.6", "Spearman rank correlation between the cell errors of each pair of models, averaged over columns.",
         STUDY.overlap.round(2))
"""))

# ---------------------------------------------------------------- 8. causes
add(md("""
## 8. Causes of failure

Each section states one candidate cause and tests it with a comparison.

### 8.1 Data limits and model limits

Part of the error cannot be removed by any model, because the data does not contain the answer. Two references separate this part.

Figures 8.1a and 8.1b divide each column's baseline error into three parts for each main model: the share removed, the share lost because linked columns were hidden, and the share that remains even with every linked column visible. Tables 8.1a and 8.1b give the values.

Tables 8.1c to 8.1e apply the generator's own definition to the visible inputs (`Formula`), on hidden cells whose defining inputs are all visible; values are MAE. Without noise the formula is exact, so any model error there is the model's own. With noise the formula inherits the noise of its inputs, and a model can do better by drawing on other columns. Table 8.1f shows how much of each column the other true columns could explain at all.
"""))
add(new("""
# Where the baseline's error goes, the generator's formula as a reference, and what the true columns could explain.
parts = ["removed by the model", "lost to hidden linked columns", "not reached even with every linked column"]
for method in MAIN:
    label = f"8.1{LETTER[method]}"
    information = STUDY.information[method].rename(columns={"skill over all hidden cells": "removed by the model"})
    fg.plot_stacked(information, parts, [fg.SERIES[0], fg.SERIES[3], fg.GRID], f"{method}: where the baseline's error goes",
                    FIGURES, f"error_split_{FILE[method]}", "Share of the baseline's error")
    fg.caption(f"Figure {label}", f"{method}: share of the baseline's error removed, lost to hidden linked columns, "
               "and not reached with every linked column visible.")
    fg.table(f"Table {label}", f"{method}: values of Figure {label}.", STUDY.information[method].round(3))
for letter, scenario in zip("cde", (NOISE_FREE, REPRESENTATIVE, HIGH_NOISE)):
    fg.table(f"Table 8.1{letter}", f"MAE of the baseline, the main models and the generator's formula on hidden cells "
             f"whose defining inputs are visible, `{scenario}`.",
             rel.relation_matrix(STUDY.relations, scenario)[COMPARED + [rel.FORMULA]].round(3))
fg.table("Table 8.1f", "Share of each column explained by the other true columns, against the skill reached.",
         STUDY.recoverable.round(3))
"""))
add(md("""
### 8.2 Reliance on linked columns

If hidden linked columns cause the decline in section 6.6, models should rely on those columns when they are visible. This is tested with single-target random forest and CatBoost models, because a chain covers every column at once and the influence of one column on one target cannot be read from it. Each predictor is shuffled in held-out rows (permutation importance; Breiman, 2001) and the added error is measured against observed labels, so truth is not used. A value of 1 means that shuffling cost as much as replacing the model with a constant.

Table 8.2a places every column a model relies on (importance above 0.02) within the generator's definition:

- **direct:** the target is computed from it, or it is computed from the target.
- **defines a column together with the target:** for example `x` and `y`, which together define `altitude`.
- **linked through another column:** for example `altitude` and `install_date`, both computed from `x` and `y`.
- **stands in for a near copy:** `x` and `install_date` are near copies, so one can replace the other.
- **no link:** the definition gives no connection.

Table 8.2b lists the strongest columns relied on outside the target's definition. Figure 8.2 shows all values for the single-target random forest, with direct links in bold.
"""))
add(new("""
# Columns the single-target models rely on, placed within the generator's definition.
with pd.option_context("display.max_colwidth", 60):
    fg.table("Table 8.2a", "Columns relied on (importance above 0.02), by their link to the target in the generator's definition.",
             STUDY.links.set_index(["method", "scenario"]).round(3).T)
    fg.table("Table 8.2b", f"Strongest columns relied on outside the target's definition, single-target {STUDY.explainer}.",
             STUDY.unexplained)
fg.plot_importance(ev.importance_matrix(STUDY.importance_means, NOISE_FREE, STUDY.explainer),
                   f"{STUDY.explainer}: importance of each column · {NOISE_FREE}", FIGURES, "importance")
fg.caption("Figure 8.2", f"Permutation importance of each predictor for each target, single-target {STUDY.explainer}, "
           f"`{NOISE_FREE}`; direct links are in bold.")
"""))
add(md("""
### 8.3 Propagated noise

Noise can reach an estimate in two ways: through the column's own visible values, from which the model learns, and through the linked columns it reads. Tables 8.3a and 8.3b give both, as the mean measurement error at the highest noise level in units of the column's standard deviation, beside the change in the model's error on the same cells.

`total_gas`, `life` and `model_id` are exact by design; if their error still rises with noise, the noise was carried in from linked columns. Figures 8.3a and 8.3b plot the change in error against the largest noise reaching a column by either path.
"""))
add(new("""
# Noise in a column and in its linked columns, against the error the model adds as noise rises.
for method in MAIN:
    label = f"8.3{LETTER[method]}"
    noise = STUDY.noise[method].copy()
    noise["largest noise reaching the column"] = noise[["noise in the column itself", "largest noise in a linked column"]].max(axis=1)
    fg.table(f"Table {label}", f"{method}: measurement noise in the column and in its linked columns at the highest noise "
             "level, and the change in error.", noise.round(3))
    fg.plot_scatter(noise, "largest noise reaching the column", "error change from noise", "target",
                    f"{method}: noise reaching a column against the error it adds", FIGURES, f"noise_carried_{FILE[method]}",
                    "Largest noise in the column or a linked column (SDs)", "Change in error, lowest to highest noise")
    fg.caption(f"Figure {label}", f"{method}: change in error from the lowest to the highest noise against the largest "
               "noise reaching the column.")
"""))
add(md("""
### 8.4 Removal of predictor blocks

Each block of columns is removed according to what it measures, and the chained models are refitted. Figures 8.4a and 8.4b give the relative change in error, so a positive value means that the block carried information for the target; cells are empty for targets inside the removed block. Table 8.4 gives the skill after each removal.

A large change identifies the block a target depends on. A small change after removing a directly linked block means that another column took its place, as a near copy can.
"""))
add(new("""
# Change in error when one block of predictors is removed and the model refitted.
removed = STUDY.ablation[STUDY.ablation.scenario.eq(REPRESENTATIVE)]
for method in MAIN:
    matrix = removed[removed.method.eq(method)].pivot(index="target", columns="variant", values="relative change").reindex(list(TARGETS))
    fg.plot_heatmap(matrix, f"{method}: change in error when a block is removed · {REPRESENTATIVE}", FIGURES,
                    f"ablation_{FILE[method]}", "Relative change in error", center=0.0, limit=1.0, fmt="{:+.0%}")
    fg.caption(f"Figure 8.4{LETTER[method]}", f"{method}: relative change in MAE when each block of predictors is removed, "
               f"`{REPRESENTATIVE}`.")
fg.table("Table 8.4", f"Skill of the main models after each block of predictors is removed, `{REPRESENTATIVE}`.",
         removed[removed.method.isin(MAIN)].pivot_table(index=["target", "variant"], columns="method", values="skill without").round(3))
"""))
add(md("""
### 8.5 Model structure

KNN, Random Forest, HistGradientBoosting and CatBoost share the ridge base, so part of their agreement may come from that structure. Figure 8.5 shows each model's skill without the ridge base (open marker) and with it (filled marker) in the representative scenario; Table 8.5a adds the two references, the ridge alone and the boosted chain without the ridge.

Table 8.5b gives the mean skill by kind of relationship for each structure. Where the ridge raises a local model, the local model approximates that relationship poorly on its own; where all structures agree, the limit lies in the data.
"""))
add(new("""
# Skill of the chained models with and without the ridge base.
structure = STUDY.structure.drop(index="random_placebo")
pairs = [(m, f"{m}: no ridge", f"{m}: with ridge", fg.METHOD_COLORS[m]) for m in SHOWN if f"{m}: no ridge" in structure]
fg.plot_pairs(structure, pairs, "Skill without and with the ridge base", FIGURES, "ridge_base_effect", "Error removed vs baseline (%)")
fg.caption("Figure 8.5", f"Skill of each chained model without (open marker) and with (filled marker) the ridge base, "
           f"`{REPRESENTATIVE}`.")
fg.table("Table 8.5a", "Skill on each column of every structure, with the ridge alone and the boosted chain as references.",
         STUDY.structure.round(3))
by_kind = STUDY.structure.assign(kind=[fl.KIND_OF[t] for t in STUDY.structure.index]).groupby("kind", sort=False).mean()
with pd.option_context("display.max_colwidth", 60):
    fg.table("Table 8.5b", "Mean skill of every structure by kind of relationship.", by_kind.round(3))
"""))
add(md("""
### 8.6 Validity checks

Three checks test whether the results could be artefacts of the evaluation.

- **Leakage.** Table 8.6a counts predictions made for a visible cell or by a model that saw the row, and hidden cells not predicted exactly once. Every count should be 0.
- **Spurious structure.** `random_placebo` is unrelated to every other column, so a model that beats the baseline on it would be fitting structure that does not exist. Tables 8.6b and 8.6c give its skill in every scenario and its importance as a predictor.
- **Fold split.** The representative scenario is repeated with two further fold seeds; Table 8.6d gives the mean skill for each seed.
"""))
add(new("""
# Leakage counts, the negative control and the fold-seed check.
fg.table("Table 8.6a", "Leakage checks: predictions for visible cells, predictions by a model that saw the row, "
         "and hidden cells not predicted exactly once.", STUDY.integrity)
placebo = STUDY.by_scenario[STUDY.by_scenario.target.eq("random_placebo")]
fg.table("Table 8.6b", "Skill on `random_placebo` in every scenario.",
         placebo.pivot(index="method", columns="scenario", values="skill").round(4))
fg.table("Table 8.6c", "Permutation importance of `random_placebo` as a predictor.", STUDY.placebo_importance.round(4))
fg.table("Table 8.6d", "Mean skill for each fold seed, representative scenario.", STUDY.seed_skill.round(4))
"""))

# ---------------------------------------------------------------- 9, 10
add(md("""
## 9. Value for downstream analysis

An analysis restricted to complete rows loses most of the table. Table 9.1 counts the rows without a missing cell in every scenario.

Table 9.2 compares eight summary values that an asset analysis might draw from the table, such as mean `life` by equipment model and the share of assets with low `life`. Each is computed from the truth and from several versions of the representative scenario:

- **rows with no missing cell:** the rows an analysis without imputation would use.
- **Median and Mode:** the baseline fill.
- **Random Forest and GMM:** the completed table of each main model, with out-of-fold values.
- **measured values with true values in the gaps:** a perfect imputation, whose error comes from measurement noise alone.

Values are relative errors against the truth. Table 9.3 gives the mean absolute relative error for the scenarios of section 2.5.
"""))
add(new("""
# Complete rows, and summary values of an asset analysis from each version of the table, against the truth.
fg.table("Table 9.1", "Rows without a missing cell in every scenario.",
         fl.complete_rows_table(STUDY.measured, STUDY.scenarios).round(3))
downstream = STUDY.downstream
order = ["rows with no missing cell", "Median and Mode"] + MAIN + ["measured values with true values in the gaps"]
with pd.option_context("display.max_colwidth", 60):
    fg.table("Table 9.2", f"Relative error of eight summary values against the truth, `{REPRESENTATIVE}`.",
             downstream.pivot(index="quantity", columns="source", values="relative error")[order].round(4))
fg.table("Table 9.3", "Mean absolute relative error of the summary values in each scenario.",
         STUDY.downstream_by_scenario.reindex(order).round(4))
"""))
add(md("""
## 10. Limitations

**Data**

- The data is synthetic, with one generated truth table, so variation between generated datasets is not measured.
- `x` and `install_date` are near copies, because assets were installed in grid order; high skill on these columns says little about model quality.
- `life`, `total_gas` and `leakage` are exact by design, so results for and from these columns are better than noisy measurements would allow.
- A noise level does not imply equal difficulty across columns; section 8.3 gives each column's noise in units of its spread.
- The scenarios of section 2.5 are simple forms of missingness and error, each run at noise 0.05 with one draw of masks or errors.

**Design**

- All development used this one dataset. Settings never saw a hidden cell, but no independent dataset was held back.
- Shared settings were chosen on three scenarios and may not be optimal elsewhere. The search covered a fixed list of candidates and does not establish any model's ceiling.
- The candidates were written with knowledge of how the data was generated. Models receive no relationships and a fixed rule made the choice, but second-order terms were offered because such relationships were expected.
- Bootstrap intervals are conditional on the fitted models, and neighbouring grid rows are related, so the intervals may be too narrow.
- Importance comes from single-target models, not from the scored chains; the predictor removal in section 8.4 tests the chains directly.

**Models**

- The five models in `vendor/` run through an adapter under the same folds, data and scoring as the others. Parts of their code that encode the generator are switched off (the gas-decay feature, the spatial installation-day feature, exact formulas and placebo blanking), so their results differ from those reported with their source code.
- MICE-PMM was written to complete a whole table; here it is fitted on training rows only, so its results differ from whole-table results.
- The settings of GMM and the masked MLP come from the selection rule applied to their documented search space; the masked MLP was still improving when selection stopped.
- The range limit of the chained models was added after they diverged at 0.3 completion, where raw estimates reached three column spans outside the range of the truth and one task failed on infinite values. Their settings were chosen before the limit existed, on scenarios where it rarely applies.
- KNN, Random Forest, HistGradientBoosting and CatBoost share the ridge base, so their agreement may partly reflect that structure (section 8.5).
- A chained model trains on estimates that used the row's own label, which is unavailable for a hidden cell. Removing this effect improved the selection score by 0.25%, so the simpler form was kept.
- Numeric estimates are single values; methods that return a range of plausible values are not tested.
- IterativeImputer can stop at its pass limit before converging; section 4.3 reports how often and the effect of other limits.
"""))

# ---------------------------------------------------------------- appendix
add(md("""
## Appendix

### A.1 Pairwise comparisons

Table A.1a gives the best model for each column; `lead` is `clear` when the gap to the next model is both statistically separated and at least 5%. Figure A.1 compares every model with the baseline on the same cells, with 95% row-bootstrap intervals (Efron & Tibshirani, 1993); the intervals resample rows for fitted models and do not include refitting or the randomness of the dataset. Table A.1b compares the best two models for each column.
"""))
add(new("""
# Best model per column, every model against the baseline, and the best two models per column.
fg.table("Table A.1a", "Best model for each numeric column, the clarity of its lead and the effect of each control.",
         STUDY.targets.round(3))
fg.plot_paired(PAIRS, REPRESENTATIVE, FIGURES)
fg.caption("Figure A.1", f"Difference in MAE between every model and the baseline on the same cells, with 95% "
           f"row-bootstrap intervals, `{REPRESENTATIVE}`.")
closest = PAIRS[PAIRS.scenario.eq(REPRESENTATIVE) & PAIRS.comparison.eq("best two")].copy()
closest["relative gap"] = closest["second error"] / closest["first error"].where(closest["first error"] > 0) - 1
closest = closest.set_index("target").reindex(list(TARGETS)).reset_index()
fg.table("Table A.1b", "Paired comparison of the best two models for each column, with 95% interval and relative gap.",
         closest[["target", "first", "second", "metric", "difference", "low", "high", "relative gap"]].round(5))
"""))
add(md("""
### A.2 Fold-seed stability

The representative scenario is repeated with two further fold seeds. The same rows are reused, so this measures sensitivity to the split rather than adding samples. Table A.2a gives the worst rank of each model across seeds and Table A.2b the results on `model_id`.
"""))
add(new("""
# Rank and error of every model under three fold seeds.
seed_numeric, seed_labels = STUDY.stability
seed_numeric["relative range"] = seed_numeric["MAE high"] / seed_numeric["MAE low"] - 1
widest = float(seed_numeric["relative range"].max())
fg.table("Table A.2a", f"Worst rank of each model across the three fold seeds; the largest relative MAE range across "
         f"seeds is {widest:.2%}.",
         seed_numeric.pivot(index="target", columns="method", values="worst rank").reindex(list(NUMERIC_TARGETS))[list(NUMERIC_METHODS)])
fg.table("Table A.2b", "Results on `model_id` across the three fold seeds.", seed_labels.round(4))
"""))
add(md("""
### A.3 Runtime and warnings

Timings are single-threaded fits measured while other workers share the machine, so they are indicative only. A chained task covers many columns, so time is shown per target. Table A.3b lists each kind of warning once.
"""))
add(new("""
# Fitting time of every model and the warnings raised during fitting.
fg.table("Table A.3a", "Fitting and prediction time of every model; a chained task's time is shared between its targets.", STUDY.cost.round(3))
fg.plot_cost(STUDY.cost, NUMERIC, REPRESENTATIVE, FIGURES)
fg.caption("Figure A.3", f"Mean skill over the numeric columns against total fitting time in the main experiment, `{REPRESENTATIVE}`.")
with pd.option_context("display.max_colwidth", None):
    fg.table("Table A.3b", "Kinds of warning raised during fitting, each listed once.", ev.warning_table(STUDY.log))
"""))
add(md("""
### A.4 Exported tables

The last cell lists the completed table written for each model (Table A.4a) and summarises the run (Table A.4b). Every table of chapters 5 to 9 is also saved in the run folder as a CSV file.
"""))
add(new("""
# Completed tables written for each model, and a summary of the run.
fg.table("Table A.4a", "Completed tables written for each model.", STUDY.completed)
fg.table("Table A.4b", "Planned, finished, failed and warned tasks of the run.", ev.run_summary(STUDY.log, STUDY.plan))
print("Run folder:", STUDY.directory)
"""))
add(md("""
### A.5 References

- Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.
- Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977). Maximum likelihood from incomplete data via the EM algorithm. *Journal of the Royal Statistical Society: Series B*, 39(1), 1–38.
- Du, T., Melis, L., & Wang, T. (2024). ReMasker: Imputing tabular data with masked autoencoding. *International Conference on Learning Representations (ICLR)*.
- Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall.
- Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. *The Annals of Statistics*, 29(5), 1189–1232.
- Ghahramani, Z., & Jordan, M. I. (1994). Supervised learning from incomplete data via an EM approach. *Advances in Neural Information Processing Systems*, 6, 120–127.
- Hoerl, A. E., & Kennard, R. W. (1970). Ridge regression: Biased estimation for nonorthogonal problems. *Technometrics*, 12(1), 55–67.
- Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems*, 30.
- Little, R. J. A. (1988). Missing-data adjustments in large surveys. *Journal of Business & Economic Statistics*, 6(3), 287–296.
- Little, R. J. A., & Rubin, D. B. (2019). *Statistical Analysis with Missing Data* (3rd ed.). Wiley.
- Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
- Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., & Gulin, A. (2018). CatBoost: Unbiased boosting with categorical features. *Advances in Neural Information Processing Systems*, 31.
- Roberts, D. R., et al. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography*, 40(8), 913–929.
- Rubin, D. B. (1976). Inference and missing data. *Biometrika*, 63(3), 581–592.
- Stekhoven, D. J., & Bühlmann, P. (2012). MissForest: Non-parametric missing value imputation for mixed-type data. *Bioinformatics*, 28(1), 112–118.
- Troyanskaya, O., Cantor, M., Sherlock, G., Brown, P., Hastie, T., Tibshirani, R., Botstein, D., & Altman, R. B. (2001). Missing value estimation methods for DNA microarrays. *Bioinformatics*, 17(6), 520–525.
- van Buuren, S., & Groothuis-Oudshoorn, K. (2011). mice: Multivariate imputation by chained equations in R. *Journal of Statistical Software*, 45(3), 1–67.
"""))

# ---------------------------------------------------------------- write
built = []
for number, (kind, text) in enumerate(cells):
    cell = {"cell_type": kind, "id": f"cell-{number:03d}", "metadata": {}, "source": text.splitlines(keepends=True)}
    if kind == "code":
        cell.update(outputs=[], execution_count=None)
        assert text.startswith("# "), ("every code cell starts with a comment", text[:80])
    built.append(cell)
for cell in built:
    source = "".join(cell["source"])
    for banned in ("CELLS", "SCORED", "Subteam", "pipeline", "this study", "branch", "Gaussian EM"):
        assert banned not in source, (banned, source[:120])
old["cells"] = built
old["nbformat"], old["nbformat_minor"] = 4, max(old.get("nbformat_minor", 0), 5)
path.write_text(json.dumps(old, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
print("cells:", len(built), "| code:", sum(c["cell_type"] == "code" for c in built))
