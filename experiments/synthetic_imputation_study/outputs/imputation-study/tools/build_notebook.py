"""Rebuilds notebooks/imputation.ipynb: chapters 1 to 4 are adapted, the evaluation chapters are new."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # the version folder
SOURCE = Path(__file__).with_name("base_notebook.ipynb")  # the version 3 notebook, whose chapters 1 to 4 are adapted
TARGET = ROOT / "notebooks/imputation.ipynb"
RUN_ID = "20261115-121714"
RUN_TIME = "2 hours 20 minutes"
old = json.loads(SOURCE.read_text(encoding="utf-8"))
kept = [("md" if c["cell_type"] == "markdown" else "code", "".join(c["source"])) for c in old["cells"][:35]]


def edit(index, before, after):
    kind, text = kept[index]
    assert text.count(before) == 1, (index, before[:60], text.count(before))
    kept[index] = (kind, text.replace(before, after))


# ---------------------------------------------------------------- chapters 1 to 4
kept[0] = ("md", """# Imputation study on the expanded synthetic dataset

**PGS219 · measured/truth file pairs · out-of-fold evaluation**

This notebook fills missing cells in a synthetic asset dataset with several imputation models and measures where they stop working. It runs the experiments and explains how to read each result. It does not state conclusions: those belong in the report. The dataset is synthetic so that noise, missingness and the relationships between columns are set in advance. That control lets the experiments address three questions:

1. The conditions under which imputation degrades or fails.
2. The form the failure takes.
3. The causes of the failure.

The project has three branches. This notebook is the main line, and it also runs the main models of the other two, Subteam A's benchmark and the team pipeline with its MICE notebook, under the same folds, data and scoring, together with experiments adopted from their designs.

Models receive measured data only, with no description of the relationships. Truth is read only after fitting, for scoring. Every comparison keeps a baseline that uses no other column: the median for numeric columns and the mode for `model_id`. The results describe this synthetic dataset and do not establish performance on real SA Power Networks assets.

**How the notebook is organised.** It goes from the whole to the detail.

| Chapters | Content |
|---|---|
| 2 to 4 | The data, the experimental design and the models of all three branches |
| 5 | Overview: the standard for failure, the overall level and a map of every condition |
| 6 to 8 | One chapter for each question: conditions, form and causes |
| 9 | What imputation gives a later analysis |
| 10 | Limits of the experiment |
| Appendix | Full comparisons of all models, stability and cost |

**To run:** create the environment once, select the kernel **Python (imputation)**, then use Restart and Run All.

```
conda create -n imputation python=3.12
conda activate imputation
pip install -r requirements.txt
python -m ipykernel install --user --name imputation --display-name "Python (imputation)"
```
""")
edit(1, "then go straight to section 5.", "then go straight to chapter 5.")
edit(19, "**Reading the results:** most of the gain over plain KNN comes from the ridge, so neighbours act here as a local correction.",
     "**Reading the results:** section 7.6 shows how much of the result comes from the ridge base and how much from the neighbours.")
edit(21, "**Reading the results:** the penalties trade exact recovery of sums for stability, so sums are recovered closely but not exactly. Second-order terms let this model handle `life` and the overgrowth threshold.",
     "**Reading the results:** the penalties trade exact recovery of sums for stability, so exact sums are the place to look for their cost. Second-order terms are what allow this model to represent `life` and the overgrowth threshold.")
edit(23, "More passes help some columns and hurt others, and ten passes gave the best overall score when the settings were chosen.",
     "Ten passes gave the best overall score when the settings were chosen.")
kept[1] = (kept[1][0], kept[1][1].rstrip("\n") + "\n\nThis copy is saved with `EVALUATION_RUN_ID` set to the latest full run, so Run All repeats the evaluation only. "
           f"Set it to `None` to fit every model again, which takes about {RUN_TIME} on the machine used here.\n")
edit(2, "EVALUATION_RUN_ID = None", f'EVALUATION_RUN_ID = "{RUN_ID}"')
edit(2, "from team19_asset_data.imputation_study import evaluation as ev\n",
     "from team19_asset_data.imputation_study import evaluation as ev\nfrom team19_asset_data.imputation_study import failure as fl\n")
edit(2, "from team19_asset_data.imputation_study.models import CATEGORY_METHODS, NUMERIC_METHODS\n",
     "from team19_asset_data.imputation_study.mechanisms import DESCRIPTIONS\n"
     "from team19_asset_data.imputation_study.models import CATEGORY_METHODS, NUMERIC_METHODS\n")
edit(13, "The raw value is kept, and section 5 reports how often this applied.", "The raw value is kept, and section 7.2 reports how often this applied.")
edit(13, "| Missing indicators | Not used | Masks are random, so an indicator carries no information about the value. |",
     "| Missing indicators | Not used | In the generator's files masks are random, so an indicator carries no information about the value. |")
edit(13, "- **Donor-matched models:** candidates within 0.01 of the best band coverage are kept, and the one with the lowest point error is chosen.\n", "")
edit(13, "- **Point models:** candidates", "- **Choice:** candidates")
edit(13, "The second-order base scored best for every model. The rule and all scores are kept in `outputs/imputation-study/optimisation/clean_selection`.",
     "The second-order base scored best for every model. The rule and all scores are kept in `outputs/imputation-study/optimisation/clean_selection`. Section 7.6 measures the effect of the base again, against the truth.\n\n"
     "**Models from the other branches.** Subteam A's package records the default settings of its models and their search space, but not the settings its tuning chose. Its two models were therefore chosen with the same rule, from their default and seven candidates drawn from their own search space. The rule kept choosing values at the edge of that space, so two further rounds added candidates beyond the edges. Each round was declared before it was scored, and selection stopped after the second. The team pipeline records every setting of its two models, and the team's MICE methodology document every setting of MICE-PMM, so these three are used as written. The rule, the candidates and all scores are kept in `outputs/imputation-study/optimisation/merge_selection`.")
edit(13, "| Skill | Share of the median's MAE removed: 0 matches the median, 1 is perfect, below 0 is worse |",
     "| Skill | Share of the baseline's error removed on the same cells: 0 matches the baseline, 1 is perfect, below 0 is worse. The baseline is the median, or the mode for `model_id` |")
edit(13, "MAE values are never averaged across columns because their units differ.",
     "MAE values are never averaged across columns because their units differ. Skill has no unit, so it is the measure used to compare columns and conditions. Section 5.1 sets the standard for failure in terms of skill.")
kept[15] = ("md", """- **main**: every method in each of the nine generator scenarios. A chained method is one task per fold, covering all its columns.
- **stability**: the representative scenario with two more fold seeds.
- **budget**: IterativeImputer at other pass limits, showing the effect of the number of passes.
- **structure**: the chained models refitted with no ridge base, in the representative scenario. This measures what the shared base adds.
- **calibration**: the same forest with calibrated probabilities for `model_id`, with all predictors and without the gas columns.
- **ablation**: refits without a block of predictors, across the three noise levels at 0.8 completion. Targets inside the removed block are skipped.
- **importance**: permutation importance on held-out observed labels, from single-target forest and CatBoost models on all five folds, because a chain covers every column at once.
- **mechanism**: every method in the six scenarios of section 2.5: cells hidden by age, by value or in related blocks, wrong visible values, and two heavier levels of random missingness.
- **spatial**: every method in the representative scenario with folds made of whole 10 × 10 blocks of the grid, so a held-out asset has no training asset beside it. Both other branches split their data this way.
""")
edit(16, "**How the chained models work.** Sections 4.2 to 4.7 and 4.9 use chained imputation, and most of them add a ridge base:",
     "**How the chained models work.** Sections 4.2 and 4.4 to 4.6 use chained imputation with a ridge base:")
edit(16, "4. The local model (trees or neighbours) fits only what the ridge leaves. The prediction is the sum of the two.",
     "4. The local model (trees or neighbours) fits only what the ridge leaves. The prediction is the sum of the two.\n"
     "5. Every estimate stays inside its column's observed training range. With few visible cells, the ridge base can otherwise extrapolate, feed on its own estimates from pass to pass and diverge.")
edit(16, "One fit covers every column, so settings are shared by all columns.",
     "One fit covers every column, so settings are shared by all columns.\n\nFour models therefore share most of their structure. Two sections act as references for it: section 4.3 uses the ridge alone, and section 4.7 uses a boosted chain with no ridge. Section 7.6 compares them.\n\n"
     "**Models from the other branches.** Sections 4.8 to 4.12 run the main models of the project's two other branches. Their code is used as written, from `src/team19_asset_data/imputation_study/vendor`, through an adapter that gives them the same folds, inputs and outputs as every other model here (`models_external.py`). Parts of their code that encode the generator's relationships are switched off, as each section states, so their results here can differ from those in their own reports.")
edit(33, "**Reading the results:** this section is the reference for sections 4.8 and 4.9, which share this engine and add donor matching.",
     "**Reading the results:** this model shares nothing with the ridge base of the other chained models, so it shows what a boosted chain reaches without it.")

cells = kept[:13]
cells += [
    ("md", """### 2.5 Scenarios added to the generator's files

The generator hides every cell with the same chance and stops at 0.7 completion. Six more scenarios are built from its files. Each differs from a generator scenario in one respect only, the truth file is the generator's, and every visible cell keeps the value the generator wrote unless the scenario is about wrong values. Four of them adopt designs from the other two branches.

| Scenario | Built from | Change | Adopted from |
|---|---|---|---|
| age | 0.9 file, to 0.8 completion | Older assets lose more cells | — |
| value | 0.9 file, to 0.8 completion | Higher values are hidden more often | — |
| group | 0.9 file, to 0.8 completion | Related columns are hidden together: location, gas readings, environment readings, and date with model | Subteam A's grouped gaps; the pipeline's hidden supporting predictors |
| errors | 0.8 file, same mask | 5% of visible installation days move by 60 days and 2% of visible `model_id` values become another model | Subteam A's structural errors; the pipeline's corrupted predictors |
| completion 0.5 and 0.3 | 0.7 file | Further cells hidden at random, nested | Subteam A's 50% and 70% missingness |

The table shows, for each scenario, the hidden share, the hidden share in each age third and in the lowest and highest fifth of `life`, the share of rows in which `x`, `y` and `altitude` are all hidden, and the number of visible values that differ from the representative scenario."""),
    ("code", """rows = []
age_third = pd.qcut(read_truth(ALL_SCENARIOS[REPRESENTATIVE]).install_date.rank(method="first"), 3,
                    labels=["oldest third", "middle third", "newest third"])
life = read_truth(ALL_SCENARIOS[REPRESENTATIVE]).life
reference = STUDY.measured[REPRESENTATIVE]
for name, frame in {REPRESENTATIVE: reference, **{n: STUDY.measured[n] for n in STUDY.mechanisms}}.items():
    hidden = frame.isna()
    by_age = hidden.mean(axis=1).groupby(age_third.reindex(frame.index), observed=True).mean()
    changed = (frame != reference) & frame.notna() & reference.reindex(frame.index).notna()
    rows.append({"scenario": name, "mechanism": STUDY.scenarios[name].mechanism, "hidden share": hidden.to_numpy().mean(),
                 **by_age.to_dict(), "life: lowest fifth": hidden.life[life.reindex(frame.index) <= life.quantile(0.2)].mean(),
                 "life: highest fifth": hidden.life[life.reindex(frame.index) >= life.quantile(0.8)].mean(),
                 "x, y and altitude all hidden": (hidden.x & hidden.y & hidden.altitude).mean(),
                 "visible values changed": int(changed.to_numpy().sum())})
display(pd.DataFrame(rows).round(3))
for name, text in DESCRIPTIONS.items():
    print(f"{name}: {text}")"""),
]
cells += kept[13:35]
cells += [
    ("md", """### 4.8 Gaussian EM (Subteam A)

The Gaussian mixture treats each row as drawn from one of several groups. Each group is a multivariate normal distribution with its own mean, and all groups share one covariance. The mixture is fitted by expectation–maximisation on incomplete rows, using only the cells each row shows. A hidden cell is estimated by its expected value given the visible cells of its row, weighted over the groups by how likely each group is for that row. `model_id` enters as one-hot columns, and its class probabilities come from the group weights.

**Setup:** six groups, eight EM steps, a covariance penalty of 0.001 and a weight of 1.8 on the `model_id` columns, chosen as section 3.3 describes. Numeric columns are centred on the training median and scaled by the training spread. Subteam A's gas-decay feature is off, because it is built from the generator's gas formula.

**Reading the results:** within one group the estimate is a straight-line function of the visible cells, so thresholds and products are approximated by the mix of groups."""),
    ("code", """STUDY.run_model("Gaussian EM")
STUDY.show_model("Gaussian EM")"""),
    ("md", """### 4.9 Masked MLP (Subteam A)

A neural network learns to rebuild cells. In each training pass, a share of the visible training cells is hidden and the network is trained to predict them from the rest. The design follows ReMasker. Each column enters standardised, with gaps set to the training median and a missing-value indicator beside it, and `model_id` enters as one-hot columns. The network returns a value and a scale for each numeric column and class probabilities for `model_id`.

**Setup:** hidden layers of 256 and 128 units, 60 training passes, learning rate 0.0015, 40% of visible cells hidden in each pass and no dropout, chosen as section 3.3 describes. Subteam A's exact-formula step and its blanking of the placebo column are off, because both use knowledge of the generator.

**Reading the results:** the selection score was still improving when selection stopped, so this setting is not the network's best."""),
    ("code", """STUDY.run_model("Masked MLP")
STUDY.show_model("Masked MLP")"""),
    ("md", """### 4.10 LightGBM chained (team pipeline)

The pipeline's chain starts by filling gaps with the training median and mode. In each of three passes it then fits one LightGBM model per column, on the rows where that column is observed, and replaces the column's gaps with its predictions. The stored models are replayed on new rows in the same order.

**Setup:** 100 trees per model, learning rate 0.07 and 31 leaves, as the pipeline's guide records. `model_id` is a categorical input.

**Reading the results:** section 4.7 runs another LightGBM chain. This one starts from the median instead of the gaps and uses fewer rounds, and neither has the ridge base of sections 4.4 to 4.6."""),
    ("code", """STUDY.run_model("LightGBM chained")
STUDY.show_model("LightGBM chained")"""),
    ("md", """### 4.11 Predictive MissForest (team pipeline)

A chain of random forests in the style of MissForest. Gaps start from the training mean and mode. In each of five passes one forest per column is fitted on the rows where that column is observed, and its predictions replace the gaps. Every forest is stored, so the passes can be replayed on new rows without refitting.

**Setup:** 100 trees, at least 10 rows per leaf and the square root of the predictors at each split, as the pipeline's guide records.

**Reading the results:** section 4.4 is also a forest chain, with the ridge base and other forest settings, and section 7.6 adds the same forest chain without the ridge."""),
    ("code", """STUDY.run_model("Predictive MissForest")
STUDY.show_model("Predictive MissForest")"""),
    ("md", """### 4.12 MICE-PMM (team pipeline)

Multivariate imputation by chained equations, from the team's MICE notebook and the pipeline's `mice_pmm_logistic.py`, which hold the same code. Gaps start from values drawn at random from the same column. In each sweep every column is fitted in turn on the current values of the others. Numeric columns use predictive mean matching: a lightly penalised Bayesian linear model predicts each gap, and the gap takes the observed value of one of the five rows with the closest predictions. `model_id` uses a logistic regression fitted on a bootstrap sample of the observed rows, and its label is drawn from the predicted class probabilities. Five independent chains run ten sweeps each.

**Setup:** five chains, ten sweeps, five donors, a penalty of 0.0001 on the linear model, and logistic regression with C = 1 and up to 500 iterations, as the methodology document records. The shared code completes a whole table, so its models see the visible cells of the rows being scored. Here every model, scaler and encoder is fitted on training rows only, and donors come from training rows only; the fitting and drawing steps are the shared code, unchanged. The estimate is the mean of the five chains for numeric columns and the share of chains choosing each class for `model_id`.

**Reading the results:** each chain copies a real observed value, so a single chain keeps the spread of a column, and the mean of five chains narrows it again. The methodology document reports this model below the median on the placebo in every scenario."""),
    ("code", """STUDY.run_model("MICE-PMM")
STUDY.show_model("MICE-PMM")"""),
]

# ---------------------------------------------------------------- chapter 5
cells += [
    ("md", """## 5. Overview

**The evaluation starts here.** Chapters 5 to 9 and the appendix need only the setup cell and a finished run. They use saved predictions and fit no model, so they can be repeated quickly after any change to the evaluation code.

The next cell collects all results. Every comparison table shows all models, and the baseline is part of every comparison. The line charts follow the strongest model of each branch, chosen by mean skill over the columns with a relationship in the representative scenario; the cell prints them. The lead model, which the failure map and chapter 8 follow, is the strongest model of this study."""),
    ("code", """STUDY.finish()
NUMERIC, CATEGORY, PAIRS, CELLS = STUDY.numeric, STUDY.category, STUDY.pairs, STUDY.cells
SCORED, FIGURES = STUDY.all_scored, STUDY.figures
MAIN, SHOWN, LEAD = list(STUDY.leads.values()), list(fl.SHOWN_METHODS), fl.LEAD_METHOD
display(ev.run_summary(STUDY.log, STUDY.plan))
for branch, method in STUDY.leads.items():
    print(f"Strongest model of {branch}: {method}")
print("Lead model:", LEAD)"""),
    ("md", """### 5.1 The standard for failure

A model is judged by its skill: the share of the baseline's error it removes on the same cells.

| State | Skill | Meaning |
|---|---|---|
| works | 0.5 or more | The model removes at least half of the baseline's error |
| weak | 0.1 to 0.5 | It removes less than half |
| fails | below 0.1 | It adds almost nothing to the baseline |

The tolerance limits of section 2.2 are not used as the standard, because the share of cells inside a limit depends on how wide a column's range is, and a baseline can already meet a limit on a narrow column. Skill includes the baseline by definition. The table shows both measures side by side, for the lead model and the baseline, so that the two can be compared."""),
    ("code", """standard = fl.skill_by(fl.main_cells(CELLS, STUDY.config, REPRESENTATIVE), ["target", "method"])
standard = standard[standard.method.isin(["Median", LEAD]) & standard.target.ne("model_id")]
table = standard.pivot(index="target", columns="method", values="within").reindex(list(NUMERIC_TARGETS))
table.columns = [f"share within loose tolerance: {c}" for c in table.columns]
table[f"skill: {LEAD}"] = standard[standard.method.eq(LEAD)].set_index("target").skill
display(table.round(3))"""),
    ("md", """### 5.2 Overall level

The chart shows skill in the representative scenario for every model of the three branches. The shaded band is the failing range, and the vertical line is the baseline.

The table adds the kind of relationship that defines each column and the baseline's own error, in the column's unit."""),
    ("code", """fg.plot_overall(STUDY.overall, SHOWN, FIGURES)
with pd.option_context("display.max_colwidth", 60):
    display(STUDY.overall.round(3))"""),
    ("md", """### 5.3 Map of every condition

Each column of the map changes one thing against the reference scenario. Values are the skill of the lead model, and bold values are below 0.5. The second table gives the state of each cell under the standard of section 5.1.

- **highest noise**, **lowest completion**, **both together**: other generator scenarios.
- **few linked columns visible**: cells of the reference scenario in which at most a third of the directly linked columns were visible in the row.
- **older assets hidden more**, **higher values hidden more**, **related columns hidden together**, **wrong recorded values**, **completion 0.5**, **completion 0.3**: the scenarios of section 2.5.
- **held out by location**: the representative scenario with folds of whole blocks of the grid.
- **most harmful block removed**: the lowest skill after one block of predictors is removed and the model is refitted.

Chapters 6 to 8 take these conditions one at a time."""),
    ("code", """fg.plot_failure_map(STUDY.failure_map, f"{LEAD}: skill under each condition", FIGURES, "failure_map", fl.WEAK)
display(STUDY.failure_map.apply(lambda column: column.map(fl.state)))"""),
    ("md", """### 5.4 Every model under every condition

The map repeats the conditions of section 5.3 for every model. Values are mean skill over the columns with a relationship, so one cell summarises one model under one condition. Bold values are below 0.5. The table adds the branch of each model. The removed-block column is empty for miceforest direct, which is not refitted without predictor blocks."""),
    ("code", """view = STUDY.branches.drop(columns="branch")
fg.plot_failure_map(view, "Mean skill over the columns with a relationship, by model and condition", FIGURES, "models_by_condition", fl.WEAK)
display(STUDY.branches.round(3))"""),
]

# ---------------------------------------------------------------- chapter 6
cells += [
    ("md", """## 6. Conditions under which results degrade

This chapter moves from the whole dataset to the single cell: the two controls of the generator, then the row a cell sits in, then the value of the cell itself, then the way cells were chosen for hiding.

### 6.1 Noise and completion

Each panel shows one column. Lines follow skill as the noise bound rises, with one line style for each completion level. Each vertical axis follows its own data so that the trend can be read, so the panels do not share a scale, and the baseline at 0 lies below most of them. All nine scenarios share one truth and nested masks, so a difference between two points comes from the controls alone.

The heatmaps then measure each control on identical cells. Values are the relative change in error from the lowest to the highest level, and bold values have a 95% row-bootstrap interval that excludes zero."""),
    ("code", """shown = STUDY.by_scenario[STUDY.by_scenario.method.isin(MAIN)]
fg.plot_lines(shown, "noise", "Skill as noise rises, at each completion level", FIGURES, "skill_by_noise",
              "Noise bound in the file name", MAIN, style="completion", style_label="completion")
completion_level, noise_level = STUDY.scenarios[REPRESENTATIVE].completion, STUDY.scenarios[REPRESENTATIVE].noise
for control, held, level, label in (("noise", "held completion", completion_level, "noise"),
                                    ("completeness", "held noise", noise_level, "completion")):
    matrix, marks = ev.effect_matrix(STUDY.effects, control, held, level)
    fg.plot_heatmap(matrix, f"Change in error from the lowest to the highest {label}, on the same cells", FIGURES,
                    f"{control}_effect", "Relative change in error", center=0.0, limit=1.0, marks=marks, fmt="{:+.0%}")"""),
    ("md", """### 6.2 Linked columns visible in the row

A column's directly linked columns are those it is computed from and those computed from it, in the generator's definition. For each hidden cell, the number of them visible in the same row is counted from measured data only. Each panel shows skill against the share that was visible, with one line style for each noise level. Groups with fewer than 30 cells are left out.

The table gives the same numbers for the lead model in the representative scenario."""),
    ("code", """seen = ev.add_scenario_levels(STUDY.visibility, STUDY.scenarios)
fg.plot_lines(seen[seen.method.isin(MAIN)], "share seen", "Skill by the share of directly linked columns visible in the row",
              FIGURES, "skill_by_visible_links", "Linked columns visible", MAIN, style="noise", style_label="noise", share_seen=True)
lead_seen = seen[seen.method.eq(LEAD) & seen.scenario.eq(REPRESENTATIVE)].copy()
lead_seen["visible"] = lead_seen.seen.astype(int).astype(str) + " of " + lead_seen.links.astype(int).astype(str)
lead_seen["order"] = lead_seen.target.map(list(TARGETS).index)
display(lead_seen.sort_values(["order", "seen"]).set_index("target")[["visible", "n", "skill"]].round(3))"""),
    ("md", """### 6.3 Position of the true value

Cells are grouped by where their true value lies in the column: the lowest tenth, the next fifth, the middle 40%, and the same on the high side. This grouping is known only afterwards, so it explains errors and sets no rule for filling data.

The chart shows the mean signed error in units of the column's spread, from the lowest band on the left to the highest on the right. A line that falls from left to right means that low values are estimated too high and high values too low. The baseline is the dashed line. The table gives the size of the error for the lead model."""),
    ("code", """fg.plot_bands(STUDY.bands, ["Median"] + MAIN, list(fl.BANDS), FIGURES)
display(STUDY.bands[STUDY.bands.method.eq(LEAD)].pivot(index="target", columns="band", values="error in SD")
        .reindex(index=list(NUMERIC_TARGETS), columns=list(fl.BANDS)).round(3))"""),
    ("md", """### 6.4 How cells were chosen for hiding

The four scenarios below hide the same share of cells: at random, more for older assets, more for higher values, and in blocks of related columns. The chart shows the skill of the lead model in each, and the first table the mean skill over the columns with a relationship for every model.

Skill alone can hide a shift. When higher values are hidden more often, the visible values of a column are lower than the hidden ones, so an estimate that leans on the visible range can be too low. The second table therefore shows the mean signed error, in units of the column's spread, for the lead model and the baseline. The third table shows mean skill by age third, to check that assets with more gaps are also filled less well."""),
    ("code", """scores = STUDY.mechanism_scores
lead_scores = scores[scores.method.eq(LEAD)].pivot(index="target", columns="mechanism", values="skill").reindex(list(TARGETS))
order = [m for m in ("random", "age", "value", "group") if m in lead_scores]
fg.plot_bars(lead_scores[order], f"{LEAD}: skill by how cells were chosen for hiding", FIGURES, "skill_by_mechanism",
             "Error removed vs baseline (%)")
signal = scores[scores.target.isin(fl.SIGNAL_TARGETS) & scores.mechanism.isin(order)]
display(signal.groupby(["method", "mechanism"]).skill.mean().unstack()[order].reindex(SHOWN).round(3))
bias = scores[scores.method.isin([LEAD, "Median"]) & scores.target.ne("model_id") & scores.mechanism.isin(order)]
display(bias.pivot(index="target", columns=["method", "mechanism"], values="bias in SD").reindex(list(NUMERIC_TARGETS)).round(3))
groups = fl.mechanism_groups(CELLS, STUDY.truth, STUDY.scenarios, STUDY.config)
groups = groups[groups.method.isin(MAIN) & groups.mechanism.isin(order)]
display(groups.pivot(index=["method", "mechanism"], columns="age third", values="skill").round(3))"""),
    ("md", """### 6.5 Kind of relationship

Columns are grouped by how the generator builds them. The table shows the mean skill of the lead model for each kind: in the reference scenario, in its best and worst generator scenario, and when few linked columns are visible."""),
    ("code", """with pd.option_context("display.max_colwidth", 70):
    display(STUDY.kinds.round(3))"""),
    ("md", """### 6.6 Heavier random missingness

The generator's files stop at 0.7 completion. Two added files hide further cells at random, down to 0.5 and 0.3 completion at noise 0.05, nested inside the 0.7 file (section 2.5). Each panel follows skill from 0.9 to 0.3 completion for the strongest model of each branch. The table gives the mean over the columns with a relationship for every model."""),
    ("code", """completion = STUDY.completion
fg.plot_lines(completion[completion.method.isin(MAIN)], "completion", "Skill as completion falls, at noise 0.05", FIGURES,
              "skill_by_completion", "Completion (share of cells visible)", MAIN)
display(completion[completion.target.isin(fl.SIGNAL_TARGETS)].groupby(["method", "completion"]).skill.mean()
        .unstack().reindex(SHOWN).round(3))"""),
    ("md", """### 6.7 Wrong recorded values

The errors scenario keeps the representative mask and makes some visible values wrong: 5% of visible installation days move by 60 days and 2% of visible `model_id` values become another model. The mask is the same, so every hidden cell is compared with itself, once with correct and once with wrong visible values. Rows are split by whether they hold a wrong value; rows without one show what the wrong values change through the fitted model alone.

The table gives mean skill over the columns with a relationship, with correct and with wrong visible values. The heatmap gives the relative change in error, by column, in rows that hold a wrong value. A cell is empty where the error was zero with correct values, so a relative change is undefined; this applies to most `model_id` cells."""),
    ("code", """wrong = STUDY.wrong_values
summary = wrong[wrong.target.isin(fl.SIGNAL_TARGETS)].groupby(["method", "row"])[["skill", "skill with wrong values"]].mean()
display(summary.unstack("row").reindex(SHOWN).round(3))
change = wrong[wrong.row.eq("row with a wrong value")].pivot(index="target", columns="method", values="relative change in error")
change = change.reindex(list(TARGETS))[[m for m in SHOWN if m in change]]
if change.notna().any().any():  # a small sample can leave too few rows with a wrong value
    fg.plot_heatmap(change, "Change in error on the same hidden cells when the row holds a wrong value", FIGURES, "wrong_values",
                    "Relative change in error", center=0.0, limit=1.0, fmt="{:+.0%}")"""),
    ("md", """### 6.8 Assets in unseen areas

Random folds leave training rows next to every held-out row. In this experiment the representative scenario is split into folds of whole 10 × 10 blocks of the grid, as both other branches split their data, so a held-out asset has no training asset beside it. The blocks are set by the grid position of a row, which no model receives. The table gives mean skill over the columns with a relationship under both splits for every model, and the heatmap the change in skill by column."""),
    ("code", """splits = STUDY.splits
display(splits[splits.target.isin(fl.SIGNAL_TARGETS)].groupby("method")[["random rows", "whole blocks", "change"]].mean()
        .reindex(SHOWN).round(3))
change = splits.pivot(index="target", columns="method", values="change").reindex(list(TARGETS))
fg.plot_heatmap(change[[m for m in SHOWN if m in change]], "Change in skill when folds are whole blocks of the grid", FIGURES,
                "fold_design", "Change in skill", center=0.0, limit=0.5, fmt="{:+.2f}")"""),
]

# ---------------------------------------------------------------- chapter 7
cells += [
    ("md", """## 7. Form of the failure

Chapter 6 showed where skill falls. This chapter describes what the estimates look like when it does.

### 7.1 Estimates collapse towards a constant

The slope of predictions against the truth is 1 when estimates follow the truth fully and 0 when they are a constant. The chart shows this slope against the share of linked columns visible. A slope that falls with the share means that a model answers a cell it cannot see into with a value near the centre of the column.

The table gives the spread of imputed values relative to the spread of the truth, over all hidden cells of the representative scenario. A value below 1 means that completed columns are narrower than the real ones."""),
    ("code", """slopes = ev.add_scenario_levels(STUDY.shrinkage, STUDY.scenarios)
fg.plot_lines(slopes[slopes.method.isin(MAIN)], "share seen", "Slope of predictions against the truth", FIGURES, "slope_by_visible_links",
              "Linked columns visible", MAIN, style="noise", style_label="noise", share_seen=True, y="slope",
              ylabel="Slope (1 follows the truth, 0 is a constant)", targets=list(NUMERIC_TARGETS))
spread = fl.overall_shrinkage(CELLS, STUDY.config)
display(spread[spread.scenario.eq(REPRESENTATIVE)].pivot(index="target", columns="method", values="spread ratio")
        .reindex(list(NUMERIC_TARGETS))[["Median"] + SHOWN].round(3))"""),
    ("md", """### 7.2 Values outside the valid range

The table shows the share of raw predictions outside the range a column allows, before they are limited to it. Scoring uses the limited values. A column with a floor or a ceiling, such as `measured_overgrowth`, is where an averaging model can leave the range."""),
    ("code", """ranges = fl.range_table(CELLS, STUDY.config)
display(ranges[ranges.scenario.eq(REPRESENTATIVE)].pivot(index="target", columns="method", values="outside valid range")
        .reindex(list(NUMERIC_TARGETS))[SHOWN].round(4))"""),
    ("md", """### 7.3 Gradual or sudden

The table describes the shape of the two declines for the lead model.

- **Noise**: the skill lost from the lowest to the highest noise, and the share of that loss that comes at the first step. A share near 1 means that the first amount of noise does most of the damage.
- **Hidden linked columns**: skill with every linked column visible and with the fewest, and the share of the loss that comes when the first one is hidden. A low share means that one hidden column is tolerated and several are not.

Shares are left empty where the loss is under 0.02."""),
    ("code", """display(fl.decline_table(STUDY.by_scenario, STUDY.visibility, STUDY.config).round(3))"""),
    ("md", """### 7.4 Relationships between columns

A completed table can have accurate cells and still distort how columns relate. The first table compares correlations in completed data with correlations in the truth, across all 91 pairs of numeric columns, on the rows that hold an imputed cell of the pair. A gap of 0 means that the correlation is unchanged.

The second table checks three definitions directly on completed rows. The reference is the residual in rows with nothing imputed, which is the level that measurement noise alone produces."""),
    ("code", """gaps = STUDY.correlations[STUDY.correlations.method.isin(["Median"] + SHOWN)]
display(gaps.pivot(index="method", columns="scenario", values="mean gap").reindex(["Median"] + SHOWN).round(4))
consistency = STUDY.consistency[STUDY.consistency.method.isin(["Median"] + MAIN)]
wide = consistency.pivot(index=["relation", "scenario"], columns="method", values="mean residual in completed rows")[["Median"] + MAIN]
wide["fully observed rows"] = consistency.groupby(["relation", "scenario"])["mean residual in fully observed rows"].first()
display(wide.round(4))"""),
    ("md", """### 7.5 Whether a model signals its own errors

For `model_id`, a model gives a probability with its answer. The tables compare that confidence with accuracy, first by the number of linked columns visible and then by scenario. `sure and wrong` is the share of cells answered wrongly with a probability of 0.9 or more. Confidence that falls together with accuracy means that the errors can be found without knowing the truth.

The numeric estimates of these models are single values and carry no such signal."""),
    ("code", """by_seen = fl.confidence_table(CELLS, STUDY.config)
display(by_seen[by_seen.method.isin(MAIN)].sort_values(["method", "seen"]).round(3))
by_scenario = fl.confidence_table(CELLS, STUDY.config, by="scenario")
display(by_scenario[by_scenario.method.eq(LEAD)].sort_values("scenario").round(3))"""),
    ("md", """### 7.6 Differences between models

Four of the models share the ridge base, so part of any agreement between them can come from their structure. The chart shows the skill of each main model with no ridge base (open marker) and with it (filled marker), in the representative scenario. The table adds the two references.

The last table covers every model of the three branches. It shows how far two models fail on the same cells: it is the rank correlation between their cell errors, averaged over columns. A value near 1 means that the models are wrong in the same places."""),
    ("code", """structure = STUDY.structure.drop(index="random_placebo")
pairs = [(m, f"{m}: no ridge", f"{m}: with ridge", fg.METHOD_COLORS[m]) for m in MAIN if f"{m}: no ridge" in structure]
fg.plot_pairs(structure, pairs, "Skill without and with the ridge base", FIGURES, "ridge_base_effect", "Error removed vs baseline (%)")
display(STUDY.structure.round(3))
display(fl.error_overlap(CELLS, STUDY.config).round(2))"""),
]

# ---------------------------------------------------------------- chapter 8
cells += [
    ("md", """## 8. Causes

Each section states one possible cause and tests it with a comparison.

### 8.1 Limits of the data and limits of the model

Some error cannot be removed by any model, because the data does not hold the answer. Two references separate that part.

The chart splits each column's baseline error into three parts for the lead model: the share removed, the share lost because linked columns were hidden, and the share that remains even when every linked column is visible.

The tables then apply the generator's own definition to the visible inputs (`Formula`), on hidden cells whose defining inputs are all visible. Values are MAE. With no noise the formula is exact, so a model's error there is its own. With noise the formula inherits the noise of its inputs, and a model can do better because other columns help to average that noise out. The last table shows how much of each column the other true columns could explain at all."""),
    ("code", """information = STUDY.information.rename(columns={"skill over all hidden cells": "removed by the model"})
fg.plot_stacked(information, ["removed by the model", "lost to hidden linked columns", "not reached even with every linked column"],
                [fg.SERIES[0], fg.SERIES[3], fg.GRID], f"{LEAD}: where the baseline's error goes", FIGURES, "error_split",
                "Share of the baseline's error")
display(STUDY.information.round(3))
for scenario in (NOISE_FREE, REPRESENTATIVE, HIGH_NOISE):
    display(Markdown(f"**{scenario}**"))
    display(rel.relation_matrix(STUDY.relations, scenario)[["Median"] + SHOWN + [rel.FORMULA]].round(3))
display(rel.recoverable_table(STUDY.truth[REPRESENTATIVE], NUMERIC, STUDY.config).round(3))"""),
    ("md", """### 8.2 Missing information

If hidden linked columns are the cause of section 6.2, the models should rely on those columns when they are visible. This is tested with single-target forest and CatBoost models, because a chain covers every column at once and a column's influence on one target cannot be read from it. Each predictor is shuffled in held-out rows, and the added error is measured against observed labels, so truth is not involved. A value of 1 means that the shuffle cost as much as replacing the model with a constant guess.

The first table places every column a model relies on (importance above 0.02) in the generator's definition:

- **direct**: the target is computed from it, or it is computed from the target.
- **defines a column together with the target**: for example `x` and `y`, which define `altitude` together.
- **linked through another column**: for example `altitude` and `install_date`, which are both computed from `x` and `y`.
- **stands in for a near copy**: `x` and `install_date` are near copies, so one can replace the other.
- **no link**: the definition gives no connection.

The second table lists the strongest columns relied on outside the target's own definition. The heatmap shows all values, with direct links in bold, for the lead model when it is one of the two single-target models, and otherwise for CatBoost, the single-target model of the same kind as the boosted lead."""),
    ("code", """with pd.option_context("display.max_colwidth", 60):
    display(STUDY.links.set_index(["method", "scenario"]).round(3).T)
    EXPLAINER = LEAD if LEAD in set(STUDY.all_importance.method) else "CatBoost"
    print("Single-target model shown:", EXPLAINER)
    display(fl.unexplained_links(STUDY.all_importance, NOISE_FREE, EXPLAINER))
mean_importance = STUDY.all_importance.groupby(["method", "scenario", "target", "predictor"], sort=False).importance.mean().reset_index()
fg.plot_importance(ev.importance_matrix(mean_importance, NOISE_FREE, EXPLAINER), f"{EXPLAINER}: importance of each column · {NOISE_FREE}",
                   FIGURES, "importance")"""),
    ("md", """### 8.3 Noise carried from linked columns

Noise can reach an estimate in two ways: through the visible values of the column itself, which the model learns from, and through the linked columns it reads. The table gives both, as the mean measurement error at the highest noise level in units of the column's spread, beside the change in the model's error on the same cells.

Columns such as `total_gas`, `life` and `model_id` are exact themselves. If their error still rises with noise, the noise was carried in from their linked columns. The chart plots the error change against the largest noise that reaches a column by either way."""),
    ("code", """noise = STUDY.noise.copy()
noise["largest noise reaching the column"] = noise[["noise in the column itself", "largest noise in a linked column"]].max(axis=1)
display(noise.round(3))
fg.plot_scatter(noise, "largest noise reaching the column", "error change from noise", "target",
                f"{LEAD}: noise reaching a column against the error it adds", FIGURES, "noise_carried",
                "Largest noise in the column or a linked column (SDs)", "Change in error, lowest to highest noise")"""),
    ("md", """### 8.4 Removed columns and substitutes

Each block of columns is removed according to what it measures, and the chained models are refitted. Values show the relative change in error, so a positive value means that the block carried information for that target. Empty cells are targets inside the removed block.

Two results matter for the causes. A large change shows which block a target depends on. A small change after removing a directly linked block shows that another column took its place, which a near copy can do."""),
    ("code", """removed = STUDY.ablation[STUDY.ablation.scenario.eq(REPRESENTATIVE)]
matrix = removed[removed.method.eq(LEAD)].pivot(index="target", columns="variant", values="relative change").reindex(list(TARGETS))
fg.plot_heatmap(matrix, f"{LEAD}: change in error when a block is removed · {REPRESENTATIVE}", FIGURES, "ablation",
                "Relative change in error", center=0.0, limit=1.0, fmt="{:+.0%}")
worst = removed[removed.method.isin(MAIN + ["IterativeImputer"])].pivot_table(index=["target", "variant"], columns="method", values="skill without")
display(worst.round(3))"""),
    ("md", """### 8.5 Structure of the model

The table gives mean skill by kind of relationship for each structure: the main models with and without the ridge base, the ridge alone and the boosted chain with no ridge. Where the ridge raises a local model, the relationship is one the local model approximates poorly on its own. Where all structures agree, the limit lies in the data."""),
    ("code", """by_kind = STUDY.structure.assign(kind=[fl.KIND_OF[t] for t in STUDY.structure.index]).groupby("kind", sort=False).mean()
with pd.option_context("display.max_colwidth", 60):
    display(by_kind.round(3))"""),
    ("md", """### 8.6 Explanations that are ruled out

Three checks test whether results could be produced by the evaluation itself.

- **Leakage.** The first table counts predictions made for a visible cell or from a model that saw the row, and hidden cells not predicted exactly once. Every count should be 0.
- **Invented structure.** `random_placebo` is unrelated to every other column. A model that beat the baseline on it would be finding structure that does not exist. The table shows its skill in every scenario, and the next one the importance given to it as a predictor.
- **The fold split.** The representative scenario is repeated with two more fold seeds. The last table shows mean skill for each seed."""),
    ("code", """display(fl.integrity_table(CELLS, STUDY.measured, STUDY.config))
placebo = STUDY.by_scenario[STUDY.by_scenario.target.eq("random_placebo")]
display(placebo.pivot(index="method", columns="scenario", values="skill").round(4))
used = STUDY.all_importance[STUDY.all_importance.predictor.eq("random_placebo") & STUDY.all_importance.target.ne("random_placebo")]
display(used.groupby(["method", "scenario"]).importance.agg(["mean", "max"]).round(4))
seeds = fl.skill_by(CELLS[CELLS.scenario.eq(REPRESENTATIVE) & CELLS.experiment.isin(["main", "stability"])
                          & CELLS.target.ne("random_placebo")], ["method", "seed", "target"])
display(seeds.groupby(["method", "seed"]).skill.mean().unstack().drop(index=["Median", "Mode"], errors="ignore").round(4))"""),
]

# ---------------------------------------------------------------- chapter 9
cells += [
    ("md", """## 9. What imputation gives a later analysis

An analysis that accepts only complete rows loses most of the table. The first table counts the rows with no missing cell in every scenario.

The second part compares eight summary values that an asset analysis might take from the table, such as mean `life` by model and the share of assets with low `life`. Each is computed from the truth and from several versions of the representative scenario:

- **rows with no missing cell**: what an analysis without imputation would use.
- **Median and Mode**: the baseline fill.
- **each model**: the completed table, with out-of-fold values.
- **measured values with true values in the gaps**: a perfect imputation, whose error comes from measurement noise alone.

Values are relative errors against the truth. The last table repeats the mean absolute error for the scenarios of section 2.5."""),
    ("code", """display(fl.complete_rows_table(STUDY.measured, STUDY.scenarios).round(3))
downstream = STUDY.downstream
order = ["rows with no missing cell", "Median and Mode"] + [m for m in SHOWN if m in set(downstream.source)] + ["measured values with true values in the gaps"]
with pd.option_context("display.max_colwidth", 60):
    display(downstream.pivot(index="quantity", columns="source", values="relative error")[order].round(4))
display(STUDY.downstream_by_scenario.reindex(order).round(4))"""),
]

# ---------------------------------------------------------------- chapter 10
cells += [
    ("md", """## 10. Limits of the experiment

- The data is synthetic, and there is one generated truth table, so the study does not measure variation between generated datasets.
- The scenarios of section 2.5 are simple forms of missingness and error. They do not cover every mechanism, each was run at noise 0.05 only, and each uses one draw of masks or errors.
- The models of the other two branches run through an adapter, under this study's folds, data and scoring. Parts of their code that encode the generator are switched off: Subteam A's gas-decay feature, spatial installation-day feature, exact formulas and placebo blanking. Their results here therefore differ from those in their own reports, and the tables here do not reproduce those reports.
- The range limit of the chained models (chapter 4, step 5) was added after the first full run of this version, in which these models diverged at 0.3 completion: raw estimates reached three column spans outside the range of the truth, and one task failed on infinite values. That run is kept as `outputs/imputation-study/runs/20261115-012041`. Their settings were chosen before the limit existed, on scenarios where it rarely applies.
- MICE-PMM was written to complete a whole table. Here it is fitted on training rows only, like every other model, so its results differ from the whole-table results of its own notebook.
- Subteam A's selected settings were not in their package, so the settings here come from this study's rule and their search space. The Masked MLP was still improving when selection stopped.
- `x` and `install_date` are near copies of each other, because assets were installed in grid order. High skill on these columns says little about model quality.
- A noise level does not mean equal difficulty across columns. Section 8.3 gives the noise of each column in units of its spread.
- Four models share the ridge base, so agreement between them can partly be a result of their structure. Section 7.6 measures that share.
- Importance comes from single-target models, not from the chained models that are scored. The ablation in section 8.4 tests chained models directly.
- The models give single values for numeric columns. The study does not test methods that return a range of plausible values.
- IterativeImputer can stop at its pass limit before it converges. Section 4.3 reports how often, and the effect of other limits.
- All development used this one dataset. Settings never saw a hidden cell, but no independent dataset was held back.
- Shared settings were chosen on three scenarios and may not be the best elsewhere. The search covered a fixed list of candidates, so it does not establish any method's ceiling.
- The candidates were written by people who knew how the data was generated. Models receive no relationship and a fixed rule made the final choice, but second-order terms were offered because such relationships were expected.
- A chained model trains on estimates that used the row's own label, which is not available for a hidden cell. Removing this effect improved the selection score by 0.25%, so the simpler form was kept.
- Row intervals are conditional on fitted models. Neighbouring grid rows are related, so intervals may be narrower than they should be.
- `life`, `total_gas` and `leakage` are exact by design, so results for and from these columns are better than noisy measurements would allow."""),
    ("md", """## Appendix

### A.1 All models, with the size of their differences

The first table gives the best method for each target, where `lead` is `clear` when the gap to the next method is both separated and at least 5%. The chart compares every method with the baseline on the same cells, with 95% row-bootstrap intervals. The intervals resample rows for fitted models. They do not include refitting or the randomness of the dataset itself. The last table compares the best two methods for each target."""),
    ("code", """display(STUDY.targets.round(3))
fg.plot_paired(PAIRS, REPRESENTATIVE, FIGURES)
closest = PAIRS[PAIRS.scenario.eq(REPRESENTATIVE) & PAIRS.comparison.eq("best two")].copy()
closest["relative gap"] = closest["second error"] / closest["first error"].where(closest["first error"] > 0) - 1
closest = closest.set_index("target").reindex(list(TARGETS)).reset_index()
display(closest[["target", "first", "second", "metric", "difference", "low", "high", "relative gap"]].round(5))"""),
    ("md", """### A.2 Different fold splits

The representative scenario is repeated with two more fold seeds. The same rows are reused across seeds, so this shows sensitivity to the split and does not add new samples."""),
    ("code", """seed_numeric, seed_labels = ev.stability_table(NUMERIC, CATEGORY, STUDY.config)
print("Worst rank of each method across seeds:")
display(seed_numeric.pivot(index="target", columns="method", values="worst rank").reindex(list(NUMERIC_TARGETS))[list(NUMERIC_METHODS)])
seed_numeric["relative range"] = seed_numeric["MAE high"] / seed_numeric["MAE low"] - 1
print("Largest relative MAE range across seeds:", round(float(seed_numeric["relative range"].max()), 4))
display(seed_labels.round(4))"""),
    ("md", """### A.3 Cost and warnings

Timings are single-threaded fits measured while other workers share the machine, so they are indicative only. A chained task covers many columns, so time is shown per target. The warning table lists each kind of warning once."""),
    ("code", """display(STUDY.cost.round(3))
fg.plot_cost(STUDY.cost, NUMERIC, REPRESENTATIVE, FIGURES)
with pd.option_context("display.max_colwidth", None):
    display(ev.warning_table(STUDY.log))"""),
    ("md", """### A.4 Saved results

The last cell writes one completed table per method to the run folder and summarises the run. Every table of chapters 5 to 9 is also saved there as a CSV file."""),
    ("code", """completed = ev.export_completions(SCORED, STUDY.measured[REPRESENTATIVE], REPRESENTATIVE, Path(STUDY.directory) / "completed")
display(completed)
display(ev.run_summary(STUDY.log, STUDY.plan))
print("Run folder:", STUDY.directory)
print("To repeat the evaluation, set EVALUATION_RUN_ID to:", Path(STUDY.directory).name)"""),
]

notebook = {"cells": [], "metadata": old["metadata"], "nbformat": old["nbformat"], "nbformat_minor": old["nbformat_minor"]}
for kind, text in cells:
    cell = {"cell_type": "markdown" if kind == "md" else "code", "metadata": {}, "source": text.rstrip("\n").splitlines(keepends=True)}
    if kind == "code":
        cell.update(execution_count=None, outputs=[])
    notebook["cells"].append(cell)
# The study's own package, independent of the team repository's layout.
RENAMES = [("team19_asset_data.transformers.miceforest_compat", "synthetic_imputation.miceforest_compat"),
           ("team19_asset_data.imputation_study", "synthetic_imputation"),
           ('(ROOT / "src/team19_asset_data")', '(ROOT / "src/synthetic_imputation")'),
           ("src/team19_asset_data/imputation_study/vendor", "src/synthetic_imputation/vendor")]
for cell in notebook["cells"]:
    source = "".join(cell["source"])
    for old_name, new_name in RENAMES:
        source = source.replace(old_name, new_name)
    assert "team19" not in source, source[:200]
    cell["source"] = source.splitlines(keepends=True)
TARGET.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
questions = sum("?" in "".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "markdown")
print(len(notebook["cells"]), "cells,", sum(c["cell_type"] == "code" for c in notebook["cells"]), "code | markdown cells with a question mark:", questions)
