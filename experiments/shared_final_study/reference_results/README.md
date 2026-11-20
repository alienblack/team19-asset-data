# Result tables and figures of the synthetic imputation study (version 5)

Everything the study produced, ready to open: result tables as CSV files (open in Excel) and figures as PNG files. No code is needed. All numbers come from the study's run `20261115-121714`.

## Where to start

- Main numbers: `tables/1_overall/overall.csv` (skill of every model on every data column) and `tables/1_overall/branches.csv` (average skill of every model under every condition).
- Main figures: `figures/project-guide/` (made for the project guide and slides).
- The filled data tables: `tables/6_completed_tables/`.

## Layout

| Folder | What is in it |
|---|---|
| `tables/1_overall/` | Start here: the overall picture in four small tables. 4 tables |
| `tables/2_scores/` | Every score, in long tables. Filter by `scenario`, `experiment`, `target` (data column) and `method` (model). 8 tables |
| `tables/3_conditions_and_causes/` | When imputation gets worse, and why. 9 tables |
| `tables/4_error_patterns/` | What the errors look like. 5 tables |
| `tables/5_use_in_analysis/` | What imputation gives a later analysis. 2 tables |
| `tables/6_completed_tables/` | The data table with every gap filled, one file per model |
| `tables/7_settings_selection/` | How each model's settings were chosen |
| `figures/notebook/` | Figures written by the notebook |
| `figures/project-guide/` | Figures of the project guide (GMM named GMM) |
| `figures/imputation-A-random-forest/` | Diagrams and charts of the Imputation A material |
| `figures/imputation-B-gmm/` | Diagrams and charts of the Imputation B material |

Not here: every single prediction, which only someone repeating the evaluation needs. It is large, so it is the separate package `03_raw_predictions_only_for_rerunning`.

## Words used in the tables

| Word | Meaning |
|---|---|
| skill | Share of the baseline's error that a model removes on the same hidden cells. 0 = no better than the baseline, 1 = perfect, below 0 = worse than the baseline |
| baseline | `Median` for numeric columns, `Mode` (most common class) for `model_id` |
| target | The data column being filled (`x`, `life`, `model_id`, …) |
| method | The model. GMM is called `Gaussian EM` in every table |
| scenario | Which version of the data. `completion0.8-noise0.05` = 80% of cells kept, 5% noise; this is the reference scenario. A suffix marks the added scenarios: `age` older assets hidden more, `value` higher values hidden more, `group` related columns hidden together, `errors` some recorded values wrong, `sparse` more cells hidden at random |
| experiment | `main` = the normal run (use this unless you need another). Others: `mechanism` (added scenarios), `stability` (other fold seeds), `spatial` (whole map areas held out), `ablation` (groups of columns removed), `structure` (no ridge base), `budget`, `calibration` |
| variant, seed | Sub-setting of an experiment; `all predictors` and seed `219` are the normal ones |
| n | Number of hidden cells scored |
| MAE, error | Mean absolute error between estimate and true value, in the column's own units |
| bias | Mean of (estimate − truth): above 0 = estimates too high |
| within tight / loose | Share of estimates close enough to the truth, by a tolerance set per column |
| related (linked) columns | The columns the data generator uses to make a column |

## tables/1_overall

Start here: the overall picture in four small tables.

| File | What it tells you | One row is | Size |
|---|---|---|---|
| `overall.csv` | How good each model is on each data column, in the reference scenario. One column per model; the values are skill. | one data column | 15 rows |
| `targets.csv` | Which model is best on each data column, the runner-up, and whether the lead is clear. | one data column | 13 rows |
| `branches.csv` | Average skill of each model under each condition (reference, more noise, fewer cells kept, few related columns visible, …), over 13 numeric columns + model_id. | one model | 11 rows |
| `cost.csv` | How long each model takes to fit and predict. | one experiment × model | 71 rows |

## tables/2_scores

Every score, in long tables. Filter by `scenario`, `experiment`, `target` (data column) and `method` (model).

| File | What it tells you | One row is | Size |
|---|---|---|---|
| `numeric_scores.csv` | All scores of the numeric columns, for every scenario and experiment: MAE, RMSE, bias, R², share within tolerance, skill. | scenario × experiment × variant × seed × data column × model | 4,878 rows |
| `category_scores.csv` | The same for model_id: accuracy, macro F1, balanced accuracy. | scenario × experiment × variant × seed × model | 339 rows |
| `skill_by_scenario.csv` | Skill in each of the nine generator scenarios. | scenario × data column × model | 1,476 rows |
| `skill_by_completion.csv` | Skill as cells go missing: 90%, 80%, 70%, 50% and 30% of cells kept. | scenario × data column × model | 820 rows |
| `mechanisms.csv` | Skill when cells are hidden in other ways (older assets, higher values, related columns together) or some recorded values are wrong. | scenario × data column × model | 895 rows |
| `skill_by_visible_links.csv` | Skill depending on how many of the column's related columns the row still has. | scenario × data column × model × related columns visible | 1,671 rows |
| `paired_intervals.csv` | Whether one model is really better than another (or than the baseline): interval of the difference in error on the same cells. | scenario × data column × pair of models | 1,611 rows |
| `scenario_effects.csv` | How much more noise or more missing cells changes the error, on the same cells. | pair of scenarios × data column × model | 2,952 rows |

## tables/3_conditions_and_causes

When imputation gets worse, and why.

| File | What it tells you | One row is | Size |
|---|---|---|---|
| `failure_map.csv` | Skill of the strongest model (HistGradientBoosting) on each data column under each condition. | one data column | 15 rows |
| `relationship_kinds.csv` | Skill by the kind of formula that makes the column (near copy, exact sum, threshold, …). | one kind of relationship | 9 rows |
| `information.csv` | How much of the error comes from related columns being hidden in the same row, and how much cannot be removed at all. | one data column | 14 rows |
| `noise.csv` | How much noise reaches each column and how much skill it costs. | one data column | 14 rows |
| `ablation.csv` | What happens when a group of columns is taken away from the model. | scenario × data column × model × removed group | 1,905 rows |
| `structure.csv` | Skill with and without the ridge base in the chained models. | one data column | 15 rows |
| `fold_design.csv` | Whether holding out whole areas of the map changes the results. | data column × model | 164 rows |
| `links.csv` | Which columns the models relied on, compared with how the data was generated. | model × scenario | 6 rows |
| `relations.csv` | Error on each formula of the data generator. | formula × scenario × model | 310 rows |

## tables/4_error_patterns

What the errors look like.

| File | What it tells you | One row is | Size |
|---|---|---|---|
| `value_bands.csv` | Whether low and high true values are estimated too close to the middle (bias by band of the true value). | data column × model × value band | 828 rows |
| `shrinkage.csv` | Slope of the estimates against the truth (1 = no pull towards the middle). | scenario × data column × model × related columns visible | 1,692 rows |
| `wrong_values.csv` | How much a wrong recorded value hurts the other filled cells of the same row. | row type × data column × model | 328 rows |
| `consistency.csv` | Whether filled rows still obey three exact formulas (for example altitude = x + y + 100). | scenario × formula × model | 108 rows |
| `correlations.csv` | Whether the filled table keeps the correlations between columns. | scenario × model | 180 rows |

## tables/5_use_in_analysis

What imputation gives a later analysis.

| File | What it tells you | One row is | Size |
|---|---|---|---|
| `downstream.csv` | Eight summary numbers (for example mean life per equipment model) from each filled table, against the true values. | source × summary number | 104 rows |
| `downstream_by_scenario.csv` | Average error of those eight numbers, per source and scenario. | one source (complete rows only, baseline, or a model) | 13 rows |

## tables/6_completed_tables

The reference scenario's data table (10,000 rows, `row_id` + 15 columns) with every gap filled by one model. Cells that were measured are unchanged. Each filled value comes from a model that never saw that row. IterativeImputer does not fill `model_id`, so those gaps stay empty in its file.

Files: `CatBoost.csv`, `Gaussian_EM.csv`, `HistGradientBoosting.csv`, `IterativeImputer.csv`, `KNN_imputer.csv`, `LightGBM_chained.csv`, `MICE-PMM.csv`, `Masked_MLP.csv`, `Median.csv`, `Predictive_MissForest.csv`, `Random_Forest.csv`, `miceforest_direct.csv`.

## tables/7_settings_selection

| Path | What it is |
|---|---|
| `clean_selection/RULE.md`, `merge_selection/RULE.md` | The selection rule, written before any candidate was scored |
| `clean_selection/<model>.csv` and `<model>_extra.csv` | Score of every candidate setting, per scenario and fold. `knn` KNN imputer, `ridge` IterativeImputer, `forest` Random Forest, `hgb` HistGradientBoosting, `cat` CatBoost, `mice` miceforest direct; `joint` and `single` are miceforest variants that were also tried |
| `<model>.chosen.json`, `<model>.final.json` | The settings that were chosen |
| `merge_selection/em*.csv`, `mlp*.csv` | The same for GMM (`em`) and Masked MLP (`mlp`) |
| `merge_selection/README.md` | Settings of all models whose settings were taken from their own documentation |

# Figures

## figures/notebook

Figures written by the notebook (model names as in the code: the GMM is `Gaussian EM`).

| File | Content |
|---|---|
| `overall_skill.png` | Skill of every model on every column, reference scenario |
| `models_by_condition.png` | Mean skill (14 columns) of every model under every condition |
| `failure_map.png` | Skill of HistGradientBoosting on every column under every condition |
| `noise_effect.png` | Change in error from the lowest to the highest noise, every model and column |
| `completeness_effect.png` | Change in error from the lowest to the highest completion, every model and column |
| `cost_against_skill.png` | Mean skill against seconds per column per fold |
| `bias_by_value_band.png` | Bias by where the true value lies in its column, for the baseline, HistGradientBoosting, GMM and LightGBM chained |
| `error_split.png` | HistGradientBoosting: the baseline's error split into what the model removes, what hidden related columns cost, and what is not reached |
| `noise_carried.png` | Error change from noise against the noise reaching each column |
| `fold_design.png` | Skill with whole map-block folds against random-row folds |
| `importance.png` | Permutation importance of each column for each target |
| `ablation.png` | Change in error when a block of columns is removed |
| `skill_by_noise.png` | Skill as noise rises, at each completion level |
| `skill_by_completion.png` | Skill as completion falls, at 5% noise |
| `skill_by_mechanism.png` | HistGradientBoosting: skill by how cells were chosen for hiding |
| `skill_by_visible_links.png` | Skill by the share of related columns visible in the row |
| `slope_by_visible_links.png` | Slope of the estimates against the truth, by the share of related columns visible |
| `wrong_values.png` | Change in error on the same hidden cells when the row holds a wrong value |
| `ridge_base_effect.png` | Skill without and with the ridge base |
| `paired_uncertainty.png` | Paired intervals of error differences between models, reference scenario |
| `model_<name>.png` (11 files) | One per model: its skill on every column in each of the nine generator scenarios |

## figures/project-guide

Figures of the project guide (13 numeric columns, GMM named GMM).

| File | Content |
|---|---|
| `workflow.png` | Workflow of one experiment |
| `overall.png` | Mean skill of every model, reference scenario |
| `skill_by_column.png` | Skill of every model on every column, Random Forest and GMM marked |
| `skill_vs_cost.png` | Mean skill against seconds per column per fold |
| `conditions.png` | Mean skill of every model under every condition (heat map) |
| `mean_by_noise.png` | Mean skill of every model against noise, one panel per completion |
| `mean_by_completion.png` | Mean skill of every model as completion falls from 0.9 to 0.3 |
| `model_id_by_completion.png` | model_id accuracy of every model as completion falls |
| `summary_values.png` | Error of eight summary values from the completed table |
| `ab_by_noise_1.png` | Random Forest and GMM: skill by column against noise, part 1 |
| `ab_by_noise_2.png` | Random Forest and GMM: skill by column against noise, part 2 (with placebo and model_id) |

## figures/imputation-A-random-forest

Diagrams and charts of the Imputation A material.

| File | Content |
|---|---|
| `chain.png` | Chained imputation, the idea |
| `pmf.png` | Predictive MissForest structure |
| `rf_plain.png` | Random Forest without the ridge step |
| `rf_full.png` | Final Random Forest with the ridge step |
| `update.png` | One column update of the final Random Forest |
| `bars.png` | Mean skill of the three versions |

## figures/imputation-B-gmm

Diagrams and charts of the Imputation B material.

| File | Content |
|---|---|
| `idea.png` | Gaussian mixture idea (made-up two-column illustration) |
| `fit.png` | Fitting the mixture by EM |
| `fill.png` | Filling one row |
| `selection.png` | Settings selection: score against penalty |
| `columns.png` | Skill by column, GMM and Random Forest |
| `completion.png` | Mean skill as completion falls |