# Transformer T3 — Feature relationship ablation

## Research objective

Quantify which input attributes the project's **Predictive MissForest** relies on when imputing unseen synthetic transformer records. This is a model-dependency diagnostic, not an experimental claim about real SAPN records and not a causal effect of a real infrastructure variable.

## Design

- Existing paired dataset: `dataset/completion0.8-noise0.05-measured.csv` and aligned `...-truth.csv`.
- **No regeneration or cleaning.** Imputation input is the measured dataset; truth is for scoring only.
- Spatial split seed 617: 8,000 training and 2,000 unseen test records, as in the nine-scenario baseline.
- Missing test cells are scored against clean truth. Existing observed values, including noise, are preserved.
- Compare a full predictor set to variants that remove `x`/`y`, `install_date`, `model_id`, `total_gas`, `altitude`, and `random_placebo`. Removed fields are excluded both from the model predictors and its scored targets.
- Model configuration from `../config.yaml`: 100 trees, `max_depth=None`, `min_samples_leaf=10`, 5 passes; model seed 219.
- **Seed control:** the experiment wrapper keeps each target's RF seed tied to its index in the *original 15-attribute schema*, so removing a column does not accidentally change other targets' RF seeds. Production model code is unchanged.
- Comparison uses the same test cells for each retained target. `ablation_comparison.csv` gives MAE, baseline-relative skill, and macro-F1.

## Run from the notebook

Open `../notebook/transformer_feature_ablation.ipynb` and set `RUN_T3=True`. Set it back to `False` to load saved outputs. `QUICK_SMOKE_TEST=True` uses smaller models and **cannot be cited as a full experiment**.

## Run from the terminal

From `pipeline/`:

```bash
python experiment/transformer/scripts/run_feature_ablation.py --variant all_attributes
python experiment/transformer/scripts/run_feature_ablation.py --variant without_coordinates
python experiment/transformer/scripts/run_feature_ablation.py --variant without_install_date
python experiment/transformer/scripts/run_feature_ablation.py --variant without_model_id
python experiment/transformer/scripts/run_feature_ablation.py --variant without_total_gas
python experiment/transformer/scripts/run_feature_ablation.py --variant without_altitude
python experiment/transformer/scripts/run_feature_ablation.py --variant without_placebo
python experiment/transformer/scripts/analyse_feature_ablation.py --require-all
```

Results are stored in `../outputs/feature_ablation_seed617/`.

## Reporting precautions

1. Report **per-target** changes, not averages over target lists that differ between variants.
2. A large MAE change from dropping a field indicates model reliance; it does not prove a causal relation between the underlying physical phenomena.
3. The coordinate ablation affects both a mathematically defined altitude relationship and an incidental coordinate/date correlation from the synthetic generator.
4. The benchmark uses one synthetic generation and one split; it lacks confidence intervals. Treat these values as a single-run diagnostic.
5. This study evaluates the reusable *Predictive MissForest* only. The existing nine-scenario benchmark already compares Median/Mode, Original MissForest, and Predictive MissForest. Original MissForest is not an unseen-record model in this codebase.

## Main files

- `../scripts/run_feature_ablation.py` — training, exact-mask evaluation, per-variant provenance and seed control
- `../scripts/analyse_feature_ablation.py` — paired comparisons, figures and Markdown report
- `../notebook/transformer_feature_ablation.ipynb` — interactive runner and result display
- `../outputs/feature_ablation_seed617/T3_report.md` — final report-ready analysis
