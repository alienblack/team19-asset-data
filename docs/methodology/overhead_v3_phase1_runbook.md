# Overhead-conductor imputation: revised-methodology Phase 1 pilot

This is the runnable companion to `Team19_Overhead_Conductor_Revised_Methodology_v3.docx` (24 October 2026). It tests recovery of deliberately hidden **material** and **installation year** on one controlled, synthetic overhead-conductor population. It does not use SAPN records, correct plausible-but-wrong values, calculate Probability of Failure, or establish downstream model stability.

## Run

From a fresh checkout with the project's Python dependencies installed:

```sh
PYTHONPATH=src python -m pytest -q tests/test_v3_pilot.py
PYTHONPATH=src python -m team19_asset_data.overhead_conductors.run_v3_pilot \
  --output outputs/v3_phase1
```

The output directory must be new or empty. Default settings are 10,000 rows, generator seed 219, all relationships at strength 100, five split/mask seeds (101-105), 30% independent MCAR hiding of each target, and 24 trees for Predictive MissForest. Forest depth, leaf size and pass count are selected on each seed's tuning rows; two passes is the starting point. No predictor noise is introduced in Phase 1. The CLI accepts `--rows`, `--seeds`, `--mask-rate`, `--trees` and `--passes` for smoke checks; changing them creates a different experiment and should be reported.

## Data boundary

The generator's complete reference is retained under `evaluator_only/`. For each seed, IDs are split 60/10/10/20 into fitting, tuning, calibration and test, stratified by material and age band where group counts permit. The split and hidden-cell manifest are frozen before fitting. Observed views contain only segment ID and seven fields: installation year, distance to sea, voltage, location, corrosion, material and covering. `conductor_type`, `asset_age`, length, feeder ID and status are absent from the model-facing files. The derived age band is calculated only from a candidate year after prediction.

Only originally observed fitting cells determine the global floor, conditional groups, learned era cuts and forest preprocessing. The conditional baseline's smoothing and minimum group size are selected on tuning cells. Forest tuning uses one-factor validation curves for depths 4/6/8/12/unlimited, leaf sizes 1/5/10/20/50 and passes 1-5. Each candidate's tuning material log loss and year MAE are divided by that seed's global-floor scores and added with equal weight. Fitting scores show possible overfitting but never select settings. A 10%/25%/50%/100% fitting-size curve checks whether more labelled records help. Calibration rows are reserved; no calibration is claimed in this pilot. The test truth is joined only after all learned candidate files pass integrity checks. Test cells are identical across learned methods.

The seven-field model views stay unchanged. Separate `missingness_flags_*.csv` files record whether each target was hidden and the count of hidden targets. These flags are not fed to the Phase 1 models: under MCAR they have no intended signal and an observed-label-only training subset would make a target's own flag constant. They are retained for the later MAR/MNAR design.

## Methods and evaluator

- **Global floor:** fitting-observed material shares and median installation year. Its material point prediction is the most common class.
- **Conditional baseline:** hierarchical material probabilities from voltage, learned era and corrosion, smoothed toward parent groups; year medians and percentiles from voltage, covering and originally observed material. Small or unavailable groups back off to broader groups. Candidate rows record the level and support count.
- **Predictive MissForest:** the repository's frozen `fit`/`transform` implementation, using 24 trees and tuning-selected depth, leaf size and passes. The runner asserts that every originally observed field stays unchanged. The fitted model is saved and reloaded; the full per-cell score table is checked for equality. Material class proportions and per-tree year quantiles are raw diagnostics, not calibrated confidence or predictive intervals.
- **Generator reference:** an evaluator-only comparator derived from the generator's exact year prior and conditional voltage, covering and material laws. Under this clean MCAR setup, voltage screens year from location, corrosion and distance; when material is observed, its likelihood contributes to the year posterior. When both targets are hidden, material probabilities average over the posterior years. This comparator is privileged generator knowledge, unavailable to a real imputation method. It must be rederived for changed generator rules or non-MCAR masking.
- **Complete-case reference:** population summaries from test rows where neither target is hidden. It is not a cell-level prediction method.

The generator check compares observed material shares in voltage x *configured material era* x corrosion groups with the configured probabilities, group sizes and binomial sampling standard errors. A small group can have a non-Steel empirical plurality without contradicting the generator law. Method era cuts are learned separately from fitting observations; they never use the generator's era boundaries.

## Outputs and interpretation

`seed_*/candidate_cells.csv` contains one row per hidden cell and learned method: seed, condition, target, point prediction, material probabilities or year quantiles, evidence and fallback. `seed_*/forest_tuning.json` records the selected settings. The generator reference files remain under `evaluator_only/`. `cell_metrics.csv` reports log loss, Brier score, accuracy, balanced accuracy, per-class recall, year MAE, signed bias, age-band accuracy, and *empirical* q10-q90 coverage and width. Material log loss clips the probability assigned to the true class to at least `1e-12` before taking its logarithm. The year-range coverage check does not turn tree disagreement into a predictive interval. `population_metrics.csv` compares material shares, probability-weighted expected shares, the share aged 50 or older at the 2014 snapshot and year-distribution distance against the complete test reference; it includes unweighted and length-weighted versions and the complete-case comparator. Length is used only by the evaluator for weighting, never as a predictor. Point-filled and expected shares answer different questions. `subgroup_diagnostics.csv` breaks errors down by configured era, voltage, location and whether both targets were hidden; `subgroup_confusion.csv` gives material confusion counts by era, voltage and location. `evidence_backoff_counts.csv` counts group tiers and distinct fallback reasons. `forest_validation_curve.csv`, `forest_learning_curve.csv` and `plots/` provide numeric and visual tuning diagnostics. `integrity_checks.json` records per-seed checks. Evaluator-only overall confusion counts and high-error review cases are saved by seed. `paired_differences.csv` and summary files show each seed plus mean and range, without significance claims. The report's gap-closed percentage is the mean of per-seed gap-closed ratios, not the ratio of mean scores.

To reproduce the tuning-only tree-count sensitivity without touching test cells, run:

```sh
PYTHONPATH=src python -m team19_asset_data.overhead_conductors.run_v3_tree_check \
  --pilot-output outputs/v3_phase1_age50_label_20261024 \
  --output outputs/v3_tree_count_tuning_recheck.csv
```

The script holds each seed's selected depth, leaf size and passes fixed and compares 24 with 200 trees on the frozen tuning masks. Its output includes zero true-class probability counts and log loss under `1e-12` and `1e-6` clipping for sensitivity.

**Read material accuracy carefully.** In the configured generator, Steel is the most likely material for every combination of material parents. The global floor and generator reference therefore share the same optimal point class in expectation. A method may still improve material *probabilities* and population shares without improving point accuracy, and a finite test sample may show a small accidental point-accuracy difference. For year, inspect bias by true age band: a low overall MAE can still make old assets appear too young.

Phase 2 (noise, MAR/MNAR, varying strength and missing rate, calibration and distribution sensitivity) is not part of this run. In particular, the MCAR generator reference cannot be reused as an MNAR ceiling without conditioning on the missingness process.
