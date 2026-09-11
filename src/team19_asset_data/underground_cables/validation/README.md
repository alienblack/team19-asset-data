# Member 3 Validation Framework

## Purpose

This reusable framework implements the Sprint 2 Member 3 validation work. It accepts file paths and field names through configuration or command-line arguments, so it can be connected to Member 1 and Member 2 outputs without changing the validation code.

## Covered Jira tasks

- UC-S2-07: hidden-known-value validation metrics
- UC-S2-08: distribution and confidence validation
- UC-S2-09: downstream model stability and results summary

## Required Inputs

Member 1 and Member 2 must provide:

- A complete reference dataset
- The corresponding masked dataset
- A baseline-imputed dataset
- An alternative-imputed dataset
- The target field name
- Masking information represented by the same rows being null in the masked target field
- Optionally, an ID field used to verify row identity and order

All four CSV files must contain the same rows in the same order. An artificial hidden-known position is a row whose target is present in the reference dataset and missing in the masked dataset. Both imputation methods are always evaluated on this single shared set of positions.

## Validation Methods

- Numeric targets: MAE and RMSE on artificial masked positions
- Categorical targets: accuracy and weighted F1 on artificial masked positions
- Numeric distribution summaries, histograms, and boxplots
- Categorical category counts and proportions
- `Observed`, `Imputed`, and reserved `Derived` data-status framework
- Preliminary `High`, `Medium`, `Low`, or `TBD` confidence framework
- Numeric target ranking stability using Spearman rank correlation

Observed values are normally marked `High`. Imputed confidence should account for overall method performance, available predictors, and uncertainty. The default is conservatively `TBD`; the command line can set another label only when supporting evidence exists. Exact thresholds must be confirmed from later experiments. These labels are internal preliminary rules, not official SAPN standards. `Derived` is supported by the interface but is not assigned unless a future documented derivation creates such a field.

## Important Limitation

The downstream validation currently uses the numeric target itself as a preliminary ranking proxy. It should not be interpreted as the official SAPN PoF or Weibull model. No business risk formula is invented here. Replace or wrap `calculate_risk_proxy` only when a documented project formula is available. Ranking stability is skipped for categorical targets because target-value ranks are not meaningful.

## How to Run

Install dependencies:

```bash
python3 -m pip install -r member3_validation/requirements.txt
```

Run all validation stages:

```bash
python3 member3_validation/run_validation.py \
  --reference data/inputs/underground_cable_reference_synthetic_data.csv \
  --masked data/inputs/underground_cable_mcar_20.csv \
  --baseline data/inputs/underground_cable_mcar_20_baseline_imputed.csv \
  --alternative data/inputs/underground_cable_mcar_20_advanced_imputed.csv \
  --target target_column \
  --id id_column
```

`--id` is optional. Defaults can instead be entered in `config.py`. CLI arguments override those defaults. Use `--output-dir` to change the output location and `--imputed-confidence` only when experiments justify a non-`TBD` label.

## Outputs

- `outputs/metrics/validation_metrics.csv`: comparable hidden-value metrics, masked count, and evaluated count for both methods. Non-applicable metrics remain blank.
- `outputs/distributions/distribution_summary.csv`: descriptive statistics for numeric targets, or category counts and proportions for categorical targets.
- `outputs/distributions/histogram.png`: numeric reference/baseline/alternative histogram.
- `outputs/distributions/boxplot.png`: numeric reference/baseline/alternative boxplot.
- `outputs/distributions/data_status_confidence.csv`: row-level method, status, and preliminary confidence records.
- `outputs/downstream/ranking_stability.csv`: Reference vs Baseline and Reference vs Alternative Spearman correlations for numeric targets.

Output folders are created automatically. Plots and ranking output are not produced for categorical targets; their relevant metric and distribution CSV files are still produced.

For the completed Sprint 2 experiment, target-specific outputs are stored under
`outputs/asset_age_years`, `outputs/groundwater_indicator`, and
`outputs/corrosion_indicator` to prevent one run from overwriting another. The
consolidated results are available in `outputs/validation_results_summary.csv`
and the interpretation is documented in `RESULTS_SUMMARY.md`.

Run the three completed experiment targets with separate `--output-dir` values
to prevent later runs from overwriting earlier results. For example:

```bash
python3 member3_validation/run_validation.py \
  --reference data/inputs/underground_cable_reference_synthetic_data.csv \
  --masked data/inputs/underground_cable_mcar_20.csv \
  --baseline data/inputs/underground_cable_mcar_20_baseline_imputed.csv \
  --alternative data/inputs/underground_cable_mcar_20_advanced_imputed.csv \
  --target asset_age_years \
  --id asset_id \
  --output-dir member3_validation/outputs/asset_age_years
```
