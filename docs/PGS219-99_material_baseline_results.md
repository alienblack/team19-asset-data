# PGS219-99: Baseline Experiment Summary

## What was completed

The existing synthetic overhead-conductor reference dataset was used without
regeneration or modification. A reproducible experiment was implemented for
missing `material` values using two explainable baselines:

- global material mode;
- feeder-group material mode with global fallback.

For each method, the code creates an 80/20 development/evaluation split and
masks known evaluation values at 10%, 30% and 50%. The same rows are masked
for both methods and the mode is fitted from development rows only. Five seeds
are used: 101, 202, 303, 404 and 505.

## Results

The values below are the mean across the five seeds. Both methods produced the
same result on this version of the synthetic data.

| Mask rate | Accuracy | Balanced accuracy | Macro-F1 |
|---:|---:|---:|---:|
| 10% | 0.7560 | 0.3333 | 0.2869 |
| 30% | 0.7350 | 0.3333 | 0.2824 |
| 50% | 0.7304 | 0.3333 | 0.2814 |

## Interpretation

The raw accuracy is substantially higher than balanced accuracy because Steel
is the dominant synthetic class. The balanced accuracy of 0.3333 indicates
that the simple mode baselines do not recover the minority material classes
well. Macro-F1 confirms the same limitation by weighting Aluminium, Copper and
Steel equally.

Feeder-group mode did not outperform global mode, and no rows required the
unseen-feeder fallback in this experiment. This indicates that the current
synthetic generator does not contain useful feeder-level material variation.
It is an honest baseline result and a reason to test richer, justified feature
relationships later, not a reason to change the data after seeing the result.

`conductor_type` was excluded because the richer reference contains a highly
predictive relationship with material that could create target leakage. The
current material-model input therefore provides a more conservative baseline.

## Jira attachment checklist

Attach these generated files from `outputs/PGS219-99/`:

- `baseline_metrics.csv`
- `confusion_matrix.csv`
- `masked_material_seed101_rate30.csv`
- `imputed_material_baselines_seed101_rate30.csv`
- `run_metadata.json`

The method, assumptions and limitations are documented in
`docs/PGS219-99_material_baseline.md`. The generated output directory is
ignored by Git so that experiment outputs are attached to Jira rather than
mistakenly committed as source data.
