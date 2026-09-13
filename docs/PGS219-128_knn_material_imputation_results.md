# PGS219-128 KNN Material Imputation Results

## Run summary

The experiment used the versioned synthetic overhead-conductor reference file
without modifying or regenerating it. It reused the PGS219-99 split and mask
helpers, giving an 80/20 development/evaluation split, five seeds
(`101, 202, 303, 404, 505`) and 10%, 30% and 50% mask rates. The KNN run
produced 9,000 masked-row predictions. Its mask manifest is byte-identical to
the baseline manifest, so the comparison is controlled.

The target was categorical `material` with the classes Aluminium, Copper and
Steel. `conductor_type` was excluded because it can reveal material directly.
`asset_age` was also excluded because it is deterministically related to
`installation_year` and would double-weight age. The predictors were
installation year, voltage category, location group, corrosion zone, distance
to sea, covering and length.

## Mean results by mask rate

The KNN candidate metrics are averaged over five seeds. Baseline values are
from the existing PGS219-99 experiment on the same masks.

| Mask rate | Method | Accuracy | Balanced accuracy | Macro-F1 | Coverage |
|---:|---|---:|---:|---:|---:|
| 10% | Mode baseline | 0.7560 | 0.3333 | 0.2869 | 1.0000 |
| 10% | KNN candidate | 0.6840 | 0.3889 | 0.3914 | 0.6600 |
| 30% | Mode baseline | 0.7350 | 0.3333 | 0.2824 | 1.0000 |
| 30% | KNN candidate | 0.6877 | 0.3922 | 0.3955 | 0.6597 |
| 50% | Mode baseline | 0.7304 | 0.3333 | 0.2814 | 1.0000 |
| 50% | KNN candidate | 0.6804 | 0.3922 | 0.3957 | 0.6660 |

Accepted KNN metrics after the confidence threshold are:

| Mask rate | Accepted accuracy | Accepted balanced accuracy | Accepted Macro-F1 | Coverage |
|---:|---:|---:|---:|---:|
| 10% | 0.7820 | 0.3542 | 0.3534 | 0.6600 |
| 30% | 0.7797 | 0.3743 | 0.3757 | 0.6597 |
| 50% | 0.7735 | 0.3751 | 0.3766 | 0.6660 |

## Interpretation

The mode baseline has higher raw accuracy because Steel is the dominant class,
but its balanced accuracy is exactly one-third and its macro-F1 is low. KNN
recovers Aluminium and Copper more often and therefore improves the
class-balanced measures at every tested rate. Its raw accuracy is lower than
the mode baseline, which is an expected trade-off when the method attempts to
represent minority classes rather than always choosing the majority class.

The current confidence policy accepts approximately 66% of masked rows.
Acceptance uses the non-calibrated combined confidence score: weighted
neighbour agreement multiplied by `1 / (1 + mean transformed neighbour
distance)`. Rows below 0.50 remain blank in `material_imputed` and include the
reason `confidence_score_below_threshold`. The retained
`material_prediction` is only a review suggestion for those rows.

Development-only tuning selected the best candidate from `k = 3, 5, 7, 9` by
mean three-fold macro-F1 for each outer seed. The selected candidate was then
refit on all development rows. See `knn_k_selection.csv` for the full tuning
evidence.

The confidence labels and distance evidence are preliminary quality indicators,
not calibrated probabilities. The threshold must be reviewed using held-out
results and client feedback before any operational interpretation.

## Required evidence

- `outputs/PGS219-128/knn_metrics.csv` contains candidate and accepted metrics,
  including per-class recall.
- `outputs/PGS219-128/knn_confusion_matrix.csv` contains candidate and accepted
  confusion matrices.
- `outputs/PGS219-128/knn_predictions.csv` contains prediction, neighbour,
  distance, confidence and abstention evidence for every masked row.
- `outputs/PGS219-128/knn_k_selection.csv` contains development-only tuning
  results for the candidate neighbour counts.
- `outputs/PGS219-128/feature_associations.csv` contains leakage diagnostics;
  the diagnostics use nominal-target normalized mutual information and a
  minimum category support of 25; none of the selected predictors triggered
  the configured warning threshold.
- `outputs/PGS219-128/run_metadata.json` records features, seeds, thresholds,
  mask compatibility and truth handling.

## Limitations and next action

This result demonstrates the method on controlled synthetic data only. The
current generator has weak local material variation, and feeder or geographic
masking has not yet been validated against authorised SAPN data. The next step
is peer review of the feature set and KNN parameters, followed by a comparable
MissForest-style experiment if Sprint 4 requires the full three-method
comparison. Downstream PoF or Weibull-style stability is not claimed until
SAPN provides or approves a fixed reference calculation.
