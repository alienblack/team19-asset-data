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
The predictors were installation year, asset age, voltage category, location
group, corrosion zone, distance to sea, covering and length.

## Mean results by mask rate

The KNN candidate metrics are averaged over five seeds. Baseline values are
from the existing PGS219-99 experiment on the same masks.

| Mask rate | Method | Accuracy | Balanced accuracy | Macro-F1 | Coverage |
|---:|---|---:|---:|---:|---:|
| 10% | Mode baseline | 0.7560 | 0.3333 | 0.2869 | 1.0000 |
| 10% | KNN candidate | 0.7050 | 0.3788 | 0.3794 | 0.9000 |
| 30% | Mode baseline | 0.7350 | 0.3333 | 0.2824 | 1.0000 |
| 30% | KNN candidate | 0.7047 | 0.3934 | 0.3981 | 0.9157 |
| 50% | Mode baseline | 0.7304 | 0.3333 | 0.2814 | 1.0000 |
| 50% | KNN candidate | 0.6988 | 0.3923 | 0.3959 | 0.9052 |

## Interpretation

The mode baseline has higher raw accuracy because Steel is the dominant class,
but its balanced accuracy is exactly one-third and its macro-F1 is low. KNN
recovers Aluminium and Copper more often and therefore improves the
class-balanced measures at every tested rate. Its raw accuracy is lower than
the mode baseline, which is an expected trade-off when the method attempts to
represent minority classes rather than always choosing the majority class.

The current confidence policy accepts approximately 90% to 92% of masked rows.
Rows below 0.50 weighted neighbour agreement remain blank in
`material_imputed` and include the reason
`weighted_neighbor_agreement_below_threshold`. The retained
`material_prediction` is only a review suggestion for those rows.

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
- `outputs/PGS219-128/feature_associations.csv` contains leakage diagnostics;
  none of the selected predictors triggered the configured warning threshold.
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
