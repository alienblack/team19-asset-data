# PGS219-128 Similar Asset KNN Material Imputation

## Purpose

This ticket adds a similar-asset KNN classifier for reconstructing masked
overhead-conductor `material` values. It extends the PGS219-99 mode baselines
without changing the controlled synthetic reference data.

The experiment is a method demonstration using synthetic data. It is not a
claim about SAPN records or production accuracy.

## Features and leakage control

The default predictors are:

- `installation_year`
- `asset_age`
- `voltage_category`
- `location_group`
- `corrosion_zone`
- `distance_to_sea_km`
- `covering`
- `length_km`

`segment_id`, `data_status` and `record_version` are excluded as identifiers or
status fields. `conductor_type` is explicitly excluded because the synthetic
relationship can reveal material directly. `material` is the hidden target and
is never a predictor.

Numeric predictors use development-set median imputation and standardisation.
Categorical predictors use development-set most-frequent imputation and
one-hot encoding. The preprocessing and neighbour index are fitted only on
the development partition.

## Evaluation design

The implementation reuses `split_indices` and `select_mask_positions` from the
PGS219-99 baseline module. It therefore uses the same 80/20 split, five seeds
(`101, 202, 303, 404, 505`), mask rates (10%, 30%, 50%) and mask-seed formula.
The hidden evaluation material values are retained only in evidence outputs.

The KNN classifier uses seven neighbours and inverse-distance voting. For each
row it records the predicted material, weighted neighbour agreement, mean and
maximum neighbour distance, a distance evidence score, quality flag, and any
abstention reason. The distance score is an evidence measure, not a calibrated
probability.

An initial confidence policy is used only for transparent experimentation:

- `High`: weighted agreement at least 0.80;
- `Medium`: weighted agreement from 0.60 to below 0.80;
- `Low`: weighted agreement from 0.50 to below 0.60;
- `Insufficient evidence`: weighted agreement below 0.50, so the candidate is
  retained as a suggestion but `material_imputed` is left blank.

These flags are internal preliminary quality labels, not SAPN standards or
calibrated per-asset probabilities.

## Metrics

Candidate predictions are scored against the hidden truth using accuracy,
balanced accuracy, macro-F1 and a labelled confusion matrix. Accepted
predictions are also scored after abstentions are removed. Coverage is the
proportion of masked rows accepted for imputation. The output also allows
review of abstention rates, error cases and feature-association warnings.

## Run command

From the repository root:

```bash
PYTHONPATH=src python -m team19_asset_data.overhead_conductors.material_knn \
  --input data/synthetic/overhead_conductors/material_model_ready_data.csv \
  --output-dir outputs/PGS219-128
```

## Evidence files

- `knn_metrics.csv`: per-seed and mask-rate candidate and accepted metrics.
- `knn_metrics_summary.csv`: metrics averaged by mask rate.
- `knn_confusion_matrix.csv`: candidate and accepted-prediction confusion matrices.
- `knn_predictions.csv`: prediction, neighbour, distance, confidence and
  abstention evidence for every masked row.
- `knn_imputed_material_seed101_rate30.csv`: convenient sample for review.
- `feature_associations.csv`: numeric correlation and categorical purity
  diagnostics for the selected predictors.
- `mask_manifest.csv`: the shared masked rows and hidden truth.
- `run_metadata.json`: features, seeds, thresholds, leakage exclusions and
  reproducibility details.

## Limitations and next step

The synthetic generator may not contain meaningful feeder-level or local
material variation. A good KNN score on this data therefore does not establish
real-world performance. Results should be checked under random, feeder-based
and geographical masking when authorised data and the required group fields
are available. Downstream PoF or Weibull-style stability must be assessed
separately against a fixed reference calculation agreed with SAPN.
