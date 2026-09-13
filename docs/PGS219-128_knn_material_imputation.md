# PGS219-128 Similar-Asset KNN Material Imputation

## Purpose

This ticket implements a transparent similar-asset KNN classifier for
reconstructing masked overhead-conductor `material` values. It extends the
PGS219-99 baseline experiment without changing the controlled synthetic
reference data.

The experiment is a method demonstration using artificial data. It is not a
claim about SAPN records, production accuracy, or a production-ready
imputation threshold.

## Features and leakage control

The default predictors are:

- `installation_year`
- `voltage_category`
- `location_group`
- `corrosion_zone`
- `distance_to_sea_km`
- `covering`
- `length_km`

`asset_age` is excluded because it is deterministically related to
`installation_year` in the synthetic data; retaining both would double-weight
age in Euclidean distance. `segment_id`, `data_status` and `record_version`
are excluded as identifier or status fields. `conductor_type` is explicitly
excluded because its synthetic relationship can reveal material directly.
`material` is the hidden target and is never a predictor. Missing target
values in development data are rejected rather than converted to string
labels.

Numeric predictors use development-set median imputation and standardisation.
Categorical predictors use development-set most-frequent imputation and
one-hot encoding. The preprocessing and neighbour index are fitted only on
the development partition.

## Evaluation design

The implementation reuses `split_indices` and `select_mask_positions` from the
PGS219-99 baseline module. It therefore uses the same 80/20 split, five seeds
(`101, 202, 303, 404, 505`), mask rates (10%, 30%, 50%) and mask-seed formula.
The hidden evaluation material values are retained only in evidence outputs.

For each outer seed, candidate values `k = 3, 5, 7, 9` are compared using
three-fold stratified validation inside the development partition. The value
with the highest mean development macro-F1 is selected, with the smaller
candidate used as the tie-break. The selected value is then refit on all
development rows and applied to the masked evaluation rows.

Inverse-distance voting is used for ordinary neighbours. Exact transformed
feature matches are handled explicitly and receive an equal exact-match vote,
avoiding an artificial `1 / 1e-12` weight.

## Evidence and acceptance policy

For each prediction the output records the candidate material, weighted
neighbour agreement, mean and maximum neighbour distance, bounded distance
evidence, combined `confidence_score`, quality flag, coverage decision and
fallback reason. `weighted_neighbor_agreement` measures label agreement only;
it is not a calibrated probability. The non-calibrated confidence score is:

```text
weighted_neighbor_agreement * 1 / (1 + mean transformed neighbour distance)
```

An exact feature match receives a distance evidence score of 1.0. A row is
accepted only when the combined confidence score is at least `0.50`; otherwise
`material_imputed` is blank, `abstained` is true, and the fallback reason is
`confidence_score_below_threshold`. The quality labels are preliminary review
flags: High at `>= 0.65`, Medium at `0.50-<0.65`, and Insufficient evidence
below `0.50`.

## Leakage diagnostics and metrics

Numeric and categorical predictors are assessed with normalized mutual
information against the nominal multiclass target. No arbitrary numeric
encoding of Steel, Copper and Aluminium is used. Categorical purity is also
reported only for categories with at least 25 observations, reducing the risk
that a tiny category appears perfectly predictive by chance.

Candidate predictions are scored against hidden truth using accuracy, balanced
accuracy, macro-F1, per-class recall and a labelled confusion matrix. Accepted
predictions are scored separately after abstentions are removed. Coverage is
the proportion of masked rows accepted for imputation.

## Run command

From the repository root:

```bash
PYTHONPATH=src python -m team19_asset_data.overhead_conductors.material_knn \
  --input data/synthetic/overhead_conductors/overhead_conductors_complete_reference.csv \
  --output-dir outputs/PGS219-128
```

## Evidence files

- `knn_metrics.csv`: per-seed and mask-rate candidate and accepted metrics.
- `knn_metrics_summary.csv`: metrics averaged by mask rate.
- `knn_confusion_matrix.csv`: candidate and accepted confusion matrices.
- `knn_predictions.csv`: prediction, neighbour, distance, confidence and
  abstention evidence for every masked row.
- `knn_k_selection.csv`: development-only candidate-k scores and selections.
- `knn_imputed_material_seed101_rate10pct.csv`: sample output generated from
  the first configured seed and rate; the filename changes with those inputs.
- `feature_associations.csv`: nominal-target association and support-aware
  leakage diagnostics for selected predictors.
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
