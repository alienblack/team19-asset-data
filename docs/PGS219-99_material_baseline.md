# PGS219-99: Explainable Material Baseline

## Purpose

This ticket establishes transparent baseline methods for reconstructing missing
`material` values for the overhead-conductor stream. The baselines provide a
defensible reference point before the team tests more complex statistical or
machine-learning methods.

The experiment uses the existing synthetic complete reference at
`data/synthetic/overhead_conductors/material_model_ready_data.csv`. No new
dataset is generated or modified by this ticket. The data is artificial and
must not be described as SAPN data.

## Methods

1. **Global mode:** replace a missing material with the most frequent material
   in the development partition.
2. **Feeder-group mode:** replace a missing material with the most frequent
   material for the feeder in the development partition. If a feeder is not
   present in development data, use the development global mode as a documented
   fallback.

`conductor_type` is intentionally excluded from this experiment. The richer
reference file contains a highly predictive relationship between conductor
type and material, which could make the benchmark unrealistically easy. The
current material-model input does not include that field.

## Evaluation design

- A reproducible 80/20 development/evaluation split is created for each of
  five seeds: 101, 202, 303, 404 and 505.
- Known material values are masked only in the evaluation partition at 10%,
  30% and 50% rates.
- Both methods use exactly the same masked rows for each seed and rate.
- Baseline modes are fitted only on development rows.
- The original values are retained in evaluation evidence, while the
  model-facing masked sample contains blank material values.
- Accuracy, balanced accuracy and macro-F1 are reported. A labelled confusion
  matrix is also saved because the material classes are imbalanced.

Run from the repository root:

```bash
PYTHONPATH=src python -m team19_asset_data.overhead_conductors.material_baseline \
  --input data/synthetic/overhead_conductors/material_model_ready_data.csv \
  --output-dir outputs/PGS219-99
```

## Evidence files

The command creates local, ignored experiment evidence under
`outputs/PGS219-99/`:

- `baseline_metrics.csv`: one row per seed, mask rate and method.
- `confusion_matrix.csv`: counts for each actual/predicted material class.
- `mask_manifest.csv`: the controlled evaluation rows and their hidden truth.
- `masked_material_seed101_rate30.csv`: model-facing example with material
  blanked on the selected rows.
- `imputed_material_baselines_seed101_rate30.csv`: predictions paired with
  truth for evaluation.
- `run_metadata.json`: seed, split, target, methods and leakage-control notes.

## Interpretation and limitations

These baselines are intentionally simple. A global mode can look reasonable on
an imbalanced dataset while performing poorly for minority classes, so
balanced accuracy and macro-F1 are necessary alongside accuracy. If feeder
group mode matches global mode, this means the current synthetic generator
does not provide useful feeder-level material variation; it is a result to
record, not a reason to alter the data after seeing the outcome.

The experiment measures imputation accuracy on controlled known values. It
does not claim improvement to a downstream Probability of Failure or Weibull
model because an agreed SAPN reference model and authorised data are not yet
available. Downstream stability should be assessed later using a fixed
reference model and the same masks.
