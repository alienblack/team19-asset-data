# Configurable model execution and optional tuning

The pipeline supports two operational modes for each asset: **train a new model**
and **load a previously fitted model**. The switches live in the relevant
`config/<asset>/imputation.yaml`, alongside the method hyperparameters. This
works for the CLI full pipeline and the `notebook/` walkthroughs.

## Example

```yaml
model:
  mode: train_new  # train_new | use_saved
  path: outputs/models/overhead_conductors/predictive_missforest.joblib
  save_after_training: true

tuning:
  enabled: false
  update_selected: false
  objective: all_attributes
  repeats: 3
  mask_rate: 0.25
  mask_seed: 219
  grid:
    max_depth: [8, 16, null]
    min_samples_leaf: [5, 10, 20]

selected_hyperparameters:
  predictive_missforest:
    max_depth: 8
    min_samples_leaf: 10
```

## Train a new model (`train_new`)

1. Load and validate the measured/partially missing CSV using asset-specific
   schema, quality, cleaning and preprocessing rules.
2. If `tuning.enabled: true`, partition the **measured** records into fit,
   validation and test using `pipeline.yaml`'s evaluation split. Fit candidates
   on the fit partition. Deliberately mask known values in validation, predict
   and score the masked values; the independent test partition is not used for
   selecting hyperparameters. Write `evaluation/tuning_candidates.csv`,
   `tuning_repeats.csv`, `tuning_baseline.csv` and the partition manifest.
3. With `tuning.update_selected: true`, use the selected grid parameters for
   the final model, then update `selected_hyperparameters.predictive_missforest`
   and `tuning_provenance` in the YAML. A copy of the previous YAML is saved as
   `evaluation/imputation_before_tuning.yaml`. With `update_selected: false`,
   tuning is diagnostic only; the final model uses the already selected values.
4. Fit the final model on the supplied measured data (never the separate truth
   file), fill missing values, postprocess, validate and report. Optionally
   save the fitted model to `model.path`, with a provenance sidecar.

**Important:** `save_after_training: true` overwrites the configured model
path after successful fitting. Choose a run/version-specific path to retain
older model versions. Set `save_after_training: false` to avoid saving. A
command-line method override uses the run-local model path rather than
overwriting the configured default model.

## Apply a saved model (`use_saved`)

1. Set `model.mode: use_saved` and point `model.path` to the saved `.joblib`.
2. Ensure `tuning.enabled: false`; no fit or tuning runs in this mode.
3. The runner loads the model, checks feature/constraint compatibility and
   fitting-record provenance, then calls `transform()` on the current measured
   file. Missing or incompatible models fail clearly; they never silently
   trigger a retrain.
4. The resulting `final_imputed.csv`, provenance, diagnostics, validation and
   report are produced in the usual run directory.
5. If a paired complete truth CSV is supplied, the pipeline scores the actual
   saved-model output on the still-missing prepared input cells. Outputs are
   `evaluation/metrics.csv`, `summary.csv` and `saved_model_mask.csv`. This is
   **not** a new independent train/test benchmark, and a record overlap with
   the model's fitting data must be disclosed. For independent generalisation
   claims, apply to records that were not used to fit the saved model.

Only methods that implement persistence can be loaded and applied. The
original, transductive MissForest cannot use `use_saved`.

## Tuning principles and approvals

- Tuning currently supports `predictive_missforest` only; other methods must
  be tuned using their own separately implemented selectors.
- `objective: all_attributes` scores each numerical attribute using MAE
  scaled by training IQR, and each categorical attribute with Macro-F1.
  `objective: year_material` remains available for overhead conductors.
- The separate truth CSV is never passed to candidate training or the main
  fit. The tuning score is only an internal validation score; keep an
  independent held-out test dataset for final comparison.
- `update_selected: true` requires **development** execution mode because
  rewriting the YAML invalidates its approved SHA-256. Review and re-approve
  the changed config before running release mode.
- `tuning_provenance` records the selected settings, mask seeds, data split
  sizes and objective values. It is not evidence of performance on real SAPN
  data.
- If a model exists already at `model.path`, turning tuning on does not reuse
  it. Tuning selects configuration for a newly fitted model.

## Reproducible workflow

1. Configure `train_new`, optionally tune with `update_selected: true` in
   development, and inspect `evaluation/tuning_*` files.
2. Review the model's validation quality and re-approve `imputation.yaml`.
3. Choose a fixed `model.path` and run `train_new` with tuning disabled on the
   approved settings to produce a controlled model artifact.
4. Switch to `use_saved` to impute later datasets with the identical frozen
   model; store each run's outputs under a separate run directory.
5. Compare MAE / Macro-F1 on eligible held-out cells; include missingness
   rates, class balance, and data-shift limitations in the report.

Existing explicit `oc-pipeline fit --model ...` and `oc-pipeline apply --model ...`
commands remain available, independent of these full-pipeline config switches.
