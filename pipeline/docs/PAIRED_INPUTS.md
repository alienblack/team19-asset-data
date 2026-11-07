# Paired measured and ground-truth files

The primary benchmark uses two source files per scenario:

1. **Measured** — observed values may be noisy; some values are missing.
2. **Ground truth** — complete reference values for the same asset records.

Example:

```bash
python -m src.cli all \
  --asset transformers \
  --measured data/transformers/completion0.9-noise0.05-measured.csv \
  --truth data/transformers/completion0.9-noise0.05-truth.csv \
  --method median_mode \
  --out outputs/transformers_noise05
```

No third masked source file is necessary. The scorer calculates the mask from
**missing cells in the measured file where truth is known**. It partitions
records into fit, validation and held-out test records using the asset
profile's evaluation settings. It saves the manifest and scoring masks for
reproducibility and never exposes test truth to an imputer.

`evaluation/summary.csv` measures imputation on the originally missing test
cells (numerical MAE/RMSE and categorical Macro-F1/accuracy).

`evaluation/observed_difference_summary.csv` separately audits **nonmissing**
measured values against truth. Its `observed_mae` and `difference_rate` can
reflect noise or other measurement error; those are **not** imputation errors.
Operational cleaning is logged, and `all` evaluates the cleaned measured
frame that went into the imputer.

## Identity and ordering

When asset IDs exist, input/truth rows are aligned by identifier, not row
position. Do not use row number as an ID for real asset registers.

The supplied transformer benchmark has no ID. Its provisional profile sets
`id_generation: row_number`, which requires identical row order and row count.
Alignment is checked against `x`, `y`, `install_date`, `model_id`, `altitude`
when those values are observed. These anchor columns are suitable only for
the supplied sample scenario, in which they are not noisy when observed.
For new noise scenarios, update the declared anchors or supply genuine IDs.

## Multiple noise levels

Run each measured/truth pair with its own output directory and retain a
scenario label (asset, completion rate, noise rate, seed). Compare identical
held-out partitions and evaluation protocols across imputation methods.
Dataset properties and observed differences can help explain **why** a
method's performance changes; performance alone cannot establish causality.

The old `evaluate --truth ...` artificial-holdout mode remains as a separate
backward-compatible experiment, but it is not needed for two-file benchmarks.
