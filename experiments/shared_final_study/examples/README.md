# Synthetic example for the dataset workbench

These 240 records are a small, newly generated demonstration, not SAPN data and
not the original research benchmark. They show that the workbench accepts
different column names and five text-valued material categories.

- `example_assets.csv`: records with artificial gaps.
- `example_truth.csv`: their complete reference values, in identical row order.
- `example_config.json`: a three-method setup with explicit field tolerances.
- [Generated example report](generated_report/report.html): a completed run with
  its CSV exports and embedded figures.
- [PDF preview](generated_report/report.pdf): a printed copy of that example report.

The numeric fields share deliberately simple relationships. Material labels are
drawn independently, so strong material recovery is not expected. A mode-like
classification result is useful evidence of weak signal, not a failed pipeline.

Use the workbench's **Try the synthetic asset example** button, or run:

```sh
python run_study.py --mode dataset --dataset examples/example_assets.csv \
  --truth examples/example_truth.csv --config examples/example_config.json \
  --out outputs/example-with-truth
```

In PowerShell or Command Prompt, enter that command on one line. Choose another
output folder for the next run; an existing run is not silently overwritten.

## Transformer product walkthrough

`transformer_workbench_config.json` is the reusable configuration from the
first product run on the bundled transformer data. It uses the existing
20%-gap, 5%-noise measured CSV and its matching truth, evaluates 1,000 rows
with three folds, adds 20% extra scattered masking and adds no further noise.
It sets explicit per-field tolerances, groups and bounds and excludes the
random placebo from mean numeric skill. These are example evaluation settings,
not a calibrated auto-fill policy.

See the [README walkthrough](../README.md#try-the-supplied-transformer-dataset)
for the CLI command or corresponding browser steps. The browser currently
accepts CSV uploads; it does not import this JSON automatically.
