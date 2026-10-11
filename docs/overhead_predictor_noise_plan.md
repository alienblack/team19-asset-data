# Controlled Predictor Noise Plan

## Purpose

This experiment creates observed-data views with controlled predictor errors
while preserving the clean complete reference as ground truth. It measures the
effect of data-quality problems without changing the `material` evaluation
target or the leakage-prone `conductor_type` field.

The outputs measure robustness under documented synthetic assumptions. They do
not estimate error rates in the SAPN asset register.

## Literature basis

- [Picket: Guarding Against Corrupted Data in Tabular Data](https://arxiv.org/abs/2006.04730)
  motivates testing random, systematic and adversarial-style corruption at both
  training and inference boundaries.
- [Framework for Testing Robustness of Machine Learning-Based Classifiers](https://pmc.ncbi.nlm.nih.gov/articles/PMC9409965/)
  motivates repeated perturbations, multiple noise types and reporting average
  performance and variability.
- [Robustness under noise](https://riunet.upv.es/server/api/core/bitstreams/b37ffc55-7765-4d0d-a275-1f690aeabf30/content)
  motivates targeted perturbation of important attributes and prediction
  stability analysis.
- [Assessing Robustness of Machine Learning Models using Covariate Perturbations](https://arxiv.org/abs/2408.01300)
  motivates global perturbation summaries and local instability checks.
- [Synthetic data generation and evaluation](https://www.nature.com/articles/s41598-024-62102-2)
  motivates checking distributions, associations and practical validity after
  synthetic-data transformations.

## Scenarios

| Scenario | Purpose |
|---|---|
| `N00` | Clean reference control. |
| `N01_05`, `N01_10`, `N01_20` | Random bounded relative noise in numeric predictors. |
| `N02_year_1`, `N02_year_2` | Discrete installation-year recording errors with linked age recalculation. |
| `N03_categorical` | Replacement errors using valid categorical values. |
| `N04_systematic_rural` | Consistent numeric bias for one operating-area group. |
| `N05_group_dependent` | Different numeric error rates by location group. |

## Data contract

- `material` is never modified because it is the hidden evaluation target.
- `conductor_type` is never modified because it directly reveals material and
  is excluded from the material-imputation feature set.
- `segment_id`, `feeder_id`, `data_status` and `record_version` are protected.
- `asset_age` is recalculated from the observed `installation_year`; it is not
  independently perturbed.
- Numeric values are clipped to the documented valid ranges.
- Categorical replacements use only valid domain values.
- Every scenario writes observed data, a clean truth copy and JSON metadata.

## Validation evidence

The runner records changed-cell counts, realised rates, SHA-256 hashes, input
and output paths, protected columns and validation status. Structural checks
verify row counts, IDs, schemas, protected fields, category domains, numeric
ranges and the installation-year/age relationship.

## Reproducibility

```bash
PYTHONPATH=src python -m team19_asset_data.overhead_conductors.run_noise \
  --input data/synthetic/overhead_conductors/overhead_conductors_complete_reference.csv \
  --config config/overhead_conductor_noise_rules.yaml \
  --output-dir outputs/PGS219-predictor-noise
```

The output directory is ignored by Git. The clean reference is not overwritten.

## Scope boundary

This ticket creates and validates noise scenarios. It does not compare
imputation methods, implement a new advanced imputer or claim downstream
PoF/Weibull stability. Those experiments should consume these observed views in
separate tickets.
