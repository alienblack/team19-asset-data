# Experiments

Everything under `experiments/` (research branch; a sibling of `pipeline/`,
not shipped inside it) is research tooling. It is never imported by the
package, it writes to `experiments/<name>/outputs/` (git-ignored), and the
result tables kept in the repository under `experiments/<name>/results/` are
the runs quoted below. All commands are run from the repository root.

> **Status of the quoted figures.** The tuning run (section 2) and the method
> comparison (section 3) both partitioned `conductors_synthetic_masked.csv`
> 75 / 25 by identifier with `split_seed 219`, so the comparison's
> `unseen_records` test records are exactly the 2,500 records on which the
> hyperparameters were selected, and the whole-table protocols score those
> records too. The tables are kept as the record of those runs, but they are
> **not** clean test estimates for the selected hyperparameters. The
> pipeline's evaluation mode (section 6; `oc-pipeline evaluate`) partitions by
> feeder into fit / validation / test, reserves the validation partition for
> tuning and scores only the test partition; its figures supersede section 3
> wherever the two are compared. All figures are on synthetic data and
> describe the generator, not SA Power Networks records.

## 1. Synthetic data

`experiments/synthetic_data/` holds the generator, the masking engine and the
predictor-noise engine (moved unchanged from the development branch apart
from their default paths).

```bash
# complete reference + model-ready file for a relationship-strength profile
python experiments/synthetic_data/generate_synthetic_data.py --strength-profile strength_100
#   -> experiments/synthetic_data/outputs/synthetic/strength_100/{overhead_conductors_complete_reference,material_model_ready_data}.csv

# masking rule MF01: 20% of every model attribute hidden independently (asset_age linked to installation_year)
python experiments/synthetic_data/run_masking.py --rules MF01 \
    --input experiments/synthetic_data/outputs/synthetic/strength_100/material_model_ready_data.csv
#   -> experiments/synthetic_data/outputs/masked/MF01/{masked_MF01,truth_MF01}.csv + metadata

# predictor-noise scenarios (N00 clean, N01_05 ... N01_100 increasing random numeric noise)
python experiments/synthetic_data/run_noise.py --scenarios N00 N01_10 N01_30 N01_50
#   -> experiments/synthetic_data/outputs/noise/<scenario>/{observed,truth}.csv + reports

# documented data-quality defects for the pipeline demonstration
python experiments/synthetic_data/make_dirty_sample.py \
    --input data/overhead_conductors/conductors_synthetic_masked.csv --output data/overhead_conductors/conductors_synthetic_dirty.csv
```

Configuration lives in `experiments/synthetic_data/config/`
(`overhead_conductors/generation.yaml`, `relationship_strength/*.yaml`,
`masking_rules.yaml`, `noise_rules.yaml`).

## 2. Hyperparameter tuning

```bash
python experiments/hyperparameter_tuning/run_tuning.py \
    --input data/overhead_conductors/conductors_synthetic_masked.csv \
    --objective all_attributes --repeats 3 --mask-rate 0.25 [--write-config]
```

Protocol: 75/25 split by identifier (optionally by feeder with
`--group-column feeder_id`); forests fitted on the 75%; on the 25%, 25% of
the observed cells of every model attribute hidden three times with different
seeds (categorical attributes stratified by class); the objective
`0.5 · mean(MAE/IQR_train)` over numeric attributes `+ 0.5 · mean(1 − macro‑F1)`
over categorical attributes, averaged over the three masks; the training
median/mode scored on the same cells as the noise floor. Selection takes the
candidate with the fewest tree nodes among those whose mean objective is within
one standard deviation of the best (`simplicity_tolerance`).

Result on the sample (`experiments/hyperparameter_tuning/results/`):

| max_depth | min_samples_leaf | objective (mean ± std) | tree nodes |
|---|---|---|---|
| 16 | 5 | 0.589 ± 0.014 | 1,649,408 |
| 8 | 5 | 0.594 ± 0.011 | 584,760 |
| 16 | 10 | 0.595 ± 0.014 | 793,116 |
| **8** | **10** | **0.597 ± 0.011** (selected) | **429,416** |
| None | 5 | 0.598 ± 0.016 | 1,872,128 |
| None | 10 | 0.598 ± 0.013 | 821,882 |
| 16 | 20 | 0.604 ± 0.008 | 381,958 |
| None | 20 | 0.604 ± 0.009 | 382,510 |
| 8 | 20 | 0.604 ± 0.009 | 290,854 |
| median/mode baseline | | 0.672 ± 0.009 | |

All nine candidates lie within 0.015 of each other and 0.07 above the
baseline: the forest helps, and the exact depth/leaf setting barely matters
on this data. `max_depth=8, min_samples_leaf=10` was selected and written to
`config/overhead_conductors/imputation.yaml` (`selected_hyperparameters`, with `tuning_provenance`).

`--write-config` replaces everything below the marker line
`# >>> managed by experiments/hyperparameter_tuning/run_tuning.py >>>` in
`config/overhead_conductors/imputation.yaml` and keeps a `.bak` copy.

## 3. Method comparison

```bash
python experiments/model_comparison/run_comparison.py \
    --input data/overhead_conductors/conductors_synthetic_masked.csv \
    --truth data/overhead_conductors/conductors_synthetic_complete.csv --repeats 3 [--set n_jobs=-1]
```

Three protocols on the masked sample (10,000 rows, 20% of every attribute
missing):

- `clean_truth` — the 2,000 originally masked cells per attribute scored against the clean reference (the real benchmark on synthetic data);
- `hidden_observed` — a further 10% of observed cells hidden (×3), as the pipeline's validation stage does;
- `unseen_records` — `predictive_missforest` fitted on 75% of the records and scored on hidden cells of the other 25% (the fit-once/apply-later situation).

Result (`experiments/model_comparison/results/comparison_summary.csv`), MAE for numeric and macro-F1 for categorical attributes:

| protocol | method | year MAE | distance MAE | voltage F1 | location F1 | corrosion F1 | material F1 | covering F1 |
|---|---|---|---|---|---|---|---|---|
| clean_truth | median_mode | 10.39 | 53.05 | 0.117 | 0.231 | 0.246 | 0.280 | 0.497 |
| clean_truth | missforest | 10.60 | 37.30 | 0.361 | 0.505 | 0.705 | 0.316 | 0.497 |
| clean_truth | predictive_missforest | 10.58 | 37.23 | 0.362 | 0.505 | 0.709 | 0.312 | 0.497 |
| hidden_observed | median_mode | 10.50 | 53.62 | 0.114 | 0.227 | 0.240 | 0.282 | 0.497 |
| hidden_observed | missforest | 10.93 | 43.76 | 0.330 | 0.496 | 0.650 | 0.324 | 0.497 |
| hidden_observed | predictive_missforest | 10.92 | 43.77 | 0.328 | 0.494 | 0.647 | 0.322 | 0.497 |
| unseen_records | median_mode | 10.71 | 52.64 | 0.114 | 0.227 | 0.241 | 0.282 | 0.498 |
| unseen_records | predictive_missforest | 10.66 | 46.21 | 0.347 | 0.435 | 0.709 | 0.280 | 0.498 |

Reading:

- The two forest methods are indistinguishable, as expected: the predictive version runs the same algorithm with a fixed number of passes. Its value is operational (saved, auditable, re-applicable), not a different accuracy.
- Both forest methods clearly beat the reference where the attributes carry information (`distance_to_sea_km`, `corrosion_zone`, `location_group`, `voltage_category`).
- `installation_year` is **not** reconstructed better than the median by any method (the median is marginally better), and `material` only slightly. This is a property of the synthetic generator, not of the methods: `installation_year` is drawn first and influences the other attributes only at a coarse era level, so the seven attributes contain almost no information about it (a gradient-boosting model fitted on the *complete* reference reaches MAE 10.6 vs 10.6 for the median). See `docs/LIMITATIONS.md`. `covering` is 99% `Bare`, so its macro-F1 is pinned near 0.5 for every method.
- `hidden_observed` tracks `clean_truth` closely for every method, which is the justification for using it as the pipeline's validation protocol on real data.
- Fitting on 75% and applying to the unseen 25% gives the same picture, with the small-sample noise of 200 cells per attribute.

Per-class recall is in `comparison_per_class.csv`. On the synthetic sample
the forest methods recover only about 3% of the `Copper` and `Aluminium`
cells (against 0% for the reference, which always answers `Steel`): the
material signal in this generator is too weak for minority classes to be
predicted, and the macro-F1 gain over the reference comes almost entirely
from a handful of correct minority predictions. On real data this per-class
table is the first thing to check before trusting imputed materials.

## 4. Noise robustness

```bash
python experiments/synthetic_data/run_noise.py --scenarios N00 N01_05 N01_10 N01_20 N01_30 N01_50
python experiments/noise_robustness/run_noise_robustness.py --repeats 2
```

Hides the same 10% of observed cells (same seed, drawn on the clean truth so
the mask is identical in every scenario) in each noisy table, fills with each
method and scores against the clean truth. Output: metrics per scenario,
summary table and a line chart of the target metrics against noise level
(`noise_robustness.png`). Not included in `results/`; run it when the noise
study is in scope.

## 5. Reproducing the example run

```bash
cd pipeline
oc-pipeline all --input data/overhead_conductors/conductors_synthetic_dirty.csv --out outputs/sample_run
oc-pipeline evaluate --truth data/overhead_conductors/conductors_synthetic_complete.csv \
                     --input data/overhead_conductors/conductors_synthetic_masked.csv --out outputs/sample_run
oc-pipeline report --run outputs/sample_run
```

`notebook/pipeline_overhead_conductors.ipynb` runs the same stages on the same files into
`outputs/notebook_demo/`. Outputs are git-ignored; every run is reproducible
from the seeds recorded in `run_metadata.json`.

## 6. Evaluation mode on the sample (pipeline, record-disjoint test partition)

`oc-pipeline evaluate` on the masked sample against the complete reference
(`evaluation.split` 60 / 15 / 25 by feeder, seed 219: fit 6,012, validation
1,450, test 2,538 records; the masked file's own gaps are scored, so the
masking configuration is rule MF01 from
`data/overhead_conductors/conductors_synthetic_masked_metadata.json`). Only hidden cells
of the **test** partition are scored (522 year cells, 516 material cells;
486-535 per attribute). MAE for numeric attributes, macro-F1 for categorical:

| protocol | method | year MAE | distance MAE | voltage F1 | location F1 | corrosion F1 | material F1 | covering F1 |
|---|---|---|---|---|---|---|---|---|
| unseen_records | predictive_missforest | 10.93 | 35.32 | 0.347 | 0.536 | 0.674 | 0.287 | 0.498 |
| unseen_records | median_mode | 10.73 | 50.36 | 0.112 | 0.234 | 0.250 | 0.281 | 0.498 |
| transductive | predictive_missforest | 10.82 | 36.60 | 0.376 | 0.517 | 0.730 | 0.296 | 0.498 |
| transductive | median_mode | 10.73 | 50.27 | 0.112 | 0.234 | 0.250 | 0.281 | 0.498 |

`unseen_records`: fitted on the fit partition, applied to the test partition
(fit-once/apply-later). `transductive`: `fit_transform` on fit + test
together. The reading is the same as in section 3 - the forest clearly beats
the reference where the attributes carry information, `installation_year` is
not reconstructed better than the median (28-30 % of hidden years within
±5 years for the forest against 43 % for the median, whose predictions
cluster at the centre), and `material` only slightly - now on records that
played no part in hyperparameter selection. Per-class recall is in
`outputs/<run>/evaluation/per_class.csv`; every scored cell with its truth and
prediction is in `scored_cells.csv`.
