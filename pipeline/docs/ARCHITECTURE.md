# Architecture

## 1. Stages and data flow

The package lives directly under `pipeline/src/`, with asset-specific YAML
profiles in `config/<asset>/`, sample data in `data/<asset>/`, and the two
controller notebooks in `notebook/`. Shared evaluation is independent of the
algorithm folder; see `PAIRED_INPUTS.md` and `ADDING_METHODS.md`.

```
config/<asset>/*.yaml + config/<asset>/approvals.json
   │  configuration.manager.ConfigurationManager   parse, cross-check, compare SHA-256 with approvals;
   │                                                release mode refuses to run anything else on an invalid
   ▼                                                or unapproved file (recorded in run_metadata.json)
source CSV
   │  ingestion.stage.ingest_csv          schema-free: structure, checksum, every value as text (ingestion_report.json)
   │  ingestion.loader.load_records       column_mapping applied, empty cells -> missing   (text frame)
   ▼
quality.validator.validate_records       quality_report.csv (errors stop the run)
   │
   ▼
quality.cleaner.plan_changes / clean_records   cleaned.csv + cleaning_log.csv (+ dropped_rows.csv)   typed frame
   │                                      rule families gated by approval; unapproved families are skipped and listed
   ├──► profiling.profiler / visualization   profile/*.csv, profile/*.png   (profile_dataset: schema-free EDA for the notebook)
   │
   ▼
preprocessing.prepare.prepare_for_imputation
   │   derived_fields.recover_sources     installation_year = snapshot_year - asset_age where only the age is known
   │                                      (range-checked; recovery_log.csv; provenance "derived")
   │   feature_selection.select_model_frame   roles checked against the schema; forbidden columns never reach a method
   ▼                                      imputation_input.csv
imputation.registry.build_imputer        median_mode | missforest | predictive_missforest   (+ reference method)
   │   fit_transform(X)  ->  ImputationOutput(imputed, diagnostics, metadata)
   │   persistence.save_imputer           model.joblib + .provenance.json (fingerprint of columns, constraints, fitting records)
   ▼
postprocessing.finalise_output
   │   derived_fields.derive_fields       asset_age filled / recomputed from the (imputed) year
   │   loader.finalise_output_types       integers as integers
   │   provenance.build_cell_provenance   cell_provenance.csv, provenance_summary.csv  (observed / derived / imputed / unresolved)
   │   integrity_checks.check_integrity   observed values unchanged, constraints hold - a failure raises
   ▼                                      final_imputed.csv, diagnostics.csv, convergence.csv, derived_fields_log.csv
evaluation.*  (operational checks)       validation/integrity_checks.csv, distribution_checks.csv,
   │                                      reconstruction_{metrics,summary,per_class}.csv, low_confidence_cells.csv, figures/
   ▼
reporting.report_generator               report.html

evaluation.protocol  (evaluation MODE, needs a complete reference; `oc-pipeline evaluate`)
    build_evaluation_inputs              feeder-grouped fit / validation / test partitions (partition_manifest.csv),
                                         masks inside partitions or the masked file's own gaps (evaluation_mask_repeat*.csv)
    run_protocols                        unseen_records (fit on fit, transform test) and transductive (fit_transform fit+test);
                                         only TEST cells are scored -> evaluation/{metrics,per_class,summary,scored_cells}.csv
```

`pipeline.py` orchestrates the stages; `cli.py` exposes them as
`oc-pipeline` commands and `notebook/pipeline_overhead_conductors.ipynb` calls the same `run_*` functions.
Stages communicate only through files in the run directory, so any stage can
be rerun alone (`validate --run`, `report --run`) and every number in the
report has a CSV behind it. `run_metadata.json` accumulates one section per
stage (configuration state and hashes, settings, versions, timings,
summaries); `run.log` records the stages as they ran.

### Where the leakage guard sits

`data_schema.yaml` lists the seven **model attributes** (`model_attributes`,
of which `installation_year` and `material` are the `primary_targets`) and
every other column (`forbidden_predictors`, including `asset_age`,
`conductor_type`, `length_km` and the identifiers). `imputation.yaml`
declares the same columns as `columns.numeric` + `columns.categorical`.
`feature_selection` cross-checks the two before any method is built and
raises `LeakageError` if a forbidden column would be used; the schema parser
itself refuses a schema that leaves a non-model column off the forbidden
list. `asset_age` is used in exactly one deterministic direction - to
*recover* a missing `installation_year` before imputation (`recover_source`)
- and is recomputed after imputation wherever the year was imputed, so a
stale age cannot contradict an imputed year. In evaluation mode the same link
is honoured in the masks: hiding a year hides its age, and recovered cells
are never scored.

### Configuration ownership and approval

Each fact has one owner: column ranges live in `data_schema.yaml`
(`imputation.yaml` no longer repeats them; `load_imputation_config` builds the
constraints from the schema), cleaning policies in `cleaning_rules.yaml`,
method hyperparameters in `imputation.yaml`, run settings in `pipeline.yaml`.
`config/overhead_conductors/approvals.json` records who approved each file at which SHA-256.
`oc-pipeline config check` validates and cross-checks the four files;
`config approve` records approval (refused for a file with error-level
findings); any later edit turns the state to `changed` and release mode
refuses to run until it is re-approved. Approvals shipped with the synthetic
sample are Team 19's, for that data only - they are not SAPN approvals.

## 2. Package map

| Module | Responsibility |
|---|---|
| `configuration/approval.py` | SHA-256 approval records (`config/overhead_conductors/approvals.json`): `approved` / `changed` / `missing` / `unapproved` |
| `configuration/manager.py` | `ConfigurationManager`: parse the four files, cross-file checks, approval states, release gate, review tables |
| `ingestion/stage.py` | schema-free ingestion: structure inspection, checksum, text frame, `ingestion_report.json` |
| `ingestion/schema.py` | `DataSchema` model and validation of `data_schema.yaml` (targets, model attributes, forbidden predictors, derive/recover rules, `max: snapshot_year`) |
| `ingestion/loader.py` | CSV → text frame aligned to the schema (`align_to_schema`); lenient typing for already-clean inputs; export typing |
| `quality/validator.py` | findings table; severity `error` / `warning` / `info` |
| `quality/cleaner.py` | rule-driven corrections with per-cell audit log and approval-gated rule families; `plan_changes` dry run; never fills, never changes unruled values |
| `profiling/profiler.py` | `profile_dataset` (schema-free EDA) and `profile_records` (schema-aware): missingness, co-missingness, patterns, summaries, missingness-by-attribute, relationships (Spearman, eta², Cramér's V) |
| `profiling/visualization.py` | in-memory discovery figures for the notebook; static PNG figures (profile, imputed-vs-observed, reconstruction) |
| `preprocessing/feature_selection.py` | role check + leakage guard; typed, ordered model frame |
| `preprocessing/derived_fields.py` | `recover_sources` (year from age, before imputation) and `derive_fields` (age from year, after) with cell masks and logs |
| `preprocessing/prepare.py` | `prepare_for_imputation`: recovery + feature selection in one step, used by `impute`, `apply` and evaluation mode |
| `postprocessing/` | `finalise_output`: derived fields, output typing, cell provenance (`provenance.py`), integrity checks |
| `imputation/config.py` | `ImputationConfig` (model columns, constraints, method defaults, `selected_hyperparameters`) |
| `imputation/input_validation.py` | input/result validators shared by the methods (unchanged legacy module) |
| `imputation/simple_imputer.py` | mean/mode helpers (unchanged legacy module) |
| `imputation/missforest.py` | original transductive MissForest (unchanged legacy module) |
| `imputation/postprocess.py` | constraints applied to imputed cells only (unchanged legacy module) |
| `imputation/predictive_missforest.py` | fit/transform imputer with frozen sequential replay |
| `imputation/predictive_tuning.py` | validation-only grid search with repeated, stratified holdouts and a baseline noise floor |
| `imputation/registry.py` | `BaseImputer` interface, the three methods, `build_imputer`, `resolve_hyperparameters` |
| `imputation/persistence.py` | `save_imputer` / `load_imputer`, the fitting-record provenance sidecar with a model fingerprint, and `check_compatibility` against the current configuration and input |
| `evaluation/holdout.py` | partitions (row or group by feeder, manifest reuse) and hide-observed-cell masks (stratified, repeated, link-aware) |
| `evaluation/protocol.py` | evaluation mode: partitions, masks within partitions, `unseen_records` / `transductive` protocols, test-only scoring |
| `evaluation/metrics.py` | hidden-cell metrics (MAE, RMSE, nRMSE, signed bias, accuracy, balanced accuracy, macro-F1) and per-class report |
| `evaluation/integrity_checks.py` | pass/fail checks on a completed table |
| `evaluation/distribution_analysis.py` | imputed-vs-observed shape diagnostics, joint decade×class shift |
| `evaluation/confidence.py` | within-column confidence ranks and the low-confidence list |
| `reporting/report_generator.py` + `templates/report.html.j2` | single-file HTML report (configuration state, cleaning, profile, imputation + provenance, operational checks, evaluation mode) |
| `settings.py` | `PipelineSettings` from `pipeline.yaml` (execution mode, evaluation-mode settings, config file paths) |
| `notebook.py` | display helpers for `notebook/pipeline_overhead_conductors.ipynb` (path display, truncated tables, figures); no pipeline logic |
| `pipeline.py`, `cli.py` | orchestration and command line |

## 3. The method interface (`imputation/registry.py`)

```python
class BaseImputer:
    name: str
    supports_persistence: bool     # can be saved and applied to new records
    transductive: bool             # must see the records it fills during fitting
    def fit(self, X) -> BaseImputer
    def transform(self, X) -> ImputationOutput
    def fit_transform(self, X) -> ImputationOutput
```

`ImputationOutput.imputed` is the full input frame with the model columns
filled (same index, same columns, non-model columns untouched);
`.diagnostics` has one row per imputed cell with fixed columns
(`row, column, kind, imputed, prediction_source, raw_prediction, max_proba,
margin, entropy, tree_std, tree_p10, tree_p90, material_probabilities`; NaN
where a method has no such field); `.metadata` is JSON-serialisable.

| Method | Persistable | Transductive | Diagnostics |
|---|---|---|---|
| `median_mode` | yes | no | value only (`training_statistic`) |
| `missforest` | no | yes | value only (`random_forest_transductive`) |
| `predictive_missforest` | yes | no | vote share, margin, entropy (categorical); tree spread (numeric); fallback flag |

Hyperparameters are resolved in three layers: `methods.<name>` defaults in
`imputation.yaml`, then the `selected_hyperparameters.<name>` block (written by
the tuning experiment; only forest hyperparameters may appear there, never
`random_state` or `n_jobs`), then call-time overrides (`--set key=value`,
development only).

## 4. Predictive MissForest and its relationship to prior work

The original MissForest (Stekhoven & Bühlmann, 2012) is transductive: for each
attribute, in order of increasing missingness, it fits a Random Forest on the
rows where that attribute is observed and predicts the rows where it is
missing, cycling over attributes until the imputed values stop changing. The
forests are discarded; the method cannot be applied to records it has not seen.

`predictive_missforest` keeps the same per-attribute, sequential (Gauss–Seidel)
update but makes it inductive:

- the number of passes is fixed (`n_passes`) rather than decided by convergence, so a transform replays exactly what fit did;
- one pipeline (one-hot encoder with a frozen vocabulary + forest) is kept for every `(pass, attribute)`;
- initial-fill statistics (mean/median, deterministic mode) and the attribute order are frozen at fit time;
- `transform` on new records replays the frozen passes; `transform(X_fit)` reproduces the fit-time table exactly (tested);
- categories unseen at fit time are handled by policy (`ignore` / `mode` / `error`);
- attributes with fewer than `min_observed_target` observed cells are filled with the training statistic and flagged (`prediction_source = training_statistic_fallback`).

This is the design of **missForestPredict** (Albu, Gao, Wynants & Van Calster,
2024; CRAN package `missForestPredict`), which saves the per-iteration models
of a missForest run so they can be applied to new data, and of scikit-learn's
**`IterativeImputer`**, which stores the fitted estimators and the imputation
order and replays them in `transform`. The project's contribution is the
adaptation to this schema and workflow (mixed-type pipelines with frozen
vocabularies, constraints, confidence diagnostics, provenance), not a new
imputation algorithm, and it should be described that way.

When the same table is fitted and transformed (`oc-pipeline impute`/`all`)
the method behaves like the original algorithm with a fixed number of passes;
on the synthetic sample the two give practically identical reconstruction
scores (`docs/EXPERIMENTS.md`). The difference is operational: the fitted
model can be saved, audited and applied to next month's records.

## 5. Two kinds of evaluation

### Operational check (every run; no truth needed)

The only accuracy that can be measured on a real extract is *reconstruction
of known values*: hide a fraction of the observed cells of each model
attribute (stratified by class for categorical attributes, `asset_age` hidden
together with `installation_year`), fill the masked table with the configured
method, and score the hidden cells against the values that were hidden. The
reference method is scored on the identical cells. Pre-existing missing values
are never scored (they have no truth). `validation.reconstruction.repeats`
draws several masks and averages. For an `apply` run the check is skipped
because the model was not fitted on that file. The report labels this
"same-table reconstruction of hidden known values (operational check)": the
method sees the very records it is scored on, so the figure is an upper bound
for unseen records.

### Evaluation mode (`oc-pipeline evaluate`; needs a complete reference)

`evaluation/protocol.py` keeps four things apart that are easy to conflate:

| | What it is | Where it is recorded |
|---|---|---|
| complete reference | every model attribute known (`--truth`) | `run_metadata.json` → `evaluation.truth` |
| masked data | the reference with cells hidden (`--input`), or the reference masked here | `evaluation.masked_input` / `evaluation_mask_repeat*.csv` |
| masking configuration | the rule that produced the gaps (travels with the masked file, e.g. `conductors_synthetic_masked_metadata.json`) or `evaluation.mask.{rate,seed,repeats}` | `evaluation.mask_source`, `mask_seed` |
| evaluation partitions | fit / validation / test by **feeder** (`evaluation.split`), so no feeder straddles partitions | `partition_manifest.csv` (reusable with `--manifest`) |

Masks are drawn (or taken from the masked file) **inside** each partition;
`installation_year` recovered from an observed `asset_age` is excluded from
scoring. Two protocols run: `unseen_records` fits on the fit partition and
transforms the test partition (persistable methods only - the operational
situation), `transductive` runs `fit_transform` on fit + test together (all
methods). Only test cells are scored; the validation partition is reserved
for hyperparameter tuning and never scored here. Outputs:
`evaluation/{metrics,per_class,summary,scored_cells}.csv`, with
`within_5_years` for the year and macro-F1 / balanced accuracy for
categorical targets, and `scored_cells.csv` listing every scored cell with
its truth and prediction. The earlier research comparison
(`experiments/model_comparison`) scored masked cells of the whole table after
tuning on a validation partition that overlapped the test records; this mode
supersedes it (`docs/EXPERIMENTS.md`). A probability-calibration partition is
not implemented: no method currently calibrates probabilities, so the split
is fit / validation / test.

## 6. Determinism and reproducibility

All randomness is seeded: forests through `random_state` in
`imputation.yaml`, masks and partitions through `seed` / `mask_seed` /
`random_seed` settings. `n_jobs: 1` keeps forest predictions bit-for-bit
reproducible (parallel prediction accumulates in thread-completion order);
`-1` is fine for experiments. `run_metadata.json` records package, Python,
scikit-learn and pandas versions; saved models record them as well and warn on
a scikit-learn mismatch at load time.
