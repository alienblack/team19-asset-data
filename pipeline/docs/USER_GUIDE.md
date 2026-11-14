# User guide

## 1. Install

Python 3.11 or newer. Everything below is run from the `pipeline/` directory.

```bash
cd pipeline
python -m venv .venv && source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'                        # pinned versions: pip install -r requirements.txt
oc-pipeline --version                                     # or: python -m src.cli --version
python -m pytest -q tests/                                # confirm the environment
```

The Python package lives **directly under `src/`** with no additional package
folder. The notebooks in `notebook/` and the `asset-pipeline` / `oc-pipeline`
commands call the same stage functions. See the main README for the two-file
benchmark interface and transformer asset profile.

## 2. Check and approve the configuration

Four files drive every run: `config/overhead_conductors/data_schema.yaml` (columns, roles,
ranges, the two primary targets and seven model attributes, forbidden
predictors, derivation rules), `config/overhead_conductors/cleaning_rules.yaml` (corrections and
who approved each rule family), `config/overhead_conductors/imputation.yaml` (model columns,
method defaults, selected hyperparameters) and `config/overhead_conductors/pipeline.yaml` (what to
run and how). `config/overhead_conductors/approvals.json` records who approved each file at which
content hash.

```bash
oc-pipeline config check                       # parse, cross-check, show approval states and findings
oc-pipeline config show                        # review tables: schema, cleaning families, methods
oc-pipeline config approve --file data_schema.yaml --by "J. Smith (SAPN)" --note "ranges confirmed"
oc-pipeline config approve --file all --by "Reviewer Name"
oc-pipeline config revoke --file cleaning_rules.yaml
```

In `release` mode (`execution.mode` in `pipeline.yaml`) every
stage refuses to run while a file is invalid, has been edited since its
approval (`changed`) or was never approved. For a development run add
`--allow-unapproved` (or set `execution.mode: development`); the state is
still recorded in `run_metadata.json`. Approving a file with error-level
findings is refused.
The two example profiles currently default to development mode; review and
approve each asset configuration separately before setting release mode.

The approvals shipped in this repository were given by Team 19 **for the
synthetic sample only**; the data owner must re-approve the files before
they are used on real records (`docs/DATA_HANDLING.md`). In particular the
`code_mappings` cleaning family (interpreting conductor-type codes such as
`ACSR` as a material) is shipped **unapproved** and is skipped until approved.

## 3. Prepare an extract

The pipeline reads a CSV with one row per conductor segment. Required
columns (canonical names):

| Column | Type | Meaning |
|---|---|---|
| `segment_id` | text | unique record identifier |
| `installation_year` | integer | year of installation (**primary target**) |
| `distance_to_sea_km` | number | distance to the coast |
| `voltage_category` | category | `LV`, `11/7.6 kV`, `19 kV SWER`, `33 kV`, `66 kV` |
| `location_group` | category | `Urban`, `Regional`, `Rural` |
| `corrosion_zone` | category | `Low`, `Severe`, `Very Severe` |
| `material` | category | `Aluminium`, `Copper`, `Steel` (**primary target**) |
| `covering` | category | `Bare`, `Covered` |

These seven are the model attributes: every method fills them and may use
them as predictors. Optional: `feeder_id` (groups records for the evaluation
partitions), `asset_age` (derived from the year; where only the age is
observed the year is **recovered** from it before imputation and marked
*derived*), and any other column, which is carried through unchanged and
never used for prediction. `length_km` is such a column: its gaps stay
unresolved and are listed as such.

If the extract uses different column names, map them in
`config/overhead_conductors/data_schema.yaml`:

```yaml
column_mapping:
  "Segment ID": segment_id
  "Install Year": installation_year
  "Conductor Material": material
```

Spellings, cases or abbreviations of an allowed label go under
`categorical.spelling_variants` in `config/overhead_conductors/cleaning_rules.yaml` (no domain
decision); interpretations of codes go under `code_mappings` and need the
data owner's approval. Set `snapshot_year` in the schema to the extract date
so `asset_age` is derived correctly. Empty cells and the tokens listed under
`missing_tokens` (`NA`, `N/A`, `unknown`, `-`, …) are treated as missing.
Every change to a configuration file must be re-approved (section 2).

## 4. Run

```bash
oc-pipeline ingest --input path/to/extract.csv --out outputs/<run-name>     # optional: structure + checksum only
oc-pipeline all --asset overhead_conductors --measured path/to/measured.csv \
  --truth path/to/truth.csv --out outputs/<run-name>
```

The run directory then contains:

```
report.html                   open this first
run_metadata.json             configuration state and hashes, settings, versions, timings, per-stage summaries
run.log
source_records.csv            byte-for-byte copy of the input
quality_report.csv            findings before cleaning
cleaned.csv                   the corrected extract
cleaning_log.csv              one row per changed cell: record_id, column, old_value, new_value, rule
dropped_rows.csv              exact duplicate rows removed (if any)
profile/                      missingness, patterns, summaries, relationships (+ PNG figures)
imputation_input.csv          the typed frame handed to the method, after deterministic recovery
recovery_log.csv              years recovered from asset_age (count, rejected out-of-range, still missing)
final_imputed.csv             the completed extract, source column order, derived fields filled, integers as integers
cell_provenance.csv           one row per derived / imputed / unresolved cell: record_id, column, status, method, value, detail
provenance_summary.csv        observed / derived / imputed / unresolved counts per column
diagnostics.csv               one row per imputed cell with the method's confidence fields
reference_imputed.csv         what the median/mode reference would have produced
convergence.csv               per-pass changes of the forest method
derived_fields_log.csv        derived columns filled or recomputed after imputation
model.joblib (+ .provenance.json)   the fitted predictive model, its fingerprint and the fitting records' identifiers
validation/                   integrity_checks, distribution_checks, reconstruction_*, low_confidence_cells, figures
```

Observed values are never changed by imputation: the integrity check
`observed_values_unchanged` compares every observed cell of
`imputation_input.csv` with `final_imputed.csv` and a failure stops the run.
Cells that are not model attributes stay empty and are listed under
`unresolved` in `provenance_summary.csv` and the report.

### Choosing the method

`config/overhead_conductors/pipeline.yaml` → `imputation.method`, or `--method` on the command
line:

- `predictive_missforest` (default) — fit/transform adaptation of MissForest; can be saved and re-applied to new records.
- `missforest` — the original algorithm run on the whole table; no saved model.
- `median_mode` — the reference; use it to see what a trivial fill would do.

The methods are interchangeable behind one interface; the report always
shows the configured method next to the reference.

### A quality error stopped the run

The quality stage refuses to continue on `error`-level findings: a required
column is missing, identifiers are missing or shared by differing records, or
a target has fewer than two observed values. Fix the extract or adjust the
schema (for example map the column name), re-approve, then rerun.
`quality_report.csv` lists the findings with examples. To proceed despite
errors set `quality.fail_on_error: false` — not recommended.

## 5. Read the report

1. **Configuration** — which files were used, at which hash, approved by whom; findings (errors, warnings such as a skipped unapproved rule family).
2. **Summary** — record count, how many cells were imputed, derived and left unresolved, whether the integrity checks passed, and the operational reconstruction headline next to the median/mode reference. If the method is not clearly better than the reference for a target, the extract does not carry enough information to reconstruct that target and its imputed values should be treated as placeholders.
3. **Data quality** and **Cleaning** — what was wrong and what was changed; which rule families were skipped for lack of approval and which code values are pending a decision. Everything the cleaner did is in `cleaning_log.csv`; nothing was filled.
4. **Profile** — where values are missing, whether missingness depends on other attributes, and how strongly attributes relate (weak relationships with a target → poor reconstruction).
5. **Imputation** — method, hyperparameters, cells per attribute, recovery, cell provenance, convergence, derived fields.
6. **Operational checks** — integrity (must all pass), same-table reconstruction metrics with per-class recall for categorical targets, distribution comparison (imputed vs observed), and the list of least-confident imputations to review by hand.
7. **Evaluation against known truth** — present only for a run that included `oc-pipeline evaluate` (section 7).

Everything in the report is also a CSV in the run directory.

## 6. Fit once, apply later

```bash
# fit on the cleaned reference extract and save the model
oc-pipeline clean --input reference_extract.csv --out outputs/ref
oc-pipeline fit   --input outputs/ref/cleaned.csv --model models/conductors_2026Q3.joblib

# later: fill gaps in new records with the saved model
oc-pipeline clean --input new_records.csv --out outputs/new
oc-pipeline apply --input outputs/new/cleaned.csv --model models/conductors_2026Q3.joblib --out outputs/new
oc-pipeline validate --run outputs/new && oc-pipeline report --run outputs/new
```

Before applying, the saved model is compared with the current configuration
and input (`check_compatibility`): different model columns are an error;
different constraints or category labels the model never saw are warnings,
recorded in `run_metadata.json`. The model file also carries SHA-256
fingerprints of the identifiers it was fitted on; `apply` reports any overlap
with the new file and with `--require-independent` refuses to run on records
that were part of fitting. New category labels are handled by
`unknown_category_policy` in `imputation.yaml` (`ignore` by default).

Refit when the population changes materially (new attribute values, a
different region, a new extract format).

## 7. Evaluation mode (when a complete reference exists)

Operational runs cannot measure how right the imputed values are: the
originally missing cells have no truth. When a complete reference exists -
on this project only the synthetic sample - evaluation mode scores the
methods on hidden cells of a **record-disjoint test partition**:

```bash
# score the originally missing cells of a measured/noisy file against truth
oc-pipeline evaluate --asset overhead_conductors \
  --measured data/overhead_conductors/conductors_synthetic_masked.csv \
  --truth data/overhead_conductors/conductors_synthetic_complete.csv --out outputs/eval

# or let the stage hide 20 % of the reference inside each partition
oc-pipeline evaluate --truth data/overhead_conductors/conductors_synthetic_complete.csv --out outputs/eval2 \
                     --manifest outputs/eval/evaluation/partition_manifest.csv      # reuse the partitions
```

Records are partitioned by `feeder_id` into fit / validation / test
(`evaluation.split`, default 60 / 15 / 25, seed 219); masks are applied
inside partitions; only test cells are scored. `unseen_records` fits on the
fit partition and transforms the test partition; `transductive` runs
`fit_transform` on fit + test. The validation partition is reserved for
hyperparameter tuning and is never scored. Results go to
`<out>/evaluation/` (`summary.csv` with MAE, `within_5_years`, macro-F1,
balanced accuracy per method and protocol; `per_class.csv`;
`scored_cells.csv` with every scored cell, its truth and prediction;
`partition_manifest.csv`; the masks) and into the report when `report` is run
on the same directory.

Results on the synthetic sample describe the generator, not SAPN's asset
population (`docs/LIMITATIONS.md`).

## 8. The notebook

`notebook/pipeline_overhead_conductors.ipynb` walks through the nine stages on the synthetic sample and is
executable from a fresh kernel (`pip install -e '.[notebook]'`, then
`jupyter nbconvert --to notebook --execute notebook/pipeline_overhead_conductors.ipynb`, or open it and
run all cells). It is a controller: every cell calls a
package function and displays the result; no cell transforms data itself
(`tests/test_notebook.py` executes the code cells in order and checks this).
Paths are set in one cell and can be overridden with the environment
variables `OC_NOTEBOOK_INPUT`, `OC_NOTEBOOK_TRUTH`, `OC_NOTEBOOK_MASKED`,
`OC_NOTEBOOK_RUN_DIR`; `OC_NOTEBOOK_OVERRIDES` (JSON) applies development
hyperparameter overrides, which are recorded in the run metadata. Setting
`APPROVE_CONFIGURATION_AS` to a name records approval of the current
configuration content from the notebook; left as `None` the notebook never
writes to `config/overhead_conductors/approvals.json`.

## 9. Hyperparameters

By default the pipeline does not tune. `config/<asset>/imputation.yaml` holds
method defaults and `selected_hyperparameters`. Set `tuning.enabled: true` to
search a grid on fit/validation records; setting `tuning.update_selected: true`
also writes selected values and provenance to the YAML in development mode.
The same file has `model.mode: train_new | use_saved` and `model.path` for
frozen-model reuse. See [model execution / tuning](MODEL_EXECUTION_AND_TUNING.md).
To change a value for one run use `--set key=value` (development only, with
tuning disabled). Any change to `imputation.yaml` requires re-approval before
release-mode execution.

## 10. Settings reference (`config/overhead_conductors/pipeline.yaml`)

| Key | Meaning |
|---|---|
| `execution.mode` | `release` (refuse unapproved configuration) or `development` (record and continue) |
| `seed` | seed for masks and partitions |
| `stages.*` | switch stages on or off |
| `quality.fail_on_error` | stop on error-level findings |
| `imputation.method`, `reference_method`, `save_model`, `overrides` | what to run, whether to save the model, development overrides |
| `validation.reconstruction.{enabled, mask_rate, mask_seed, repeats, stratify_categorical, include_reference}` | the operational same-table check |
| `validation.distribution.{enabled, year_bin_width}` | distribution comparison and decade bins for the joint shift table |
| `confidence.{low_confidence_fraction, min_rows}` | size of the low-confidence list |
| `evaluation.split.{fit, validation, test, seed, strategy, group_column}` | evaluation-mode partitions |
| `evaluation.mask.{rate, seed, repeats}` | masks drawn when no masked input is given |
| `evaluation.protocols`, `evaluation.methods` | which protocols and methods evaluation mode runs |
| `report.{title, embed_figures}` | report header and whether PNGs are embedded |

Command-line equivalents: `--method`, `--reference none`, `--no-model`,
`--mask-rate`, `--repeats`, `--seed`, `--schema`, `--rules`, `--config`,
`--pipeline-config`, `--allow-unapproved`, `--set key=value`.
