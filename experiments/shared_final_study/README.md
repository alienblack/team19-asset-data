# Team 19 · Asset data imputation

A local workbench for comparing ways to fill missing asset attributes. Load a
CSV, check its fields, run a controlled comparison and download completed data
with a report of what worked and where it failed.

**Version 5.2.1 · Python 3.12 · Local browser interface**

## Start here

| What you want to do | Where to go |
| --- | --- |
| Try the app for the first time | [Install](#2-installation), then follow the [five-minute test](#your-first-test-about-five-minutes) |
| Run your own CSV | [Browser walkthrough](#3-guided-browser-workflow) and [input requirements](#4-input-requirements) |
| Understand the results | [Metrics and interpretation](#8-metrics-and-interpretation) |
| Repeat a run from a saved configuration | [Command-line workflow](#10-repeat-a-new-csv-run-from-the-command-line) |
| Read the original study | [Published report](reports/published/report.html) |
| Test uploads, settings and downloads | [Testing guide](docs/TESTING.md) |
| Prepare the final submission | [Submission checklist](SUBMISSION.md) |

### Launch after installation

From this project's folder, with its environment activated:

```sh
python run_study.py --mode guided
```

Open **http://127.0.0.1:8765** if your browser does not open automatically. Keep
the terminal running while you use the app. Press **Ctrl+C** to stop it.

### Your first test (about five minutes)

1. Click **Use example data**. This loads the supplied synthetic demonstration;
   no upload or reference file is needed.
2. Check that the summary shows **240 records, 7 columns and 207 missing cells**.
   `asset_id` should be ignored and `material` categorical. Open **Advanced field
   settings** if you want to inspect limits, integer rounding or field groups.
3. Keep the three selected methods: **Median / mode**, **Predictive MissForest**
   and **Gaussian EM**. Leave **20%** additional masking and **Scattered gaps**.
4. Click **Run study & generate report** and wait for **Report ready**. Then
   select **Open report**. Inspect the method comparison, field results and
   limitations; an error from a method must be visible rather than silently ignored.
5. Download a completed CSV and the run configuration from the report. The CSV
   should have the same **240 records and 7 columns**, with recorded values
   retained. For a PDF, use the report's print action and choose **Save as PDF**.

The demonstration has weakly predictable material labels: a low material score
can be a valid finding. There is no fixed skill value every successful run must
reach. Check completion, preserved inputs and honest reporting of failures.

The app runs on your computer and does not send your dataset to a hosted model.
New runs are saved under `outputs/dataset-workbench/`; the original published
results are retained separately.

## 1. Choose the correct workflow

| Workflow | Purpose | Command |
| --- | --- | --- |
| Guided dataset study | Choose a CSV, define its fields, configure an experiment and generate a report in a local browser | `python run_study.py --mode guided` |
| Dataset CLI | Repeat a configured CSV study without the browser | `python run_study.py --mode dataset --dataset INPUT.csv --config SETTINGS.json --out NEW_FOLDER` |
| Archived-results report | Present the supplied original results; no model fitting | `python run_study.py` |
| Quick original benchmark | Check every original stage on 600 rows with small models | `python run_study.py --mode quick --workers 2` |
| Full original benchmark | Refit the original 3,580-task experiment and rebuild its outputs | `python run_study.py --mode full --workers 4` |
| Saved-prediction evaluation | Reevaluate an existing original-format run after its contract check | `python run_study.py --mode evaluate --run-id RUN_ID` |

### What remains unchanged

The new dataset route does not edit the original 25 scientific modules, 30 input
CSVs, 58 result CSVs or selected original settings. Their exact hashes are recorded
in `ORIGINAL_SOURCES.json`. The archived report has a new layout; its scientific
summary and result values are preserved.

**A new CSV run is a new experiment.** Its generic adapters and configurable
defaults are not the exact original benchmark protocol. Selecting the original
transformer CSV in the new workbench does not reproduce the archived scores.
Use the original benchmark commands when that is the intended task.

## 2. Installation

Use Python **3.12**, not the Python 3.11 environment for the separate Subteam A
benchmark. The pinned dependencies include the original model libraries, Jinja2,
the notebook tools and pytest. The workbench uses Python's standard-library HTTP
server; no hosted service, API key, external fonts or frontend build is required.

### macOS / Linux

```sh
cd /path/to/Team19_Automated_Handover
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
```

### Windows PowerShell

```powershell
cd C:\path\to\Team19_Automated_Handover
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip check
```

If activation is restricted, use `.venv\Scripts\python.exe` in place of `python`.
Do not install the repository root's requirements as a substitute for this folder's
requirements. Do not upgrade numerical/model libraries casually: the miceforest
compatibility implementation is specifically audited for `miceforest==6.0.5`.

The launcher adds the local `src/` directory itself. An editable installation is
optional: `python -m pip install -e .`. It also exposes `imputation-study` as an
alias while working from this source tree. The complete source folder contains
the data and templates; a bare installed wheel is not a substitute for the client
handover archive.

## 3. Guided browser workflow

```sh
python run_study.py --mode guided
```

The app opens at `http://127.0.0.1:8765`. It binds only to the local computer.
Keep the terminal open; Ctrl+C stops it. To choose a different port or suppress
automatic browser opening:

```sh
python run_study.py --mode guided --port 8766 --no-browser
```

After environment setup, `Start_Workbench.command` on macOS or
`Start_Workbench.bat` on Windows launches the same app. If a macOS archive tool
does not preserve executable permissions, use the terminal command above.

### Step 1 - choose data

Select the CSV containing your recorded values and gaps. A second clean truth CSV
is available under **Reference data (optional)**. Click **Load dataset**. The app reports row count, fields, missing
cells and suggested column types. Changing a selected file requires loading it
again before a run can start.

For a first demonstration, click **Use example data**. The example
has seven columns, five text-valued material labels and 240 records. It is a new
demonstration, not SAPN data or the archived study. Its independently drawn
material labels deliberately have little predictable signal.

### Step 2 - confirm fields

The main table shows column names, suggested types and tolerances. Use
**Advanced field settings** for bounds, groups, integer rounding and controls.
Choosing **Grouped gaps** also reveals these settings. The app suggests field
types; you remain responsible for confirming their meaning.

| Setting | Meaning | Example |
| --- | --- | --- |
| Numeric | A measured quantity or numeric time/year field | `installation_year`, `length_km` |
| Categorical | An unordered label; arbitrary text or numeric codes are supported | `material`, `equipment_model` |
| Date (ISO) | ISO dates modelled in days; recorded strings are preserved and new fills are exported as `YYYY-MM-DD` | `install_date` |
| Ignore / ID | Retain this column in exports, but exclude it from prediction and scoring | `asset_id`, comments, record identifiers |
| Tolerance | Absolute units or relative percentage for evaluation; blank means not assessed | ±5 years, ±10% concentration |
| Minimum / maximum | Optional limits on newly filled numeric values | length ≥ 0 |
| Group | Fields with the same group name disappear together in a grouped-mask test | `location` for x/y/altitude |
| Integer | Round newly filled numeric values; recorded values are preserved | installation year |
| Control | Keep a numeric negative control in the experiment but exclude it from mean skill | an independently generated placebo |

Unique ID-like fields and fields with fewer than two observations are suggested
as ignored. These are suggestions to review, not knowledge of the dataset.
Exclude outcome-derived or future-information fields when they would reveal a
target illegitimately. Row holdout alone cannot identify semantic leakage.

No transformer-specific limits are imposed on a new schema: a column called `x`
is not automatically clipped to 99, and a field called `life` does not inherit the
synthetic study's ceiling. Only your specified numeric constraints apply.

For date fields, absolute tolerances and errors are in **days**, not a percentage
of the timestamp origin. For calendar years stored numerically, use an absolute
tolerance such as 5 years. Date bounds, if supplied, use days since 1970; leaving
them blank is usually clearer than imposing an arbitrary date range.

### Step 3 - configure the experiment

The three initial choices are median/mode, Predictive MissForest and Gaussian EM.
Open **More methods** to choose additional models. Median/mode is always retained
as the common reference.

Set the additional masking rate, missingness pattern and evaluation sample.
**Evaluation settings** contains folds, repetitions and seed; **Model settings**
contains model sizes and iteration budgets. Their defaults are recorded in every
run. The optional noise test perturbs only explicitly selected numeric fields.

### Step 4 - run and review

Press **Run study & generate report**. Progress shows the current method/fold,
then the separate final-completion fits and report generation. One job runs at a
time in the local app. Model-stage errors are retained and shown in the report;
an incomplete validation method is excluded from the paired ranking.

Open the resulting report, inspect the field results and limitations, then
download completed CSVs or individual fills ledgers. Each run gets a new folder.
Nothing is automatically written into an operational register.

## 4. Input requirements

- UTF-8 CSV with a header, unique nonempty column names and one record per row.
- At least 12 records; up to 100,000 rows and 200 columns in this local workflow.
- Browser uploads are limited to 50 MB per CSV.
- Blank cells and pandas' standard NA tokens are treated as missing. If a literal
  label such as `NA` is a real category, replace it with an unambiguous label first.
- Identifier text is retained, including leading zeros. Numeric conversion applies
  only to modelled numeric/date fields. Numeric-like category codes in a truth
  file are aligned to the input labels when that mapping is unambiguous (for
  example, `0` versus `0.0`); their recorded label formatting is retained.
- Numeric fields must parse as finite numbers; infinity is rejected.
- Categorical fields support up to 200 observed labels. Exclude identifiers or
  free text rather than treating thousands of unique strings as equipment classes.
- ISO date fields accept ISO-formatted values. Convert ambiguous regional date
  strings explicitly before loading. Date fills use whole days.
- Every modelled field needs at least two observed values. Entirely unobserved
  fields cannot be learned from this dataset; exclude them or provide external
  information through a separately justified process.

These are tabular input requirements, not a promise that every dataset contains
enough signal for useful recovery. Large datasets, many categories and many
missing patterns can be expensive. Start with a few methods and the default
evaluation sample. Gaussian EM is limited to 80 encoded dimensions and a 512 MB
estimated working-array budget; exceeding it is a reported method error.

### Optional truth CSV

The truth file must have the same row count, column names and column order. Rows
must correspond exactly. A unique, fully observed ignored identifier is checked
for matching row alignment when present. The program does not silently reorder
or join records by guessed IDs.

With truth, held-out missing cells are scored where their reference is available,
including naturally missing cells with known truth. Without truth, only additional
hidden recorded values can be scored. The latter measures recovery of recorded
values; recording errors may still be present. Truth is not passed to model fitting.

## 5. What happens inside a new dataset run

1. Read the input, validate field definitions and exclude IDs/ignored fields.
2. Select the configured evaluation sample using the seed.
3. Build shuffled row folds. Each repetition uses another deterministic split.
4. Copy training and held-out rows; add the configured hiding pattern. Existing
   gaps remain. The training mask retains at least two available observations per
   field when the original training fold has them, to avoid destroying its last
   usable training labels. Actual counts are saved; nominal rates are not assumed
   to be exact realised rates.
5. Optionally add symmetric multiplicative noise to selected visible numeric
   entries, including training labels. Hidden scoring references are not perturbed.
   Training masks, test masks and their noise use separate seeded random streams.
   Changing only noise amplitude keeps the hidden cells fixed and scales the same
   underlying noise draws, so those effects are not confounded.
6. Fit each model and its preprocessing on training folds only. Training medians,
   scaling, category vocabularies, models and PMM donors do not use held-out rows.
7. Predict the held-out rows using stored models. All methods share the same rows,
   hidden cells and stress inputs in that fold.
8. Score against the separate reference and save the per-cell predictions.
9. Pool errors by method and attribute; keep numeric and categorical measures
   separate. Only methods completing every validation fold enter the paired ranking.
10. Fit each selected method again on the full supplied input, without the added
    experimental masking/noise, to produce its completed dataset. These final fits
    are separate from the scored fold models.
11. Apply configured constraints to new fills, restore recorded values, and write
    completed CSVs and fill ledgers.
12. Generate figures, data-derived findings, the HTML report, settings, hashes,
    warning records and failure details.

The workflow processes **one dataset per run**. An overhead CSV and a transformer
CSV can both be used in separate runs with their own schemas and reports; they
are not automatically concatenated into one training table.

## 6. Methods available for new schemas

| ID | Method | Important detail |
| --- | --- | --- |
| `median` | Median / mode | Training-only constant reference; always included. |
| `forest` | Predictive MissForest | Reuses the team's stored per-pass forest implementation for arbitrary numeric/categorical columns. |
| `gaussian` | Gaussian EM | Reuses Subteam A's shared-covariance incomplete-data EM; train-only category coding, with dataset-specific output formatting. Gaussian one-hot categories are an approximation. |
| `ridge` | Chained ridge | Reuses stored linear/logistic conditional models with numeric scaling and categorical encoding. |
| `knn` | k-NN | Reuses overlap-aware training-donor predictions for arbitrary schemas. |
| `mlp` | Masked MLP | Reuses the reconstruction network; requires at least one numeric field. The generic workflow uses independent training remasking. |
| `lightgbm` | Chained LightGBM | Reuses the team's frozen conditional-model replay. |
| `hgb` | Chained histogram boosting | Generic per-target boosting models fitted on observed training targets and replayed in fixed passes. |
| `catboost` | Chained CatBoost | Generic per-target CatBoost models using training-fitted preprocessing and fixed replay. |
| `iterative` | Iterative ridge + category mode | Numeric IterativeImputer only; categorical gaps explicitly use the mode. It is not a mixed-type MICE classifier. |
| `mice` | MICE-PMM | Reuses the team's PMM/logistic draws with training-only fitting and donor pools; pools numeric chains by mean and categories by mode. |
| `miceforest` | miceforest / LightGBM | Uses the audited compatibility implementation, with training models applied to new records. |

The generic HGB/CatBoost adapters do not reproduce the historical benchmark's
ridge-plus-residual chains. The generic report labels its protocol and settings
separately. Every method is a point-fill comparison here; MICE's generated draws
are pooled for the reported point estimates. Multiple-imputation inference or
calibrated per-asset confidence intervals are not claimed.

## 7. Parameter reference

| Setting | Default | Meaning |
| --- | --- | --- |
| `folds` | 3 | Number of held-out row folds; 2-10. |
| `repeats` | 1 | Number of deterministic repeat splits; 1-5. |
| `seed` | 219 | Base seed; repetition r uses seed + 1000r. |
| `mask_rate` | 0.20 | Additional hiding probability for visible cells, not total missingness. |
| `masking` | `scattered` | `scattered` or `grouped`; grouped needs a group with at least two fields. |
| `noise` | 0 | Uniform multiplicative perturbation ±this fraction during validation. |
| `noise_columns` | none | Explicit numeric fields receiving noise; dates are excluded. |
| `evaluation_rows` | 1000 | Evaluation sample limit; 0 means all rows. Final completion uses all input rows. |
| `trees` | 50 | Forest trees, boosting rounds or miceforest rounds. |
| `passes` | 3 | Conditional replay passes / iterative budget / MICE sweeps. |
| `clusters` | 4 | Gaussian components, capped by training-row count. |
| `em_steps` | 12 | Maximum EM iterations. |
| `regularisation` | 0.025 | Diagonal covariance stabilisation in encoded units. |
| `epochs` | 30 | Masked MLP training epochs. |
| `neighbors` | 15 | Neighbours used by k-NN, capped by available donors. |
| `ridge_alpha` | 10 | Chained-ridge penalty. |
| `chains` | 3 | MICE-PMM chains. |
| `donors` | 5 | Candidate PMM donors. |

The default selected methods are median/mode, Predictive MissForest and Gaussian
EM. Selecting more methods, repeats and passes increases runtime. There is no
automatic parameter tuning against the evaluation scores.

## 8. Metrics and interpretation

### Numeric recovery

MAE is the average absolute difference between fills and references. RMSE gives
greater weight to large errors. Both are in the field's units; date errors are in
days. MAE for years must not be compared directly with MAE for gas concentration.

```text
skill(field) = 1 - MAE(method, field) / MAE(training-median reference, field)
```

- 0: matches the median reference.
- 1: perfect reconstruction on scored cells.
- 0.60: 60% less MAE than the median for that field.
- Negative: worse than the median.
- Undefined: the median already has zero error; report a dash rather than divide
  by zero.

Mean numeric skill is an equal-weight mean of valid field skills, excluding
declared controls. It is not a percentage of correct fills. The report also counts
numeric fields below their median reference so the mean cannot hide all severity.

### Categorical recovery

Accuracy is the share of correctly recovered categories. Macro-F1 gives every
category equal weight and is useful when one class dominates. Categorical results
are not mixed into mean numeric skill. Category-only datasets can be ranked on
macro-F1. Tied methods are not presented as a unique winner.

### Within tolerance

Tolerances are optional and user-defined. Absolute ±5 years and relative ±5% are
different rules. Relative tolerance excludes zero reference values; its denominator
is saved in `field_scores.csv`. No generic 5% tolerance is silently applied to
calendar years, dates or other arbitrary-origin quantities.

### Limits of these findings

Cross-validation measures the configured recovery problem. It does not certify
every natural gap, identify all wrong recorded values, establish causation, or
validate future missingness patterns. Point fills can compress variation and
distort counts at the extremes. The report includes matched distribution charts
for this reason. Generic confidence calibration, record screening and operational
auto-fill gates are not implemented in this dataset route.

This route runs one configured condition at a time. A nonzero noise run alone
does not establish a noise sensitivity curve; use separate, recorded conditions
for that comparison. The fixed historical benchmark already supplies its own
multi-condition study.

## 9. Reports and files produced

Each dataset run contains:

```text
report.html                   Offline interactive report; charts embedded
summary.json                  Dataset, configuration and scored summary
config.json                   Reusable field/method/experiment settings
manifest.json                 Input hashes, code hashes and environment
run_status.json               Success, partial completion or failure
field_scores.csv              Numeric and categorical metrics by field/method
validation_predictions.csv    Held-out per-cell references, fills and baseline
runtime.csv                   Validation and final-fit timings
completed_METHOD.csv          Full input with that method's new fills
fills_METHOD.csv              Only original gaps and their completed values
warnings.json                 Model/library warnings
failures.json                 Failed methods/folds or export stages
figures/                      Comparison, missingness and distribution PNGs
```

Ignored columns remain in completed exports, including any gaps they had. Recorded
values in modelled columns are restored before export. CSV reserialisation may
change formatting or tiny decimal roundoff; it is not a byte-for-byte copy of the
input text. The fills ledger explicitly says individual fills are not validated.
Its `row_number` is the one-based record number, excluding the CSV header.

A generic report includes a dataset/design section, method comparison, searchable
field table, figures with interpretations, limits, completed data and reproducibility
details. Numeric and categorical sections appear when applicable. Detailed
distribution plots cover up to eight numeric fields, the skill map up to 24 and the
association plot up to 16; complete CSV metrics cover all evaluated fields.

Use **Print / save PDF** in the report. The print action loads all figures and
clears filters for the printed result. All charts travel inside the HTML. Keep
the HTML in its run folder for CSV downloads; moving the HTML alone breaks those
local download links.

## 10. Repeat a new CSV run from the command line

```sh
python run_study.py --mode dataset \
  --dataset examples/example_assets.csv \
  --truth examples/example_truth.csv \
  --config examples/example_config.json \
  --out outputs/example-with-truth
```

Without truth, omit `--truth`. Without a config, the program suggests field types
and uses the default three methods. Reviewing a schema in the browser first is
preferable to relying on inference for numeric category codes or identifier fields.

Download `config.json` from a completed report to repeat its settings. Use a new
output directory; existing nonempty runs are rejected. Dataset-mode jobs do not
currently resume partially fitted models. Do not use the original benchmark's
`--resume` or `--run-id` flags as a substitute.

Dataset CLI exit codes are 0 for a complete run, 2 for a report with model-stage
errors, and 1 for a failed run or invalid input. A partial report is useful for
diagnosis, but its missing methods are not presented as a complete comparison.

## 11. Original benchmark and saved predictions

The preserved shared study uses one 10,000-row synthetic transformer population,
five row folds plus additional experiments, eleven imputers and median/mode. Its
reference condition is 20% missingness and 5% noise. The original report averages
13 numeric fields, excluding the random placebo. Model-ID accuracy is separate.

The original GMM has six components, up to eight EM steps, regularisation 0.001
and category coordinate weight 1.8. Its generator-specific features are disabled
in that shared protocol. These are different settings from a generic new CSV run
and from the earlier Subteam A spatial/confidence study.

```sh
python run_study.py
python run_study.py --mode quick --workers 2
python run_study.py --mode full --workers 4
```

The first command regenerates the report from `reference_results/` without any
fitting. Quick/full execute the original notebook's Python cells, model runner,
evaluation and figures. Quick is a diagnostic, not research evidence. The original
full run took about 2 hours 20 minutes with 12 workers; runtime depends on hardware.

### Package 03 and original-run reevaluation

The client ZIP combines the original code/input package 01 and result/figure
package 02. Original package 03 is **not** required to read results or fit again.
It is required for reevaluating its original saved predictions without fitting.
Place that run at:

```text
outputs/imputation-study/runs/20261115-121714/
```

Then use:

```sh
python run_study.py --mode evaluate --run-id 20261115-121714
```

The run contract must match its input hashes, prediction code and configuration.
Fresh quick/full fits do not need package 03. Their predictions are saved in their
own timestamped run folders.

```sh
python run_study.py --mode quick --resume RUN_ID --workers 2
python run_study.py --mode evaluate --run-id QUICK_RUN_ID --quick-evaluation
```

Use the matching original configuration file when resuming or evaluating it.
Successful checkpoints are reused; failed tasks are retried. Prediction changes
require a new original-format run. `--keep-inputs` skips original input generation.

## 12. Parameter-selection evidence

```sh
python scripts/selection_audit.py
python scripts/selection_audit.py --model em --round 1
```

The audit reads the original candidate scores and selection rules. It recovers all
eight supplied final choices without fitting. Intermediate round 2 has no supplied
chosen JSON, so no historical-match assertion is made for that round.

The original selection `run.py`, `run2.py` and `run3.py` were unavailable. The
explicit replacement runner can refit their recorded candidate configurations:

```sh
python scripts/refit_selection.py --model em
python scripts/refit_selection.py --model em --round 1 --rows 600 --limit 2
```

It never reads truth CSVs. It hides another 10% of visible test-fold cells, computes
numeric MAE ratios, rejects placebo ratios above 1.01 and selects the fastest
candidate within 0.002 of the best. Its remasking seed is `seed + 1000 + fold`;
historical remasking draws were unavailable. New choices are written separately
and are never silently substituted into published results.

## 13. Project structure and code walkthrough

```text
START_HERE.md                 Short setup and run guide
README.md                     Setup, usage and methodology guide
docs/TESTING.md               Browser test checklist and automated UI checks
SUBMISSION.md                 GitHub/Jira/client submission map
run_study.py                  One entry point; adds local src to the import path
Start_Workbench.command       macOS launcher after setup
Start_Workbench.bat           Windows launcher after setup
requirements.txt              Original pinned dependency environment
src/synthetic_imputation/
  cli.py                      Dispatch report/benchmark/guided/dataset commands
  dataset_app.py              Local upload, configuration and progress server
  dataset.py                  Schema checks, paired evaluation and CSV exports
  dataset_models.py           Generic-schema model adapters
  dataset_reporting.py        New-dataset figures, findings and report
  reporting.py                Archived/fresh original-benchmark report
  execution.py                Original notebook/generator execution helpers
  runner.py, study.py          Original fixed experiment plan and orchestration
  evaluation.py, failure.py    Original metrics and failure analyses
  models.py, models_external.py Original benchmark models and adapters
  vendor/                     Preserved team implementations, including GMM
  templates/                  Inlined report/app HTML, CSS and interactions
data/                         Exact original synthetic input pairs/generator
examples/                     Separate arbitrary-schema demonstration
reference_results/            Original result exports, settings and figures
reports/published/            Ready-to-open original-results report
outputs/                      New runs/uploads; excluded from Git/client ZIP
notebooks/imputation.ipynb    Original interactive benchmark
tests/                        Original and guided-workflow tests
tools/build_handover.py       Clean client ZIP builder and integrity gate
```

For a short walkthrough: `run_study.py` enters `cli.py`; guided mode opens
`dataset_app.py`; the chosen config goes to `dataset.run_dataset`; the selected
adapters fit through `dataset_models.py`; `dataset_reporting.py` saves figures
and renders the report. The GMM equations remain in
`vendor/subteam_a/mixture.py`. The new adapter changes schema handling and output
formatting, not those EM equations.

## 14. Verification and release construction

Use the [testing guide](docs/TESTING.md) for the browser walkthrough, negative
checks, grouped/noise tests and the optional Playwright browser smoke test.
The Python tests cover the scientific workflow and server behaviour; browser
tests also check the controls, progress, report and downloads as a user sees them.

```sh
python -m pip check
python -m pytest -q
python scripts/selection_audit.py
python run_study.py
python tools/build_handover.py --out /path/outside/project/Team19_Automated_Handover.zip
```

Build with the Python script rather than compressing a working directory in
Finder. It verifies the original scientific hashes, refreshes the package
manifest, and excludes environments, caches, build files, macOS metadata and
generated run/upload folders. `.gitattributes` preserves original file bytes
across platform checkouts.

Extract the ZIP to a fresh folder and check the documented entry points before
sharing it. Keep its checksum and source commit with the release. Tests, actual
completed runs and limitations are recorded in `VERIFICATION.md` and audit JSONs.

The shared source and the client ZIP belong to PGS219-265 (Final Product). The
agreed written report belongs to PGS219-266 (Final Report). See `SUBMISSION.md`.
Do not describe a pushed PR as merged, a report rebuild as a fresh fit, or quick
diagnostics as the full research experiment.

## 15. Troubleshooting

| Symptom | What to check |
| --- | --- |
| `ModuleNotFoundError` | Activate this folder's Python 3.12 environment and install its requirements. Use its interpreter explicitly if needed. |
| Wrong Python version | Check `python --version`; do not reuse the separate Subteam A Python 3.11 environment. |
| Compiled-library loading error | Check the installed wheel/platform and the model library's required system runtime. Preserve the pinned package versions when resolving environment issues. |
| Workbench port is already used | Reuse the existing workbench or start with `--port 8766`. |
| Run button stays disabled | Choose the file and click Load dataset. Changing a file invalidates the previous inspection. |
| Column cannot parse numerically | Change its role to categorical/date or correct its values; do not silently coerce errors into missing values. |
| Date cannot parse | Use ISO dates, or explicitly convert the source before running. Numeric years should remain numeric. |
| Grouped-mask error | Give at least two related modelled fields the same nonempty group name. |
| Noise-column error | Select the numeric measurement fields to perturb; leave dates and category codes out. |
| No training observations in a fold | Reconsider sparse fields, folds or seed; exclude fields that cannot be learned rather than inventing values. |
| Model error but report available | Inspect failures.json and the report. Incomplete validation models are not given a favourable partial score. |
| Iterative convergence warning | The configured iteration budget may be exhausted. Warnings are retained separately; this is not automatically a failed fit. |
| Existing output folder rejected | Choose a new folder. Dataset runs are not silently overwritten or resumed. |
| Missing saved original contract | Install package 03 for the original archived run, or choose fresh quick/full fitting. |
| HTML charts work but downloads do not | Keep the HTML with its accompanying run/reference folders. Embedded images are portable; CSV links are relative. |
| Results differ from the old report | Check workflow, dataset, fold design, missingness, feature treatment and parameters. Generic runs and the earlier Subteam A study are separate protocols. |

## 16. Scope

The new workbench is a local research/evaluation tool for supported CSV schemas,
not an operational SAPN register integration. It preserves recorded inputs and
produces reviewable candidate fills. It does not automatically merge different
asset populations, infer all engineering relationships, correct wrong recorded
values, justify tolerances for a client, or certify automatic acceptance.

The original synthetic studies have their own limits: one designed population,
controlled degradation and no validation on live SAPN records. The earlier
Subteam A uncertainty analysis remains useful supporting evidence, with its
documented weak-pattern limitations; it is not silently transferred to a new CSV.
