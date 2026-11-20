# Team 19 — automated study handover

Start by opening **[the complete report](reports/published/report.html)** in a web browser. It contains the supplied full-study results, every chart, a searchable field-level table and links to the CSV exports. No installation is needed to read it. The charts are embedded; keep the report inside this folder for CSV links to work.

The report is based on the supplied run **20261115-121714**. It is labelled as archived full-study evidence, rather than a new model fit. `reference_results/` contains the original result tables, completed datasets and figures from package 02.

## Generate the report again

Install Python **3.12**, then run these commands in this folder:

```sh
python -m venv .venv
# macOS / Linux:
source .venv/bin/activate
# Windows PowerShell instead:
# .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python run_study.py
```

Open `reports/published/report.html`. The command reads the supplied result exports, creates the report and records its source hashes. It does not train models.

## Run the experiment

```sh
# Small installation check: all stages, 600 rows, small models.
python run_study.py --mode quick --workers 2

# Rebuild the full study: inputs, models, evaluation, CSVs, charts and report.
python run_study.py --mode full --workers 4
```

Each command prints the report location. A quick-run report is prominently marked **not research findings**. The full run has 3,580 tasks; the original full run took about 2 hours 20 minutes with 12 workers. Time depends on hardware. Allow the process to finish; its detailed progress is in the printed `execution.log`.

The report offers model/field filtering, full-size chart inspection, PNG downloads, CSV links and **Print / save PDF** through the browser. `summary.json` contains machine-readable results; `manifest.json` records source and code hashes.

## What is included

- `run_study.py`: one command for all execution modes.
- `src/`, `data/`, `notebooks/`: study code, exact synthetic inputs and the interactive notebook.
- `reference_results/`: supplied full-study results and tuning records.
- `reports/published/`: generated full-study report, summary and provenance.
- `scripts/selection_audit.py`: checks the eight published parameter-selection choices.
- `scripts/refit_selection.py`: a new, explicit measured-data-only tuning runner.
- `README.md`: resume/evaluation commands, configuration and limitations.
- `CHANGELOG.md`: exactly what changed in this handover update.

The separate original package 03 is optional for a fresh model run. It is required only to reevaluate its original saved per-cell predictions without refitting. No confidential SAPN records are included.

For GitHub, Jira and client submission, see [SUBMISSION.md](SUBMISSION.md). Build
the client ZIP with `tools/build_handover.py`; do not compress a working folder
containing a virtual environment or generated experiment runs.

`FINAL_SUBMISSION_AUDIT.json` records the latest unit tests, fresh extracted
quick-run check, original-file integrity checks and report checks.
