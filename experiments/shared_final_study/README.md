# Team 19 synthetic asset imputation study

Version 5.1 — automated execution and report handover.

**Read [START_HERE.md](START_HERE.md), or open [the complete supplied-results report](reports/published/report.html).**

This project compares eleven imputers and a median/mode baseline on synthetic asset data. It evaluates when accuracy degrades, how errors change and which missing relationships explain the failures. Results describe the synthetic generator; they do not establish performance on operational SAPN data.

## Installation

Use Python **3.12** and a separate virtual environment. The pins in `requirements.txt` preserve the numerical/model dependencies used by the original study. Jinja2 is explicitly pinned for the new report template.

```sh
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The launcher adds the local `src/` directory automatically. An editable installation is optional: `python -m pip install -e .` also exposes `imputation-study` as an alias for the launcher.

## Execution modes

| Command | Behaviour | Output |
|---|---|---|
| `python run_study.py` | Builds a report from the supplied full-study CSVs and PNGs. No fitting | `reports/published/report.html` |
| `python run_study.py --mode quick --workers 2` | Regenerates original inputs; runs every stage on 600 rows with small models | Timestamped report and run; clearly labelled diagnostic |
| `python run_study.py --mode full --workers 4` | Regenerates inputs; runs the full 3,580-task experiment, evaluation, charts and report | Timestamped report and run |
| `python run_study.py --mode evaluate --run-id RUN_ID --workers 2` | Repeats evaluation from saved predictions after the original contract check | Timestamped report |

The original full run took about 2 hours 20 minutes using 12 workers; time depends on hardware. Choose worker count for available memory. Both fresh modes execute the same Python cells as `notebooks/imputation.ipynb`, without needing a notebook server. The launcher uses the original Study, runner, models, scoring and figure functions.

All chart-generating notebook cells are executed before a fresh report is created. The interactive notebook also generates a report automatically after Run All. Its default is full fresh fitting; set `QUICK = True` for a diagnostic run.

### Saved predictions and resume

Original package 03 is **not required** to train again. To reevaluate the original archived predictions, place its run at `outputs/imputation-study/runs/20261115-121714/`, then use `--mode evaluate --run-id 20261115-121714`. Package 02 alone contains aggregate results, not those raw predictions.

```sh
# Resume a matching interrupted quick run. Copy RUN_ID from execution.log.
python run_study.py --mode quick --resume RUN_ID --workers 2

# Reevaluate a completed quick run (its configuration must also be quick).
python run_study.py --mode evaluate --run-id RUN_ID --quick-evaluation
```

Resume/evaluation requires the same input hashes, prediction code and configuration. Successful task checkpoints are reused; failed task markers are retried during resume. Use a fresh run after changing prediction code or configuration. The original prediction modules were retained in this update, so their hashes have not been changed by report/launcher work.

### Options and outputs

- `--out PATH`: choose the report directory.
- `--source PATH`: in report mode, use another complete result export or completed run.
- `--keep-inputs`: skip original-generator execution before a fresh run.
- `--config PATH`: JSON overrides of `StudyConfig`, including nested model settings. These become part of the run contract. Use the same file when resuming/evaluating that configuration.

For example, a new investigation with a different GMM component count can use:

```json
{"parameters": {"Gaussian EM": {"clusters": 4}}}
```

This produces a separately labelled new run; it does not alter the supplied published results or justify relabelling them.

A generated report directory contains `report.html`, `summary.json`, `manifest.json` and `run_status.json`. Fresh runs also have `execution.log`. Detailed predictions, task metadata, metrics, charts and completed reference tables live under the printed `outputs/imputation-study/runs/RUN_ID/`. Links in the report point to those CSVs. The HTML embeds every chart and needs no network connection; CSV downloads require the accompanying handover/run folders. The browser's Print / save PDF includes the charts.

## Parameter selection

The result package contains the original candidates, scores, rules and chosen/final JSON settings. Check them without fitting:

```sh
python scripts/selection_audit.py
python scripts/selection_audit.py --model em --round 1
```

The default audit recovers all eight published choices. Intermediate round 2 does not have a supplied chosen JSON, so its computed choice is reported without a historical-match assertion.

The original `run.py`, `run2.py`, `run3.py` selection scripts were absent. A new replacement can refit their recorded candidate configurations:

```sh
# Full candidate evaluation, three folds and three 70%-complete measured datasets.
python scripts/refit_selection.py --model em

# Small diagnostic check only; not a research selection result.
python scripts/refit_selection.py --model em --round 1 --rows 600 --limit 2
```

Available model IDs: `knn`, `ridge`, `forest`, `hgb`, `cat`, `mice`, `em`, `mlp`. The new runner never reads truth CSVs. It hides another 10% of visible test-fold cells, computes numeric MAE ratios, rejects placebo ratios above 1.01 and selects the fastest candidate within 0.002 of the best. Its explicit remasking seed is `seed + 1000 + fold`; historical draws were not supplied. Fresh choices can differ from the archived choices and are written separately, never applied silently to the main study. Donor-matched exploratory variants are retained as historical records, not presented as reproduced implementations.

## Scientific definitions and limits

- The original nine scenarios cross 10%, 20%, 30% missingness with 0%, 5%, 10% noise, on one 10,000-row population. The six added scenarios test age/value-dependent gaps, grouped gaps, recorded-value errors and 50%/70% missingness.
- Fit on four measured row folds; fill the fifth; score only originally missing cells. Additional experiments use whole spatial blocks. Truth is used for evaluation, not model fitting.
- Numeric skill is `1 − model MAE / median MAE`. The presentation's mean excludes the independent placebo and uses 13 numeric fields. Model-ID accuracy is separate. Some notebook condition tables additionally include categorical skill; labels in the report explain this distinction.
- IterativeImputer supplies no model-ID classifier; its 1,965 missing reference model IDs remain empty. This is intentional and reported.
- The GMM uses six components, a shared full covariance and incomplete-data EM; at most eight steps, regularisation 0.001, category weight 1.8. It is imported from Subteam A and adapted to this shared protocol with generator-specific features disabled.
- Synthetic relationships and one population limit generalisation. Numeric estimates are points; calibrated per-asset uncertainty intervals are not evaluated here. Error-comparison intervals are a different quantity.

The original complete study/method description is retained in [docs/ORIGINAL_STUDY_V5.md](docs/ORIGINAL_STUDY_V5.md) as historical documentation. Its old run-mode instructions are superseded by this README. [CHANGELOG.md](CHANGELOG.md) records the update, and `VERIFICATION.md` records validation.

## Tests

```sh
python -m pytest -q
```

The original model tests are retained. Added checks cover report provenance, numeric/categorical aggregation, invalid inputs, the selection rule and configuration overrides.
