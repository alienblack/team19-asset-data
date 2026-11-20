# Team 19 - start here

This package offers two separate routes:

| What you want to do | Start with |
| --- | --- |
| Read the supplied final comparison | Open [reports/published/report.html](reports/published/report.html); no installation is needed. |
| Choose your own CSV and generate a new report | Install the environment below, then run `python run_study.py --mode guided`. |
| Reproduce the original synthetic benchmark | Use `--mode quick` or `--mode full`; see the README. |

The **new dataset workbench does not change the original benchmark's model code,
inputs or saved scores**. A new CSV run has its own schema, parameters, validation
protocol and outputs. It is not automatically a reproduction of the old study.

## Install once

Use **Python 3.12** in this folder. Keep this environment separate from the
earlier Subteam A benchmark's Python 3.11 environment.

macOS / Linux:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
```

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip check
```

If PowerShell activation is restricted, use `.venv\Scripts\python.exe` instead
of `python` in the subsequent commands; changing execution policy is unnecessary.

## Open the dataset workbench

```sh
python run_study.py --mode guided
```

Your browser opens at `http://127.0.0.1:8765`. Choose a CSV, confirm field types,
choose methods and parameters, then press **Run study & generate report**. Click
**Use example data** for a first check: expect 240 records, 7 columns and 207
missing cells. For uploads, choose the file and click **Load dataset**. A matching clean truth CSV is
optional. Without it, validation hides known recorded values on held-out rows.

After setup, macOS users can also open `Start_Workbench.command`; Windows users
can open `Start_Workbench.bat`. Keep the terminal running while the app is in use.
Ctrl+C stops the local workbench. Uploaded data and generated runs remain local
under `outputs/dataset-workbench/`; they are excluded from the client ZIP and Git.

The report includes comparison tables, field results, charts with interpretations,
completed CSVs, fill ledgers, settings, warnings and provenance. Use **Print / save
PDF** in the report for a printable copy. Keep the HTML with its run folder for
CSV download links to work; all charts are embedded.

## The original study

```sh
python run_study.py                           # rebuild archived-results report
python run_study.py --mode quick --workers 2 # small diagnostic, not research findings
python run_study.py --mode full --workers 4  # fresh full synthetic benchmark
```

The archived full-study result is run `20261115-121714`. Original package 03 is
not needed to read it or fit models again. It is required to reevaluate that
original run's saved per-cell predictions without fitting.

Read [README.md](README.md) for the full methodology, field/parameter reference,
file map, commands, troubleshooting and limits. [SUBMISSION.md](SUBMISSION.md)
maps this release to GitHub and Jira. [The testing guide](docs/TESTING.md) gives
a browser checklist and optional automated UI checks. Validation is recorded in `VERIFICATION.md`
and the audit JSON files.
