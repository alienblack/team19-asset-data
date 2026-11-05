> [!IMPORTANT]
> **Archive: version 3 of the synthetic imputation study.** This folder holds the code of version 3 as it ran
> in run `20261105-234621` (2026-11-05). The code was developed locally and imported later under PGS219-274 (subtask PGS219-277).
> Data and saved runs are kept outside the repository, so paths to `data/` and `outputs/imputation-study/runs/`
> below do not resolve here; the notebook's saved outputs are the record of the run. The later versions follow in the Git history of this folder. This branch is an archive and is not merged into `main`.

# Synthetic imputation study, version 3: evaluation from saved predictions

2026-11-06. One version of the imputation study on a synthetic asset dataset. Each version of the study has its own
folder in `gh`, and `gh/README.md` lists them.

The evaluation was rebuilt so that it can start from a finished run without fitting (`EVALUATION_RUN_ID`), and the
run contract covers only the code that makes predictions. Two flaws were fixed: the calibrated forest now uses the same
forest as the raw result, and every chained model was offered the same range of passes. New checks compare the models
with the generator's relationships: recovery of each relationship, the columns models rely on, correlation between
columns and accuracy when related columns are hidden. The shareable package in
`archive/shared-packages/imputation-study-package` was made from this version.

## Runs

| Run | Tasks | What it is |
|---|---:|---|
| `20261105-234621` | 1,580 | The full run of this version. The notebook shows it. |

## Layout

| Path | Content |
|---|---|
| `notebooks/imputation.ipynb` | The notebook, saved with the outputs of the run named below |
| `src/team19_asset_data/imputation_study/` | The study code, exactly as it ran |
| `src/team19_asset_data/transformers/miceforest_compat.py` | The miceforest 6.0.5 compatibility code the study imports |
| `tests/` | The tests of this version |
| `data/` | The measured and truth files this version read |
| `outputs/imputation-study/runs/` | Every saved prediction, table and figure of this version's runs |
| `outputs/imputation-study/optimisation/clean_selection/` | The selection rule and scores, which section 5 reads |
| `outputs/imputation-study/baselines/v1/run_tables/numeric_scores.csv` | Version 1 scores, which section 5 compares with |

The code keeps the package path it had in the team repository, `team19_asset_data.imputation_study`, so that the
hashes in each run's `contract.json` still match the code. This was checked when the version was moved here:
the code and data of this folder reproduce the contract of run `20261105-234621`.

## Running

Create the environment from `requirements.txt` (Python 3.12) and register it as the kernel **Python (imputation)**:

```
conda create -n imputation python=3.12
conda activate imputation
pip install -r requirements.txt
python -m ipykernel install --user --name imputation --display-name "Python (imputation)"
```

Then open `notebooks/imputation.ipynb` from this folder. Restart and Run All fits every model again (about 1 hour 25 minutes). To repeat only the evaluation, set `EVALUATION_RUN_ID = "20261105-234621"` in the setup cell.
