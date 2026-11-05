> [!IMPORTANT]
> **Archive: version 2 of the synthetic imputation study.** This folder holds the code of version 2 as it ran
> in run `20261105-221748` (2026-11-05). The code was developed locally and imported later under PGS219-274 (subtask PGS219-276).
> Data and saved runs are kept outside the repository, so paths to `data/` and `outputs/imputation-study/runs/`
> below do not resolve here; the notebook's saved outputs are the record of the run. The later versions follow in the Git history of this folder. This branch is an archive and is not merged into `main`.

# Synthetic imputation study, version 2: models optimised one at a time

2026-11-05. One version of the imputation study on a synthetic asset dataset. Each version of the study has its own
folder in `gh`, and `gh/README.md` lists them.

Each model was optimised on its own terms. The chained models gained a ridge base on second-order terms with a
local model fitted to what the ridge leaves, and every setting was chosen by a fixed rule on cells that no scenario
hides, without reading truth (`outputs/imputation-study/optimisation/clean_selection/RULE.md`). Tuning was taken out
of the run, so the plan fell to 1,580 tasks.

## Runs

| Run | Tasks | What it is |
|---|---:|---|
| `20261105-221748` | 1,580 | The full run of this version. The notebook shows it. |
| `20261105-221141` | 825 | A run that was stopped part way. |

## Layout

| Path | Content |
|---|---|
| `notebooks/imputation.ipynb` | The notebook, saved with the outputs of the run named below |
| `src/team19_asset_data/imputation_study/` | The study code, exactly as it ran |
| `src/team19_asset_data/transformers/miceforest_compat.py` | The miceforest 6.0.5 compatibility code the study imports |
| `tests/` | The tests of this version |
| `data/` | The measured and truth files this version read |
| `outputs/imputation-study/runs/` | Every saved prediction, table and figure of this version's runs |
| `outputs/imputation-study/optimisation/` | Every development round of the optimisation and the final clean selection |
| `history/` | Two states of the code during the optimisation: before KNN was optimised, and before a leakage audit fix |

The code keeps the package path it had in the team repository, `team19_asset_data.imputation_study`, so that the
hashes in each run's `contract.json` still match the code. This was checked when the version was moved here:
the code and data of this folder reproduce the contract of run `20261105-221748`.

## Running

Create the environment from `requirements.txt` (Python 3.12) and register it as the kernel **Python (imputation)**:

```
conda create -n imputation python=3.12
conda activate imputation
pip install -r requirements.txt
python -m ipykernel install --user --name imputation --display-name "Python (imputation)"
```

Then open `notebooks/imputation.ipynb` from this folder. Restart and Run All fits every model again. `RESUME_RUN_ID` continues a run with the same code, data and settings.
