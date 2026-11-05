> [!IMPORTANT]
> **Archive: version 1 of the synthetic imputation study.** This folder holds the code of version 1 as it ran
> in run `20261105-133845` (2026-11-05). The code was developed locally and imported later under PGS219-274 (subtask PGS219-275).
> Data and saved runs are kept outside the repository, so paths to `data/` and `outputs/imputation-study/runs/`
> below do not resolve here; the notebook's saved outputs are the record of the run. The later versions follow in the Git history of this folder. This branch is an archive and is not merged into `main`.

# Synthetic imputation study, version 1: the first full study

2026-11-05. One version of the imputation study on a synthetic asset dataset. Each version of the study has its own
folder in `gh`, and `gh/README.md` lists them.

The first complete run of the study on the nine generator scenarios (completion 0.7, 0.8 and 0.9 by noise 0,
0.05 and 0.1). Models: median and mode, KNN imputer, IterativeImputer, Random Forest, HistGradientBoosting,
CatBoost, miceforest direct, miceforest PMM and joint miceforest PMM. Settings were tuned inside the run, and the
experiments were main, stability, defaults (tuned against default settings), budget, pmm, joint, calibration,
ablation and importance.

## Runs

| Run | Tasks | What it is |
|---|---:|---|
| `20261105-133845` | 8,820 | The run of this version's code on the generator's regenerated files (`data/datasets`). The notebook shows it. |
| `20261105-114544` | 5,699 | An earlier run on the first release of the files (`data/datasets_v1`), before the dataset was regenerated with the generator's author. The code of that run was not kept. |

## Layout

| Path | Content |
|---|---|
| `notebooks/imputation.ipynb` | The notebook, saved with the outputs of the run named below |
| `src/team19_asset_data/imputation_study/` | The study code, exactly as it ran |
| `src/team19_asset_data/transformers/miceforest_compat.py` | The miceforest 6.0.5 compatibility code the study imports |
| `tests/` | The tests of this version |
| `data/` | The measured and truth files this version read |
| `outputs/imputation-study/runs/` | Every saved prediction, table and figure of this version's runs |

The code keeps the package path it had in the team repository, `team19_asset_data.imputation_study`, so that the
hashes in each run's `contract.json` still match the code. This was checked when the version was moved here:
the code and data of this folder reproduce the contract of run `20261105-133845`.

## Running

Create the environment from `requirements.txt` (Python 3.12) and register it as the kernel **Python (imputation)**:

```
conda create -n imputation python=3.12
conda activate imputation
pip install -r requirements.txt
python -m ipykernel install --user --name imputation --display-name "Python (imputation)"
```

Then open `notebooks/imputation.ipynb` from this folder. Restart and Run All fits every model again. `RESUME_RUN_ID` continues a run with the same code, data and settings.
