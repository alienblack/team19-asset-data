> [!IMPORTANT]
> **Archive: version 4 of the synthetic imputation study.** This folder holds the code of version 4 as it ran
> in run `20261114-114941` (2026-11-14). The code was developed locally and imported later under PGS219-274 (subtask PGS219-278).
> Data and saved runs are kept outside the repository, so paths to `data/` and `outputs/imputation-study/runs/`
> below do not resolve here; the notebook's saved outputs are the record of the run. The later versions follow in the Git history of this folder. This branch is an archive and is not merged into `main`.

# Synthetic imputation study, version 4: when, how and why imputation fails

2026-11-14. One version of the imputation study on a synthetic asset dataset. Each version of the study has its own
folder in `gh`, and `gh/README.md` lists them.

The evaluation was reorganised around three questions: the conditions under which imputation degrades or fails,
the form the failure takes, and its causes (`failure.py`). The notebook gives an overview first and the detail after,
explains how to read each result and states no conclusions. The two donor-matched miceforest variants were dropped.
Experiments were added: scenarios with gaps that depend on asset age or on the value, the chained models without
their ridge base, a wider removal of predictor blocks and importance on all five folds.

## Runs

| Run | Tasks | What it is |
|---|---:|---|
| `20261114-114941` | 2,230 | The full run of this version. The notebook shows it. |
| `20261114-015422` | 2,230 | An earlier run of the same code, with identical predictions. |

## Layout

| Path | Content |
|---|---|
| `notebooks/imputation.ipynb` | The notebook, saved with the outputs of the run named below |
| `src/team19_asset_data/imputation_study/` | The study code, exactly as it ran |
| `src/team19_asset_data/transformers/miceforest_compat.py` | The miceforest 6.0.5 compatibility code the study imports |
| `tests/` | The tests of this version |
| `data/` | The measured and truth files this version read |
| `outputs/imputation-study/runs/` | Every saved prediction, table and figure of this version's runs |
| `data/datasets_mechanisms/` | The scenarios with gaps by age and by value |
| `outputs/imputation-study/optimisation/clean_selection/` | The selection rule and scores of the settings |
| `outputs/imputation-study/tools/build_notebook.py` | The script that built this notebook |

The code keeps the package path it had in the team repository, `team19_asset_data.imputation_study`, so that the
hashes in each run's `contract.json` still match the code. This was checked when the version was moved here:
the code and data of this folder reproduce the contract of run `20261114-114941`.

## Running

Create the environment from `requirements.txt` (Python 3.12) and register it as the kernel **Python (imputation)**:

```
conda create -n imputation python=3.12
conda activate imputation
pip install -r requirements.txt
python -m ipykernel install --user --name imputation --display-name "Python (imputation)"
```

Then open `notebooks/imputation.ipynb` from this folder. The notebook is saved with `EVALUATION_RUN_ID = "20261114-114941"`, so Run All repeats only the evaluation. Set it to `None` to fit every model again (about 1 hour 45 minutes).
