# Tools used to build and run the study

- `build_notebook.py`: rebuilds `notebooks/imputation.ipynb` from `base_notebook.ipynb`, the version 3 notebook,
  (chapters 1 to 4 are adapted, the rest is written here). `RUN_ID` and `RUN_TIME` at the top set the run the
  saved notebook points to.
- `fit_driver.py`: fits every model family outside the notebook. `python fit_driver.py full [run_id]` from the
  repository root; with a run id it continues that run.
- `run_nb.py`: executes a notebook in the `imputation` kernel and saves its outputs.

The latest full run is `../runs/20261115-030132` (3,580 tasks). `../runs/20261115-012041` is the earlier run of the
same version without the range limit in the chained models, kept as the record of their divergence at 0.3
completion; see `PROVENANCE.md` in the latest run.
