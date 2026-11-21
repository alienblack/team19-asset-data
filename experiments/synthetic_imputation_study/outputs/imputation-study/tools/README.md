# Tools

Scripts that build and run the notebook and fit the models outside it. Run them from the project folder.

| Script | Use |
|---|---|
| `build_notebook.py` | Writes `notebooks/imputation.ipynb`: every text and code cell, with numbered captions for every table and figure, is written in the script. Run `python outputs/imputation-study/tools/build_notebook.py`, then execute the notebook to save its outputs. |
| `base_notebook.ipynb` | Supplies the notebook metadata (kernel and language) to `build_notebook.py` |
| `run_nb.py` | Executes a notebook in the `imputation` kernel and saves its outputs |
| `fit_driver.py` | Fits every model section outside the notebook: `python outputs/imputation-study/tools/fit_driver.py full [run_id]`; with a run id it continues that run |

The final run is `../runs/20261115-121714` (3,580 tasks).
