# Transformer Asset Imputation Study

The single active notebook for the transformer experiments is:

`notebook/transformer_imputation_experiments.ipynb`

It loads the supplied measured/truth synthetic CSV pairs, explores distributions and generator relationships, runs the compatible imputation methods, and evaluates T2–T6 and the relationship experiments. The main notebook compares Median/Mode, Original MissForest, Predictive MissForest, Chained Ridge, LightGBM chained and MICE-PMM/logistic. TabPFN is evaluated separately at `tabpfn_separate/transformer_tabpfn_experiment.ipynb`.

For installation, complete experiment definitions, evaluation protocols, reproducibility controls, results and limitations, see [`../../TRANSFORMER_EXPERIMENT_GUIDE.md`](../../TRANSFORMER_EXPERIMENT_GUIDE.md).

Run from the `pipeline/` root with the project environment activated:

```bash
python -m jupyter lab experiment/transformer/notebook/transformer_imputation_experiments.ipynb
```

Quick mode is intended for setup testing; full mode uses the research configurations. The measured/truth source CSVs are never overwritten by the notebook.
