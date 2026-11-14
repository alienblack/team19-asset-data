# Capstone Asset Imputation Pipeline

Team 19 · University of Adelaide · SA Power Networks collaboration

This repository contains the synthetic-data imputation research pipeline, including the transformer experiments and shared imputation implementations.

**Transformer study:** [Experimental methodology and reproducibility guide](TRANSFORMER_EXPERIMENT_GUIDE.md). Run `experiment/transformer/notebook/transformer_imputation_experiments.ipynb` for data exploration, T2–T6 experiments, relationship analyses, model comparison and visualisation. The TabPFN investigation is in its own notebook under `experiment/transformer/tabpfn_separate/`.

**Fresh installation (Python 3.12):**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[notebook,test]"
python -m jupyter lab experiment/transformer/notebook/transformer_imputation_experiments.ipynb
```

The main notebook defaults to a quick fresh run. Choose `MODE='full'` in its settings cell to reproduce the research-scale experiments. Set `EXECUTE_EXPERIMENTS=False` and select `RESULTS_SOURCE_DIR` to analyse an existing run without refitting. See the guide for configurations, output descriptions, method-protocol distinctions and scientific limitations.

---

University of Adelaide / SA Power Networks capstone prototype for configurable
asset-data quality checking, profiling, missing-value imputation, and paired
ground-truth evaluation. The included data are **synthetic examples**, not
production SAPN asset records. Results on them do not imply real-world accuracy.

## Project layout

```text
pipeline/
  config/
    overhead_conductors/    # Independent schema, cleaning, imputation and run config
    transformers/           # Provisional schema for the supplied paired sample
    underground_cables/     # Placeholder until teammate supplies an asset schema
  data/
    overhead_conductors/
    transformers/
  notebook/
    pipeline_overhead_conductors.ipynb
    pipeline_transformer.ipynb
  src/                       # Python modules live DIRECTLY under src/
    configuration/ ingestion/ quality/ profiling/ preprocessing/
    imputation/
      median_mode/ missforest/ predictive_missforest/
    evaluation/ postprocessing/ reporting/
    cli.py  pipeline.py  settings.py  notebook.py
  tests/ docs/ outputs/
  pyproject.toml  requirements.txt
```

No extra `capstone_pipeline/` package directory is required. The Python
package name is `src`, and both `oc-pipeline` (legacy alias) and
`asset-pipeline` invoke `src.cli:main`.

## Install

From this `pipeline/` directory:

```bash
python3 -m venv .pipeline
source .pipeline/bin/activate
python -m pip install -e '.[test,notebook]'
python -m src.cli --help
python -m pytest -q tests/
```

Never include `.venv/`, `__pycache__/`, `outputs/`, or private asset records in
a distributable archive or Git commit.

## Run two-file experiments

Each experiment has **exactly two source files**: a measured CSV, which may be
incomplete and noisy, and a matching ground-truth CSV. No separate masked CSV
is required. Record partitions and the scoring mask are generated as outputs.

```bash
python -m src.cli all \
  --asset transformers \
  --measured data/transformers/completion0.9-noise0.05-measured.csv \
  --truth data/transformers/completion0.9-noise0.05-truth.csv \
  --method median_mode \
  --out outputs/transformers_noise05
```

For overhead conductors:

```bash
python -m src.cli all \
  --asset overhead_conductors \
  --measured data/overhead_conductors/conductors_synthetic_masked.csv \
  --truth data/overhead_conductors/conductors_synthetic_complete.csv \
  --method median_mode \
  --out outputs/conductors_baseline
```

`--input` remains a backward-compatible alias of `--measured`. `--truth` is
optional for operational imputation, but required for scoring originally
missing cells. `evaluate --measured ... --truth ...` can run the same paired
evaluation independently of `all`.

Results include `final_imputed.csv`, `cell_provenance.csv`, cleaning and
validation logs, `evaluation/summary.csv`, `evaluation/scored_cells.csv`,
`evaluation/observed_difference_summary.csv`, a partition manifest, and
`report.html` (when the reporting dependency is installed).

The reported imputation errors score **originally missing test cells only**.
Observed-data differences from truth are a separate analysis. The true values
are never provided to model fitting. See [paired-input protocol](docs/PAIRED_INPUTS.md)
and [model execution / tuning](docs/MODEL_EXECUTION_AND_TUNING.md) for the
`train_new` versus `use_saved` switch and optional validation-only tuning.

## Asset profiles

`--asset NAME` selects `config/NAME/pipeline.yaml` and its sibling schema,
cleaning, and imputation files. To add another asset class, copy and adapt a
config folder, explicitly review predictor leakage, and supply its own two
aligned source CSVs. Do not reuse another asset's categorical vocabulary or
numeric limits without reviewing the domain.

The transformer sample has no genuine unique record ID, so its provisional
config generates row-number IDs and checks declared **unnoised** anchor
columns. This is acceptable only when the two files are in matching row order.
Prefer a genuine stable asset ID for real data.

## Method plugins and evaluation

Shared method-independent scoring lives in `src/evaluation/`. Each algorithm
has its own folder inside `src/imputation/`. Methods implement the same
`BaseImputer` contract (`fit`, `transform`, diagnostics) and receive only the
configured model attributes. New teammate methods can register via their own
`plugin.py`; see [adding an imputation method](docs/ADDING_METHODS.md).

All methods are evaluated with matching partitions, missing-cell masks, and
metrics (MAE/RMSE for numeric columns, Macro-F1/accuracy for categorical).
Model-specific diagnostics may live alongside each method; common scoring
remains central for comparability.

## Notebooks

Open either notebook from `notebook/`. They resolve the project root from
the notebook directory, the `pipeline/` directory, or its parent. They use the
same reusable functions as the CLI and take only measured/truth inputs.

The notebook path defaults can be overridden with `OC_NOTEBOOK_INPUT`,
`OC_NOTEBOOK_TRUTH`, and `OC_NOTEBOOK_RUN_DIR`. The optional
`OC_NOTEBOOK_OVERRIDES` JSON mapping changes method hyperparameters.

## Configuration approvals

After restructuring, asset profiles are **development-only until reviewed**.
Their previous approval records must not be treated as approval for a new
layout, new dataset, or real SAPN data. To inspect:

```bash
python -m src.cli config check --asset overhead_conductors
python -m src.cli config check --asset transformers
```

Before using `execution.mode: release`, review, validate, and explicitly
approve the current profile for its intended data. See `docs/USER_GUIDE.md`
and `docs/DATA_HANDLING.md` for operational safeguards.
