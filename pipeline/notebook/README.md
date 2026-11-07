# Notebook controllers

The notebooks are kept in this folder as requested. All processing code is
imported from the sibling `../src/` package; notebooks contain orchestration,
explanations and displays only.

- `pipeline_overhead_conductors.ipynb`: overhead-conductor synthetic example.
- `pipeline_transformer.ipynb`: paired measured/truth transformer example.

Start Jupyter from the `pipeline/` project root, or directly from this
`notebook/` folder. The code discovers the project root in both cases.

Each notebook accepts just two source inputs (`INPUT_CSV`, `TRUTH_CSV`).
Override them with `OC_NOTEBOOK_INPUT` and `OC_NOTEBOOK_TRUTH`, and set a
different `OC_NOTEBOOK_RUN_DIR` for each scenario. No separate masked file is
needed. The notebooks generate reports under the run directory, which they
reset when `FRESH_RUN = True`.
