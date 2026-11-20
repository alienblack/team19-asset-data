# Final product and supporting research

The client product is in
[experiments/shared_final_study](../experiments/shared_final_study/README.md).
That folder has its own Python 3.12 requirements, entry point, tests, inputs and
reports. Its curated client ZIP is built by its `tools/build_handover.py` script.

The rest of the repository records the team's research and development. These
files are not all obsolete: several preserve methods, assumptions, tests or
results needed to explain earlier sprints. They are kept at their original paths
so imports, configurations, notebooks and Jira evidence links remain valid.

## Which area to use

| Path | Contents | Intended use |
| --- | --- | --- |
| `experiments/shared_final_study/` | Final workbench, original shared benchmark and client handover | Client installation and new dataset studies |
| `experiments/subteam_a_transformer_benchmark/` | Subteam A's separate transformer benchmark, Gaussian EM/forest work, calibration and stress analyses | Supporting research and Sprint 6 evidence; follow its own README and Python 3.11 environment |
| `src/team19_asset_data/` | Earlier reusable stream-specific generators, masking/noise tools and imputation methods | Reproduce overhead, cable and transformer pilot work |
| `config/` | Earlier generation, relationship-strength and imputation configurations | Use with the corresponding earlier runners |
| `data/synthetic/` | Versioned synthetic populations and controlled scenarios | Reproduce the associated studies; not live SAPN records |
| `notebooks/` | Exploratory analyses and stream-specific demonstrations | Explain earlier investigation and development |
| `experiments/*.py` | Earlier comparison and predictive experiment entry points | Run with their corresponding root environment/configuration |
| `docs/methodology/`, root ticket documents | Assumptions, runbooks, results and Jira-linked evidence | Trace decisions and earlier sprint contributions |
| Selected tracked `outputs/` | Deliberately retained historical evidence and shared synthetic scenarios | Trace earlier results; not the workbench's current run directory |
| Root `tests/`, `requirements.txt`, `pyproject.toml` | Tests and environment for the earlier stream package | Separate research setup; not the client workbench environment |

The [Subteam A integration record](../experiments/subteam_a_transformer_benchmark/INTEGRATION.md)
identifies its imported source commit. Its
[product-review handover](../experiments/subteam_a_transformer_benchmark/docs/product_review/README.md)
maps implementation, demonstrations and notebooks to the Sprint 6 tickets.
The final shared study's submission guide explains the different evaluation
protocols. Scores from these studies are not interchangeable.

## Earlier stream-specific setup

To work on the root research package, follow the corresponding runbook and its
recorded Python/dependency environment. Its original installation route is:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pytest
```

This is retained for research contributors. For the client workbench, use the
Python 3.12 setup in `experiments/shared_final_study/START_HERE.md` instead.

## Cleaning and archival policy

The curated client archive excludes the earlier root package, development
environments, caches, uploaded client data and scratch runs. It intentionally
retains the final product's required synthetic inputs and reference results.

A permanent `dev` branch would be another repository snapshot; it would not by
itself separate the client files from the research history. Use `main` for
reviewed integration, short Jira branches for changes, and version tags/releases
for the exact client delivery.

If an earlier implementation is later confirmed obsolete, archive it through a
separate reviewed change after checking imports, configuration paths, notebook
paths, tests and evidence links. Preserve its reproducible revision and explain
its replacement. Moving all older folders simply because their dates are earlier
would break traceability and can break runnable experiments.
