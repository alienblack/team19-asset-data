# Repository contents

## Included files

- `.gitattributes`
- `.gitignore`
- `README.md`
- `docs/Final_Report.md`
- `docs/Final_Report.pdf`
- `docs/Repository_Contents.md`
- `docs/figures/01_skill_limit_map.png`
- `docs/figures/02_mcar_grid.png`
- `docs/figures/03_safe_fill_map.png`
- `docs/figures/04_visible_inputs.png`
- `docs/figures/05_confidence.png`
- `docs/figures/06_distribution.png`
- `docs/figures/07_downstream.png`
- `docs/figures/08_stress.png`
- `docs/figures/09_overhead_transfer.png`
- `docs/figures/10_tuning.png`
- `docs/figures/11_convergence.png`
- `docs/figures/12_svm_demo.png`
- `docs/figures/13_consistency.png`
- `docs/figures/14_category_calibration.png`
- `docs/figures/15_mnar_age_bias.png`
- `docs/figures/attribute-skill.png`
- `docs/figures/grouped_model_id_probabilities.png`
- `docs/figures/masking-by-attribute.png`
- `docs/figures/noise-family-heatmap.png`
- `docs/figures/where-it-breaks.png`
- `pyproject.toml`
- `scripts/verify_run.py`
- `src/subteam_a/__init__.py`
- `src/subteam_a/cli.py`
- `src/subteam_a/data.py`
- `src/subteam_a/diagnostics.py`
- `src/subteam_a/evaluate.py`
- `src/subteam_a/failures.py`
- `src/subteam_a/features.py`
- `src/subteam_a/generator.py`
- `src/subteam_a/mixture.py`
- `src/subteam_a/models.py`
- `src/subteam_a/screening.py`
- `src/subteam_a/stress.py`
- `src/subteam_a/study.py`
- `tests/test_pipeline.py`

## Excluded from the original working project

The paths and patterns below describe exclusions relative to the original Subteam A working project. Directory entries cover every descendant.

| Path or pattern | Reason |
| --- | --- |
| `src/subteam_a/report.py`, `diagnostics_report.py`, `failures_report.py`, `stress_report.py` in the same directory | Report renderers. The PDF, Markdown and PNGs are the report deliverables. |
| `scripts/review_checks.py` | Original report-review script depends on the renderer and archived CSV tables. Its numerical checks are reported in the static report with source paths. |
| `requirements-tested.txt` | Exact dependency pins are in the packaged `pyproject.toml`. |
| `docs/METHODS.md`, `docs/PROJECT_REVIEW.md` | Working methods and planning documents. The final report describes the protocol and evidence. |
| `data/source/SyntheticDataset.ipynb` | The adapted generator is in `generator.py`. The original notebook is credited and available in the course bundle. |
| `data/source/completion*-truth.csv`, `data/source/completion*-measured.csv` | The original transformer pairs are local inputs. See the README for acquisition and audit. |
| `data/overhead/completion*-truth.csv`, `data/overhead/completion*-measured.csv` | The prepared conductor pairs are local inputs. See the README for acquisition. |
| `outputs/**` | Saved task JSON, CSVs, ledgers, renderer exports and previous run revisions are generated or archived outside the clean repository. Static PNG copies appear in `docs/figures/`. |
| `__pycache__/**`, `*.pyc`, `.pytest_cache/**`, `.DS_Store`, virtual environments | Local caches, OS metadata and environments. |
| `.submission-build/**`, tool configuration, agent instructions, logs, transcripts and scratch material | Local authoring and review material outside the submission scope. |

## Verification

The packaged test suite passed 32 tests. The verifier reproduced the four archived verification records. It also reproduced the full-run record from task JSON and original ledgers without the renderer’s CSVs.

| Archived run | Tasks | Exports | Hidden cells | Tuning trials | Screening rows |
| --- | ---: | ---: | ---: | ---: | ---: |
| final | 123 | 192 | 1,495,404 | 900 | — |
| anchor-stress | 18 | 24 | 216,780 | 0 | — |
| anchor-diagnostics | 171 | 57 | 136,005 | 0 | — |
| failure-audit | 9 | 2 | 17,996 | 0 | 50,440 |

The fresh smoke run verified 4 tasks, 42 exported completions and 103,098 hidden cells. The archived full run also verified 24 overhead year exports.

The modeling, data and evaluation code has the same executable syntax tree after stripping docstrings. The packaging changes remove renderer calls and let the verifier read task JSON directly.
