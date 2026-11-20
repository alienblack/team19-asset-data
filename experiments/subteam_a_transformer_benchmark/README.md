# Missing asset attributes | Team 19, Subteam A

**Rishank Goyal · University of Adelaide · SA Power Networks capstone**

How reliably can we recover missing asset attributes, preserve the underlying population and identify cases that need review? This benchmark evaluates three primary imputers and three references on synthetic transformer data, with a separate overhead conductor transfer. The [final report](docs/Final_Report.pdf) includes the methods, complete anchor tables, stress tests, interpretations and figures. [Markdown](docs/Final_Report.md) · [Repository contents](docs/Repository_Contents.md).

## Data sources

Subteam C supplied the synthetic transformer generator and nine aligned clean/measured pairs. The benchmark uses the generator’s draw order, including the held-out calibration masks. The synthetic overhead conductor pairs come from the team’s overhead work. [SA Power Networks’ October 2014 Asset Management Plan 3.1.10](https://www.aer.gov.au/documents/sapn-2063-public-sapn-asset-management-plan-3110-overhead-conductor-2014-2025) provides historical context for the overhead design.

Obtain the original files from the Team 19 course bundle or ask the authors for the prepared clean/measured pairs. Run the following from this repository after setting `TEAM19_BUNDLE` to the extracted bundle directory:

```sh
mkdir -p data/source data/overhead
cp "$TEAM19_BUNDLE/Subteam c/datasets/"completion*-truth.csv data/source/
cp "$TEAM19_BUNDLE/Subteam c/datasets/"completion*-measured.csv data/source/
cp "$TEAM19_BUNDLE/Subteam c/overhead-imputation-study/data/datasets/"completion*-truth.csv data/overhead/
cp "$TEAM19_BUNDLE/Subteam c/overhead-imputation-study/data/datasets/"completion*-measured.csv data/overhead/
```

The same prepared pairs are available under `data/source` and `data/overhead` in the original Subteam A working project. The source audit checks all nine transformer pairs cell for cell. Use the prepared overhead pairs with the `feeder_id` field. The public asset management plan provides context for those files.

## Installation

Use Python 3.11. The submitted contract recorded Python 3.11.9. The direct numerical dependencies in `pyproject.toml` are pinned to the versions in that contract.

```sh
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[test]'
```

## Reproduce and verify

Run the data setup above before running the tests or benchmark.

```sh
asset-impute audit --data data --out outputs/source-check
pytest -q
asset-impute run --profile smoke --data data --out outputs/smoke --workers 2
python scripts/verify_run.py --data data --out outputs/smoke
asset-impute run --profile full --data data --out outputs/final --workers 4
python scripts/verify_run.py --data data --out outputs/final
```

The packaging smoke run took about 9 seconds before verification. The archived full run took about 6 minutes 26 seconds on the original machine. Expect timing to vary with hardware and worker count. Use fewer workers if memory is limited. The `pilot` profile uses one generator/split pair with four tuning trials per trainable method.

The CLI writes task JSON, source and run contracts, and representative prediction ledgers. The verifier reads those task records directly or accepts the original renderer’s saved CSV summaries. For an archived run, pass its directory to `--out`. An output directory resumes completed tasks only when its input, code, environment and settings match. Use a fresh directory when any of those change, including when moving from the archived code to this documented package.

Optional follow-ups reuse a full run from this checkout and its corresponding stress run.

```sh
asset-impute stress --anchor-run outputs/final --out outputs/anchor-stress
python scripts/verify_run.py --out outputs/anchor-stress
asset-impute diagnose --anchor-run outputs/final --stress-run outputs/anchor-stress --out outputs/anchor-diagnostics
python scripts/verify_run.py --out outputs/anchor-diagnostics
asset-impute failures --anchor-run outputs/final --stress-run outputs/anchor-stress --out outputs/failure-audit
python scripts/verify_run.py --out outputs/failure-audit
```

## Project structure

| Path | Purpose |
| --- | --- |
| `src/subteam_a/data.py`, `generator.py` | Input audit, reproducible masks and noise, blocked splits |
| `features.py`, `models.py`, `mixture.py` | Exact rules, train-only features and imputation models |
| `evaluate.py`, `study.py`, `cli.py` | Scoring, calibration, tuning and experiment commands |
| `stress.py`, `diagnostics.py`, `failures.py`, `screening.py` | Stress comparisons, grouped MLP training and recorded-value screening |
| `tests/`, `scripts/verify_run.py` | Unit and integration checks plus independent verification of saved runs |
| `docs/` | Report PDF and Markdown, figure PNGs and repository manifest |

## Methods

The floor uses training medians or modal categories. Chained ridge, mixed-type k-NN, inductive Predictive MissForest, conditional Gaussian mixture EM and a ReMasker-inspired masked MLP complete the comparison. The transformer benchmark compares model-only and exact-rules-first arms. Transformations preserve the supplied observations.

## Evaluation and tolerances

The five layers are hidden-cell accuracy, distribution preservation, uncertainty and routing, downstream life prediction, and record consistency. Numeric skill is one minus MAE divided by the median-floor MAE. Categorical skill uses misclassification error. Numeric decisions use field-specific tight and loose tolerances, including absolute limits for days, years and life. Model-based auto-fill also needs independent evidence for the relevant missing pattern. The report lists every tolerance and the source table for each result.

## Reproducibility

Generator seeds are `219`, `220` and `221`. Split seeds are `617`, `631` and `653`. The run contract stores input hashes, code hashes, seeds, the direct dependency versions and key scoring settings. Spatial tiles and overhead feeders keep held-out groups separate. Settings were tuned at the 30% MCAR transformer anchor. The grouped MLP follow-up reused the selected settings.

## Key results

- MissForest scored **0.656** at the original feature-enabled anchor and **0.264** under grouped gaps. Gaussian EM scored **0.435** under grouped gaps, and the MLP trained with matched group masks scored **0.453**.
- Under grouped gaps, MissForest and Gaussian EM abstained on every y, altitude, humidity, salt and overgrowth gap in the rules-first arm. MissForest’s grouped Gas A eligibility still missed the precision target.
- Wrong recorded values caused larger errors on affected rows than the aggregate suggested. At a nominal 1% false-alarm setting, the Gaussian EM model-ID screen caught **73.5%** of swaps at **68.0%** flag precision in the synthetic pilot.

See the [final report](docs/Final_Report.md) for the field-by-field results and visual interpretations.

## Limitations

The benchmark uses a synthetic transformer generator, a synthetic overhead transfer and designed gaps and errors. Settings were selected at one anchor. Coarse uncertainty groups can miss coverage targets. Screening requires verified clean examples. Performance on the live SAPN register still needs validation.

## Acknowledgements

Thank you to Team 19, Subteam C for the generator and synthetic data, and SA Power Networks and the University of Adelaide supervisors for the project guidance.

AI coding assistants were used for parts of drafting, debugging and documentation. The author reviewed all design decisions and results.
