# Team 19 · Asset data imputation

SA Power Networks capstone project. This repository contains the final client
product and the supporting research that led to it.

**Client starting point: [Final product — setup and user guide](experiments/shared_final_study/README.md).**

| What you need | Start here |
| --- | --- |
| Install and run the final product | [Short setup guide](experiments/shared_final_study/START_HERE.md) |
| Use the dataset workbench, transformer example and generated reports | [Client README](experiments/shared_final_study/README.md) |
| Test uploads, experiments, reports and downloads | [Testing guide](experiments/shared_final_study/docs/TESTING.md) |
| Download a named client release | [GitHub Releases](https://github.com/alienblack/team19-asset-data/releases) — use the attached client handover ZIP when published |
| Check submission contents and version-control steps | [Handover checklist](experiments/shared_final_study/SUBMISSION.md) |
| Review earlier methods, experiments and contribution evidence | [Research and repository guide](docs/REPOSITORY_GUIDE.md) |

## Run the final product

Use **Python 3.12** and the requirements inside `experiments/shared_final_study`.
From a repository checkout on macOS/Linux:

```sh
cd experiments/shared_final_study
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_study.py --mode guided
```

Open `http://127.0.0.1:8765`. The client README includes Windows instructions,
screenshots, the transformer input/truth files, field settings, masking/noise
experiments and report interpretation. The root-level requirements belong to
the earlier stream-specific research; use the final product's own environment.

For the client handover, download the attached **Team19_Automated_Handover.zip**
and its checksum from the named release, then open `START_HERE.md` after
extraction. This curated archive contains the runnable product and its required
synthetic inputs/reference evidence. GitHub's automatic whole-repository source
ZIP also contains the team's supporting research.

## How this repository is organised

| Area | Role |
| --- | --- |
| `experiments/shared_final_study/` | Final client product, workbench, tests, synthetic inputs, curated results and detailed client documentation. |
| `experiments/subteam_a_transformer_benchmark/` | Earlier Subteam A benchmark, uncertainty/stress analyses and product-review evidence; a separate Python 3.11 study. |
| Root `src/`, `config/`, `data/`, `notebooks/`, `tests/` | Supporting overhead-conductor, underground-cable and transformer research. These are separate from the packaged workbench. |
| Root `docs/` and selected tracked `outputs/` | Methodology, Jira evidence and historical study results retained for traceability. |

Earlier work is preserved because it explains the project's decisions and
contributions. Its presence does not make it a second installation route for the
client product. See the research guide before reusing an older script or score.
New scratch runs, environments, caches and client uploads remain excluded from
Git and from the curated handover ZIP.

## Project purpose and data handling

The project investigates missing-value recovery to support SA Power Networks'
existing statistical, Probability of Failure and Weibull-style analyses. It does
not replace a PoF model. The research covered overhead conductors, underground
cables and pole-mounted transformers.

Only synthetic data, public information and approved literature-derived
assumptions belong in this repository. Confidential SAPN data, credentials,
meeting recordings, signed documents and unapproved client material must stay
outside it. The supplied demonstration datasets are synthetic.

## Team workflow

Use a short branch for each Jira ticket, review its PR into `main`, then tag the
merged revision for a client release. Keep existing release tags fixed and use a
new patch version for later fixes. The final-product submission guide describes
the release assets and the exact checks. A permanent `dev` or `final` branch is
not needed for this handover.
