# Team 19 Asset Data

Reproducible synthetic-data experiments for Team 19's SA Power Networks capstone project.

## Project purpose

The project investigates justified, explainable approaches for improving selected
missing asset attributes that support SA Power Networks' existing statistical,
Probability of Failure (PoF), and Weibull-style analyses. It does not develop a
replacement PoF model.

The initial asset streams are:

- overhead conductors;
- underground cables; and
- pole-mounted transformers.

## Data handling rule

Only synthetic data, public information, and literature-derived assumptions may
be committed to this repository. Do not commit, upload, or paste confidential SA
Power Networks data, credentials, signed documents, meeting recordings, or
unapproved client material. Any future authorised data must follow the storage
arrangements confirmed by SA Power Networks; it is not automatically suitable for
GitHub.

## Repository layout

```text
data/synthetic/                 Versioned, controlled test data only
docs/                           Decisions, methodology and data-handling notes
notebooks/                      Exploratory work that is later moved into src/
src/team19_asset_data/common/   Shared validation and utility code
src/team19_asset_data/<stream>/ Asset-stream-specific code
tests/                          Automated checks for reusable code
```

## Getting started

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest
```

## Team workflow

1. Create one branch per Jira ticket, for example `feature/PGS219-XX-overhead-baselines`.
2. Make small commits using the Jira key, for example `PGS219-XX add material mode baseline`.
3. Open a pull request for another team member to review before merging into `main`.
4. Record the source, assumptions, seed, masking mechanism and metric for every experiment.
5. Keep exploratory notebooks lightweight; move reusable logic into `src/`.

The first overhead-conductor experiment is controlled material reconstruction:
hide known synthetic material values, apply comparable imputation methods, and
measure recovery accuracy and coverage against the hidden truth.

## Final shared study and client handover

The [combined final study](experiments/shared_final_study/START_HERE.md) contains
the team's eleven-imputer comparison, median/mode baseline, synthetic inputs,
archived results, completed datasets and automated offline report. Its
[submission guide](experiments/shared_final_study/SUBMISSION.md) explains the
GitHub/Jira handover and the optional original saved-prediction package.

Run this study in its own **Python 3.12** environment, using its local
`requirements.txt`; the root setup above does not install its dependencies.
From the `experiments/shared_final_study` folder, `python run_study.py` regenerates
the archived-results report without fitting. Quick/full modes fit fresh models.
`python run_study.py --mode guided` opens the local dataset workbench: choose a
CSV, define its fields and experiment settings, then generate completed datasets
and a new report. This new-dataset protocol is documented separately from the
archived benchmark, whose original scientific modules and scores are preserved.
The earlier Subteam A benchmark below uses a separate **Python 3.11** environment
and a different evaluation protocol; its figures are supporting evidence, not
interchangeable scores for the shared final study.

## Subteam A transformer benchmark and product review

The [integrated transformer benchmark](experiments/subteam_a_transformer_benchmark/README.md) contains Rishank Goyal's existing Gaussian EM, Predictive MissForest, masked MLP and reference-method study using Subteam C's synthetic data. The [integration record](experiments/subteam_a_transformer_benchmark/INTEGRATION.md) identifies the source repository and imported commit.

PGS219-258 coordinates its integration and product-review handover. The child PRs add a runnable demonstration, independent verification and documentation.

The [product-review handover](experiments/subteam_a_transformer_benchmark/docs/product_review/README.md) maps the Gaussian EM implementation, fresh demonstration, validation notebooks and final report to the Sprint 6 integration tickets.
