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
