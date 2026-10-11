# PGS219-129 review corrections and results

## Status

The two comments from Zisan on PR 14 have been addressed locally against
current main `cabec8848bc9ff70be719de0b86fe7aa861638fc`. No commit or push has
been made. The existing Jira branch is unchanged in name:

`PGS219-129-overhead-conductors-validate-controlled-synthetic-relationships-and-leakage`

Latest main was merged with `--no-commit --no-ff`, so Git intentionally has a
pending merge. Its staged KNN/noise files come from main, not new work for this
ticket. The new audit edits remain uncommitted. Review the working tree against
`origin/main` to distinguish this ticket from those already-merged changes.

## Corrections

1. Voltage and material now have separate era definitions matching their
   generator rules. The middle material era explicitly represents no added
   era effect, not a missing category.
2. Feature-recoverability screening now runs across all 11 strengths and three
   seeds, not only strength 100. Every row records profile, seed and whether
   it concerns a direct answer/duplicate or a profile-dependent predictor.
3. Support counts now describe the statistic: evaluation rows for classifier
   scores, eligible category support for purity, all paired rows for association.
4. Every seed is regenerated from current code/configuration. Existing CSVs
   and old reports remain untouched. Hashes record source and generated data.
5. Checks include finite values, valid years/covering, exact model-ready row
   and value alignment, and the generator's configured row/route-share checks.
6. Heatmap strengths are ordered numerically. Documentation distinguishes
   descriptive thresholds, seed variability and uncalibrated screening from
   statistical significance, confidence intervals and proof of leakage.

## New results

The full rerun uses seeds 219, 319 and 419 and 10,000 rows per run.

- 33 of 33 profile-seed runs passed structural/generator checks.
- 792 relationship-statistic rows and 264 cross-seed summary rows.
- 1,287 audit rows covering all 33 profile-seed combinations.
- 66 classifier warning rows, all for conductor_type: two metrics across
  11 profiles and three seeds. This means one flagged feature, not 66 features.
- The deterministic installation-year/age duplication remains detected.

| Configured relationship | Old strength-100 NMI | Corrected strength-100 NMI |
|---|---:|---:|
| Installation era to voltage | 0.007844 | 0.083804 |
| Installation era to material | 0.007786 | 0.049750 |

The changed values reflect better-matched measurement categories, not improved
imputation or changes to the generator. Both relationships now exceed the
audit's descriptive endpoint-change threshold. Voltage-to-location and
voltage-to-material also remain visible. Corrosion-related endpoint changes
remain smaller under these synthetic assumptions.

No ordinary predictor exceeded the configured 0.80 single-feature classifier
warning threshold in this run. This does not establish freedom from nonlinear
or multi-feature leakage. Length calibration remains a documented potential
source of target-related information; a with/without-length comparison belongs
in the subsequent method robustness experiment.

## Tests and evidence

Full local suite after incorporating main: **38 passed**. New tests cover era
boundaries, distinct relationship groupings, row/value alignment, invalid
reference values, audit scope/support, all-profile execution, invalid run
parameters and regeneration instead of stale-cache loading. `git diff --check`
passed. The generated heatmap was visually inspected.

New evidence is in `outputs/PGS219-129-review-2026-10-11/`:

- `relationship_validation_report.md`
- `relationship_profile_metrics.csv`
- `relationship_direction_checks.csv`
- `relationship_seed_stability.csv`
- `leakage_audit.csv`
- `schema_and_invariant_checks.json`
- `run_metadata.json`
- `relationship_heatmaps/material_association_heatmap.png`

This output directory is ignored by Git and can be used for Jira attachments.
The old `outputs/PGS219-relationship-validation/` evidence is retained as history.

## Proposed commit scope for approval

The correction changes three existing ticket files:

- `src/team19_asset_data/overhead_conductors/relationship_validation.py`
- `tests/test_relationship_validation.py`
- `docs/overhead_relationship_validation_plan.md`

It adds this results note. The original literature-source file remains part
of the PR but was not edited. No datasets, generator probabilities, baseline,
KNN or noise algorithms were changed by this correction.

Proposed message, not executed:

`PGS219-129 Address relationship validation review and refresh against main`

After approval, stage only the audit corrections/results note in addition to
the existing main-merge changes, finish the pending merge commit, and push the
same branch to update PR 14. Request Zisan's re-review before merging the PR.
Do not create a second PR for the same ticket.

## Client explanation

We checked whether our synthetic data behaves according to its design. The
first version grouped installation years too broadly, hiding some of the
relationships we had configured. After aligning the checks with the actual
generation rules, these relationships became clearer. We also tested every
strength setting rather than assuming all predictors behave the same way.
The direct conductor-type shortcut remains excluded from material prediction.
This gives us a better-audited foundation for the next noise and imputation
comparison, not a claim about SAPN's real data or failure probabilities.
