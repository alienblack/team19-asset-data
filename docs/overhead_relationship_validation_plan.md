# Overhead Conductor Relationship Validation Plan

## Purpose

This plan defines the first validation experiment for the controlled synthetic
overhead-conductor dataset. It addresses the client requirement that the
synthetic data have understandable, controlled relationships before it is used
to compare imputation methods.

The dataset is synthetic and SAPN-inspired. The results measure recoverability
under the documented assumptions; they do not estimate relationships in the
SAPN asset register.

## Evidence base

The design is informed by the following sources:

1. Troyanskaya et al. (2001), *Missing value estimation methods for DNA
   microarrays*. The paper compares weighted KNN with simpler alternatives and
   shows that scaling, the missingness rate and the choice of neighbourhood
   size affect performance.
2. Stekhoven and Buhlmann (2012), *MissForest: non-parametric missing value
   imputation for mixed-type data*. The paper supports separating mixed-type
   predictors, nonlinear relationships and imputation-error estimation rather
   than assuming one method fits every dataset.
3. van Buuren and Groothuis-Oudshoorn (2011), *mice: Multivariate Imputation
   by Chained Equations in R*. The paper emphasises the missingness process,
   plausible imputation models and uncertainty across multiple imputations.
4. Kaufman et al. (2012), *Leakage in Data Mining: Formulation, Detection, and
   Avoidance*. The paper treats leakage as illegitimate information about the
   target entering through features or the training process, not merely as a
   high Pearson correlation.
5. Kuhnel et al. (2024), *Synthetic data generation for a longitudinal cohort
   study - evaluation, method extension and reproduction of published data
   analysis results*. The evaluation framework checks individual distributions,
   correlation structures, direct mathematical dependencies and practical
   utility rather than relying on summary statistics alone.

## Current controlled inputs

- 10,000-row complete reference dataset.
- Default generation seed: `219`.
- Snapshot year: `2014`.
- Relationship profiles: `strength_0` through `strength_100` in 10-point
  increments.
- Complete reference output retains `conductor_type` for audit purposes.
- Material-model-ready output excludes `conductor_type` because it directly
  reveals material.
- `asset_age` is deterministically derived from `installation_year` and must
  not be treated as independent evidence in the same model.

## Experiments

### E0 Schema and invariant audit

For every relationship profile, verify:

- no missing cells in the complete reference;
- unique segment identifiers;
- `asset_age = 2014 - installation_year` for every row;
- valid categories and bounded numeric values;
- material, voltage and corrosion class counts;
- route-length totals and documented route-length targets; and
- the complete and model-ready schemas differ only by the intended leakage
  field.

### E1 Relationship-strength sensitivity

For each strength profile, calculate the observed association between the
configured field pairs. Use measures appropriate to variable type:

- Cramer's V or normalized mutual information for categorical pairs;
- Spearman correlation for ordered or numeric variables;
- conditional material composition tables for material against voltage, age
  band and corrosion; and
- support counts for every association table.

The expected result is directional, not necessarily perfectly linear. A
strength of zero removes a configured direct effect, but upstream dependencies
and route-length calibration can still create indirect associations. Every
such effect must be reported rather than treated as a failed test.

Minimum relationship checks:

| Relationship | Expected interpretation |
|---|---|
| Installation year to voltage | Stronger profile should increase the configured era effect. |
| Voltage to location | Stronger profile should make SWER more rural and LV more urban. |
| Location/voltage to corrosion | Context shifts should become more visible as strength increases. |
| Age/voltage/corrosion to material | Material composition should change in the configured direction, without becoming deterministic. |
| Feeder to material | Feeder should not be treated as useful unless the audit shows meaningful signal. |

### E2 Leakage and duplicated-evidence audit

Run the following checks before any imputation comparison:

1. Measure material purity and multiclass association for `conductor_type`.
2. Confirm that `conductor_type` is absent from all material-model features.
3. Confirm that only one of `installation_year` and `asset_age` is used as a
   timing predictor in a model.
4. Measure the association created by post-generation length calibration with
   material and corrosion.
5. Run single-feature screening using balanced accuracy and macro-F1, not raw
   accuracy alone. A feature that nearly reconstructs material by itself is a
   leakage warning or a feature requiring explicit justification.
6. Check that no truth columns, mask indicators or evaluation-only values are
   available during development fitting.

### E3 Seed repeatability

Repeat the profile audit with seeds `219`, `319` and `419` while keeping the
configuration and relationship strengths unchanged. Report mean, standard
deviation and range for each association statistic. This separates a genuine
configured relationship from a result caused by one random sample.

## Outputs

The experiment should produce:

- `relationship_profile_metrics.csv`;
- `relationship_direction_checks.csv`;
- `leakage_audit.csv`;
- `schema_and_invariant_checks.json`;
- `relationship_heatmaps/` or equivalent plots;
- `run_metadata.json`; and
- `relationship_validation_report.md`.

Each output must record the profile, seed, input path, feature pair, statistic,
support count, expected direction and interpretation.

## Pass criteria

The ticket is complete when:

- every profile passes the schema and invariant checks;
- configured relationships are visible in the expected direction or have a
  documented explanation for indirect or calibration effects;
- direct leakage and duplicate timing evidence are explicitly flagged;
- results are stable enough across the three seeds to support the next
  experiment;
- the material-model feature list is frozen and documented; and
- a short client-readable conclusion explains what the dataset can and cannot
  demonstrate.

## Boundaries

This ticket does not add predictor noise, run the KNN experiment, or claim
downstream PoF/Weibull stability. Those are subsequent experiments. The clean
reference must remain unchanged; later noise and missingness layers must be
generated as separate observed-data views with the clean values retained for
evaluation.

## Recommended interpretation for the client

The team is first validating that the artificial dataset behaves according to
its documented design. This prevents a later imputation result from being
mistaken for evidence about SAPN data when it may only be exploiting an
unintended synthetic correlation.
