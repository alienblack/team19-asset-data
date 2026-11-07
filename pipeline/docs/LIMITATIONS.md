# Limitations

## What the accuracy figures do and do not show

- **Only known values can be scored.** Every accuracy figure in a report or
  experiment comes from hiding values that were observed and reconstructing
  them - on the same table (the operational check in every run) or on a
  record-disjoint test partition of a complete reference (evaluation mode,
  which on this project exists only for synthetic data). The originally
  missing values of a real extract have no truth; nothing here measures how
  right their imputations are. If values go missing for reasons related to
  the value itself (for example very old assets having no recorded year), the
  hidden-known-values check overstates accuracy on the cells that matter.
- **Synthetic results do not establish real-world performance.** Every
  number in `docs/EXPERIMENTS.md`, in the sample reports and in the notebook
  describes the synthetic generator. The pipeline is demonstrated end to end
  on that data; how well it reconstructs SA Power Networks' records can only
  be measured on their data, by them.
- **The reference is the yardstick.** A method that is not clearly better
  than `median_mode` on a target has not learned anything about that target
  from the available attributes; its imputations are then a sophisticated way
  of writing the median. The report shows both side by side for this reason.
- **Confidence fields are rankings, not probabilities.** `max_proba`,
  `margin` and `tree_std` order the imputed cells from most to least decisive;
  they are not calibrated and should not be read as "70% chance of Steel".
  Calibration against held-out truth would be the next step and is not
  implemented.
- **Repeated masks reduce but do not remove noise.** With 10% of observed
  cells hidden once, a 10,000-row file gives about 800 scored cells per
  attribute; for a class with 100 observed cells that is ten hidden cells.
  Increase `repeats` or `mask_rate` before drawing conclusions about minority
  classes.

## Known properties of the synthetic data

All validation to date is on synthetic records (`data/overhead_conductors/`,
`experiments/synthetic_data/`). Two properties of the generator shape every
number obtained so far:

1. **`installation_year` is close to unpredictable from the approved
   attributes.** The generator draws the year first and lets it influence
   voltage and material only at a coarse era level. A gradient-boosting model
   fitted on the complete reference reaches MAE 10.6 years against 10.6 for the
   median; the forest imputers reach 10.6 on originally masked cells. Any
   improvement on real data would come from information the synthetic data
   does not contain (feeder installation campaigns, construction standards,
   neighbouring assets), not from the method.
2. **`material` carries a weak signal and `covering` almost none.** Macro-F1
   for material rises from 0.28 (always `Steel`) to about 0.31–0.35; minority
   classes are recovered at about 3% recall. `covering` is 99% `Bare`.

Consequently the synthetic benchmark demonstrates the machinery (leak-free
protocol, reproducibility, reporting) and the methods' behaviour on
attributes that do carry signal (`distance_to_sea_km`, `corrosion_zone`,
`location_group`, `voltage_category`), but it cannot show how well the two
project targets will be reconstructed on real records. That is a statement
about the data, not the methods, and should be repeated wherever the numbers
are quoted.

## Methodological limits

- **Missing at random is assumed by the forest methods.** The profiling stage
  reports whether a target's missingness depends on other attributes (which
  the forests can exploit); it cannot detect dependence on the missing value
  itself. Sensitivity analysis under not-at-random assumptions is not
  implemented.
- **Single imputation.** Each cell receives one value. Downstream statistical
  models that treat imputed values as observed will understate their
  uncertainty. The diagnostics file carries enough information (vote shares,
  tree spread) to draw multiple imputations or to down-weight uncertain cells,
  but the pipeline does not do this.
- **No survival or Weibull claims.** The project improves input attributes;
  it does not estimate failure probabilities, and no claim is made about the
  effect of imputed values on such models.
- **The approved attributes are fixed by configuration.** Adding predictors
  (for example `length_km` or `conductor_type`) requires a deliberate schema
  change; the leakage guard exists to make that change explicit, and the
  synthetic relationships between those columns and the targets are generator
  artefacts that must not be relied on.
- **Evaluation mode partitions by feeder; the research experiments did
  not.** Records of the same feeder are likely to share installation
  campaigns, so `oc-pipeline evaluate` groups the fit / validation / test
  partitions by `feeder_id` (`evaluation.split.strategy: group`). The earlier
  tuning and comparison experiments split by identifier with the same seed,
  which also made their test records the tuning records; their tables are
  marked superseded in `docs/EXPERIMENTS.md`. The synthetic feeder
  identifiers carry no campaign structure, so the grouped split is the honest
  mechanism without yet being a hard test; it becomes one on real feeders.
- **No probability calibration.** The validation partition is reserved for
  tuning; there is no calibration partition and no method calibrates its
  vote shares (see "Confidence fields are rankings").

## What the pipeline does not do

- It does not deduplicate records that differ in any column, resolve
  conflicting identifiers, or check geometry; conflicting identifiers stop the
  run so the source can be fixed.
- It does not fill values during cleaning. Values that fail a rule become
  missing and are then imputed; the cleaning log says which.
- It does not run hyperparameter tuning; tuning is an experiment whose
  result is committed to `config/overhead_conductors/imputation.yaml`.
- It does not fill columns outside the seven model attributes. Gaps in
  `length_km` (or any carried-through column) are counted as *unresolved* in
  `provenance_summary.csv` and the report; filling them would need an
  approved rule or a schema change that makes the column a model attribute.
- It does not apply domain interpretations that nobody has approved: the
  `code_mappings` cleaning family ships unapproved and is skipped, and the
  values it would have mapped are listed as pending (`pending_code_mappings`)
  and then treated as unknown labels.
- It does not handle extracts that do not fit in memory. 10,000 records run
  through every stage, the operational checks and evaluation mode in about a
  minute and a few hundred megabytes; 100,000 should still be practical on a
  workstation, beyond that the forest settings (fewer trees, `max_depth`)
  would need revisiting.
- It has been tested on Linux with the pinned versions in `requirements.txt`
  and the ranges in `pyproject.toml`; other platforms should work but have
  not been exercised.
