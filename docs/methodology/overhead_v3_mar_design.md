# PGS219-192: Predeclared MAR year-missingness experiment

Team 19, Subteam A | 25 October 2026 | Controlled synthetic population

## Question

The MCAR age-share experiment showed that replacing point years with distributions removes most population distortion, even for a global floor. This follow-up asks whether conditioning on observed asset fields helps when year missingness is **not** random but depends on those fields. It is a separate experiment; the clean/noisy MCAR results are frozen.

## Fixed design before running

- Reuse the same 10,000 clean synthetic rows, the five frozen 60/10/10/20 partitions, the 30% MCAR material masks, and the clean-tuning-selected baseline/forest settings. Refit on each MAR fitting view; do not retune on MAR test outcomes.
- Change only the `installation_year` mask. Draw independent Bernoulli masks within each partition from the probabilities below, with deterministic seeds. All methods see the same observed rows and masked IDs within an arm. The target average rate is about 30%, but actual rates and subgroup counts may vary.
- For the material-driven arm, the driver is the **observed** material. Rows whose material was already masked use probability 0.30; no hidden material is consulted to decide year missingness.

| Arm | Observed driver | Year-mask probability by category | Expected total rate | Expected hidden test rows in small named groups |
| --- | --- | --- | ---: | ---: |
| Voltage | `voltage_category` | 19 kV SWER .35; 11/7.6 kV .27; LV .12; 33 kV .55; 66 kV .35 | .298 | SWER ~281; 66 kV ~15 |
| Location | `location_group` | Rural .38; Regional .18; Urban .25 | .299 | Urban ~111 |
| Covering | `covering` | Bare .294; Covered .80 | .300 | Covered ~18 |
| Material | observed `material` | Steel .19; Copper .48; Aluminium .75; unavailable .30 | .300 | Aluminium ~177; unavailable varies |

Expected counts use the fixed population's category sizes and a 20% test fraction. They are design calculations, not realised results. Covered and 66 kV remain sparse; their per-group estimates are exploratory even when the counts meet expectation. Mask probabilities are deliberately stronger for older-leaning voltage groups and newer-leaning Aluminium/Covered groups to make selection effects visible, but no outcome threshold or method winner is chosen from test results.

## Predeclared expectations

| Driver | Complete cases and global floor | Conditional baseline: voltage x covering x material backoff | Forest |
| --- | --- | --- | --- |
| Voltage | May bias population age share because voltage and year are related | Should reduce bias when voltage groups have support | Should reduce bias if fitted relationship is recovered |
| Location | May show weak bias because location is mostly linked to year through voltage | May be close to unbiased after conditioning on voltage, but indirect paths and backoff remain | Same qualitative expectation; not guaranteed |
| Covering | May bias population age share, but rare Covered records limit power | Should reduce bias where group support is sufficient; fallback may dominate rare Covered rows | Same qualitative expectation; rare-group uncertainty remains |
| Observed material | May bias population age share | Should reduce bias for known material; unavailable material requires backoff | May reduce bias if training support and predictors suffice |

These are hypotheses, not success criteria. The primary comparison is absolute error in the **aged-50-or-older population share**, together with signed error, age-band errors, group counts, per-seed ranges and the complete-case estimate. Also inspect calibration-partition Brier scores and hidden-year MAE, without claiming calibrated probability or downstream PoF improvement.

The evaluator-only generator reference remains applicable to these MAR arms **only because** mask probability depends entirely on recorded model fields (or recorded absence of material) and a random draw, not the hidden year after conditioning on those fields. The same reference is not valid unchanged for an MNAR mask that directly depends on unobserved year. The simulated MAR rates are not SAPN missingness estimates.
