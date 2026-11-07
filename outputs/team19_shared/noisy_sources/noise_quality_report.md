# Controlled Predictor Noise Report

This report describes synthetic observed-data views created from the clean overhead-conductor reference. It does not estimate SAPN data-quality rates.

## Scenario results

| Scenario | Strategy | Column | Changed rate | Validation |
|---|---|---|---:|---|
| N06_mixed_50 | random_mixed | distance_to_sea_km | 0.500 | passed |
| N06_mixed_50 | random_mixed | length_km | 0.500 | passed |
| N06_mixed_50 | random_mixed | voltage_category | 0.500 | passed |
| N06_mixed_50 | random_mixed | location_group | 0.500 | passed |
| N06_mixed_50 | random_mixed | corrosion_zone | 0.500 | passed |
| N06_mixed_50 | random_mixed | material | 0.500 | passed |
| N06_mixed_50 | random_mixed | covering | 0.500 | passed |
| N06_mixed_50 | random_mixed | installation_year | 0.499 | passed |
| N06_mixed_50 | random_mixed | asset_age | 0.499 | passed |

## Distribution and relationship drift

`distribution_l1` is zero when the binned numeric or categorical distribution is unchanged. `material_association_delta` records the change in nominal association with the material target; it is diagnostic only.

| Scenario | Column | Distribution L1 | Material association delta | Truth-observed correlation |
|---|---|---:|---:|---:|
| N06_mixed_50 | installation_year | 0.086 | -0.146 | 0.999 |
| N06_mixed_50 | asset_age | 0.086 | -0.157 | 0.999 |
| N06_mixed_50 | voltage_category | 0.180 | -0.186 | - |
| N06_mixed_50 | location_group | 0.134 | -0.098 | - |
| N06_mixed_50 | corrosion_zone | 0.170 | -0.047 | - |
| N06_mixed_50 | distance_to_sea_km | 0.022 | -0.041 | 0.958 |
| N06_mixed_50 | material | 0.295 | +0.000 | - |
| N06_mixed_50 | covering | 0.488 | -0.010 | - |
| N06_mixed_50 | length_km | 0.025 | -0.030 | 0.951 |

## Interpretation

- Random numerical levels specify both the selected fraction and maximum relative error; categorical levels specify the selected fraction.
- Same-seed random scenarios of the same family use nested corruption selections across increasing levels.
- Apply the same original MF01 missingness mask downstream; this generator alone does not add missingness.
- Do not treat a 100% categorical replacement scenario as a plausible real-world error rate; it is an extreme stress test.
- Systematic and group-dependent scenarios represent structured data-quality problems and should not be averaged together with random noise.
- Installation-year noise is propagated to asset age to preserve the source data contract.
- `material` and `conductor_type` remain unchanged in every scenario.
- Method comparisons are intentionally deferred to the next ticket.
