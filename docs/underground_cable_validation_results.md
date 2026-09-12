# Sprint 2 Member 3 Validation Results

## Scope

This report covers UC-S2-07, UC-S2-08, and UC-S2-09 for the supplied underground-cable synthetic datasets. The complete reference, 20% MCAR masked, baseline-imputed, and advanced-imputed datasets each contain 5,000 records with aligned `asset_id` values. The masked dataset contains 1,000 hidden-known values in each of the following fields:

- `asset_age_years`
- `groundwater_indicator`
- `corrosion_indicator`

Both methods were evaluated against the same masked positions. Each method supplied 1,000 usable predictions for every target.

## UC-S2-07: Hidden-Known-Value Metrics

| Target | Method | MAE | RMSE | Accuracy | Weighted F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| `asset_age_years` | Baseline | 13.914 | 16.834 | n.a. | n.a. |
| `asset_age_years` | Advanced | 0.001 | 0.032 | n.a. | n.a. |
| `groundwater_indicator` | Baseline | n.a. | n.a. | 0.876 | 0.818 |
| `groundwater_indicator` | Advanced | n.a. | n.a. | 0.923 | 0.925 |
| `corrosion_indicator` | Baseline | n.a. | n.a. | 0.838 | 0.764 |
| `corrosion_indicator` | Advanced | n.a. | n.a. | 0.941 | 0.941 |

The Advanced method performed better for all three targets. For `groundwater_indicator`, accuracy increased by 4.7 percentage points and weighted F1 increased by 0.107. For `corrosion_indicator`, accuracy increased by 10.3 percentage points and weighted F1 increased by 0.177.

The almost perfect `asset_age_years` result requires careful interpretation. In the supplied reference data, `asset_age_years` is deterministically linked to `installation_year` through `asset_age_years = 2026 - installation_year`. The result therefore demonstrates recovery of a retained deterministic relationship, rather than performance on an independently uncertain field.

## UC-S2-08: Distribution and Confidence Validation

### Asset age

| Dataset | Mean | Median | Standard deviation | 25% | 75% |
| --- | ---: | ---: | ---: | ---: | ---: |
| Reference | 29.271 | 27.0 | 16.975 | 16.0 | 42.0 |
| Baseline | 28.836 | 27.0 | 15.273 | 19.0 | 37.0 |
| Advanced | 29.271 | 27.0 | 16.975 | 16.0 | 42.0 |

The Baseline method preserves the median but compresses the distribution toward 27 years. Its standard deviation is lower than the reference and its interquartile range narrows from 26 years to 18 years. The Advanced distribution is effectively identical to the reference distribution.

### Groundwater indicator

| Dataset | Low | Medium | High |
| --- | ---: | ---: | ---: |
| Reference | 3.78% | 87.14% | 9.08% |
| Baseline | 3.28% | 89.62% | 7.10% |
| Advanced | 3.92% | 86.92% | 9.16% |

The Baseline method over-represents the majority `Medium` category by 2.48 percentage points and under-represents `High` by 1.98 percentage points. The Advanced proportions remain within 0.22 percentage points of the reference for every category.

### Corrosion indicator

| Dataset | Low | Medium | High |
| --- | ---: | ---: | ---: |
| Reference | 5.86% | 84.34% | 9.80% |
| Baseline | 4.72% | 87.58% | 7.70% |
| Advanced | 6.02% | 84.32% | 9.66% |

The Baseline method over-represents `Medium` by 3.24 percentage points and under-represents `High` by 2.10 percentage points. The Advanced proportions remain within 0.16 percentage points of the reference.

### Status and confidence

For each target and method, 4,000 original values are marked `Observed / High` and 1,000 filled values are marked `Imputed / TBD`. `Derived` remains available as a future status but was not assigned because no new derived field was created during validation.

`TBD` is intentionally conservative. Per-value High, Medium, or Low confidence cannot be justified from aggregate test performance alone. A later confidence policy should incorporate method performance, predictor availability, uncertainty, and whether direct derivation or target leakage is present. These flags are internal preliminary labels and are not official SAPN standards.

## UC-S2-09: Downstream Ranking Stability

| Comparison | Spearman correlation | Records |
| --- | ---: | ---: |
| Reference vs Baseline | 0.886875 | 5,000 |
| Reference vs Advanced | 1.000000 | 5,000 |

The Advanced method preserves the `asset_age_years` ordering, while the Baseline method changes some ranks by concentrating imputed values near the median. Ranking stability was not calculated for the two categorical indicators because the current framework does not assume that category labels form a validated continuous risk score.

This is a preliminary target-ranking stability test. It should not be interpreted as the official SAPN probability-of-failure or Weibull model. No documented project PoF formula was available for this validation.

## Overall Conclusion

The Advanced method outperformed the Baseline method on hidden-known-value accuracy and distribution preservation for all evaluated targets. The Baseline method shows the expected central-value bias: it reduces numeric variability and shifts categorical distributions toward the majority class. The Advanced method closely reproduces the reference data.

The Advanced results should not be generalized without qualification. `asset_age_years` is directly derivable from `installation_year`, and the categorical indicators appear strongly linked to their corresponding exposure scores. A stricter follow-up experiment should remove or isolate direct source fields when testing derived targets. Performance should also be tested under non-random missingness, feeder/geographical masking, and repeated random seeds before assigning production confidence levels.

## Files Used

- Reference: `data/inputs/underground_cable_reference_synthetic_data.csv`
- Masked: `data/inputs/underground_cable_mcar_20.csv`
- Baseline: `data/inputs/underground_cable_mcar_20_baseline_imputed.csv`
- Advanced: `data/inputs/underground_cable_mcar_20_advanced_imputed.csv`
- Identifier: `asset_id`
- Masking scenario: 20% MCAR per evaluated target
