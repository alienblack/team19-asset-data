# Gaussian EM product-review demonstration

Rishank Goyal · Team 19 · Subteam A · PGS219-258

## Run the demonstration

From the repository root, use the project Python 3.11 environment.

```sh
python scripts/product_review_demo.py
```

On the original development machine, this interpreter already has the tested project dependencies.

```sh
/Users/alien/anaconda3/bin/python3.11 scripts/product_review_demo.py
```

For a fresh installation, follow the repository README and install the pinned package with `python -m pip install -e '.[test]'`. The demo recreates the synthetic transformer population using the credited generator included in the source. It can run without the original prepared CSV bundle. The full archived benchmark uses the separate input setup described in the README.

Each execution writes a fresh subdirectory under `outputs/product-review-demo`. The terminal prints the path. Open its three PNG figures and its example CSVs in VS Code.

```sh
python scripts/product_review_demo.py --noise 0.10
python scripts/product_review_demo.py --missing 0.50
python scripts/product_review_demo.py --no-gas-correction
```

Each command runs new fits. The command-line help lists all row-count, seed and output options. The script also contains `# %%` markers so VS Code can navigate the workflow as Python cells. The terminal command is the rehearsed entry point.

## A focused demonstration order

1. State the missing-value problem and identify the implementation being demonstrated.
2. Open `src/subteam_a/mixture.py` and explain `encode`, `preparation`, `expect`, `fit` and `predict`.
3. Run `scripts/product_review_demo.py` from the terminal.
4. Open `gmm_example_record.csv` and show the recorded values, gaps, filled values and clean scoring reference for one asset.
5. Open `gmm_components.png` to explain how component weights and conditional predictions combine for one missing date.
6. Open `gmm_fitting_and_intervals.png` to explain the EM trace and the calibrated intervals.
7. Open `live_comparison.png` to compare the methods by field and gap pattern.
8. Finish with `verification.json`, the known-answer test and the limitations.

The existing A01/A02 notebooks rebuild analysis from embedded archived scores. They are supporting evidence for the larger study. The demonstration script performs fresh fitting and prediction.

## The question behind the model

An asset can have a missing installation day while its location, equipment model or some measurements remain available. The task is to use that observed evidence to estimate the missing entry and quantify the uncertainty.

A Gaussian mixture model describes the joint measurements as several overlapping Gaussian components. Each component has a centre and the covariance describes how the fields vary together. Component weights describe their frequency. An incomplete asset receives a probability for each component according to its observed fields.

The Gaussian mixture is the statistical model. Expectation maximisation is the fitting algorithm. This implementation uses four components and a shared full covariance matrix. Four was prespecified. The components are learned statistical groups and their correspondence to equipment classes was left open.

## What the implementation does

| Function | Implementation and explanation |
| --- | --- |
| `Features.fit` | Learn medians, standard deviations and category frequencies from measured training rows. |
| `MaskedMixture.encode` | Standardise numeric fields, encode categories and apply the optional reversible gas representation. |
| `MaskedMixture.preparation` | Preserve the observation mask and group rows with the same missing pattern. Temporary numeric placeholders support array calculations and initialisation. The mask determines which entries contribute to the observed-data likelihood. |
| `MaskedMixture.expect` | Calculate posterior component probabilities, conditional means and conditional covariance using the observed coordinates. |
| `MaskedMixture.fit` | Initialise with K-means, then update component means, shared covariance and mixture weights. Include conditional uncertainty in the covariance update and add diagonal regularisation. |
| `MaskedMixture.predict` | Combine conditional component predictions and uncertainty, return category probabilities and restore the original numeric scale. |
| `Features.finish` | Apply admissible candidate ranges and preserve recorded values. |

## Explain the E and M steps

In the E-step, calculate how strongly each component explains the observed part of an asset. For each component, estimate the missing values and the uncertainty around them.

In the M-step, use those expectations to update the means, covariance and component weights. The covariance update includes conditional uncertainty about missing entries. The loop stops at the configured iteration limit or when the absolute change in average observed-data log likelihood is below the stopping threshold after at least five iterations.

The component plot comes from the actual newly fitted model. To expose soft component membership, the example is the hidden date with the largest component-membership entropy. Its selection uses the observed inputs and fitted component probabilities, independently of the clean answer or prediction error. `gmm_components.csv` gives the weights, conditional means, conditional standard deviations and weighted contributions. Their weighted mean produces the candidate date before admissible-range clipping. The script checks this relationship.

Categorical outputs combine smoothed class frequencies within each component using the posterior component weights. Equipment model codes are encoded as categories. Separate temperature scaling calibrates their probabilities.

## Why these implementation choices matter

- Numeric fields have different units, so the model uses training-derived centring and scaling.
- The observation mask allows an incomplete row to contribute its available measurements.
- A shared covariance reduces the number of fitted parameters while retaining cross-field relationships.
- Diagonal regularisation stabilises the covariance calculations.
- Grouping missingness patterns reuses conditional matrix calculations across similar rows.
- Numeric uncertainty includes variation within components and disagreement between component predictions.
- Independent calibration adjusts the model scale using held-out errors. A 90% target remains an empirical validation question within particular missing patterns.

## Data, fitting and evaluation

The generator creates 10,000 synthetic assets on a 100 by 100 grid. It is the Subteam C generator adapted in the existing repository. Fixed generator and split seeds are 219 and 617 by default.

The existing spatial splitter reserves disjoint blocks for training, tuning, calibration and test. The fast example uses 1,500 training, 600 calibration and 600 test rows from those partitions. The tuning partition stays reserved because this demo uses fixed benchmark defaults.

The scattered condition independently masks nominally 30% of entries. The grouped condition hides location, gas, environmental and date/model fields together. Both use the same asset partitions and the same nominal gap rate. Their precise target cells can differ. All four methods receive identical inputs and hidden cells within a condition.

The default noise amplitude is plus or minus 5% on the generator-designated measured fields. It does not corrupt every field. The calibration view uses a separate mask draw on its reserved rows. Models fit on measured training inputs. Clean references are supplied separately for calibration and evaluation.

The live demonstration uses the model-only arm. It applies no exact-rule fills. The `--no-gas-correction` option changes the feature representation for a new run.

## Explain the comparison

Numeric skill is `1 - MAE(method) / MAE(median floor)`. Zero matches the floor, one is perfect recovery and negative values indicate larger error than the floor. A skill of 0.8 means an 80% reduction in MAE for that field. The displayed mean gives the 13 numeric non-placebo fields equal weight.

The field plot compares Gaussian EM and Predictive MissForest under scattered gaps. The condition plot compares all four methods under scattered and grouped gaps. Both plots show the fresh demonstration values.

The break rule flags a field when skill is below 0.10 or fewer than half of its fills meet the loose field tolerance. A method can improve on the median and still fail the tolerance requirement. `field_metrics.csv` contains the exact counts and metrics behind the summaries.

Gaussian EM describes conditional relationships within Gaussian components. Predictive MissForest fits separate target-wise forest models and reuses them while updating missing entries. Chained ridge uses linear numeric conditionals and logistic categorical conditionals. The median/mode floor uses training summaries. Different assumptions can lead to different strengths as the visible information changes.

## Explain the two GMM figures

`gmm_components.png` displays weighted conditional densities for one missing date. The combined density is their sum. The point estimate is the weighted conditional mean. The clean reference is attached for scoring. These model densities precede interval calibration.

`gmm_fitting_and_intervals.png` shows the observed-data log-likelihood trace during fitting and the first twelve hidden dates in fixed test order. Read the trace within its own training condition. For each displayed date, compare the point fill, calibrated band and clean reference. The interval target is 90%. The examples were selected by fixed order, independently of their prediction errors.

Wide intervals can lead to abstention because the estimate is insufficiently precise for the field tolerance. An interval containing the truth can still be too wide for an automatic fill.

## Verification and a small test to explain

For a two-field Gaussian with mean zero and covariance `[[1, 0.6], [0.6, 2]]`, observing the first value as 1 gives a conditional mean of 0.6 for the second field and variance `2 - 0.6^2 = 1.64`. With both values missing, the result returns the prior moments. The demo checks these known answers using the existing conditional calculation.

It also checks disjoint spatial partitions, finite completed outputs, positive scales, normalised categorical probabilities and preservation of recorded inputs. After writing the evidence, it reads the saved predictions again and recomputes field MAEs and floor errors. It checks preservation from the saved completed tables as well.

`verification.json` records those checks, model settings, source hashes and runtime. `hidden_predictions.csv` provides row-level provenance for each evaluated gap. The final report and repository test suite provide the broader validation context.

## Relate the demo to the final study

The full tuned feature-enabled anchor reported mean skills of 0.656 for MissForest and 0.641 for Gaussian EM. Under grouped gaps they were 0.264 and 0.435. Gaussian EM broke three fields at the anchor and five with grouped gaps. The grouped-trained MLP reached 0.453 in the follow-up.

Removing the gas correction changed the anchor ranking. Gaussian EM scored 0.671 and MissForest 0.657 without it. The cause of the observed Gaussian EM loss with correction remains unresolved.

These figures belong to the archived repeated benchmark. The live run uses smaller partitions, fixed defaults and one seed pair, so its numbers and even a close ranking can differ. Use the original A01/A02 validation notebooks for the saved full-study evidence. Their source hashes and result scope are documented in those files.

## Limitations to explain

- The data are synthetic and include strong designed relationships. The generator does not establish live SAPN performance.
- This is a fast subset demonstration with one split and fixed defaults. It illustrates functionality and conditional sensitivity.
- Related fields can vanish together, removing both learned clues and available identities.
- Gaussian components approximate nonlinear relationships. A point estimate can compress population tails.
- Calibration can be insufficient for specific missing patterns. Candidate auto-fill labels require further pattern-level validation before operational use.
- Gap filling preserves recorded values. Errors in those recorded inputs need a separate screening process.

## Review material and handover

The source implementation is in `src/subteam_a/mixture.py`. Target-wise forest and ridge implementations are in `src/subteam_a/models.py`. Calibration and scoring are in `src/subteam_a/evaluate.py`. The generator, masks and spatial splits are in `generator.py` and `data.py`.

The existing A01/A02 validation notebooks and Word summaries support the corresponding completed tickets. The final report documents the assumptions, field tolerances, repeated results, uncertainty failures and references.

- Repository — https://github.com/alienblack/SAPN_endtoend
- Final report — https://github.com/alienblack/SAPN_endtoend/blob/main/docs/Final_Report.pdf
- Original data and generator — Team 19, Subteam C, credited in the report and source.

Use this demonstration to explain the implementation, make one observable change and interpret the resulting evidence. The ticket text file maps the code, verification and handover work to PGS219-259, PGS219-260 and PGS219-261.
