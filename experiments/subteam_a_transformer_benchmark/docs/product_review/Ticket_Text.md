# PGS219-258

## Replacement title

Integrate and demonstrate the final imputation pipeline

## Description

Prepare a reproducible product-review demonstration of the transformer imputation work developed across the earlier sprints. Use the existing Gaussian EM implementation as the main technical walkthrough, with Predictive MissForest, chained ridge and the median/mode floor as comparisons.

Connect the implementation to a complete workflow from synthetic reference data through controlled masking, measured training inputs, fitting, prediction, independent calibration and hidden-cell evaluation. Demonstrate how scattered and grouped gaps change the information available to each method.

Provide a runnable VS Code/terminal script, fresh example predictions, comparison figures, verification results and a short technical handover guide. Link the completed MissForest and Gaussian EM validation work in PGS219-250 and PGS219-251. Retain the grouped-MLP findings in the final report as the wider benchmark context.

The live example uses one spatial split, smaller partitions and fixed benchmark defaults. The final report and previously submitted notebooks contain the tuned repeated benchmark. Explain the settings and sample size whenever presenting a result.

## Acceptance criteria

- The demonstration runs from the supplied source package with Python 3.11 and the documented dependencies.
- It fits Gaussian EM, Predictive MissForest, chained ridge and the floor freshly under scattered and grouped gaps.
- Each condition uses the same held-out assets and inputs for all four methods.
- Training, calibration and test spatial blocks are disjoint. Preprocessing is fitted on measured training rows.
- Recorded test values are preserved in the completed exports.
- A known-answer Gaussian conditional calculation returns mean 0.6 and variance 1.64.
- Saved per-gap predictions reproduce the displayed field MAEs and floor errors.
- The output includes an incomplete/completed asset example, actual GMM component contributions, fitting and interval plots, a method comparison, CSV evidence and a verification record.
- The handover explains the EM steps, the meaning of skill and tolerance, the limitations of the live example and how to reproduce it.

## Scope update comment

Updated the remaining Sprint 6 ticket to cover final integration, verification and the product-review demonstration. The completed model-validation tickets remain the evidence for the earlier experiments. The existing grouped-MLP results remain documented in the final report. This ticket adds a fresh, small demonstration and a technical walkthrough of the implemented pipeline.

# PGS219-259

## Replacement title

Build the Gaussian EM live demonstration

## Description

Create a VS Code/terminal entry point that reuses the existing project implementation to fit Gaussian EM on incomplete transformer records. Include the median/mode floor, chained ridge and Predictive MissForest in the same small experiment.

Use the supplied generator implementation, fixed generator/split seeds and training-only preprocessing. Demonstrate numerical standardisation, missingness masks, four shared-covariance Gaussian components, the expectation/maximisation loop, conditional prediction and separate confidence calibration.

Export a complete before-and-after example and the actual component weights, conditional means and variances behind a missing installation-day estimate. Show the relevant implementation functions during the review.

## Completion evidence

- `scripts/product_review_demo.py`
- `gmm_example_record.csv` and `gmm_example_dates.csv`
- `gmm_components.csv` and `gmm_components.png`
- `gmm_fit_trace.csv` and `gmm_fitting_and_intervals.png`

# PGS219-260

## Replacement title

Verify model comparisons and failure analysis

## Description

Run the live example under scattered and grouped gaps at a fixed nominal missing rate and noise amplitude. Score every method on the same hidden cells within each condition. Calculate numeric MAE, floor-relative skill, loose-tolerance agreement, calibrated coverage and field-break counts using the existing evaluator.

Verify the Gaussian conditional calculation, spatial partition separation, preservation of recorded values, valid categorical probabilities and exported error calculations. Run a changed-noise example to demonstrate that the scenario configuration affects the actual execution.

Explain why grouped gaps remove helper fields and why beating the median can still fail an operational tolerance. Compare the small live example with the archived repeated study using their respective settings and sample sizes.

## Completion evidence

- `live_comparison.png`
- `comparison.csv` and `field_metrics.csv`
- `hidden_predictions.csv`, `measured_test.csv` and `completed_test.csv`
- `partitions.csv`, `runtime.csv` and `verification.json`
- Changed-noise run and its verification record

# PGS219-261

## Replacement title

Prepare product review documentation and handover

## Description

Prepare a concise explanation of the GMM model and incomplete-data EM implementation, with a demonstration order suitable for the academic supervisor. Map the explanation to the actual code functions and generated outputs.

Document setup, execution, changes to the missingness/noise settings, interpretation of the figures, verification checks and remaining limitations. Include the earlier A01/A02 validation notebooks as saved-result evidence and link the final report and GitHub repository.

Assemble the runnable source, guide, verification output, original validation evidence and example figures into an attachment package that can be extracted and rerun.

## Completion evidence

- `Product_Review_Walkthrough.pdf`
- `Product_Review_Walkthrough.md`
- `PGS219-258_Ticket_Text.md`
- `PGS219-258_Product_Review_Pack.zip`
- Existing A01/A02 notebooks and validation summaries
- Final report and repository links

## Work-log wording

Use the entries that match the activity and record the actual time spent.

- PGS219-259 — Integrated the existing Gaussian EM, forest, ridge and floor implementations into a reproducible live demonstration. Added an asset-level example and visualised the GMM component contributions.
- PGS219-260 — Executed and checked the scattered/grouped comparison, verified the exported prediction errors and preserved observations, and tested a changed-noise configuration.
- PGS219-261 — Prepared the technical explanation, run instructions, evidence package and product-review walkthrough.
- Parent comment — Consolidated the completed demonstration, verification evidence and handover attachments. Linked the earlier model-validation tickets.

Log shared time once, on the child where the work occurred. A parent summary can link the evidence without duplicating the same time entry.
