# PGS219-261 product review and handover

Rishank Goyal · Team 19 · Subteam A

This guide connects the implementation to the runnable demonstration, evidence and earlier Jira work.

## Start the demonstration

From the shared repository root, after the PGS219-259 demo is integrated, use the project Python 3.11 environment.

```sh
python experiments/subteam_a_transformer_benchmark/scripts/product_review_demo.py
```

The longer walkthrough uses `experiments/subteam_a_transformer_benchmark` as its project root. Change into that directory before following its shorter `scripts/...` commands.

For a fresh demo environment, install `experiments/subteam_a_transformer_benchmark/requirements-demo.txt`. The demo recreates the credited synthetic generator inputs and performs fresh model fits.

## Material for the supervisor

- [Walkthrough PDF](Product_Review_Walkthrough.pdf) and [editable guide](Product_Review_Walkthrough.md).
- [A01 MissForest evidence](validation/A01%20MissForest/A01_MissForest_Validation.ipynb).
- [A02 Gaussian EM evidence](validation/A02%20Gaussian%20EM/A02_Gaussian_EM_Validation.ipynb).
- [Full capstone report](../Final_Report.pdf).
- [Suggested ticket descriptions](Ticket_Text.md).

The A01/A02 notebooks recalculate tables and figures from embedded saved scores. The demo script fits the implemented models freshly. The guide includes actual small-demo figures and explains their scope relative to the full tuned study.

## Source walkthrough

1. Open `src/subteam_a/mixture.py` within this benchmark directory.
2. Explain training-derived scaling and categorical encoding.
3. Explain observed masks, component responsibilities and conditional moments in `expect`.
4. Explain the parameter updates and regularisation in `fit`.
5. Explain the weighted predictions and uncertainty in `predict`.
6. Run the demo and inspect the asset example, component plot, interval plot and comparisons.
7. Open the verification record and the independent PGS219-260 checker.

## Development and review trace

PGS219-258 imports the existing implementation from the published SAPN_endtoend repository with commit provenance. PGS219-259 adds the demo entry point, PGS219-260 adds independent evidence checks, and PGS219-261 provides this handover. The original implementation, synthetic data source and earlier work remain attributed in the final report and source integration record.

[Pull-request links, Jira comment text and review order](PR_And_Jira_Trace.md).
