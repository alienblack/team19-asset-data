# Architecture and Contribution Guide

Shared reusable logic belongs in `src/team19_asset_data/common/`. Asset-specific
logic belongs in its matching stream directory. Each stream should keep data
generation, masking, imputation methods and evaluation separate so methods can be
compared using the same held-out records.

For overhead conductors, the planned modules are:

```text
generate_synthetic_data.py  Create controlled, documented synthetic data
masking.py                  Hide known values reproducibly
baselines.py                Simple transparent comparison methods
classifier.py               Interpretable comparison model
evaluate.py                 Common metrics and result summaries
```

Do not use a target's direct answer as one of its predictors. Split evaluation
records before fitting an imputer or classifier, and compare each approach on the
same deliberately hidden values.

## Pull request checklist

- Jira ticket key appears in the branch and commits.
- No confidential or unapproved data is included.
- Assumptions and provenance are documented.
- A peer has reviewed the change before it enters `main`.
