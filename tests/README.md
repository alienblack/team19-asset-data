## PGS219-184-implement-predictive-miss-forest-with-hyperparameter-tuning-and-experimental-evaluation

Two new pytest files for Predictive MissForest:

- `tests/test_predictive_missforest_review.py` — five regression tests for fit/transform, observed-value preservation, excluded-column preservation, imputation on new rows with an entirely missing field, save/load exact replay with a guard against retraining, training labels excluding placeholder values, and missing required-column errors.
- `tests/test_predictive_experiment_runner_review.py` — two tests for the 60/10/10/20 disjoint experiment-runner workflow: verify fit sees only fitting rows, hidden-test-mask scoring against clean truth, baseline and predictive outputs and completeness, and failure on incomplete reference data.

## Run the new tests

```bash
PYTHONPATH=src python -m pytest -q \
  tests/test_predictive_missforest_review.py \
  tests/test_predictive_experiment_runner_review.py
```

## Run entire regression suite

```bash
PYTHONPATH=src python -m pytest -q tests/
```