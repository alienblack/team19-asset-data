# Adding a teammate's imputation method

Do not duplicate the pipeline or shared metrics. Put your implementation in
`src/imputation/<method_name>/` and implement the common
`src.imputation.registry.BaseImputer` interface.

Example folder:

```text
src/imputation/catboost/
  __init__.py
  imputer.py         # implements BaseImputer
  diagnostics.py     # optional method-specific analysis, NOT shared metrics
  plugin.py          # registers the method with the CLI
```

The example `plugin.py` is:

```python
from ..registry import register_method
from .imputer import CatBoostImputer


def make_imputer(config, params):
    return CatBoostImputer(
        numeric_columns=config.numeric_columns,
        categorical_columns=config.categorical_columns,
        constraints=config.constraints,
        **params,
    )


register_method("catboost", make_imputer)
```

The registry discovers each subfolder's optional `plugin.py` automatically.
Your factory must return a `BaseImputer` subclass, whose `transform(X)` returns
`ImputationOutput(imputed=<DataFrame>, diagnostics=<DataFrame>, metadata=<dict>)`.
The imputed DataFrame must have the same index and columns as `X`, must not
alter observed cells, and must fill only the approved model attributes.
Set `transductive=True` if the method cannot be fitted on one partition and
applied to unseen records; the unseen-record evaluation protocol will skip it.

Provide optional defaults in the selected asset's `imputation.yaml`:

```yaml
methods:
  # Keep the existing baseline and MissForest sections.
  catboost:
    iterations: 250
    depth: 6
```

Then invoke `--method catboost` or use `--methods catboost,median_mode` with
the `evaluate` command. The common scorer uses the same fit/test split,
missingness mask, and metrics for all methods.

Never access the ground-truth CSV inside an imputer. It is restricted to
`run_evaluate` and the central evaluation protocol.
