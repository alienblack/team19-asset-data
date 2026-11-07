# Underground cables: configuration placeholder

This asset class has not yet supplied a schema or benchmark dataset. When the
teammate's data is available, create the four profile files beside this note:

- `data_schema.yaml`: unique asset ID, types, allowed categories, legitimate
  predictors, forbidden/leakage columns, and optional grouping.
- `cleaning_rules.yaml`: asset-specific normalisation and validation rules.
- `imputation.yaml`: the attributes to fill and method parameters.
- `pipeline.yaml`: the processing stages and evaluation split policy.

Do not copy conductor or transformer ranges/relationships into this profile
without validating them against underground-cable data. Once created, select
it with `--asset underground_cables` and provide measured/truth CSVs.
