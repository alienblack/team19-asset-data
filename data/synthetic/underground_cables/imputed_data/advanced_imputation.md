# Advanced Imputation using MICE-style Predictive Mean Matching

## Purpose

`advanced_imputation.py` demonstrates a more advanced multivariable imputation approach using a MICE-style chained-equations procedure combined with Predictive Mean Matching (PMM).

Unlike the baseline method, the advanced approach uses relationships between multiple asset variables when selecting replacement values.

The target fields are:

- `asset_age_years`
- `groundwater_indicator`
- `corrosion_indicator`

The environmental indicators use the categories:

```text
Low
Medium
High
```

## Why Predictive Mean Matching?

A regression model can produce continuous predictions such as:

```text
asset_age_years = 23.74
```

or, for an internally encoded indicator:

```text
groundwater_indicator = 2.41
```

These values may not directly match realistic or valid field values.

PMM avoids this problem by using regression predictions only to identify similar observed records. The final imputed value is copied from an actual observed donor.

For example, if the model predicts that a missing age is approximately 24 years, PMM may identify observed assets with similar predicted ages:

```text
21
23
24
25
27
```

One donor is selected and its actual observed age is copied into the missing cell.

## Handling Categorical Indicators

`groundwater_indicator` and `corrosion_indicator` are categorical variables with the valid values:

```text
Low
Medium
High
```

For modelling, they are temporarily encoded as:

```text
Low    = 1
Medium = 2
High   = 3
```

These numeric codes are used only internally.

PMM selects an observed donor value rather than directly saving the regression prediction. After imputation, the codes are converted back to their original categories.

For example:

```text
Internal donor value = 3
```

is saved as:

```text
High
```

Therefore, the final output will not contain values such as:

```text
2.37
```

The categorical fields remain strictly:

```text
Low
Medium
High
```

## MICE-style Chained Process

The target variables can contribute information to the imputation of each other.

Conceptually:

```text
Predict asset_age_years
using groundwater, corrosion, and other predictors
        ↓
Predict groundwater_indicator
using age, corrosion, and other predictors
        ↓
Predict corrosion_indicator
using age, groundwater, and other predictors
        ↓
Repeat for several iterations
```

The default implementation performs:

```text
10 iterations
```

## Initialisation

Before chained equations begin, temporary initial values are required so variables containing missing data can still be used as predictors.

The script uses:

- Median initialisation for `asset_age_years`.
- Mode initialisation for `groundwater_indicator`.
- Mode initialisation for `corrosion_indicator`.

These are temporary starting values rather than the final PMM imputations.

## Predictor Selection

The method can use other available fields from the dataset as predictors.

Numeric fields are used directly.

Low-cardinality categorical fields are one-hot encoded so they can be used by the regression model.

Examples may include:

```text
voltage level
insulation type
environmental categories
other numeric asset characteristics
```

High-cardinality text fields, such as unique asset identifiers, are excluded because they generally do not provide useful predictive information.

The current implementation excludes categorical predictor fields with more than 20 unique values.

## Regression Model

The implementation uses:

```text
Bayesian Ridge Regression
```

with predictor standardisation.

The regression prediction itself is not saved as the imputed value. It is used only to identify similar donor records for PMM.

## PMM Donor Selection

For each missing value:

1. A regression model predicts the target for the missing record.
2. Predictions are generated for records where the target is observed.
3. Observed records with the closest predicted values are identified.
4. The nearest records form the PMM donor pool.
5. One donor is randomly selected.
6. The donor's actual observed value is copied into the missing cell.

The default donor pool size is:

```text
5
```

The default random seed is:

```text
42
```

Using a fixed random seed makes the experiment reproducible.

## Workflow

```text
Reference synthetic dataset
        ↓
PGS219-117 artificial missingness
        ↓
Masked CSV
        ↓
PGS219-119 MICE-style PMM
        ↓
Advanced-imputed CSV
```

The current implementation stops after producing the imputed CSV.

No comparison with the original reference dataset or validation metric calculation is performed.

## Input

The input CSV must contain:

```text
asset_age_years
groundwater_indicator
corrosion_indicator
```

`asset_age_years` must be numeric.

The two indicator variables must contain only:

```text
Low
Medium
High
```

or missing values.

Capitalisation differences such as `low`, `LOW`, or `Low` are normalised internally.

## Output

If no explicit output path is supplied, the script creates:

```text
<input_name>_mice_pmm_imputed.csv
```

Example:

```text
underground_cable_mcar_10.csv
```

becomes:

```text
underground_cable_mcar_10_mice_pmm_imputed.csv
```

The categorical output remains:

```text
Low
Medium
High
```

## Usage

Basic usage:

```bash
python advanced_imputation.py \
    --input data/synthetic/underground_cables/missingness_scenarios/underground_cable_mcar_10.csv
```

Custom output:

```bash
python advanced_imputation.py \
    --input masked.csv \
    --output imputed/advanced_imputed.csv
```

Custom number of iterations:

```bash
python advanced_imputation.py \
    --input masked.csv \
    --iterations 15
```

Custom donor pool:

```bash
python advanced_imputation.py \
    --input masked.csv \
    --donors 3
```

Custom random seed:

```bash
python advanced_imputation.py \
    --input masked.csv \
    --random-state 100
```

## Dependencies

```text
pandas
numpy
scikit-learn
```

Install with:

```bash
pip install pandas numpy scikit-learn
```

## Advantages

- Uses relationships between multiple asset variables.
- Can generate different imputations for different assets.
- Uses actual observed values as PMM donors.
- Preserves realistic age values.
- Preserves `Low`, `Medium`, and `High` indicator categories.
- Reproducible when the same random seed is used.
- Provides a useful advanced comparison to the simple baseline approach.

## Limitations

This implementation is intentionally a small experimental example for the synthetic underground-cable dataset.

It should not be treated as a complete production multiple-imputation framework.

Current limitations include:

- Only one completed imputed dataset is produced.
- Multiple independent imputations and statistical pooling are not performed.
- Categorical indicators are ordinally encoded for modelling rather than handled using dedicated ordinal regression models.
- Results depend on relationships present in the synthetic dataset.
- Predictor selection is intentionally simple.
- PMM requires suitable observed donor records.
- An imputed value should not be interpreted as the known true missing value.

The method is intended to demonstrate a more statistically informed approach than median/mode baseline imputation while remaining understandable and reproducible.
