# Explainable Baseline Imputation

## Purpose

`baseline_imputation.py` provides a simple, transparent, and reproducible method for filling missing values in the synthetic underground-cable dataset.

It is intended to act as an easily understandable baseline before more advanced imputation methods are applied.

The default target fields are:

- `asset_age_years`
- `groundwater_indicator`
- `corrosion_indicator`

## Imputation Method

The script selects the imputation method according to the data type of each target field.

| Field type | Method |
|---|---|
| Numeric | Median |
| Categorical / text | Mode |

For the current underground-cable dataset:

| Field | Method |
|---|---|
| `asset_age_years` | Median |
| `groundwater_indicator` | Mode |
| `corrosion_indicator` | Mode |

### Median Imputation

For numeric fields such as `asset_age_years`, the median of all available non-missing values is calculated.

Example:

```text
Observed ages:
10, 12, 15, 20, 25

Median = 15
```

A missing age would therefore be replaced with `15`.

Median was selected instead of mean because it is less sensitive to unusually young or old assets.

### Mode Imputation

For categorical fields, the most frequently occurring category is used.

Example:

```text
Groundwater indicators:
Low
Medium
Low
High
Low

Mode = Low
```

A missing `groundwater_indicator` would therefore be replaced with `Low`.

## Workflow

```text
Reference synthetic dataset
        ↓
PGS219-117 artificial missingness
        ↓
Masked CSV
        ↓
PGS219-118 baseline imputation
        ↓
Baseline-imputed CSV
```

Only the selected target columns are modified. Other columns remain unchanged.

## Input

The script accepts a masked CSV produced by PGS219-117.

Example:

```text
underground_cable_mcar_10.csv
```

The CSV must contain the requested target columns.

## Output

If an explicit output path is not supplied, the script creates a CSV beside the input file using:

```text
<input_name>_baseline_imputed.csv
```

Example:

```text
underground_cable_mcar_10.csv
```

becomes:

```text
underground_cable_mcar_10_baseline_imputed.csv
```

## Usage

Basic usage:

```bash
python baseline_imputation.py \
    --input data/synthetic/underground_cables/missingness_scenarios/underground_cable_mcar_10.csv
```

Custom output path:

```bash
python baseline_imputation.py \
    --input masked.csv \
    --output imputed/baseline_imputed.csv
```

Custom target columns:

```bash
python baseline_imputation.py \
    --input masked.csv \
    --columns asset_age_years groundwater_indicator corrosion_indicator
```

## Dependencies

```text
pandas
```

Install with:

```bash
pip install pandas
```

## Advantages

- Simple and easy to explain.
- Reproducible.
- Computationally inexpensive.
- Suitable as a reference method.
- Requires very few assumptions.

## Limitations

Median and mode imputation do not use relationships between different asset characteristics.

For example, two cables with very different installation environments may receive the same median age because the baseline method does not consider their other attributes.

The method can also reduce natural variation because many missing values may receive the same replacement value.

The baseline should therefore be treated as a simple reference method rather than an estimate of the true missing value.
