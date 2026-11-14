# Synthetic Asset Dataset

A synthetic dataset of 10,000 transformer assets for testing how well missing data can be filled in. Each scenario comes as a pair of tables: one with missing cells and measurement noise, and one with the true values. Because the true value of every hidden cell is known, any imputation method can be scored exactly.

All values are generated. Nothing here comes from real assets or real network data.

## What is in this folder

```text
synthetic-asset-dataset/
├── README.md
├── completion0.7-noise0.0-measured.csv    18 CSV files: 9 scenarios × measured and truth
├── completion0.7-noise0.0-truth.csv
├── ...
└── generator/
    └── SyntheticDataset.ipynb             Notebook that creates the CSV files
```

## Quick start

```python
import pandas as pd

measured = pd.read_csv("completion0.8-noise0.05-measured.csv")
truth = pd.read_csv("completion0.8-noise0.05-truth.csv")

# Fill the gaps with any method. This example uses the column median.
column = "measured_humid"
hidden = measured[column].isna()
filled = measured[column].fillna(measured[column].median())

# Score only the cells that were hidden.
error = (filled[hidden] - truth.loc[hidden, column]).abs().mean()
print(f"{hidden.sum()} hidden cells, MAE {error:.4f}")
```

The two files of a pair match row by row. There is no ID column, so keep the original row order when comparing them.

## Scenarios

Each file name sets two controls:

```text
completion0.8-noise0.05-measured.csv
           │        │      └── measured or truth
           │        └── noise: readings can be off by up to ±5%
           └── completion: each cell has an 80% chance of staying visible
```

There are nine scenarios:

| | noise 0.0 | noise 0.05 | noise 0.1 |
|---|---|---|---|
| **completion 0.7** (about 30% of cells missing) | ✓ | ✓ | ✓ |
| **completion 0.8** (about 20% of cells missing) | ✓ | ✓ | ✓ |
| **completion 0.9** (about 10% of cells missing) | ✓ | ✓ | ✓ |

- **measured** files are what an imputation method should see. Missing cells are empty.
- **truth** files hold the true value of every cell and have no gaps. Use them only for scoring.

### Comparing scenarios

All nine scenarios describe the same assets, so the nine truth files are identical. Noise and missingness are also linked between scenarios, which means a difference between two scenarios comes only from the control you changed:

- At a given completion level, the same cells are hidden at every noise level.
- A cell hidden at completion 0.9 is also hidden at 0.8 and 0.7.
- The error on a cell scales with the noise level: at noise 0.1 it is exactly twice the error at noise 0.05.

## Columns

Every column has missing cells in the measured files. The ranges below are for the truth values.

| Column | Meaning | Type | Range | Noisy |
|---|---|---|---|---|
| `x` | Grid row of the asset | Whole number | 0 to 99 | No |
| `y` | Grid column of the asset | Whole number | 0 to 99 | No |
| `install_date` | Installation day, counted from the start of the sequence | Whole number | 2 to 1,389 | No |
| `model_id` | Transformer model | Category (0, 1, 2, 3) | | No |
| `altitude` | Ground altitude | Whole number | 100 to 298 | No |
| `measured_humid` | Humidity exposure | Decimal | 0 to 1 | Yes |
| `measured_salt` | Salt exposure | Decimal | 0 to 1 | Yes |
| `measured_overgrowth` | Vegetation overgrowth | Decimal | 0 to about 1.8 | Yes |
| `life` | Remaining-life indicator | Decimal | about 95,500 to 97,900 | No |
| `random_placebo` | Random number unrelated to any other column | Decimal | 0 to 1 | Yes |
| `measured_gas_a` | Gas A level | Decimal | about 7,200 to 12,100 | Yes |
| `measured_gas_b` | Gas B level | Decimal | about 7,700 to 12,200 | Yes |
| `measured_gas_c` | Gas C level | Decimal | about 7,000 to 11,500 | Yes |
| `total_gas` | Sum of the three true gas levels | Decimal | about 24,600 to 30,600 | No |
| `leakage` | Leakage rate | Decimal | 0.8 to 1.2 | No |

## How the data is generated

### Assets

Assets are generated row by row across a 100 × 100 grid, so row `r` has `x = r // 100` and `y = r % 100`.

Installation days increase along that order. The sequence starts at day 2 and moves to the next day after every 5 to 15 installations. When it reaches a day divisible by 7, it jumps forward two days, so no asset is installed on those days.

Each asset's model depends on its installation day:

| Installation day | Possible models |
|---|---|
| Below 200 | 0 |
| 200 to 399 | 0, 1 |
| 400 to 699 | 0, 1, 2 |
| 700 to 999 | 1, 2, 3 |
| 1000 to 1199 | 2, 3 |
| 1200 and above | 3 |

### Column definitions

`U(a, b)` means a uniform random draw between `a` and `b`.

```text
altitude    = x + y + 100
humid       = U(0, 1) × altitude / 300
salt        = U(0, 1) × (y + 200) / 300
overgrowth  = max(2 × humid − salt, 0)
leakage     = U(0.8, 1.2)
placebo     = U(0, 1)
```

Gas levels fall with age and leakage. Each gas also has its own random factor between 0.975 and 1.025, written `factor` below. The gas levels are floored at zero.

```text
gas_a     = factor × start_a − day × leakage
gas_b     = factor × start_b − 0.0005 × day² × leakage
gas_c     = factor × start_c − day × leakage − 0.00025 × day² × leakage
total_gas = gas_a + gas_b + gas_c
```

Each model starts high in a different gas:

| Model | `start_a` | `start_b` | `start_c` |
|---|---|---|---|
| 0 | 10,000 | 10,000 | 10,000 |
| 1 | 12,000 | 9,000 | 9,000 |
| 2 | 9,000 | 12,000 | 9,000 |
| 3 | 9,000 | 9,000 | 12,000 |

Remaining life starts at 100,000 and falls with the hazards that affect each model, and with gas:

```text
life = 100000 − hazards − 25000 × total_gas / 300000
```

| Model | `hazards` |
|---|---|
| 0 | `day × (humid + salt + overgrowth)` |
| 1 | `day × (humid + overgrowth)` |
| 2 | `day × (salt + overgrowth)` |
| 3 | `day × overgrowth` |

### Noise

Seven columns are noisy: humidity, salt, overgrowth, the placebo and the three gas readings. Each cell gets one draw `u` from `U(−1, 1)`:

```text
measured = true × (1 + u × noise)
```

All other columns are exact in every file. This is deliberate: some columns stay reliable while others degrade.

### Missing cells

Each cell in all 15 columns gets one draw from `U(0, 1)` and is hidden if the draw is above the completion level. Cells are hidden independently of each other and of their values, so the data is missing completely at random.

## Things to watch for

- **Do not use the row number as a feature.** Rows are in grid order, so the row number reveals `x`, `y` and roughly `install_date`, even in cells that are hidden.
- **Keep truth away from your model.** Use truth files for scoring only.
- **`model_id` is a category.** Its numbers have no order, so treat it as a class rather than a quantity.
- **A zero in `measured_overgrowth` is a real reading.** About a third of assets have no overgrowth. Only empty cells are missing.
- **Noisy values can leave the true range.** For example, `random_placebo` can reach about 1.1 at noise 0.1.
- **`total_gas` can differ from the sum of the gas readings.** It is the sum of the true levels, so the two only match at noise 0.0.
- **Some column names start with `measured_` in truth files too.** Column names are the same in both file types, and the values in truth files are the true ones.
- **Few rows are complete.** Dropping incomplete rows leaves about 50 rows at completion 0.7, 350 at 0.8 and 2,060 at 0.9.
- **`random_placebo` is a negative control.** Nothing can predict it, so a method that appears to beat the median on it is finding structure that is not there.

## Regenerating the data

You only need this to change the settings. The notebook reproduces the provided files exactly with the default settings.

1. Open `generator/SyntheticDataset.ipynb` in Jupyter or VS Code. The working directory must be the `generator` folder, which is the default for both.
2. Change the settings at the top of the first code cell if needed:

   ```python
   SEED = 219
   COMPLETION_RATIOS = (0.7, 0.8, 0.9)
   NOISE_RATIOS = (0.0, 0.05, 0.1)
   ```

3. Run all cells. The CSV files are written to the folder above `generator`, replacing any files with the same name.

Generating the files needs only Python 3. The checks at the end of the notebook also use `pandas` and `matplotlib`.

A different `SEED` creates a different set of assets. Copy the existing files elsewhere first if you want to keep them.
