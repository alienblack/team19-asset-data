# Transformer Asset Imputation - Experimental Methodology, Results and Reproduction

**Team 19 · Master of Computer Science Capstone · University of Adelaide · SA Power Networks collaboration**

**Study scope:** Controlled imputation on synthetic transformer asset records. These data are not operational SAPN records, and the results are not estimates of field-data performance.

**Main notebook:** `experiment/transformer/notebook/transformer_imputation_experiments.ipynb`  
**Separate foundation-model notebook:** `experiment/transformer/tabpfn_separate/transformer_tabpfn_experiment.ipynb`

This document describes the experiment as implemented and reports the observed results. It also provides instructions to run the study from a fresh Python environment, reproduce its configurations, analyse saved results, and interpret the exported files.

## At a glance

- **Data:** One 10,000-row, 15-attribute synthetic transformer population supplied as nine measured/truth dataset pairs; 10%, 20%, 30% missingness crossed with 0%, ±5%, ±10% measurement-noise amplitude.
- **Methods:** Median/Mode; Original MissForest; Predictive MissForest; Chained Ridge; LightGBM chained; MICE-PMM/logistic. TabPFN is examined separately.
- **Protocols:** Joint-table reconstruction and unseen-record prediction are reported separately.
- **Experiments:** T1 synthetic dataset characterisation and relationship analysis; T2 noise × missingness; T3 feature ablation and predictor redundancy; T4 grouped missingness; T5 observed-value corruption; T6 training–application robustness. Additional analyses investigate conditional remaining-life sensitivity, MICE pooling and gas-identity consistency.
- **Evaluation:** MAE, RMSE and baseline-relative skill for numerical targets; Macro-F1 and related metrics for `model_id`; paired scored cells, spatially grouped uncertainty and per-attribute analysis.

## Run the study from a fresh environment - short version

```bash
cd pipeline
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[notebook,test]"
python -m jupyter lab experiment/transformer/notebook/transformer_imputation_experiments.ipynb
```

The notebook's **Settings** cell starts in quick execution mode. Select **Kernel → Restart Kernel and Run All Cells** to test the setup; switch to full mode and run again to reproduce research-scale experiments.

```python
MODE = 'quick'              # change to 'full' for research-scale execution
EXECUTE_EXPERIMENTS = True
RUN_RELATIONSHIP_EXPERIMENTS = True
RUN_T2 = RUN_T3 = RUN_T4 = RUN_T5 = RUN_T6 = True
```

A fresh execution writes to `experiment/transformer/outputs/transformer_study_<timestamp>_<mode>/`. Existing run folders are not overwritten. Further instructions appear in [Section 8](#8-running-the-notebook-from-a-fresh-installation).

---

## 1. Research background, purpose and contribution

Electricity distribution asset databases can contain missing, noisy or incorrect measurements and descriptive fields. An imputation model that performs well under ordinary missingness may deteriorate when related predictors disappear together, when observed values are incorrect, or when its training data quality differs from the data encountered at application time.

Our industry partner is interested in **explainable experimental outcomes**: which models perform well, under which conditions they deteriorate, and which available data relationships explain the outcome. This study therefore goes beyond an accuracy leaderboard by using known synthetic-generator dependencies and deliberately controlled data-quality interventions.

### Research questions

| Question | Research objective | Main evidence |
|---|---|---|
| **RQ1 - Comparative performance** | Which imputation families reconstruct numerical and categorical transformer attributes most accurately? | T2, per-target scores, compatible-protocol comparisons, repeated spatial splits |
| **RQ2 - Robustness and deterioration** | How does performance change with missingness, measurement noise, jointly missing predictors, incorrect observed values and training/application mismatch? | T2, T4, T5, T6, row-volume-matched controls |
| **RQ3 - Explanation** | Which deterministic, conditional, redundant and probabilistic dependencies account for resilience or failure? | Synthetic-generator analysis, T3, joint `model_id` ablation, T4 controls, T5 propagation, conditional-life experiment |

### The two contributions must be separated

1. **Scientific/experimental contribution:** Controlled comparison and explanation of imputation methods under synthetic data-quality degradation, with paired scoring, attribute-level analysis and uncertainty assessments.
2. **Engineering contribution:** A reusable **Predictive MissForest** implementation that saves per-pass conditional Random Forest models and replays them on previously unseen records (`fit`, `transform`, `save`, `load`). The Original MissForest and Predictive MissForest implementations use the **same iterative conditional-Random-Forest reconstruction mechanism** under matched conditions. Their distinction is principally **stopping policy and inductive versus transductive inference**, **not** a novel forest-learning algorithm or an intrinsic accuracy advantage.

The project does **not** claim that its synthetic error rates, failure thresholds or model rankings represent actual SAPN field performance.

---

## 2. Repository layout and authoritative files

The workflow is organised around one active main transformer experiment notebook, with algorithms imported from existing Python modules.

```text
pipeline/
├── pyproject.toml                        # Core and optional Python dependencies
├── README.md
├── src/
│   ├── imputation/
│   │   ├── predictive_missforest/        # Reusable predictive forest implementation
│   │   ├── missforest/                   # Original transductive MissForest
│   │   └── registry.py
│   └── ...                               # Shared pipeline modules
├── notebook/
│   └── pipeline_overhead_conductors.ipynb  # Separate asset-specific pipeline example
├── data/
│   └── transformers/                     # DIFFERENT generator population; not the main T2 data
└── experiment/
    └── transformer/
        ├── dataset/
        │   ├── completion0.9-noise0.0-measured.csv
        │   ├── completion0.9-noise0.0-truth.csv
        │   ├── ...                       # Nine supplied measured/truth pairs
        │   └── generator/
        │       └── SyntheticDataset.ipynb
        ├── notebook/
        │   └── transformer_imputation_experiments.ipynb
        ├── mice_pmm_logistic.py
        ├── scripts/
        │   ├── comparison_models.py
        │   ├── run_model_comparison.py
        │   ├── plot_model_comparison.py
        │   ├── test_model_comparison.py
        │   └── test_robustness_suite.py
        ├── tabpfn_separate/
        │   └── transformer_tabpfn_experiment.ipynb
        └── outputs/
            ├── single_notebook_20261113_215529_full/  # Recorded T2–T6 experiment
```

**Important provenance distinction:** `experiment/transformer/dataset/` is the main experiment population, reproduced exactly from generator seed **219**. `pipeline/data/transformers/` contains a second population made by the same generator at an **unrecorded** seed. Do **not** silently combine the two, or treat them as interchangeable data files. A population-level replication would require a clearly recorded generator seed and a distinct output directory.


---

## 3. Data source and synthetic generator

### 3.1 Dataset shape, schema and attribute roles

The generator creates **10,000 records** arranged on a 100 × 100 coordinate grid, with **15 modelled attributes**: 14 numerical and one categorical.

| Attribute | Type | Generator role / important relationship |
|---|---|---|
| `x` | Numerical | Grid coordinate; with `y`, determines `altitude` and reveals the generator's row-order shortcut |
| `y` | Numerical | Grid coordinate; drives the distribution of `measured_salt` |
| `install_date` | Numerical | Generated installation-day count; highly related to grid iteration order, **not** an independent physical date mechanism |
| `model_id` | Categorical, classes `0`–`3` | Age-dependent model-category draw; gas signatures and life relationships provide additional information about it |
| `altitude` | Numerical | Exact: `x + y + 100` |
| `measured_humid` | Numerical | Uniform draw scaled by `altitude / 300` |
| `measured_salt` | Numerical | Uniform draw scaled by `(y + 200) / 300` |
| `measured_overgrowth` | Numerical | Exact clipped function: `max(2 × humid − salt, 0)` |
| `life` | Numerical | Model-dependent remaining-life equation incorporating installation day, environmental hazards and total gas |
| `random_placebo` | Numerical | Independent `U(0, 1)` negative-control attribute |
| `measured_gas_a` | Numerical | Model-specific starting gas, age-dependent decay and latent leakage |
| `measured_gas_b` | Numerical | Model-specific starting gas, squared-age decay and latent leakage |
| `measured_gas_c` | Numerical | Model-specific starting gas, linear and squared-age decay and latent leakage |
| `total_gas` | Numerical | Exact in ground truth: sum of the three gas channels |
| `leakage` | Numerical | Independently generated `U(0.8, 1.2)` factor; can be inferred indirectly from gas consequences |

`install_date` is the generator's day count, not a calendar-formatted timestamp. The generator populates rows in grid order and advances the day counter as it does so. The near-deterministic relationship between coordinates and installation date is a **synthetic generation artefact**; unusually high date-imputation accuracy must not be treated as evidence of a general real-world spatial law.

### 3.2 Known generator equations

For coordinates `x`, `y`, humidity `h`, salt `s` and age/day count `d`:

```text
altitude         = x + y + 100
humid            ~ Uniform(0, altitude / 300)
salt             ~ Uniform(0, (y + 200) / 300)
overgrowth       = max(2 * humid - salt, 0)
leakage          ~ Uniform(0.8, 1.2)
random_placebo   ~ Uniform(0, 1)
```

The gas channels use model-specific starting values (in units of the synthetic generator):

| `model_id` | Gas A start | Gas B start | Gas C start |
|---|---:|---:|---:|
| 0 | 10,000 | 10,000 | 10,000 |
| 1 | 12,000 | 9,000 | 9,000 |
| 2 | 9,000 | 12,000 | 9,000 |
| 3 | 9,000 | 9,000 | 12,000 |

With independently drawn hidden multipliers `h_a`, `h_b`, `h_c` in `[0.975, 1.025]` and leakage `L`:

```text
gas_a     = max(h_a * start_a - d * L, 0)
gas_b     = max(h_b * start_b - 0.0005 * d^2 * L, 0)
gas_c     = max(h_c * start_c - d * L - 0.00025 * d^2 * L, 0)
total_gas = gas_a + gas_b + gas_c
```

The synthetic life equation is **conditional on model identity**. Let `gas_penalty = 25000 × total_gas / 300000`. Then:

```text
model 0: life = 100000 - d*humid - d*salt - d*overgrowth - gas_penalty
model 1: life = 100000 - d*humid          - d*overgrowth - gas_penalty
model 2: life = 100000          - d*salt - d*overgrowth - gas_penalty
model 3: life = 100000                   - d*overgrowth - gas_penalty
```

These equations are **synthetic experimental ground truth**. In particular, independent `leakage` has no generator parents, but its downstream effects on gas make inverse estimation possible. The placebo is genuinely independent and should not show a meaningful, reproducible improvement over a suitable marginal baseline.

### 3.3 Nine supplied input scenarios

Every scenario has two files: `*-measured.csv` (observed/noisy/masked input) and `*-truth.csv` (complete reference).

| Nominal missingness | Noise ±0% | Noise ±5% | Noise ±10% |
|---|---|---|---|
| 10% | `completion0.9-noise0.0` | `completion0.9-noise0.05` | `completion0.9-noise0.1` |
| 20% | `completion0.8-noise0.0` | **`completion0.8-noise0.05` (anchor)** | `completion0.8-noise0.1` |
| 30% | `completion0.7-noise0.0` | `completion0.7-noise0.05` | `completion0.7-noise0.1` |

To form a filename, append `-measured.csv` or `-truth.csv` to the stem.

The original generator creates one **shared noise draw and one shared masking draw per cell** and reuses these draws at different levels:

- Noise is **uniform multiplicative amplitude**, not the fraction of noisy records: for eligible numerical fields, `observed = truth × (1 + amplitude × U[-1,1])`.
- Only seven attributes receive measurement noise: `measured_humid`, `measured_salt`, `measured_overgrowth`, `random_placebo`, `measured_gas_a`, `measured_gas_b`, `measured_gas_c`.
- Every column, including exact and categorical fields, can be independently **masked**.
- Masks are **nested**: cells missing at 10% also remain missing at 20% and 30%. Noise amplitudes scale the same underlying noise draws.
- All nine scenario truth files refer to the **same 10,000 synthetic records**. The nine settings are **not nine independent data populations or replications**.
- Ground-truth relationships such as `total_gas = gas_a + gas_b + gas_c` can become inconsistent among observed values when the individual gas channels are measured with noise but `total_gas` is not.

The notebook **loads these supplied datasets directly** in T2; it does **not** regenerate or clean them. T4, T5 and T6 then construct separately labelled research interventions for questions the original nine scenarios cannot answer.

---

## 4. Evaluation protocols and scientific safeguards

### 4.1 Two different experimental protocols

| Protocol | How it works | Methods | Interpretation |
|---|---|---|---|
| **Unseen-record / inductive** (`unseen_records`) | Fit on **8,000 training records**, apply frozen fitted model to **2,000 held-out records** | Median/Mode, Predictive MissForest, Chained Ridge, LightGBM chained | Tests generalisation to records not used during model fitting |
| **Joint-table / transductive** (`joint_table_transductive`) | Fit the imputer on a **single joint 10,000-row table**, then score the designated held-out row positions | Median/Mode, Original MissForest, Predictive MissForest, Chained Ridge, LightGBM chained, MICE-PMM/logistic | Measures table-level reconstruction when rows being scored are visible during imputation fitting |

The joint protocol gives the imputer additional access to the 2,000 test rows during reconstruction. **Do not rank a joint-table score against an unseen-record score as though they are equivalent experiments.** A joint-vs-unseen gap is descriptive and also reflects different training exposure.

The full unseen split is constructed using coordinate tiles (`x//10`, `y//10`), with split seed **617**. Two of ten `y` tiles per `x` band are held out, giving exactly **8,000 train / 2,000 test** records. Additional split seeds **631** and **653** are repeated partitions of **the same population**, not new generator seeds.

The split prevents row overlap but does **not** eliminate the generator's coordinate-to-installation-date shortcut: surrounding training tiles can still make that relationship easy to learn.

### 4.2 Methods and major full-mode parameters

| Method | Core procedure | Use in this study |
|---|---|---|
| `median_mode` | Marginal numerical median and categorical mode | Universal baseline; no cross-attribute prediction |
| `missforest` | Iterative conditional Random Forest with early stopping | Joint-table Original MissForest; max **20** iterations |
| `predictive_missforest` | Same underlying conditional Random Forest mechanism, stores each pass's models for frozen replay | **5** fixed passes; **100** trees; minimum leaf **10**; `max_features='sqrt'`; `n_jobs=1` |
| `chained_ridge` | Repeated conditional Ridge models | **3** passes; Ridge `alpha=10` |
| `lightgbm_chained` | Repeated conditional LightGBM models | **3** passes; **100** estimators; learning rate **0.07**; **31** leaves |
| `mice_pmm_logistic` | PMM for numeric variables and logistic modelling for the categorical attribute; multiple stochastic draws | **5** chains, **10** iterations; joint-table only |
| `tabpfn_unsupervised` | Separate experimental stochastic/foundation-model imputation extension | **Not** executed by the main notebook; separate optional comparison |

The baseline for normalised skill is fitted/evaluated within the appropriate comparison protocol. The same masked target cells are used in paired method comparisons. For frozen-model experiments, observed predictors are left unchanged unless they are the **explicit target of the intervention**.

### 4.3 Metrics

For numerical attributes, scores are computed **only on deliberately missing target cells**, using complete synthetic truth:

```text
MAE        = mean(|truth - imputation|)
RMSE       = sqrt(mean((truth - imputation)^2))
Signed bias = mean(imputation - truth)
Numeric skill = 1 - MAE_method / MAE_median_mode
```

Interpretation of numerical skill:

- `1` indicates zero MAE on the evaluated missing cells.
- `0` indicates the same MAE as the marginal median baseline.
- `> 0` indicates improvement over the baseline.
- `< 0` indicates worse-than-baseline *point-estimate* performance; statistical breakdown requires suitable uncertainty evidence.

For `model_id`, the notebook reports **accuracy**, **balanced accuracy** and **Macro-F1**, using the fixed label set `['0','1','2','3']`.

The cross-attribute **mean numerical skill** is a compact descriptive summary over **13 numerical targets**, deliberately excluding independent `random_placebo`. It mixes exact, shortcut-driven and stochastic attributes; it is **not the only or primary conclusion**. Use **per-target results**, relationship-specific categories and target-wise uncertainty wherever possible.

**Statistical interpretation:** Paired spatial-block bootstrap intervals describe uncertainty conditional on the tested population, partition, mask realisation and fitted models. Three repeated spatial splits also provide a limited stability check. No single “±0.014 resolution floor” applies universally to all comparisons. A nonzero paired difference, consistent sign across repeats and independently generated populations provide different levels of evidence; this project currently has strong **within-population** evidence, not generalisation across SAPN data or many generator populations.

---

## 5. Detailed experimental design and completed findings

### T1 - Synthetic Dataset Characterisation and Relationship Analysis

**Objective:** Understand the structure, distributions, missingness characteristics and generator dependencies of the synthetic transformer data before evaluating imputation algorithms.

**Dataset:** The study uses 10,000 synthetic transformer records with 15 attributes (14 numerical and one categorical). Nine measured/ground-truth dataset pairs represent combinations of 10%, 20% and 30% missingness with 0%, ±5% and ±10% multiplicative measurement noise.

**Analysis performed:**
1. Examine dataset dimensions, data types and attribute distributions.
2. Summarise missing-value counts and percentages across attributes.
3. Visualise missingness patterns and numerical correlations.
4. Examine known generator relationships, including:
   - `altitude = x + y + 100`
   - `measured_overgrowth = max(2 × measured_humid − measured_salt, 0)`
   - `total_gas = measured_gas_a + measured_gas_b + measured_gas_c`
5. Identify probabilistic and conditional attribute dependencies.
6. Examine the installation-date/coordinate relationship introduced by the generator.
7. Identify `random_placebo` as an independent negative-control attribute.

**Findings:** The synthetic dataset contains exact algebraic relationships, conditional and probabilistic dependencies, and a predictable installation-date pattern associated with generator ordering. These relationships provide useful information for reconstructing missing values but also introduce synthetic-specific characteristics that influence model performance.

**Outcome:** T1 establishes the data characteristics and known information pathways used to design and interpret the subsequent imputation experiments. It is a descriptive analysis rather than an additional imputation benchmark.

> The dataset generation and scenario details are documented in Section 3.

### T2 - Independent missingness × measurement noise

**Question:** How accurately does each imputation method reconstruct hidden numerical/categorical values under increasing original MCAR missingness and measurement noise?

- Inputs: **all nine supplied measured/truth pairs** (10/20/30% missingness × ±0/5/10% noise).
- Full unseen-record evaluation: 8,000/2,000 spatial split. Joint-table methods are reported separately.
- All **15 targets** are scored whenever target cells are missing.
- The same scenario, split and scored cells are used across compatible methods.
- Across different missingness levels, nested masks give partially overlapping-not identical-scored-cell sets. The  notebook therefore also computes a comparison restricted to cells already masked at 10%.

**Full-mode reference scenario (20% missing, ±5% noise), unseen-record protocol:**

| Method | Mean numerical skill, excluding placebo | `model_id` Macro-F1 |
|---|---:|---:|
| Median/Mode | 0.000 | 0.113 |
| Chained Ridge | 0.699 | 0.973 |
| Predictive MissForest | 0.709 | **0.985** |
| **LightGBM chained** | **0.769** | 0.980 |

The mean skill ordering is not uniform over attributes: LightGBM's advantage is concentrated in environmental, coordinate and some structural targets; the gas-channel scores are much closer. The Chained Ridge versus Predictive MissForest difference is **not stable across all three repeated spatial splits**. Original MissForest is **not** a separate algorithm-family accuracy victory over PMF when both run the same iterations.

**PMF mean numerical skill across T2:**

| Missingness | 0% noise | ±5% noise | ±10% noise |
|---|---:|---:|---:|
| 10% | 0.7610 | 0.7447 | 0.7396 |
| 20% | 0.7231 | 0.7093 | 0.6952 |
| 30% | 0.6704 | 0.6538 | 0.6386 |

The observed degradation is monotonic in the aggregate, but that does not establish an operational breakdown threshold. Report per-attribute curves and the shared-mask design alongside the aggregate.

### T3 - Feature ablation and information pathways

**Question:** How does reconstruction change if selected predictors are unavailable during both fitting and application?

Original ablations include removal of `x`/`y`, `install_date`, `model_id`, `total_gas`, `altitude` and the placebo. The original T3 compares PMF, Chained Ridge and LightGBM at the **20%-mask, ±5%-noise anchor**. An ablated model is **refitted without those predictor columns**; the evaluated target's hidden rows are preserved when that target remains present.

Illustrative original findings:

- Removing coordinates sharply degrades `altitude` reconstruction because the exact `altitude=x+y+100` pathway is lost.
- Removing `altitude` harms `y`, which can be recovered from `altitude-x-100` when the siblings are known.
- Removing `model_id` reduces gas reconstruction, e.g. PMF gas A skill around **0.898 → 0.815**. This measures the *value of model identity as a predictor for gas*, not the recoverability of `model_id` itself.
- Removing the independent placebo has very little effect, an important negative control.

**Joint `model_id` ablation experiment:** Three nested predictor removals specifically examine **recoverability of `model_id` as the target**, using full-mode PMF, `max_features=3` held fixed, and the same **409** masked target cells.

| Predictor removal | `model_id` Macro-F1 |
|---|---:|
| Full-feature reference | ~0.9854 |
| `x`, `y`, `install_date` | **0.9825** |
| Above, plus three gas channels | **0.6299** |
| Above, plus `life` | **0.5193** |

**Interpretation:** Coordinate/date information is redundant under the retained predictors; the gas channels provide a major additional model-identity pathway; `life` also carries conditional model information. The remaining Macro-F1 of 0.519 is still well above the majority-class baseline and **does not mean all identifying information has disappeared**.

**Caveat:** Ablation with refitting estimates the effect of **available information in the full modelling procedure**, not a causal feature attribution or feature importance in a fixed fitted model.

### T4 - Grouped missingness and predictor availability

**Question:** Is it especially damaging when the hidden target also loses related supporting predictors, rather than an equal number of arbitrary predictors?

Three generator-motivated groups are tested:

| Group | Primary anchor target | Supporting predictors hidden together |
|---|---|---|
| Coordinates | `altitude` | `x`, `y` |
| Environment | `measured_humid` | `measured_salt`, `measured_overgrowth` |
| Gas | `measured_gas_a` | `measured_gas_b`, `measured_gas_c` |

**Primary frozen-model intervention:** Fit on the original anchor training data and hold the model fixed. On test rows where the anchor is already missing, hide additional currently observed supporting predictors. The anchor's scored cells remain fixed, but **overall missingness increases**.

**Row-volume-matched missingness control:** On the **same anchor-missing rows**, hide the same number of additional observed predictors **per row**, chosen randomly from *non-group* attributes. Repeat across five random seeds without refitting the model. This matches **row-level missingness volume**, **not per-column missingness counts**, and tests whether losing the structurally related predictors is especially harmful.

**Completed full-mode anchor-target numerical skill** (random controls averaged across five draws):

| Group / target | Method | Original MCAR | Random-volume control | Related predictors hidden |
|---|---|---:|---:|---:|
| Coordinates / altitude | Ridge | 0.7919 | 0.7828 | 0.1538 |
| Coordinates / altitude | LightGBM | 0.7610 | 0.7545 | 0.2286 |
| Coordinates / altitude | PMF | **0.6755** | **0.6739** | **0.1920** |
| Environment / humidity | Ridge | 0.5995 | 0.5918 | 0.3428 |
| Environment / humidity | LightGBM | 0.7272 | 0.7197 | 0.4161 |
| Environment / humidity | PMF | **0.6348** | **0.6180** | **0.4097** |
| Gas / gas A | Ridge | 0.8798 | 0.8628 | 0.8151 |
| Gas / gas A | LightGBM | 0.8807 | 0.8679 | 0.7985 |
| Gas / gas A | PMF | **0.8984** | **0.8889** | **0.8185** |

Additional hidden test predictor cells: **645** coordinates, **652** environment, **620** gas. The consistent and larger grouped drop supports a **structural predictor-availability explanation** in this synthetic setting.

**Separate secondary T4 design:** A **count-preserving grouped-mask** scenario redistributes source-column missingness positions, re-exposes previously missing synthetic source values using truth plus independently generated matched-amplitude noise, and **refits** the model. This is a constructed new synthetic dataset, **not** a frozen-model intervention or actual recovery of field measurements. It must be labelled and analysed separately; do not average it with the primary frozen design.

**Coverage caveat:** The expanded output scores each group's other members as **spillover targets**, but its intervention is still keyed to the **primary anchor's hidden rows**. The non-anchor results are not equivalent to separately rotating the anchor and testing each group member independently.

### T5 - Incorrect *observed* predictors and error propagation

**Question:** What happens when influential values are *present but incorrect* rather than missing?

- Starting point: 20% missing, ±5% noise anchor, same held-out rows and target masks.
- Perturb only **originally observed** `model_id` or `install_date` values by distribution-preserving donor swaps.
- Nominal selection fractions: **0%, 5%, 10%, 25%, 50%**. Test the frozen imputer at each dose; score the **same hidden target cells** before and after corruption.
- There is also a separately labelled **training corruption** arm; it must not be mistaken for frozen-model application-time corruption.
- The imputer may propagate the incorrect observed predictor into predictions of other attributes. The baseline is unaffected by these predictor swaps because marginal Median/Mode does not use them.

**Effective corruption prevalence:** Some selected `model_id` donor values equal the recipient's original category. The **nominal selected fraction is not the true corruption prevalence**.

```text
nominal_selected_fraction = selected_candidates / eligible_observed_cells
swap_efficiency          = actually_changed / selected_candidates
effective_prevalence     = actually_changed / eligible_observed_cells
```

Use **effective prevalence as the dose-response plot's x-axis** and keep the nominal selection level as metadata. At the nominal 50% level:

| Corrupted predictor | Actually changed | Eligible observed | Actual effective prevalence |
|---|---:|---:|---:|
| `model_id` | 564 | 1,591 | **35.45%** |
| `install_date` | 801 | 1,608 | **49.81%** |

At the same *nominal* 50% `model_id` swap setting, representative PMF gas A numerical skill declines from approximately **0.898 to 0.635**; LightGBM declines from about **0.881 to 0.505** and Ridge from **0.880 to 0.573**. Since the true changed-value prevalence differs between predictors, **do not directly compare equal nominal levels across `model_id` and `install_date`**.

The T5 cell-level export permits **paired error propagation**: compare absolute error for the same row, target and model between the uncorrupted and corrupted conditions. Untouched rows should show exactly zero prediction-error change under deterministic frozen replay. The recorded error directions are consistent with known gas-start and date-decay dependencies, but exact propagation magnitudes are conditional on the synthetic generator.

### T6 - Frozen-model training/application mismatch

**Question:** Can the same trained imputer be reused accurately when application records have different data quality?

- Train with noise amplitudes **0%, ±5%, ±10%** and apply the frozen model at **0%, ±5%, ±10%, ±20%, ±40%**.
- A separate matrix trains under **10%, 20%, 30% missingness** and applies to those missingness levels, with the other relevant conditions controlled.
- The ±20% and ±40% inputs are **synthetic stress tests** obtained by scaling the original stored ±5% measurement-noise deviations. They are **not** separate original teammate datasets.
- Original synthetic truth is used only to **construct these explicitly simulated stress-test observations** and to evaluate performance. It is not an application-time model predictor.
- Apply-time error and baselines are evaluated against the appropriate observed scenario; the reusable models are not refitted at transform time.

**Representative PMF noise results:** At application noise 0%, 5%, 10%, the corresponding matched training level performs best in the completed anchor study (mean skill approximately **0.7231**, **0.7093**, **0.6952**, respectively). For harsher ±20% and ±40% application noise, training at the highest tested noise level (±10%) is generally more robust than training at 0%.

**Missingness finding:** Training at 10% missingness had slightly higher estimated aggregate performance than condition-matched training in the tested matrices, but key margins were only **~0.003–0.004** skill. These fine differences are **not resolved** by the present stability evidence; do not make a firm claim that lower missingness training is universally better.

**Interpretation:** This is **observed data-quality mismatch**, not proven concept drift, because the underlying synthetic generator functions do not change.

### Additional RQ3 relationship - Conditional remaining-life sensitivity

**Question:** Does corruption of humidity/salt propagate differently into `life` when the related overgrowth field is available versus deliberately missing, and how does this vary by true `model_id`?

This experiment uses the **frozen full-mode PMF**, fixed hidden `life` target cells, and paired 25% candidate corruption of originally observed humidity and salt. Two conditions are compared: overgrowth remains observed, or overgrowth is additionally hidden on the scored `life` rows.

For an apples-to-apples contrast, the analysis restricts both conditions to **319 identical originally overgrowth-observed scoring records**, then to **128 records whose humidity and/or salt values actually changed**. The hidden-arm file alone contains extra rows with already missing overgrowth; these are **not** included in the paired table below.

| True `model_id` | Paired corrupted records | Extra absolute-error change: overgrowth available | Extra absolute-error change: overgrowth hidden |
|---|---:|---:|---:|
| 0 | 43 | +39.1 | +56.0 |
| 1 | 29 | +78.7 | +193.7 |
| 2 | 22 | +84.8 | +99.9 |
| 3 | 34 | +78.4 | **+261.9** |

These values are increases in the synthetic `life` error units. The model-3 increase when overgrowth becomes hidden is approximately **183.5** additional units relative to the observed-overgrowth condition; model 1 also shows pronounced amplification. The positive model-3 effect even with overgrowth observed demonstrates that **learned predictive reliance** is not identical to direct dependency in the generator's life formula. Subgroup sample sizes are modest; report these as exploratory conditional contrasts rather than formal universal mechanisms without additional paired uncertainty or independent populations.

### Additional analysis - MICE numeric pooling sensitivity

MICE creates stochastic imputation draws. The notebook computes **both mean-pooled and median-pooled numerical point estimates from the same draws**, with identical scored cells; categorical predictions are the same.

The expected MAE advantage of median pooling was **not observed** in the completed full grid:

| Missingness / noise | Mean pooled numerical skill | Median pooled numerical skill |
|---|---:|---:|
| 10% / 0% | 0.7810 | 0.7786 |
| 10% / ±5% | 0.7487 | 0.7357 |
| 10% / ±10% | 0.7223 | 0.6980 |
| 20% / 0% | 0.7212 | 0.7129 |
| **20% / ±5% (anchor)** | **0.6824** | 0.6695 |
| 20% / ±10% | 0.6591 | 0.6337 |
| 30% / 0% | 0.6567 | 0.6483 |
| 30% / ±5% | 0.6329 | 0.6153 |
| 30% / ±10% | 0.6153 | 0.5908 |
| **Mean across nine scenarios** | **0.6911** | 0.6759 |

The numerical aggregation uses 13 target attributes excluding the placebo. Median pooling improved only **28 of 117** numeric scenario–attribute comparisons. This does not refute the conditional median's theoretical optimality for absolute-error loss: a **finite empirical median of stochastic draws** is not guaranteed to have lower realised MAE than their mean.

**Reporting approach:** Keep **mean-pooled MICE as the primary observed implementation** and present median pooling as a sensitivity analysis. Both outputs are joint-table/transductive; **do not insert them into the unseen-record leaderboard**.

### Additional analysis - Gas identity observability

The truth satisfies `total_gas = gas_a + gas_b + gas_c`, but measured components carry noise while total gas is exact when observed. Define the checkable residual:

```text
r = total_gas - (measured_gas_a + measured_gas_b + measured_gas_c)
```

This requires **all four fields to be observed**. In the supplied scenarios the observed checkable fraction is approximately **64.92%**, **40.30%** and **23.46%** at 10%, 20% and 30% masking. At 20% masking, mean absolute residual was approximately **0** at 0% noise, **389.95** at ±5%, and **779.89** at ±10%.

This is a useful **dataset-level measurement-consistency diagnostic**. The measured record-level association was insufficient between the residual and subsequent imputation error; **do not market it as a validated per-record imputation-confidence flag**.

---

## 6. Interpretation and research findings

1. **RQ1:** On the tested main synthetic population, LightGBM chained has the strongest **aggregate unseen-record numerical skill** at the anchor (0.769 versus PMF 0.709); its advantage is attribute-specific. Across three spatial partitions, paired LightGBM-minus-PMF mean skill difference is approximately **+0.0600**, with a three-split 95% t-interval of approximately **[+0.0442, +0.0757]**. This describes stability *within this synthetic population*, not general superiority on real asset data.
2. **RQ1 qualification:** PMF versus Ridge is not resolved across repeats: approximate mean difference **−0.0052** for Ridge-minus-PMF, interval **[−0.0242, +0.0138]**, with a sign change across partitions.
3. **RQ2:** Increasing supplied masking and measurement-noise severity reduces aggregate reconstruction skill. Co-missing related predictors can be substantially more damaging than losing the same per-row volume of unrelated predictors. Present-but-wrong predictors can propagate error even where target cells are unchanged.
4. **RQ3:** Exact identities explain strong altitude/overgrowth/total-gas recoverability; gas-channel patterns reveal redundant information about `model_id`; and conditional life relationships create more complicated learned error propagation than a single feature-importance chart can explain.
5. **Implementation:** PMF provides demonstrated saved-model replay on unseen records; its distinctive contribution is inductive reuse. Original and predictive MissForest can give bit-identical results at matched iterations by construction, so the two should not be described as algorithmically independent forest families.
6. **Caveats:** Results use one recorded-seed synthetic population and three splits for selected comparisons. `install_date` is strongly shortcut-driven. Several targets are algebraically constrained. TabPFN has a separate, differently resourced exploratory experiment. None of the results establish SAPN operational outcomes or thresholds.

Distinguish three claims in reports: **degradation** (a decline in performance), **baseline parity/advantage not established** (uncertainty overlaps baseline), and **breakdown** (evidence of worse-than-baseline performance under an explicitly defined comparison). Never infer a breakdown point outside tested conditions.

---

---

# Part II - Running the experiments and reproducing the results

## 7. Prerequisites and repository layout

Use **Python 3.12** on macOS, Linux or Windows. The project declares its dependencies in `pipeline/pyproject.toml`; install them into a clean virtual environment. Allow sufficient free disk space for the saved cell-level predictions and model training. The 8,000/2,000 full evaluation with many tree ensembles, multiple imputations and numerous experimental conditions can be computationally demanding.

The active transformer workflow is:

```text
pipeline/
├── pyproject.toml
├── README.md
├── TRANSFORMER_EXPERIMENT_GUIDE.md
├── src/
│   ├── imputation/
│   │   ├── median_mode/
│   │   ├── missforest/
│   │   └── predictive_missforest/
│   └── ...
├── experiment/
│   └── transformer/
│       ├── dataset/
│       │   ├── completion0.9-noise0.0-measured.csv
│       │   ├── completion0.9-noise0.0-truth.csv
│       │   └── ... eight additional measured/truth pairs
│       ├── notebook/
│       │   └── transformer_imputation_experiments.ipynb
│       ├── mice_pmm_logistic.py
│       ├── scripts/comparison_models.py
│       ├── tabpfn_separate/transformer_tabpfn_experiment.ipynb
│       └── outputs/
└── tests/
```

Run the notebook from inside the extracted `pipeline/` directory. The notebook locates the project root automatically; do not copy the notebook out of the project and run it alone. The main input population is always `experiment/transformer/dataset/`. The separate `data/transformers/` folder contains a different synthetic realisation and is **not** substituted for the main experiment data.

### Install on macOS / Linux

```bash
cd /path/to/pipeline
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[notebook,test]"
python -m ipykernel install --user --name capstone-transformer --display-name "Python (Capstone Transformer)"
python -m jupyter lab experiment/transformer/notebook/transformer_imputation_experiments.ipynb
```

On Windows PowerShell:

```powershell
cd path\to\pipeline
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[notebook,test]"
python -m ipykernel install --user --name capstone-transformer --display-name "Python (Capstone Transformer)"
python -m jupyter lab experiment/transformer/notebook/transformer_imputation_experiments.ipynb
```

Select the **Python (Capstone Transformer)** kernel in Jupyter. Ensure installation uses the same Python interpreter as the notebook kernel. There is no YAML configuration step for the transformer experiment.

### Optional separate TabPFN setup

TabPFN is not required for the main notebook. It has different hardware, dependency, model-weight and licensing considerations; see its own notebook and check its current package requirements before installing it. Where compatible with the local environment:

```bash
python -m pip install -e ".[tabpfn]"
```

## 8. Running the notebook from a fresh installation

### A. Quick end-to-end execution

In the **Settings** cell:

```python
MODE = 'quick'
EXECUTE_EXPERIMENTS = True
RUN_RELATIONSHIP_EXPERIMENTS = True
RUN_MICE_POOLING_COMPARISON = False
RUN_T2 = RUN_T3 = RUN_T4 = RUN_T5 = RUN_T6 = True
```

Select **Restart Kernel and Run All Cells**. Quick mode uses about 1,000 fitting and 1,000 held-out records, 10-tree forests, one Predictive MissForest pass and reduced chained/MICE iteration budgets. It checks data loading, model execution, evaluation and exports; it is **not the source of the reported research scores**.

### B. Reproduce the full T1–T6 study and relationship experiments

Set:

```python
MODE = 'full'
EXECUTE_EXPERIMENTS = True
RUN_RELATIONSHIP_EXPERIMENTS = True
RUN_JOINT_MODEL_ID_ABLATION = True
RUN_T4_VOLUME_CONTROLS = True
RUN_LIFE_HAZARD_TEST = True
RUN_MICE_POOLING_COMPARISON = False
RUN_T2 = RUN_T3 = RUN_T4 = RUN_T5 = RUN_T6 = True
```

Restart and run all cells. Full mode uses a spatial 8,000/2,000 train/test split, 100-tree forest models, five Predictive MissForest passes and the full experiment configurations. The original nine T2 input datasets are **loaded as supplied**; they are not regenerated or re-noised by T2. Controlled T4/T5/T6 interventions are constructed explicitly in their own sections.

During T2, each MICE imputation run produces **both** mean- and median-pooled numerical point estimates from the same stochastic draws. The main reported MICE point summary is the mean; the median is a sensitivity comparison. There is no need to execute the separate MICE pooling option during a fresh complete run.

The full run is computationally expensive. To run only a particular section, switch the other `RUN_T2`–`RUN_T6` flags off. Ensure the anchor reference models and datasets needed by the selected experiments remain available. A small run should be done first whenever the environment has changed.

### C. Analyse saved results without refitting

Set:

```python
MODE = 'full'
EXECUTE_EXPERIMENTS = False
RUN_RELATIONSHIP_EXPERIMENTS = False
RESULTS_SOURCE_DIR = 'experiment/transformer/outputs/single_notebook_20261113_215529_full'
```

Replace `RESULTS_SOURCE_DIR` with the saved study directory you wish to inspect. A path relative to `pipeline/` is supported; an absolute path also works. Alternatively, set `RESULTS_SOURCE_DIR = None` to discover a completed full run. Restart and run all cells.

The notebook loads the metric tables and scored-cell exports, regenerates the analysis and figures, and writes derived files to a separate `transformer_analysis_<timestamp>_<mode>/` directory. The source results remain unchanged. Earlier saved runs with differently structured per-cell CSVs are supported by the notebook's CSV reader; fresh runs write a consistent export schema.

### D. Run relationship experiments against an existing full run

When the primary T2–T6 results already exist and you need the joint ablation, matched-volume controls and conditional-life experiment only:

```python
MODE = 'full'
EXECUTE_EXPERIMENTS = False
RESULTS_SOURCE_DIR = 'experiment/transformer/outputs/single_notebook_20261113_215529_full'
RUN_RELATIONSHIP_EXPERIMENTS = True
RUN_JOINT_MODEL_ID_ABLATION = True
RUN_T4_VOLUME_CONTROLS = True
RUN_LIFE_HAZARD_TEST = True
RUN_MICE_POOLING_COMPARISON = False
```

Some fitted anchor models are constructed in memory during this operation if they were not persisted in the saved results. These are local anchor fits; the entire T2–T6 grid is **not** rerun. Results are written to the `transformer_analysis_...` directory.

### E. Run the MICE pooling comparison only

To evaluate MICE across the T2 nine-scenario grid without executing other methods:

```python
MODE = 'full'
EXECUTE_EXPERIMENTS = False
RESULTS_SOURCE_DIR = 'experiment/transformer/outputs/single_notebook_20261113_215529_full'
RUN_RELATIONSHIP_EXPERIMENTS = True
RUN_JOINT_MODEL_ID_ABLATION = False
RUN_T4_VOLUME_CONTROLS = False
RUN_LIFE_HAZARD_TEST = False
RUN_MICE_POOLING_COMPARISON = True
```

The MICE-PMM algorithm generates multiple completed tables for each joint-table scenario and computes the mean and median numerical point summaries from the same draws. Scores are written separately as `mice_pmm_logistic_mean` and `mice_pmm_logistic_median`. This experiment is transductive and should not be placed in the unseen-record leaderboard.

### F. Run TabPFN separately

Open:

```text
experiment/transformer/tabpfn_separate/transformer_tabpfn_experiment.ipynb
```

Follow the notebook's installation and model-weight instructions and choose its own compute settings. Its smaller context, stochastic extension, separate execution and possible hardware differences mean its scores are presented as an **exploratory comparison**, not an identically resourced main-protocol arm.

## 9. Results and file descriptions

A fresh notebook run creates:

```text
experiment/transformer/outputs/
└── transformer_study_YYYYMMDD_HHMMSS_full/
    ├── all_per_target_metrics.csv
    ├── analysis_per_target_metrics.csv
    ├── all_masked_cells.csv
    ├── metrics_checkpoint.csv
    ├── run_manifest.json
    ├── runtime_log.csv
    ├── spatial_split.csv
    ├── relationship_experiment_metrics.csv
    ├── t5_effective_prevalence.csv
    ├── paired_method_split_comparisons.csv
    ├── paired_spatial_block_method_differences.csv
    ├── attribute_wise_reference.csv
    ├── relationship_group_skill.csv
    ├── gas_identity_availability.csv
    ├── conditional_life_overgrowth_observed.csv
    ├── conditional_life_overgrowth_hidden.csv
    └── figures/
```

Files depend on selected experiment flags. `metrics_checkpoint.csv` and `all_masked_cells.csv` are written incrementally as work is completed. `all_per_target_metrics.csv` stores the primary T2–T6 metric table; `relationship_experiment_metrics.csv` stores the joint-ablation and volume-control results. `analysis_per_target_metrics.csv` contains the full analysis view with standardised labels and estimated T5 effective prevalence. The original measured/truth CSVs are not modified.

For numerical targets, inspect `skill`, `mae`, `rmse`, `bias`, `baseline_mae` and `n_masked`; for `model_id` inspect `macro_f1`, `accuracy`, `balanced_accuracy` and `n_masked`. Cell-level prediction exports support independent metric recomputation and paired error-change analysis. The analysis outputs include selected spatial-block bootstrap intervals and repeated-split comparisons; they are conditional on the synthetic population and model settings, not field-data performance guarantees.

### Recorded study results

The supplied project includes the recorded T2–T6 full run in `outputs/single_notebook_20261113_215529_full/` and, when the companion results are included, the three additional relationship experiments and MICE pooling study. These are **completed experimental observations**, not required inputs for fresh model training. You can reproduce the experimental procedure using the new fresh-run controls described above.

## 10. Verification and reproducibility checks

1. Confirm all nine measured/truth dataset pairs exist and have matching row alignment. `completion0.8-noise0.05` is the shared reference condition.
2. Verify the full spatial split reports **8,000 training and 2,000 held-out records**; the quick split uses smaller sets.
3. Check the runtime prints `MODE`, source data path and output directory; keep the JSON manifest with the result tables.
4. Confirm evaluations score only cells missing in the measured input and use complete truth strictly for scoring and explicitly declared synthetic interventions.
5. Keep joint-table and unseen-record comparisons separate, including their different training exposure.
6. Inspect T5's *actual changed-value prevalence* rather than treating selected swaps as changed values.
7. Use the same scored target cells for paired comparisons; group spatial bootstrap resampling by coordinate tile, not individual prediction cells.
8. Distinguish repeated spatial splits of the same 10,000 records from independently generated synthetic populations.
9. Preserve the original dataset files and software version information in the manifest.

Run the relevant code checks from the `pipeline/` root:

```bash
python -m pytest experiment/transformer/scripts/test_model_comparison.py experiment/transformer/scripts/test_robustness_suite.py -q
```

If the separate shared-pipeline test suite is needed, use `python -m pytest tests -q`; results can depend on optional example notebook outputs not generated by the main transformer workflow. Avoid describing the entire suite as passing unless it has been executed in the current environment.

## 11. Troubleshooting

| Symptom | Cause to check | Action |
|---|---|---|
| ImportError for LightGBM | Package not installed in notebook kernel | Activate `.venv`; reinstall `-e ".[notebook,test]"`; select the environment's kernel |
| No saved results found with execution disabled | `RESULTS_SOURCE_DIR` does not point to a completed run | Provide the path or switch `EXECUTE_EXPERIMENTS=True` |
| Python output shows wrong dataset | Notebook opened outside project or wrong `pipeline/` root | Launch Jupyter from `pipeline/` and inspect printed `DATA` |
| Numerical count assignment TypeError | Count columns inferred as strings from a CSV | Use the provided notebook, which casts swap-count columns to nullable integer types |
| Cell CSV parsing exception from older runs | Optional metadata fields were appended with varying widths | Use the notebook's compatible scored-cell reader; fresh runs have fixed-column exports |
| Very long runtime | Full forests, transductive MICE and repeated scenarios | Run in quick mode first; select individual experiment flags for full runs |
| New analysis seems to duplicate a result | Both joint-table and unseen-record scores displayed together | Filter by `protocol` and by `experiment` before ranking |
| MICE scores differ between summaries | Mean and median are distinct point estimators | Compare within the same joint-table scenario and keep both aggregation labels |
| TabPFN installation or weights unavailable | Optional separate model has additional hardware/dependency requirements | Run the main notebook without TabPFN; follow its separate documented setup |

## 12. Scope and handover

The study provides reproducible evidence about missing-value reconstruction and model deterioration **within the supplied synthetic generator**. It does not estimate real-world SAPN error frequencies, deployment readiness or transferable model-failure thresholds. The generator has strong deterministic and shortcut-driven relationships, so all aggregate results should be accompanied by per-attribute detail and explicit study boundaries.

When handing over the study, provide the exact source commit, `pyproject.toml`, the measured/truth input pairs, the main notebook, its exported run manifest and result tables, and the separate TabPFN workflow when included. These artifacts make it possible to inspect existing results and run the experimental procedure again without access to proprietary data.
