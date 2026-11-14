# Transformer model-family comparison (T2 extension)

This addition is self-contained under `experiment/transformer/`. It does not
modify `pipeline/src/`, the existing nine-scenario T2 experiment, or the T3
feature-ablation experiment. It is designed for the uploaded **`experiment/`
subtree**, which does **not** contain the shared production imputation code.

## What is compared

At the reference condition (20% MCAR missingness and ±5% measurement noise),
all newly fitted methods use exactly the same **8,000 measured fitting rows**
and **2,000 spatially held-out measured test rows** (split seed 617). The
complete synthetic truth is used for scoring only. All numerical and
categorical results are scored on the test rows' **originally missing cells**.

Methods:

- `median_mode`: baseline fitted on the measured training partition.
- `chained_ridge`: 3-pass stored-model ridge regression and categorical logistic
  regression, with standardized numerical predictors and one-hot encoding.
- `lightgbm_chained`: 3-pass stored-model LightGBM regression/classification,
  declaring `model_id` as categorical. **Custom chained imputation, not the
  `miceforest` library.**
- `tabpfn_unsupervised`: **optional, experimental** official Prior Labs
  `TabPFNUnsupervisedModel` extension. No silent fallback is used if TabPFN
  is unavailable. The configured 2,000-row context subset means the result is
  a *reduced-context feasibility test*, **not a strictly matched 8,000-row
  learner comparison**. Increase the context when resources and licensing permit.
- `predictive_missforest`: imported **only as a reference row** from the
  completed original T2 CSV. This ZIP does not include `pipeline/src`, so this
  comparison runner does not retrain Original or Predictive MissForest.

The comparison is explicitly **unseen-record**; do not mix scores with the
T2 `joint_table_transductive` results.

## Dependencies

```bash
pip install numpy pandas scikit-learn pyyaml matplotlib lightgbm jupyter
```

TabPFN only (optional): the config selects `v3.5-fast` explicitly.
Consult the current versions of [`tabpfn`](https://github.com/PriorLabs/TabPFN)
and [`tabpfn-extensions`](https://github.com/PriorLabs/tabpfn-extensions).
Model weights may require authentication, a download, compatible hardware,
and licence acceptance. TabPFN 3.5 weights are non-commercial/non-production;
check whether a SAPN-facing use fits those terms. Assess academic / SAPN use against the **weight
licence**, which can differ from the extension's source-code licence.
Do not assume the extension supports cheap frozen-model replay: it can do
new in-context fitting during `impute()`.

## Run

From the repository root containing `experiment/`:

```bash
# Small pipeline/functionality test — NOT publication results:
python experiment/transformer/scripts/run_model_comparison.py --quick

# Full 8k/2k matched reference-condition comparison:
python experiment/transformer/scripts/run_model_comparison.py \
  --methods median_mode,chained_ridge,lightgbm_chained

# TabPFN feasibility pilot (can be expensive, requires optional packages):
python experiment/transformer/scripts/run_model_comparison.py \
  --methods median_mode,tabpfn_unsupervised

# Extend to supplied scenarios after reference condition is validated:
python experiment/transformer/scripts/run_model_comparison.py \
  --cases 0.9:0.0,0.8:0.05,0.7:0.1
```

Every invocation creates a **fresh output directory** and never overwrites an
existing result; pass `--output-dir` to choose a new empty directory. Outputs:

- `per_target_metrics.csv`: numeric MAE/RMSE/bias/skill; categorical
  Macro-F1, balanced accuracy and accuracy.
- `method_summary.csv`: secondary, macro-averaged scores.
- `cells_<method>_<case>.csv`: individual masked-cell predictions and errors,
  with spatial tile and record ID; appropriate for paired/block bootstrap.
  Pre-existing T2 Predictive MF is aggregate-only and does not have these
  per-cell exports in the supplied archive.
- `run_manifest.json`: parameters, intervention, split and limitations.

Generate figures without fitting any model:

```bash
python experiment/transformer/scripts/plot_model_comparison.py \
  --output-dir experiment/transformer/outputs/<YOUR_RUN>
```

## Data-quality and fairness checks

- Original observations are preserved exactly by every new method.
- No test records or test truth are used for fitting, initial statistics or
  feature encoding.
- Scoring uses **only originally masked** test cells.
- The full reference 20%/±5% split must reproduce T2 Median/Mode exactly
  (comparison aborts otherwise).
- Test rows, observed values and masks are identical across models.
- Method hyperparameters and iteration budgets are recorded; these are
  preliminary/default configurations, **not independently tuned optima**.
- Per-target results are primary; mean skill is a secondary descriptive summary.
- One generator, one split, no CIs. A foundation-model context subset is an
  additional limitation for TabPFN.

## Research note

A difference in numerical skill is **not** proof of superiority under
unseen degradation conditions. Do not claim an algorithm is more robust until
its advantage is tested on additional masks/noise/corruption mechanisms and
with independent synthetic populations/splits. A high `install_date` score
partly exploits the synthetic generator's coordinate ordering shortcut.
