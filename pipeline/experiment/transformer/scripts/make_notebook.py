"""Generate the transformer experiment notebook from the reusable runner."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
nb = {
    "nbformat": 4,
    "nbformat_minor": 4,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "cells": [],
}


def md(source: str) -> None:
    nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)})


def code(source: str) -> None:
    nb["cells"].append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(keepends=True)})


md("""# Transformer imputation: masking × measurement noise

**Methods:** Median/Mode, Original MissForest, Predictive MissForest.  
**Source:** Nine *unchanged* synthetic transformer measured/truth CSV pairs.  
**Experiment:** 10%, 20%, 30% nominal MCAR masking crossed with 0%, ±5%, ±10%
multiplicative measurement noise.

The notebook uses the team's production model classes through
`transformer_noise_masking.py`. Ground truth is **never** a model input.
Two protocols are reported separately: joint-table transductive and
unseen-record generalisation. The random placebo is a negative control.

**Notebook workflow:** review inputs → configure → execute (or reuse saved
results) → inspect overall and target-level performance. Running the notebook
with its default settings does **not** retrain the models.
""")

md("## 1. Setup and locate the existing pipeline files")
code("""from pathlib import Path
import sys
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

# Locate this experiment from a Jupyter session opened within the pipeline.
def find_experiment_root():
    current = Path.cwd().resolve()
    for candidate in (current, *current.parents):
        for possible in (candidate, candidate / "experiment" / "transformer"):
            if (possible / "dataset").is_dir() and (possible / "scripts" / "transformer_noise_masking.py").is_file():
                return possible
    raise FileNotFoundError("Cannot locate experiment/transformer; open notebook from the pipeline workspace")

EXPERIMENT_DIR = find_experiment_root()
DATA_DIR = EXPERIMENT_DIR / "dataset"
PIPELINE_DIR = EXPERIMENT_DIR.parent.parent
OUTPUT_DIR = EXPERIMENT_DIR / "outputs" / "full_grid_seed617"
CONFIG_PATH = EXPERIMENT_DIR / "config.yaml"
RUNNER = EXPERIMENT_DIR / "scripts" / "transformer_noise_masking.py"

assert DATA_DIR.exists(), f"Dataset not found: {DATA_DIR}"
assert CONFIG_PATH.is_file(), f"Experiment YAML not found: {CONFIG_PATH}"
assert (PIPELINE_DIR / "src" / "imputation" / "registry.py").exists()
print("Dataset:", DATA_DIR)
print("Pipeline:", PIPELINE_DIR)
print("Experiment config:", CONFIG_PATH)
""")

md("## 2. Inspect the unchanged dataset and source audit")
code("""sys.path.insert(0, str(EXPERIMENT_DIR / "scripts"))
from transformer_noise_masking import audit_source_files, NUMERIC, CATEGORICAL, COLUMNS

audit = audit_source_files(DATA_DIR)
display(audit)

sample = pd.read_csv(DATA_DIR / "completion0.8-noise0.05-measured.csv")
display(sample.head(5))
print("Data shape:", sample.shape)
print("Missing fraction by attribute:")
display(sample.isna().mean().rename("missing_fraction").to_frame())
""")

md("""## 3. Configure the experiment

Choose one mode:

- `"load_saved"`: display the previous complete benchmark immediately (**default**).
- `"pilot"`: run a quick 1,000-train / 1,000-test experiment on one condition.
- `"full"`: run the selected masking × noise grid on all 10,000 rows.

All modes use the existing **Median/Mode, Original MissForest and Predictive
MissForest** implementations and the *effective* settings from `config.yaml`,
including `selected_hyperparameters`. Pilot mode samples fewer rows but does
**not** reduce model settings. Historical 30-tree/3-pass outputs must not be
presented as results from the newly selected configuration.
""")
code("""EXPERIMENT_MODE = "load_saved"  # "load_saved", "pilot", or "full"

# Used for full runs; edit these lists to run only a subset of the nine pairs.
MASKING_RATES = [10, 20, 30]  # percent missing: 10, 20, 30
NOISE_AMPLITUDES = [0, 5, 10]  # maximum multiplicative noise: 0, ±5, ±10 percent

# A pilot deliberately uses one scenario for a short end-to-end check.
PILOT_CASE = (20, 5)  # (masking percent, noise percent)
SPLIT_SEEDS = [617]

assert EXPERIMENT_MODE in {"load_saved", "pilot", "full"}
assert MASKING_RATES and set(MASKING_RATES) <= {10, 20, 30}
assert NOISE_AMPLITUDES and set(NOISE_AMPLITUDES) <= {0, 5, 10}
assert PILOT_CASE[0] in {10, 20, 30} and PILOT_CASE[1] in {0, 5, 10}
assert SPLIT_SEEDS and all(isinstance(seed, int) for seed in SPLIT_SEEDS)
print("Selected mode:", EXPERIMENT_MODE)
from transformer_noise_masking import load_experiment_config
ACTIVE_CONFIG = load_experiment_config(CONFIG_PATH)
print("MissForest:", ACTIVE_CONFIG["effective_methods"]["missforest"])
print("Predictive MissForest:", ACTIVE_CONFIG["effective_methods"]["predictive_missforest"])
""")

md("""## 4. Run the experiment (or load saved results)

Run **this cell** to execute your selection. It calls the existing model runner
from inside Jupyter, saves all results under `outputs/`, and then points the
analysis cells at the new output. Each new training run gets a timestamped
folder, so the original `full_grid_seed617` results are not overwritten.
""")
code("""from datetime import datetime
import json

if EXPERIMENT_MODE == "load_saved":
    OUTPUT_DIR = EXPERIMENT_DIR / "outputs" / "full_grid_seed617"
    print("Using existing results:", OUTPUT_DIR)
    saved = json.loads((OUTPUT_DIR / "experiment_manifest.json").read_text())
    if saved.get("effective_methods") != ACTIVE_CONFIG["effective_methods"]:
        print("WARNING: These are historical results from the old configuration. Run pilot/full for the new 100-tree configuration.")
else:
    chosen_pairs = (
        [PILOT_CASE] if EXPERIMENT_MODE == "pilot"
        else [(mask, noise) for mask in MASKING_RATES for noise in NOISE_AMPLITUDES]
    )
    cases = ",".join(f"{1 - mask / 100:.1f}:{noise / 100:.2f}" for mask, noise in chosen_pairs)
    run_label = f"notebook_{EXPERIMENT_MODE}_{datetime.now():%Y%m%d_%H%M%S}"
    RUN_OUTPUT = EXPERIMENT_DIR / "outputs" / run_label
    command = [
        sys.executable, str(RUNNER), "--data-dir", str(DATA_DIR),
        "--pipeline-dir", str(PIPELINE_DIR), "--config", str(CONFIG_PATH),
        "--output-dir", str(RUN_OUTPUT),
        "--cases", cases, "--split-seeds", ",".join(map(str, SPLIT_SEEDS)),
    ]
    if EXPERIMENT_MODE == "pilot":
        command.append("--pilot")
    print(f"Running {len(chosen_pairs)} scenario(s) × {len(SPLIT_SEEDS)} split seed(s)")
    print("Saving new outputs to:", RUN_OUTPUT)
    subprocess.run(command, check=True)
    OUTPUT_DIR = RUN_OUTPUT
""")

md("## 5. Read the selected experiment results")
code("""assert (OUTPUT_DIR / "scenario_summary.csv").exists(), (
    f"Run section 3 or point OUTPUT_DIR at existing results: {OUTPUT_DIR}"
)
summary = pd.read_csv(OUTPUT_DIR / "scenario_summary.csv")
detail = pd.read_csv(OUTPUT_DIR / "per_target_metrics.csv")
summary["masked_percent"] = (100 * (1 - summary["completion"])).round().astype(int)
summary["noise_percent"] = (100 * summary["noise"]).round().astype(int)
display(summary.sort_values(["protocol", "masked_percent", "noise_percent", "method"]))
""")

md("""## 6. Robustness curves

Higher numeric skill is better; 0 means equal MAE to the protocol-matched
Median/Mode baseline. **Do not compare curves across protocols as if the
training and application arrangements were identical.**
""")
code("""from IPython.display import Image
from plot_results import make_plots

# Post-processing only: uses saved CSVs, never retrains a model.
PLOT_PATHS = make_plots(OUTPUT_DIR)
print(f"Created {len(PLOT_PATHS)} figures in {OUTPUT_DIR}")
display(Image(filename=str(OUTPUT_DIR / "noise_masking_curves.png")))
""")

md("## 7. Target-level errors and negative control")
code("""anchor_scenarios = detail[["completion", "noise"]].drop_duplicates().sort_values(
    ["completion", "noise"], ascending=[True, True]
)
anchor_completion, anchor_noise = next(
    ((c, n) for c, n in anchor_scenarios.itertuples(index=False, name=None)
     if abs(c - 0.7) < 1e-9 and abs(n - 0.05) < 1e-9),
    tuple(anchor_scenarios.iloc[0]),
)
print(f"Target-level scenario: {1 - anchor_completion:.0%} masking, ±{anchor_noise:.0%} noise")
anchor = detail.loc[
    (detail.completion == anchor_completion) & (detail.noise == anchor_noise)
    & (detail.protocol == "unseen_records")
]
cols = ["variable", "method", "n_masked", "mae", "rmse", "signed_bias", "macro_f1"]
display(anchor[cols].sort_values(["variable", "method"]))
display(summary[["protocol", "method", "masked_percent", "noise_percent", "placebo_skill"]])
""")

md("""## 8. Additional diagnostic figures

These visualisations are generated from the saved results, so they can be
recreated after the experiment without running the models again. Method
comparisons are protocol-matched; random placebo is shown separately.
""")
code("""for plot_path in PLOT_PATHS:
    if plot_path.name != "noise_masking_curves.png":
        print(plot_path.name)
        display(Image(filename=str(plot_path)))
""")

md("""## 9. Interpretation and limitations

- The dataset is synthetic and represents one generator realization.
- The coordinates encode installation-date sequence in this generator.
- Derived `altitude`, `total_gas`, `overgrowth`, and `life` may make some targets easier.
- The six measured environmental/gas fields plus placebo are subject to
  measurement noise; all 15 attributes are subject to missingness.
- Report per-target MAE and model-ID macro-F1 alongside any aggregate.
- Check raw observations and their noise separately from reconstruction errors.
- Before causal claims about attribute relationships, run controlled predictor
  ablations and source-target corruption experiments.
- Later runs should repeat split seeds and, if available, generator seeds.
""")

output = ROOT / "notebook" / "transformer_noise_masking.ipynb"
output.write_text(json.dumps(nb, indent=2) + "\n", encoding="utf-8")
print(output)
