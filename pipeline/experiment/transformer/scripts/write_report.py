"""Turn a completed transformer grid into a compact reproducible results note."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def markdown_table(frame: pd.DataFrame) -> str:
    labels = list(frame.columns)
    out = ["| " + " | ".join(labels) + " |", "| " + " | ".join("---" for _ in labels) + " |"]
    for row in frame.itertuples(index=False, name=None):
        out.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(out)


def write_report(result_dir: Path) -> Path:
    s = pd.read_csv(result_dir / "scenario_summary.csv")
    p = pd.read_csv(result_dir / "per_target_metrics.csv")
    a = pd.read_csv(result_dir / "scenario_audit.csv")
    m = json.loads((result_dir / "experiment_manifest.json").read_text())
    if s.split_seed.nunique() != 1:
        raise ValueError("This note is for a single split; multi-seed CI needs a separate aggregation")

    def skill_table(protocol: str) -> pd.DataFrame:
        subset = s.loc[s.protocol == protocol].copy()
        subset = subset.sort_values(["completion", "noise"], ascending=[False, True])
        subset["Masking"] = (100 * (1 - subset.completion)).round().astype(int).astype(str) + "%"
        subset["Noise"] = "±" + (100 * subset.noise).round().astype(int).astype(str) + "%"
        values = subset.pivot_table(index=["Masking", "Noise"], columns="method",
                                    values="mean_numeric_skill_ex_placebo", sort=False).reset_index()
        order = ["Masking", "Noise", "median_mode", "missforest", "predictive_missforest"]
        values = values[[c for c in order if c in values]]
        return values.rename(columns={
            "median_mode": "Median/Mode", "missforest": "MissForest", "predictive_missforest": "Predictive MF"
        }).round(3)

    def f1_table(protocol: str) -> pd.DataFrame:
        subset = s.loc[(s.protocol == protocol) & np.isclose(s.noise, 0.05)].copy()
        subset["Masking"] = (100 * (1 - subset.completion)).round().astype(int).astype(str) + "%"
        values = subset.pivot_table(index="Masking", columns="method", values="model_id_macro_f1").reset_index()
        return values.rename(columns={"median_mode": "Median/Mode", "missforest": "MissForest", "predictive_missforest": "Predictive MF"}).round(3)

    anchor = p.loc[
        (p.protocol == "joint_table_transductive") & np.isclose(p.completion, 0.7)
        & np.isclose(p.noise, 0.05) & (p.variable_type == "numeric")
    ][["method", "variable", "mae"]]
    baseline = anchor.loc[anchor.method == "median_mode", ["variable", "mae"]].rename(columns={"mae": "baseline_mae"})
    targets = anchor.merge(baseline, on="variable", validate="many_to_one")
    targets = targets.loc[targets.method != "median_mode"].copy()
    targets["skill"] = 1 - targets.mae / targets.baseline_mae
    target_table = targets.pivot(index="variable", columns="method", values="skill").reset_index().round(3)
    target_table = target_table.rename(columns={"variable": "Target", "missforest": "MissForest", "predictive_missforest": "Predictive MF"})

    audit_table = a[["mask_rate_target", "noise_amplitude", "mask_rate_actual", "changed_observed_numeric"]].copy()
    audit_table["mask_rate_actual"] = (100 * audit_table.mask_rate_actual).round(3).astype(str) + "%"
    audit_table["mask_rate_target"] = (100 * audit_table.mask_rate_target).round().astype(int).astype(str) + "%"
    audit_table["noise_amplitude"] = (100 * audit_table.noise_amplitude).round().astype(int).astype(str) + "%"
    audit_table.columns = ["Nominal masking", "Max noise", "Actual missingness", "Changed observed numerics"]

    total_seconds = s.runtime_seconds.sum()
    if "effective_methods" in m:
        forest = m["effective_methods"]["missforest"]
        predictive = m["effective_methods"]["predictive_missforest"]
        settings_text = (
            f"- Hyperparameters: Original MissForest {forest['n_estimators']} trees / up to {forest['max_iter']} iterations; "
            f"Predictive MissForest {predictive['n_estimators']} trees / {predictive['n_passes']} passes; "
            f"model seeds {forest['random_state']}/{predictive['random_state']}. "
            "See experiment_manifest.json for effective selected overrides."
        )
    else:
        settings_text = f"- Historical hyperparameters: {m['n_estimators']} trees, {m['passes']} passes / maximum iterations, seed {m['model_seed']}."

    left = p.loc[(p.method == "missforest") & (p.protocol == "joint_table_transductive")].sort_values(
        ["completion", "noise", "split_seed", "variable"]
    )
    right = p.loc[(p.method == "predictive_missforest") & (p.protocol == "joint_table_transductive")].sort_values(
        ["completion", "noise", "split_seed", "variable"]
    )
    equal_scores = len(left) == len(right) and all(
        np.allclose(left[column], right[column], equal_nan=True)
        for column in ["mae", "rmse", "accuracy", "macro_f1"]
    )
    comparison_note = (
        "The two forests achieved identical joint-table error metrics in this run. "
        "This does not imply that their persistence/deployment capabilities are the same."
        if equal_scores else
        "The two forests achieved different joint-table metrics. Compare per-target scores and "
        "their iteration histories before attributing differences to model design."
    )
    lines = [
        "# Transformer masking × measurement-noise: single-split benchmark",
        "",
        "**Status:** Complete for the listed source scenarios and split. Results are descriptive; no generator-seed confidence interval is available.",
        "",
        "## Experimental contract",
        "",
        f"- Data: nine supplied paired synthetic scenarios; one generator realization; {len(m['numeric_columns'])} numeric and {len(m['categorical_columns'])} categorical attributes.",
        f"- Methods: {', '.join(m['methods'])}.",
        f"- Split seed: {m['split_seeds'][0]}, spatially blocked 10×10 tiles; 8,000 train-labelled and 2,000 test-labelled rows.",
        settings_text,
        "- Joint-table protocol: fit all 10,000 partially observed rows and score only the hidden cells in 2,000 test-labelled rows. All three methods are included. This is transductive, not an unseen-record holdout.",
        "- Unseen-record protocol: Median/Mode and Predictive MF fit the train rows, then transform unseen test rows.",
        "- The ground-truth file was never passed to any imputer.",
        f"- Total imputation runtime across protocol runs: {total_seconds/60:.1f} minutes (hardware-dependent).",
        "",
        "## Dataset audit",
        "",
        markdown_table(audit_table),
        "",
        "Masks are aligned across noise severity and nested across missingness levels; the noise-10 errors are twice noise-5 for matching observed cells.",
        "",
        "## Joint-table numeric skill (comparable access protocol)",
        "",
        markdown_table(skill_table("joint_table_transductive")),
        "",
        comparison_note,
        "",
        "## Unseen-record numeric skill (deployment-like)",
        "",
        markdown_table(skill_table("unseen_records")),
        "",
        "Numerical skill is 1 − method MAE / the same-protocol Median/Mode MAE, averaged equally across numerical targets other than the random placebo. It is not percentage accuracy.",
        "",
        "## Model-ID macro-F1 at ±5% noise",
        "",
        "Joint-table protocol:",
        "",
        markdown_table(f1_table("joint_table_transductive")),
        "",
        "Unseen-record protocol:",
        "",
        markdown_table(f1_table("unseen_records")),
        "",
        "## Per-target numerical skill: 30% masked, ±5% noise, joint-table",
        "",
        markdown_table(target_table),
        "",
        "## Interpretation safeguards",
        "",
        "1. Original MissForest cannot be evaluated on genuinely unseen records without refitting; its transductive scores must not be blended with unseen-record scores.",
        "2. The all-15-feature configuration contains exact derived relationships and a coordinate/date shortcut; separate feature ablations are necessary to explain high performance.",
        "3. The supplied nine scenarios contain one generator and missing-mask realization. Repeated spatial splits alone cannot establish uncertainty across independently generated synthetic populations.",
        "4. The noise factor applies to seven observed numeric attributes, not all attributes. Scoring only missing cells measures robustness to noisy *predictors*, not correction of corrupted observed values; all original observed values remain unchanged by the imputer.",
        "5. Weak placebo improvement is expected; apparently positive placebo performance would require investigation.",
        "6. These results do not establish operational SAPN thresholds or actual-register performance.",
        "",
        "## Outputs",
        "",
        "- `scenario_summary.csv` — method and protocol aggregation",
        "- `per_target_metrics.csv` — target-level errors and scores",
        "- `spatial_split_manifest.csv` — held-out records and tile assignments",
        "- `scenario_audit.csv` — observed corruption and missingness",
        "- `experiment_manifest.json` — method/settings contract",
        "- `convergence_history.csv` — iteration-wise changes and early-stop acceptance, available for newer YAML-driven runs",
        "- `noise_masking_curves.png`, `per_target_anchor.png`, `model_id_macro_f1.png` — main figures",
        "- `per_attribute_skill_heatmap.png` — target-level numerical skill across scenarios, excluding placebo",
        "- `forest_method_differences.png` — paired original/predictive forest comparisons on joint-table rows",
        "- `placebo_skill_heatmap.png` — negative-control behaviour by method and scenario",
        "- `protocol_gap.png` — joint-table versus unseen-record predictive-forest skill",
        "- `runtime_comparison.png` — scenario cost, comparable only within a run",
        "- `convergence_diagnostics.png` — saved iteration/pass changes where convergence history is available",
    ]
    output = result_dir / "results_report.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    print(write_report(args.result_dir))
