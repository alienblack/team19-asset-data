"""Report-safe plots for the transformer model comparison CSV (no model fits)."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ORDER = ("median_mode", "chained_ridge", "predictive_missforest", "lightgbm_chained", "tabpfn_unsupervised")
DISPLAY = {
    "median_mode": "Median/Mode", "chained_ridge": "Chained Ridge",
    "predictive_missforest": "Predictive MF (T2)",
    "lightgbm_chained": "LightGBM chained", "tabpfn_unsupervised": "TabPFN extension",
}


def make_figures(output_dir: Path) -> list[Path]:
    df = pd.read_csv(output_dir / "per_target_metrics.csv")
    if df.empty:
        raise ValueError("No comparison metrics")
    methods = [m for m in ORDER if m in set(df.method)]
    numeric = df.loc[(df.variable_type == "numeric") & (df.variable != "random_placebo")]
    matrix = numeric.pivot_table(index="variable", columns="method", values="numeric_skill")
    matrix = matrix[methods].sort_index()
    values = matrix.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(max(9, len(methods)*2.05), 8.5))
    plot = ax.imshow(values, aspect="auto", vmin=-0.1, vmax=1.0, cmap="viridis")
    ax.set_yticks(np.arange(len(matrix.index)), labels=matrix.index)
    ax.set_xticks(np.arange(len(methods)), labels=[DISPLAY.get(m,m) for m in methods])
    ax.tick_params(axis="x", rotation=25)
    for r in range(values.shape[0]):
        for c in range(values.shape[1]):
            if np.isfinite(values[r,c]):
                ax.text(c, r, f"{values[r,c]:.2f}", ha="center", va="center",
                        color="white" if values[r,c]<0.42 else "black", fontsize=9)
    ax.set_title("Transformer imputation: attribute-level numerical skill\n20% missingness, ±5% noise, spatially held-out records")
    fig.colorbar(plot, ax=ax, shrink=.82, label="Skill relative to training-only Median/Mode")
    fig.tight_layout()
    path1 = output_dir / "model_comparison_attribute_skill.png"
    fig.savefig(path1, dpi=160)
    plt.close(fig)

    summary = pd.read_csv(output_dir / "method_summary.csv")
    summary = summary.set_index("method").reindex(methods)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    x = np.arange(len(methods))
    axes[0].bar(x, summary.mean_numeric_skill_ex_placebo.to_numpy(dtype=float))
    axes[0].axhline(0, color="grey", linewidth=.7)
    axes[0].set_ylabel("Mean numerical skill (excluding placebo)")
    axes[0].set_ylim(-.10, 1)
    axes[1].bar(x, summary.model_id_macro_f1.to_numpy(dtype=float))
    axes[1].set_ylabel("Model-ID Macro-F1")
    axes[1].set_ylim(0, 1.05)
    for ax in axes:
        ax.set_xticks(x, labels=[DISPLAY.get(m,m) for m in methods], rotation=28, ha="right")
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("Unseen-record comparison — descriptive single-split results")
    fig.tight_layout()
    path2 = output_dir / "model_comparison_overview.png"
    fig.savefig(path2, dpi=160)
    plt.close(fig)
    return [path1, path2]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    print("Generated:", *(str(path) for path in make_figures(args.output_dir)), sep="\n  ")
