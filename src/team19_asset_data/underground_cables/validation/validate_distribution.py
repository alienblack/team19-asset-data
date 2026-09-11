"""UC-S2-08: distribution, source-status, and confidence validation."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

try:
    from .utils import infer_target_type
except ImportError:
    from utils import infer_target_type


DISPLAY_NAMES = {
    "reference": "Reference",
    "baseline": "Baseline",
    "alternative": "Alternative",
}


def compare_distributions(
    datasets: dict[str, pd.DataFrame], target_column: str, output_dir: str | Path
) -> tuple[str, pd.DataFrame]:
    """Save numeric summaries/plots or categorical counts and proportions."""
    output_dir = Path(output_dir)
    target_type = infer_target_type(datasets["reference"][target_column])
    selected = {DISPLAY_NAMES[key]: df[target_column] for key, df in datasets.items() if key != "masked"}
    if target_type == "numeric":
        summary = pd.DataFrame(selected).describe().T
        summary.insert(0, "dataset", summary.index)
        summary.reset_index(drop=True, inplace=True)
        summary["median"] = summary["50%"]
        summary = summary[
            ["dataset", "count", "mean", "median", "std", "min", "25%", "50%", "75%", "max"]
        ]
        summary.to_csv(output_dir / "distribution_summary.csv", index=False)
        _save_numeric_plots(selected, target_column, output_dir)
        return target_type, summary

    rows = []
    for dataset_name, values in selected.items():
        counts = values.fillna("<MISSING>").astype(str).value_counts(dropna=False)
        total = counts.sum()
        rows.extend(
            {"dataset": dataset_name, "category": category, "count": int(count), "proportion": count / total}
            for category, count in counts.items()
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "distribution_summary.csv", index=False)
    return target_type, summary


def _save_numeric_plots(selected: dict[str, pd.Series], target_column: str, output_dir: Path) -> None:
    """Create clearly labelled histogram and boxplot files."""
    colors = {"Reference": "#333333", "Baseline": "#377eb8", "Alternative": "#e41a1c"}
    fig, ax = plt.subplots(figsize=(9, 6))
    for name, values in selected.items():
        ax.hist(values.dropna(), bins=30, alpha=0.45, label=name, color=colors[name])
    ax.set(title=f"Distribution of {target_column}", xlabel=target_column, ylabel="Frequency")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "histogram.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6))
    labels, values = [], []
    for name, series in selected.items():
        labels.append(name)
        values.append(series.dropna().to_numpy())
    ax.boxplot(values, tick_labels=labels)
    ax.set(title=f"Boxplot of {target_column}", ylabel=target_column)
    fig.tight_layout()
    fig.savefig(output_dir / "boxplot.png", dpi=160)
    plt.close(fig)
