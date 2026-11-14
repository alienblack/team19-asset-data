"""Create reproducible plots from the transformer experiment CSV outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


COLORS = {
    "median_mode": "#64748b",
    "missforest": "#0f766e",
    "predictive_missforest": "#2563eb",
}
LABELS = {
    "median_mode": "Median/Mode",
    "missforest": "Original MissForest",
    "predictive_missforest": "Predictive MissForest",
}
PROTOCOLS = ("joint_table_transductive", "unseen_records")
NOISE_LEVELS = (0, 5, 10)
METHOD_STYLES = {
    "median_mode": {"marker": "o", "linestyle": ":", "markersize": 6},
    "missforest": {"marker": "s", "linestyle": "--", "markersize": 7},
    # Hollow circles let coincident MissForest points remain visible.
    "predictive_missforest": {"marker": "o", "linestyle": "-", "markersize": 9,
                              "markerfacecolor": "none", "markeredgewidth": 1.8},
}


def _save(fig: plt.Figure, result_dir: Path, filename: str) -> Path:
    path = result_dir / filename
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _scenario_labels(frame: pd.DataFrame) -> list[tuple[float, float]]:
    """Available masking/noise combinations, ordered consistently across plots."""
    return sorted(
        {(round(float(c), 8), round(float(n), 8))
         for c, n in zip(frame["completion"], frame["noise"])},
        key=lambda pair: (1 - pair[0], pair[1]),
    )


def _short_case(completion: float, noise: float) -> str:
    return f"M{100 * (1 - completion):.0f}\nN±{100 * noise:.0f}"


def _baseline_relative_target_skill(detail: pd.DataFrame) -> pd.DataFrame:
    """Score each hidden numeric target against its same-case, same-split baseline.

    Keep placebo as a separate negative-control diagnostic. A target with zero
    baseline MAE has no defined relative skill and remains NaN (never zero).
    """
    numeric = detail.loc[detail.variable_type.eq("numeric")].copy()
    keys = ["protocol", "completion", "noise", "split_seed", "variable"]
    baseline = numeric.loc[numeric.method.eq("median_mode"), keys + ["mae"]]
    baseline = baseline.rename(columns={"mae": "baseline_mae"})
    merged = numeric.loc[~numeric.method.eq("median_mode")].merge(
        baseline, on=keys, how="inner", validate="many_to_one"
    )
    merged["skill"] = np.where(
        merged.baseline_mae.gt(0), 1 - merged.mae / merged.baseline_mae, np.nan
    )
    return merged


def _anchored_case(frame: pd.DataFrame) -> tuple[float, float]:
    cases = _scenario_labels(frame)
    if not cases:
        raise ValueError("Cannot plot an empty experiment.")
    return min(cases, key=lambda case: abs(case[0] - 0.8) + abs(case[1] - 0.05))


def make_plots(result_dir: Path) -> list[Path]:
    summary = pd.read_csv(result_dir / "scenario_summary.csv")
    detail = pd.read_csv(result_dir / "per_target_metrics.csv")
    if summary.split_seed.nunique() != 1:
        summary = summary.groupby(["completion", "noise", "method", "protocol"], as_index=False).agg({
            "mean_numeric_skill_ex_placebo": "mean", "model_id_macro_f1": "mean",
            "placebo_skill": "mean", "runtime_seconds": "mean",
        })
    summary["mask_pct"] = 100 * (1 - summary.completion)
    summary["noise_pct"] = 100 * summary.noise
    paths: list[Path] = []

    # Primary line charts: two separate protocol rows, never a blended ranking.
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)
    protocols = PROTOCOLS
    for row, protocol in enumerate(protocols):
        for col, noise_pct in enumerate(NOISE_LEVELS):
            ax = axes[row, col]
            for method in COLORS:
                sub = summary.loc[
                    (summary.protocol == protocol) &
                    np.isclose(summary.noise_pct, noise_pct) &
                    (summary.method == method)
                ].sort_values("mask_pct")
                if len(sub):
                    ax.plot(sub.mask_pct, sub.mean_numeric_skill_ex_placebo,
                            color=COLORS[method], label=LABELS[method], lw=2,
                            **METHOD_STYLES[method])
            ax.set_title(f"{['Joint-table', 'Unseen-record'][row]} protocol | noise ±{noise_pct}%")
            ax.set_xticks([10, 20, 30])
            ax.set_ylim(0, 1)
            ax.grid(alpha=0.25)
            if col == 0:
                ax.set_ylabel("Mean numeric skill (excl. placebo)")
            if row == 1:
                ax.set_xlabel("Masked cells (%)")
    handles = [Line2D([0], [0], color=COLORS[m], lw=2, **METHOD_STYLES[m]) for m in COLORS]
    fig.legend(handles, [LABELS[m] for m in COLORS], loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.025))
    fig.suptitle("Transformer imputation robustness: three methods, two protocols", fontsize=15)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    paths.append(_save(fig, result_dir, "noise_masking_curves.png"))

    # At anchor 30% masking / ±5% noise, show per-target MAE reductions.
    protocol = "joint_table_transductive"
    anchor = detail.loc[
        (detail.protocol == protocol) & np.isclose(detail.completion, 0.7)
        & np.isclose(detail.noise, 0.05) & (detail.variable_type == "numeric")
        & detail.variable.ne("random_placebo")
    ].copy()
    if len(anchor):
        anchor = anchor.groupby(["method", "variable"], as_index=False).mae.mean()
        baseline = anchor.loc[anchor.method == "median_mode", ["variable", "mae"]].rename(columns={"mae": "baseline_mae"})
        anchor = anchor.merge(baseline, on="variable", validate="many_to_one")
        anchor = anchor.loc[anchor.method != "median_mode"]
        anchor["skill"] = 1 - anchor.mae / anchor.baseline_mae
        targets = sorted(anchor.variable.unique())
        y = np.arange(len(targets))
        fig, ax = plt.subplots(figsize=(10, 7))
        for offset, method in [(-0.19, "missforest"), (0.19, "predictive_missforest")]:
            series = anchor.loc[anchor.method == method].set_index("variable").reindex(targets)
            ax.barh(y + offset, series.skill, height=0.35, color=COLORS[method], label=LABELS[method])
        ax.axvline(0, color="#333333", linewidth=0.8)
        ax.set_yticks(y, targets)
        ax.set_xlabel("Numerical skill relative to Median/Mode")
        ax.set_title("Per-target recoverability: 30% MCAR and ±5% noise (joint-table)")
        ax.grid(axis="x", alpha=0.2)
        ax.legend()
        fig.tight_layout()
        paths.append(_save(fig, result_dir, "per_target_anchor.png"))

    # Categorical results are separate from numerical skill, across all noise levels.
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)
    for row, protocol in enumerate(protocols):
        for col, noise_pct in enumerate(NOISE_LEVELS):
            ax = axes[row, col]
            chunk = summary.loc[(summary.protocol == protocol)
                                & np.isclose(summary.noise_pct, noise_pct)]
            for method in COLORS:
                subset = chunk.loc[chunk.method == method].sort_values("mask_pct")
                if len(subset):
                    ax.plot(subset.mask_pct, subset.model_id_macro_f1,
                            label=LABELS[method], color=COLORS[method], lw=2,
                            **METHOD_STYLES[method])
            ax.set_title(f"{['Joint-table', 'Unseen-record'][row]} | noise ±{noise_pct}%")
            ax.set_xticks([10, 20, 30])
            ax.set_ylim(0, 1.05)
            ax.grid(alpha=0.25)
            if row == 1:
                ax.set_xlabel("Masked cells (%)")
            if col == 0:
                ax.set_ylabel("model_id Macro-F1")
    fig.legend(handles, [LABELS[m] for m in COLORS], loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.025))
    fig.suptitle("Categorical reconstruction across masking and noise")
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    paths.append(_save(fig, result_dir, "model_id_macro_f1.png"))

    paths.extend(make_diagnostic_plots(result_dir, summary, detail))
    return paths


def make_diagnostic_plots(result_dir: Path, summary: pd.DataFrame,
                          detail: pd.DataFrame) -> list[Path]:
    """Build further figures solely from saved results: no model fitting."""
    paths: list[Path] = []
    cases = _scenario_labels(summary)
    labels = [_short_case(c, n) for c, n in cases]
    target_skill = _baseline_relative_target_skill(detail)

    # 1. Per-target robustness matrix: 13 meaningful numerical targets; placebo separate.
    comparisons = [
        ("joint_table_transductive", "missforest"),
        ("joint_table_transductive", "predictive_missforest"),
        ("unseen_records", "predictive_missforest"),
    ]
    targets = [v for v in detail.loc[detail.variable_type.eq("numeric"), "variable"].unique()
               if v != "random_placebo"]
    if targets and not target_skill.empty:
        fig, axes = plt.subplots(len(comparisons), 1, figsize=(max(12, len(cases) * 1.3), 13),
                                 squeeze=False)
        heatmap_norm = TwoSlopeNorm(vmin=-0.5, vcenter=0, vmax=1)
        cmap = plt.colormaps["RdYlGn"].copy()
        cmap.set_bad("#e2e8f0")
        for row, (protocol, method) in enumerate(comparisons):
            ax = axes[row, 0]
            part = target_skill.loc[
                target_skill.protocol.eq(protocol) & target_skill.method.eq(method)
                & target_skill.variable.ne("random_placebo")
            ].groupby(["variable", "completion", "noise"], as_index=False).skill.mean()
            values = {(r.variable, round(float(r.completion), 8), round(float(r.noise), 8)):
                      r.skill for r in part.itertuples(index=False)}
            matrix = np.array([[values.get((target, c, n), np.nan) for c, n in cases]
                               for target in targets], dtype=float)
            heat = ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap,
                             norm=heatmap_norm, aspect="auto")
            ax.set_yticks(range(len(targets)), targets)
            ax.set_xticks(range(len(cases)), labels)
            ax.tick_params(axis="x", labelrotation=0)
            ax.set_title(f"{LABELS[method]} — {protocol.replace('_', ' ')}")
            ax.set_xlabel("Masking (M%) and maximum noise (N%)")
            ax.set_xticks(np.arange(-0.5, len(cases), 1), minor=True)
            ax.set_yticks(np.arange(-0.5, len(targets), 1), minor=True)
            ax.grid(which="minor", color="white", linewidth=0.9)
            ax.tick_params(which="minor", length=0)
        fig.colorbar(heat, ax=axes.ravel().tolist(), fraction=0.018, pad=0.03,
                     label="Numeric skill vs same-protocol Median/Mode (placebo excluded)")
        fig.suptitle("Per-attribute robustness (negative = worse than baseline)", fontsize=15)
        fig.subplots_adjust(left=0.2, right=0.87, hspace=0.55, top=0.93, bottom=0.05)
        paths.append(_save(fig, result_dir, "per_attribute_skill_heatmap.png"))

    # 2. Paired difference where both methods share the joint-table access protocol.
    joint = summary.loc[summary.protocol.eq("joint_table_transductive")]
    left = joint.loc[joint.method.eq("missforest")]
    right = joint.loc[joint.method.eq("predictive_missforest")]
    compare = left.merge(right, on=["completion", "noise", "protocol"],
                         suffixes=("_mf", "_pmf"), validate="one_to_one")
    if not compare.empty:
        fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
        for row, (metric, title) in enumerate([
            ("mean_numeric_skill_ex_placebo", "Numerical skill difference"),
            ("model_id_macro_f1", "model_id Macro-F1 difference"),
        ]):
            diffs = compare[f"{metric}_pmf"] - compare[f"{metric}_mf"]
            limit = max(0.005 if row == 0 else 0.001, float(np.nanmax(np.abs(diffs))) * 1.25)
            for col, noise_pct in enumerate(NOISE_LEVELS):
                ax = axes[row, col]
                data = compare.loc[np.isclose(compare.noise * 100, noise_pct)].copy()
                data = data.sort_values("completion", ascending=False)
                ax.plot(100 * (1 - data.completion),
                        data[f"{metric}_pmf"] - data[f"{metric}_mf"], "o-",
                        color=COLORS["predictive_missforest"], linewidth=2)
                ax.axhline(0, color="#64748b", linestyle="--", linewidth=1)
                ax.set_ylim(-limit, limit)
                ax.set_xticks([10, 20, 30])
                ax.grid(alpha=0.25)
                ax.set_title(f"{title} | noise ±{noise_pct}%")
                if row == 1:
                    ax.set_xlabel("Masked cells (%)")
                if col == 0:
                    ax.set_ylabel("Predictive MF minus Original MF")
        fig.suptitle("Paired joint-table method differences (zero means identical scores)")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        paths.append(_save(fig, result_dir, "forest_method_differences.png"))

    # 3. Placebo is a separate negative control, not part of the main skill heatmap.
    if "placebo_skill" in summary:
        methods = [(p, m) for p in PROTOCOLS for m in COLORS
                   if not (p == "unseen_records" and m == "missforest")]
        fig, ax = plt.subplots(figsize=(max(12, len(cases) * 1.3), 5))
        matrix = np.full((len(methods), len(cases)), np.nan)
        for row, (protocol, method) in enumerate(methods):
            sub = summary.loc[summary.protocol.eq(protocol) & summary.method.eq(method)]
            lookup = {(round(float(r.completion), 8), round(float(r.noise), 8)):
                      r.placebo_skill for r in sub.itertuples(index=False)}
            for col, key in enumerate(cases):
                matrix[row, col] = lookup.get(key, np.nan)
        finite = matrix[np.isfinite(matrix)]
        cap = max(0.05, float(np.max(np.abs(finite))) * 1.15) if finite.size else 0.05
        norm = TwoSlopeNorm(vmin=-cap, vcenter=0, vmax=cap)
        cmap = plt.colormaps["RdBu"].copy()
        cmap.set_bad("#e2e8f0")
        im = ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap, norm=norm,
                       aspect="auto")
        ax.set_xticks(range(len(cases)), labels)
        ax.set_yticks(range(len(methods)),
                      [f"{LABELS[m]} — {('Joint-table' if p == PROTOCOLS[0] else 'Unseen')}"
                       for p, m in methods])
        for row in range(len(methods)):
            for col in range(len(cases)):
                if np.isfinite(matrix[row, col]):
                    ax.text(col, row, f"{matrix[row, col]:+.3f}",
                            ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, label="Placebo skill (0 = Median/Mode baseline)")
        ax.set_title("Random-placebo negative control across scenarios")
        fig.tight_layout()
        paths.append(_save(fig, result_dir, "placebo_skill_heatmap.png"))

    # 4. How far joint-table performance differs from genuinely unseen records.
    predictive = summary.loc[summary.method.eq("predictive_missforest")]
    joint_p = predictive.loc[predictive.protocol.eq("joint_table_transductive")]
    unseen_p = predictive.loc[predictive.protocol.eq("unseen_records")]
    gap = joint_p.merge(unseen_p, on=["completion", "noise", "method"],
                        suffixes=("_joint", "_unseen"), validate="one_to_one")
    if not gap.empty:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for noise_pct in NOISE_LEVELS:
            chunk = gap.loc[np.isclose(gap.noise * 100, noise_pct)].sort_values(
                "completion", ascending=False)
            if len(chunk):
                ax.plot(100 * (1 - chunk.completion),
                        chunk.mean_numeric_skill_ex_placebo_joint
                        - chunk.mean_numeric_skill_ex_placebo_unseen,
                        "o-", label=f"Noise ±{noise_pct}%", linewidth=2)
        ax.axhline(0, color="#64748b", linestyle="--", linewidth=1)
        ax.set_xlabel("Masked cells (%)")
        ax.set_ylabel("Joint-table minus unseen-record numeric skill")
        ax.set_title("Predictive MissForest: evaluation-protocol gap")
        ax.set_xticks([10, 20, 30])
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        paths.append(_save(fig, result_dir, "protocol_gap.png"))

    # 5. Run-time is directly comparable only within the current hardware/run.
    if "runtime_seconds" in summary:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
        for ax, protocol in zip(axes, PROTOCOLS):
            chunk = summary.loc[summary.protocol.eq(protocol)]
            means = chunk.groupby("method").runtime_seconds.mean()
            methods_here = [m for m in COLORS if m in means]
            values = [float(means[m]) for m in methods_here]
            ax.bar(range(len(methods_here)), values,
                   color=[COLORS[m] for m in methods_here])
            ax.set_xticks(range(len(methods_here)),
                          [LABELS[m].replace(" ", "\n") for m in methods_here])
            ax.set_title(protocol.replace("_", " "))
            ax.set_yscale("log")
            ax.grid(axis="y", alpha=0.25)
            for i, value in enumerate(values):
                label = f"{value:.2f}s" if value < 1 else f"{value:.1f}s"
                ax.annotate(label, (i, value), xytext=(0, 5),
                            textcoords="offset points", ha="center", fontsize=9)
        axes[0].set_ylabel("Average scenario runtime (seconds, log scale)")
        fig.suptitle("Computational cost within this run (hardware dependent)")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        paths.append(_save(fig, result_dir, "runtime_comparison.png"))

    # 6. Iteration history, if saved. Shows stability, not validation accuracy.
    history_file = result_dir / "convergence_history.csv"
    if history_file.is_file():
        history = pd.read_csv(history_file)
        if not history.empty:
            completion, noise = _anchored_case(history)
            focus = history.loc[np.isclose(history.completion, completion)
                                & np.isclose(history.noise, noise)]
            fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex="col")
            for row, protocol in enumerate(PROTOCOLS):
                for col, (metric, ylabel) in enumerate([
                    ("numeric_change", "Numeric change between iterations"),
                    ("categorical_change", "Categorical change between iterations"),
                ]):
                    ax = axes[row, col]
                    for method in ("missforest", "predictive_missforest"):
                        data = focus.loc[focus.protocol.eq(protocol)
                                         & focus.method.eq(method)]
                        if data.empty or metric not in data:
                            continue
                        points = data.groupby("step", as_index=False)[metric].mean()
                        ax.plot(points.step, points[metric],
                                color=COLORS[method], label=LABELS[method], lw=2,
                                **METHOD_STYLES[method])
                        if "accepted" in data:
                            rejected = data.loc[data.accepted.astype(str).str.lower().eq("false")]
                            if len(rejected):
                                ax.scatter(rejected.step, rejected[metric], marker="x",
                                           s=70, color="#dc2626", zorder=4)
                    if col == 0:
                        ax.set_yscale("symlog", linthresh=1e-8)
                    ax.set_title(f"{protocol.replace('_', ' ')} — {ylabel}")
                    ax.set_xlabel("Iteration / pass")
                    ax.set_ylabel(ylabel)
                    ax.grid(alpha=0.25)
                    ax.legend(fontsize=8)
            fig.suptitle(
                f"Iteration changes at {100 * (1 - completion):.0f}% masking, ±{100 * noise:.0f}% noise\n"
                "Stability does not establish correctness; red × = rejected step")
            fig.tight_layout(rect=(0, 0, 1, 0.91))
            paths.append(_save(fig, result_dir, "convergence_diagnostics.png"))

    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    for path in make_plots(args.result_dir):
        print(path)
