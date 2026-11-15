"""Static figures for the notebook. Every figure is shown and saved to the run folder."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np

from .columns import CATEGORY_TARGET, GENERATOR_LINKS, NAMES, NUMERIC_TARGETS, TARGETS
from .models import NUMERIC_METHODS

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948",
          "#8a5a2b", "#0f9fb5", "#9a9a2e")
METHOD_COLORS = dict(zip(NUMERIC_METHODS[1:], SERIES))
SEQUENTIAL = LinearSegmentedColormap.from_list("blue", ["#f4f7fb", "#a9c9ef", "#2a78d6", "#123f75"])
DIVERGING = LinearSegmentedColormap.from_list("polarity", ["#2a78d6", "#efeeea", "#eb6834"])
MARKERS = ("o", "s", "^", "D", "v", "P", "X")


def apply_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "axes.titlecolor": INK,
        "axes.titlesize": 10, "axes.labelsize": 9, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.spines.top": False,
        "axes.spines.right": False, "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.frameon": False, "legend.fontsize": 8,
        "figure.titlesize": 12, "font.size": 9, "lines.linewidth": 2,
    })


def finish(figure, folder, name):
    figure.tight_layout()
    if folder is not None:
        Path(folder).mkdir(parents=True, exist_ok=True)
        figure.savefig(Path(folder) / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.show()


def _grid(count, columns=4, width=3.6, height=2.7):
    rows = -(-count // columns)
    figure, axes = plt.subplots(rows, columns, figsize=(columns * width, rows * height), squeeze=False)
    for axis in axes.ravel()[count:]:
        axis.set_visible(False)
    return figure, axes.ravel()


def plot_missingness(field_audit, folder):
    """Observed share per column in the three completeness levels."""
    table = field_audit.groupby(["column", "completion"], sort=False)["observed share"].mean().unstack()
    table = table.reindex(list(NAMES))
    figure, axis = plt.subplots(figsize=(10, 3.2))
    width = 0.26
    positions = np.arange(len(table))
    for index, level in enumerate(table.columns):
        axis.bar(positions + (index - 1) * width, table[level], width * 0.9, color=SERIES[index],
                 label=f"file completion {level}")
    axis.set_xticks(positions, table.index, rotation=35, ha="right")
    axis.set_ylabel("Observed share of cells")
    axis.set_ylim(0, 1.08)
    axis.grid(axis="x", visible=False)
    axis.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.2))
    finish(figure, folder, "observed_share")


def plot_heatmap(matrix, title, folder, name, label, center=None, limit=None, marks=None, fmt="{:.2f}"):
    """Annotated heatmap: one hue for magnitude, two poles when values have a sign."""
    values = matrix.to_numpy(dtype=float)
    figure, axis = plt.subplots(figsize=(max(6, 0.82 * matrix.shape[1] + 2.5), 0.42 * matrix.shape[0] + 1.6))
    axis.grid(False)
    if center is None:
        image = axis.imshow(values, cmap=SEQUENTIAL, aspect="auto", vmin=0, vmax=limit)
    else:
        reach = limit or float(np.nanmax(np.abs(values - center))) or 1.0
        image = axis.imshow(values, cmap=DIVERGING, aspect="auto",
                            norm=TwoSlopeNorm(center, center - reach, center + reach))
    axis.set_xticks(range(matrix.shape[1]), matrix.columns, rotation=35, ha="right")
    axis.set_yticks(range(matrix.shape[0]), matrix.index)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if np.isfinite(values[i, j]):
                shade = image.cmap(image.norm(values[i, j]))
                light = 0.299 * shade[0] + 0.587 * shade[1] + 0.114 * shade[2] > 0.55
                axis.text(j, i, fmt.format(values[i, j]), ha="center", va="center", fontsize=7.5,
                          color=INK if light else "white",
                          fontweight="bold" if marks is not None and marks[i, j] else "normal")
    colorbar = figure.colorbar(image, ax=axis, fraction=0.03, pad=0.02)
    colorbar.set_label(label)
    colorbar.outline.set_visible(False)
    axis.set_title(title, loc="left")
    for spine in axis.spines.values():
        spine.set_visible(False)
    finish(figure, folder, name)


def plot_importance(matrix, title, folder, name):
    """Learned importance. Bold values are links in the generator's definition."""
    ordered = matrix[[c for c in NAMES if c in matrix.columns]]
    marks = np.array([[p in GENERATOR_LINKS[t] for p in ordered.columns] for t in ordered.index])
    plot_heatmap(ordered, title, folder, name, "Error added when shuffled (1 = constant guess)",
                 limit=max(0.2, float(np.nanmax(ordered.to_numpy()))), marks=marks)


def plot_paired(pairs, scenario, folder):
    """MAE difference from the median with row-bootstrap intervals."""
    block = pairs[pairs.scenario.eq(scenario) & pairs.comparison.eq("versus baseline") & pairs.target.ne(CATEGORY_TARGET)]
    figure, axes = _grid(len(NUMERIC_TARGETS))
    methods = list(NUMERIC_METHODS[1:])
    for axis, target in zip(axes, NUMERIC_TARGETS):
        part = block[block.target.eq(target)].set_index("second").reindex(methods)
        positions = np.arange(len(methods))
        axis.hlines(positions, part.low, part.high, color=SERIES[0], lw=2)
        axis.plot(part.difference, positions, "o", color=SERIES[0], markersize=4.5)
        axis.axvline(0, color=MUTED, lw=0.8)
        axis.set_yticks(positions, methods if axis in axes[::4] else [""] * len(methods))
        axis.invert_yaxis()
        axis.set_title(target, loc="left")
        axis.set_xlabel("MAE minus median MAE")
        axis.grid(axis="y", visible=False)
    figure.suptitle(f"Paired error difference with 95% row-bootstrap interval · {scenario}", x=0.01, ha="left")
    finish(figure, folder, "paired_uncertainty")


def plot_cost(cost, numeric, scenario, folder):
    """Mean skill across numeric targets against total main-experiment time."""
    block = numeric[numeric.experiment.eq("main") & numeric.scenario.eq(scenario) & numeric.method.ne("Median")
                    & numeric.target.ne("random_placebo")]
    skill = block.groupby("method").skill.mean()
    seconds = cost[cost.experiment.eq("main")].set_index("method")["seconds per target and fold"]
    figure, axis = plt.subplots(figsize=(7, 3.6))
    for method, value in skill.items():
        axis.plot(seconds[method], value, "o", color=SERIES[0], markersize=7)
        axis.annotate(method, (seconds[method], value), xytext=(6, 4), textcoords="offset points", fontsize=8, color=INK)
    axis.set_xscale("log")
    axis.set_xlabel("Seconds per target and fold (log scale)")
    axis.set_ylabel("Mean skill across targets")
    axis.set_title(f"Accuracy against cost · {scenario}", loc="left")
    finish(figure, folder, "cost_against_skill")


# ------------------------------------------------------------ failure analysis

STYLES = ("-", "--", ":", "-.")
BASELINE_LABEL = "baseline (median or mode)"


def _percent(axis):
    low, high = axis.get_ylim()
    digits = 0 if high - low > 0.25 else 1
    axis.yaxis.set_major_formatter(lambda value, _: f"{100 * value:.{digits}f}")


def plot_lines(table, x, title, folder, name, xlabel, methods, style=None, style_label="", targets=None,
               y="skill", ylabel="Error removed vs baseline (%)", columns=4, share_seen=False):
    """One panel per target: a line per method, and a line style per level of ``style``.

    The zero line is the baseline: a model on it is no better than the median or the mode.
    """
    targets = [t for t in (targets or TARGETS) if t in set(table.target)]
    levels = sorted(table[style].unique()) if style else [None]
    figure, axes = _grid(len(targets), columns=columns, width=3.5, height=2.6)
    for axis, target in zip(axes, targets):
        part = table[table.target.eq(target)]
        for method in methods:
            for index, level in enumerate(levels):
                line = part[part.method.eq(method)] if level is None else part[part.method.eq(method) & part[style].eq(level)]
                line = line.sort_values(x)
                axis.plot(line[x], line[y], color=METHOD_COLORS[method], linestyle=STYLES[index % len(STYLES)],
                          marker="o", markersize=3.5, lw=1.6)
        if y == "skill":
            # The axis follows the data so that trends stay readable. The baseline is drawn where
            # it is in range; the legend states its value for the panels where it lies below.
            low, high = axis.get_ylim()
            if low <= 0 <= high:
                axis.axhline(0, color=MUTED, lw=1)
            _percent(axis)
        axis.set_title(target, loc="left")
        axis.set_xlabel(xlabel)
        if share_seen:
            axis.set_xlim(-0.05, 1.05)
            axis.xaxis.set_major_formatter(lambda value, _: f"{100 * value:.0f}%")
        else:
            axis.set_xticks(sorted(part[x].unique()))
    for axis in axes[::columns]:
        axis.set_ylabel(ylabel)
    handles = [plt.Line2D([], [], color=METHOD_COLORS[m], lw=2, label=m) for m in methods]
    if style:
        handles += [plt.Line2D([], [], color=MUTED, lw=1.6, linestyle=STYLES[i % len(STYLES)], label=f"{style_label} {level}")
                    for i, level in enumerate(levels)]
    if y == "skill":
        handles.append(plt.Line2D([], [], color=MUTED, lw=1, label=BASELINE_LABEL + " = 0, below most panels"))
    figure.legend(handles=handles, ncol=min(len(handles), 7), loc="lower center", bbox_to_anchor=(0.5, -0.03))
    figure.suptitle(title, x=0.01, ha="left")
    finish(figure, folder, name)


def plot_overall(overall, methods, folder):
    """Skill of each model on each target; the baseline is the zero line."""
    table = overall[[m for m in methods if m in overall]]
    figure, axis = plt.subplots(figsize=(9, 0.36 * len(table) + 1.6))
    positions = np.arange(len(table))
    for index, method in enumerate(table.columns):
        axis.plot(table[method], positions, MARKERS[index % len(MARKERS)], color=METHOD_COLORS[method], markersize=6,
                  label=method, markeredgecolor=SURFACE, markeredgewidth=0.8)
    axis.axvline(0, color=MUTED, lw=1, label=BASELINE_LABEL)
    axis.axvspan(-0.1, 0.1, color=GRID, alpha=0.6, lw=0)
    axis.set_yticks(positions, table.index)
    axis.invert_yaxis()
    axis.set_xlim(-0.1, 1.03)
    axis.xaxis.set_major_formatter(lambda value, _: f"{100 * value:.0f}")
    axis.set_xlabel("Error removed vs baseline (%). The shaded band is the failing range")
    axis.grid(axis="y", visible=False)
    axis.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.02))
    finish(figure, folder, "overall_skill")


def plot_failure_map(matrix, title, folder, name, weak):
    """Skill by target and condition. Bold values are below the weak threshold."""
    values = matrix.to_numpy(dtype=float)
    plot_heatmap(matrix, title, folder, name, "Share of the baseline's error removed", center=weak, limit=0.5,
                 marks=np.nan_to_num(values, nan=1.0) < weak)


def plot_bands(bands, methods, order, folder, value="bias in SD", name="bias_by_value_band",
               title="Bias by where the true value lies in its column"):
    """Signed error across the value range. A falling line means estimates are pulled towards the centre."""
    targets = [t for t in NUMERIC_TARGETS if t in set(bands.target)]
    figure, axes = _grid(len(targets), columns=5, width=2.9, height=2.4)
    for axis, target in zip(axes, targets):
        part = bands[bands.target.eq(target)]
        for method in methods:
            line = part[part.method.eq(method)].set_index("band").reindex(order)
            axis.plot(range(len(order)), line[value], color=METHOD_COLORS.get(method, MUTED), marker="o", markersize=3.5,
                      lw=1.6, linestyle="--" if method == "Median" else "-")
        axis.axhline(0, color=MUTED, lw=1)
        axis.set_title(target, loc="left")
        axis.set_xticks(range(len(order)), ["lowest"] + [""] * (len(order) - 2) + ["highest"])
    for axis in axes[::5]:
        axis.set_ylabel("Mean error, in SDs of the column")
    handles = [plt.Line2D([], [], color=METHOD_COLORS.get(m, MUTED), lw=2, linestyle="--" if m == "Median" else "-",
                          label="Median (baseline)" if m == "Median" else m) for m in methods]
    figure.legend(handles=handles, ncol=len(handles), loc="lower center", bbox_to_anchor=(0.5, -0.04))
    figure.suptitle(title, x=0.01, ha="left")
    finish(figure, folder, name)


def plot_stacked(table, parts, colors, title, folder, name, xlabel):
    """Horizontal bars that split each row into the given parts."""
    figure, axis = plt.subplots(figsize=(9, 0.36 * len(table) + 1.6))
    left = np.zeros(len(table))
    positions = np.arange(len(table))
    for part, color in zip(parts, colors):
        width = np.nan_to_num(table[part].clip(lower=0).to_numpy(dtype=float))
        axis.barh(positions, width, left=left, color=color, height=0.62, label=part, edgecolor=SURFACE, linewidth=1.5)
        left += width
    axis.set_yticks(positions, table.index)
    axis.invert_yaxis()
    axis.set_xlim(0, 1)
    axis.xaxis.set_major_formatter(lambda value, _: f"{100 * value:.0f}%")
    axis.set_xlabel(xlabel)
    axis.grid(axis="y", visible=False)
    axis.legend(ncol=len(parts), loc="lower center", bbox_to_anchor=(0.5, 1.02))
    axis.set_title(title, loc="left", pad=28)
    finish(figure, folder, name)


def plot_pairs(table, pairs, title, folder, name, xlabel):
    """For each row, pairs of values joined by a line. The open marker is the first value of each pair."""
    title = f"{title} (open marker: without, filled: with)"
    figure, axis = plt.subplots(figsize=(9, 0.4 * len(table) + 1.8))
    positions = np.arange(len(table))
    offsets = np.linspace(-0.25, 0.25, len(pairs)) if len(pairs) > 1 else [0]
    for offset, (label, first, second, color) in zip(offsets, pairs):
        axis.hlines(positions + offset, table[first], table[second], color=color, lw=1.6, alpha=0.6)
        axis.plot(table[first], positions + offset, "o", color=SURFACE, markeredgecolor=color, markersize=5.5, markeredgewidth=1.6)
        axis.plot(table[second], positions + offset, "o", color=color, markersize=5.5, label=label)
    axis.axvline(0, color=MUTED, lw=1, label=BASELINE_LABEL)
    axis.set_yticks(positions, table.index)
    axis.invert_yaxis()
    axis.xaxis.set_major_formatter(lambda value, _: f"{100 * value:.0f}")
    axis.set_xlabel(xlabel)
    axis.grid(axis="y", visible=False)
    axis.legend(ncol=len(pairs) + 1, loc="lower center", bbox_to_anchor=(0.5, 1.02))
    axis.set_title(title, loc="left", pad=28)
    finish(figure, folder, name)


def plot_scatter(table, x, y, labels, title, folder, name, xlabel, ylabel):
    """Labelled points, one per target."""
    figure, axis = plt.subplots(figsize=(7.5, 4.2))
    axis.plot(table[x], table[y], "o", color=SERIES[0], markersize=7, markeredgecolor=SURFACE)
    for _, row in table.iterrows():
        axis.annotate(row[labels], (row[x], row[y]), xytext=(6, 4), textcoords="offset points", fontsize=8, color=INK)
    axis.axhline(0, color=MUTED, lw=1)
    axis.yaxis.set_major_formatter(lambda value, _: f"{100 * value:+.0f}%")
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_title(title, loc="left")
    finish(figure, folder, name)


def plot_bars(table, title, folder, name, ylabel, colors=None):
    """Grouped bars: one group per row of ``table`` and one bar per column. The zero line is the baseline."""
    figure, axis = plt.subplots(figsize=(max(8, 0.75 * len(table) + 2), 3.6))
    width = 0.8 / table.shape[1]
    positions = np.arange(len(table))
    for index, column in enumerate(table.columns):
        axis.bar(positions + (index - (table.shape[1] - 1) / 2) * width, table[column], width * 0.9,
                 color=(colors or SERIES)[index], label=column)
    axis.axhline(0, color=MUTED, lw=1)
    axis.set_xticks(positions, table.index, rotation=35, ha="right")
    axis.yaxis.set_major_formatter(lambda value, _: f"{100 * value:.0f}")
    axis.set_ylabel(ylabel)
    axis.grid(axis="x", visible=False)
    axis.legend(ncol=table.shape[1], loc="lower center", bbox_to_anchor=(0.5, 1.02))
    axis.set_title(title, loc="left", pad=28)
    finish(figure, folder, name)
