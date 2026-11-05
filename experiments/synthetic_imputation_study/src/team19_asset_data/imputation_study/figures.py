"""Static figures for the notebook. Every figure is shown and saved to the run folder."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np

from .columns import CATEGORY_TARGET, GENERATOR_LINKS, NAMES, NUMERIC_TARGETS
from .models import NUMERIC_METHODS

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
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


def plot_skill(numeric, scenario, folder):
    """Share of the median's error removed, one panel per target."""
    block = numeric[numeric.experiment.eq("main") & numeric.scenario.eq(scenario) & numeric.method.ne("Median")]
    figure, axes = _grid(len(NUMERIC_TARGETS))
    methods = list(NUMERIC_METHODS[1:])
    for axis, target in zip(axes, NUMERIC_TARGETS):
        part = block[block.target.eq(target)].set_index("method").reindex(methods)
        positions = np.arange(len(methods))
        axis.barh(positions, part.skill, color=SERIES[0], height=0.62)
        axis.axvline(0, color=MUTED, lw=0.8)
        axis.set_yticks(positions, methods if axis in axes[::4] else [""] * len(methods))
        axis.invert_yaxis()
        axis.set_title(target, loc="left")
        axis.set_xlim(min(-0.1, float(part.skill.min()) - 0.05), 1.05)
        axis.grid(axis="y", visible=False)
        axis.set_xlabel("Skill vs median")
    figure.suptitle(f"Error removed relative to the median · {scenario}", x=0.01, ha="left")
    finish(figure, folder, "skill_by_method")


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


def plot_noise_lines(numeric, series_scenarios, level_name, folder, name, title):
    """Skill along one file parameter with the other held fixed; lines never join all nine files."""
    block = numeric[numeric.experiment.eq("main") & numeric.scenario.isin(series_scenarios) & numeric.method.ne("Median")]
    figure, axes = _grid(len(NUMERIC_TARGETS))
    for axis, target in zip(axes, NUMERIC_TARGETS):
        part = block[block.target.eq(target)]
        for index, method in enumerate(NUMERIC_METHODS[1:]):
            line = part[part.method.eq(method)].sort_values(level_name)
            axis.plot(line[level_name], line.skill, color=METHOD_COLORS[method], marker=MARKERS[index],
                      markersize=4.5, label=method)
        axis.axhline(0, color=MUTED, lw=0.8)
        axis.set_title(target, loc="left")
        axis.set_xlabel(level_name)
        axis.set_ylabel("Skill vs median")
        axis.set_xticks(sorted(part[level_name].unique()))
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, ncol=6, loc="lower center", bbox_to_anchor=(0.5, -0.04))
    figure.suptitle(title, x=0.01, ha="left")
    finish(figure, folder, name)


def plot_groups(groups, scenario, method, folder):
    """Skill by which companion columns were visible, for one method."""
    block = groups[groups.scenario.eq(scenario) & groups.method.eq(method) & groups.target.ne("random_placebo")
                   & groups.target.ne(CATEGORY_TARGET)]
    targets = [t for t in NUMERIC_TARGETS if t in set(block.target)]
    figure, axes = _grid(len(targets), columns=2, width=6.8, height=2.0)
    for axis, target in zip(axes, targets):
        part = block[block.target.eq(target)].sort_values("skill")
        positions = np.arange(len(part))
        axis.barh(positions, part.skill, color=SERIES[0], height=0.6)
        axis.set_yticks(positions, [f"{g}  (n={n})" for g, n in zip(part.group, part.n)], fontsize=7)
        axis.axvline(0, color=MUTED, lw=0.8)
        axis.set_xlim(min(-0.1, float(part.skill.min()) - 0.05), 1.05)
        axis.set_title(target, loc="left")
        axis.grid(axis="y", visible=False)
    figure.suptitle(f"{method}: skill by which related columns were visible · {scenario}", x=0.01, ha="left")
    finish(figure, folder, "skill_by_visible_inputs")


def plot_spread(distribution, folder):
    """SD of imputed values relative to the truth; one means the spread is preserved."""
    block = distribution[distribution.scope.eq("missing cells")]
    order = [m for m in NUMERIC_METHODS if m != "Median"]
    order += [m for m in block.method.unique() if m not in order and m != "Median"]
    matrix = block.pivot(index="target", columns="method", values="SD ratio").reindex(list(NUMERIC_TARGETS))[order]
    plot_heatmap(matrix, "Spread of imputed values relative to the truth (1 = preserved, below 1 = shrunk)",
                 folder, "spread_ratio", "SD ratio", center=1.0, limit=1.0)


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
