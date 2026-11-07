"""Figures for schema-free discovery and legacy validation stages.

All figures are written as PNG files with a fixed, colour-blind-safe palette.
Colour carries identity only (observed vs imputed, one class per hue); magnitude
uses a single-hue ramp and polarity a two-hue diverging ramp with a neutral
midpoint. Marks are thin and axes recessive so the data dominates.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

# Validated categorical order (identity), used in fixed order, never cycled.
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
OBSERVED, IMPUTED = CATEGORICAL[0], CATEGORICAL[1]
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
SURFACE = "#fcfcfb"
SEQUENTIAL = LinearSegmentedColormap.from_list("oc_blue", ["#eef3fb", "#9dbfe9", "#2a78d6", "#143b6b"])
DIVERGING = LinearSegmentedColormap.from_list("oc_div", ["#eb6834", "#f3f3f1", "#2a78d6"])

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT,
        "axes.titlecolor": TEXT,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "font.size": 9.5,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "legend.frameon": False,
        "savefig.dpi": 140,
        "savefig.bbox": "tight",
    }
)


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def _bar(ax: plt.Axes, labels: Sequence[str], values: Sequence[float], color: str, horizontal: bool = False) -> None:
    positions = np.arange(len(labels))
    if horizontal:
        ax.barh(positions, values, color=color, height=0.62)
        ax.set_yticks(positions, labels)
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
    else:
        ax.bar(positions, values, color=color, width=0.62)
        ax.set_xticks(positions, labels, rotation=20, ha="right")
        ax.grid(axis="x", visible=False)


# ---------------------------------------------------------------------------
# Profiling figures
# ---------------------------------------------------------------------------


def render_discovery_figures(
    frame: pd.DataFrame,
    profile: Any,
    max_numeric: int = 6,
    max_categorical: int = 6,
) -> dict[str, plt.Figure]:
    """Build notebook-ready figures from a schema-free ``ProfileResult``.

    Returns figures in memory; nothing is written to disk. The notebook can
    call ``display(fig)`` and then ``plt.close(fig)`` for each figure.
    """
    figures: dict[str, plt.Figure] = {}
    columns = profile.tables["column_profile"]

    if not columns.empty:
        fig, ax = plt.subplots(figsize=(8, max(2.8, 0.36 * len(columns) + 1)))
        labels = [f"{row.column} [{row.position}]" for row in columns.itertuples()]
        rates = columns["missing_rate"].fillna(0).to_numpy(dtype=float)
        _bar(ax, labels, rates, OBSERVED, horizontal=True)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Share of records blank or null")
        ax.set_title("Missing values by column")
        fig.tight_layout()
        figures["missingness"] = fig

    numeric = list(profile.summary.get("numeric_columns", []))[:max_numeric]
    if numeric:
        rows = int(np.ceil(len(numeric) / 2))
        fig, axes = plt.subplots(rows, 2, figsize=(11, max(3.2, 3.1 * rows)), squeeze=False)
        for ax, column in zip(axes.flat, numeric):
            vals = pd.to_numeric(frame[column], errors="coerce").dropna()
            if not vals.empty:
                ax.hist(vals, bins="auto", color=OBSERVED, edgecolor=SURFACE)
                ax.axvline(vals.median(), color=TEXT, linestyle="--", linewidth=1)
            ax.set_title(f"{column} (n={len(vals):,})")
            ax.set_xlabel(column)
            ax.set_ylabel("Records")
        for ax in axes.flat[len(numeric):]:
            ax.axis("off")
        fig.suptitle("Numerical distributions", fontweight="bold")
        fig.tight_layout()
        figures["numeric_distributions"] = fig

    categorical = list(profile.summary.get("categorical_columns", []))[:max_categorical]
    if categorical:
        rows = int(np.ceil(len(categorical) / 2))
        fig, axes = plt.subplots(rows, 2, figsize=(11, max(3.2, 3.1 * rows)), squeeze=False)
        for ax, column in zip(axes.flat, categorical):
            values = frame[column].astype("string")
            observed = values.loc[values.notna() & values.str.strip().ne("").fillna(False)]
            counts = observed.value_counts().head(10)
            _bar(ax, [str(v)[:22] for v in counts.index], counts.to_numpy(), OBSERVED)
            ax.set_title(f"{column} (top {len(counts)} categories)")
            ax.set_ylabel("Records")
        for ax in axes.flat[len(categorical):]:
            ax.axis("off")
        fig.suptitle("Categorical distributions", fontweight="bold")
        fig.tight_layout()
        figures["categorical_distributions"] = fig

    if len(numeric) >= 2:
        numeric_frame = frame[numeric].apply(pd.to_numeric, errors="coerce")
        matrix = numeric_frame.corr(method="spearman")
        fig, ax = plt.subplots(figsize=(max(4.5, len(numeric) * 1.0), max(4, len(numeric) * 0.9)))
        image = ax.imshow(matrix.to_numpy(dtype=float), cmap=DIVERGING, vmin=-1, vmax=1)
        ax.set_xticks(range(len(numeric)), numeric, rotation=35, ha="right")
        ax.set_yticks(range(len(numeric)), numeric)
        ax.grid(False)
        for i in range(len(numeric)):
            for j in range(len(numeric)):
                val = matrix.iat[i, j]
                if np.isfinite(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)
        fig.colorbar(image, ax=ax, label="Spearman correlation")
        ax.set_title("Numerical relationships")
        fig.tight_layout()
        figures["numeric_relationships"] = fig

    co = profile.tables.get("co_missingness")
    if co is not None and not co.empty and profile.summary.get("missing_cells", 0) > 0:
        labels = co["column"].astype(str).tolist()
        matrix = co.drop(columns="column").to_numpy(dtype=float)
        fig, ax = plt.subplots(figsize=(max(4.5, len(labels) * 0.65), max(4, len(labels) * 0.58)))
        image = ax.imshow(matrix, cmap=SEQUENTIAL, vmin=0, vmax=1)
        ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
        ax.set_yticks(range(len(labels)), labels)
        ax.grid(False)
        fig.colorbar(image, ax=ax, label="Joint missing rate")
        ax.set_title("Co-missingness")
        fig.tight_layout()
        figures["co_missingness"] = fig

    return figures


def render_profile_figures(frame: pd.DataFrame, schema: Any, out_dir: str | Path) -> dict[str, Path]:
    """Render the profiling figures for a typed frame. Returns ``{name: path}``."""
    out_dir = Path(out_dir)
    figures: dict[str, Path] = {}
    declared = [c for c in schema.column_names if c in frame.columns]
    model_columns = [c for c in schema.model_columns if c in frame.columns]
    numeric = [c for c in schema.numeric_model_columns if c in frame.columns]
    categorical = [c for c in schema.categorical_model_columns if c in frame.columns]
    n = len(frame)

    # 1. missing rate per column -------------------------------------------------------
    rates = [float(frame[c].isna().mean()) if n else 0.0 for c in declared]
    colors = [OBSERVED if c in model_columns else "#b9b8b2" for c in declared]
    fig, ax = plt.subplots(figsize=(7.2, 0.32 * len(declared) + 1.4))
    positions = np.arange(len(declared))
    ax.barh(positions, rates, color=colors, height=0.62)
    ax.set_yticks(positions, declared)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of records missing")
    ax.set_title("Missing values per column (model attributes in blue)")
    for pos, rate in zip(positions, rates):
        if rate > 0:
            ax.text(min(rate + 0.01, 0.98), pos, f"{rate:.1%}", va="center", color=MUTED, fontsize=8.5)
    figures["missingness"] = _save(fig, out_dir / "missingness.png")

    # 2. co-missingness heatmap ---------------------------------------------------------------
    if model_columns:
        miss = frame[model_columns].isna().astype(float)
        joint = (miss.T @ miss) / n if n else miss.T @ miss
        fig, ax = plt.subplots(figsize=(6.4, 5.4))
        image = ax.imshow(joint.to_numpy(), cmap=SEQUENTIAL, vmin=0, vmax=max(float(joint.to_numpy().max()), 1e-9))
        ax.set_xticks(range(len(model_columns)), model_columns, rotation=35, ha="right")
        ax.set_yticks(range(len(model_columns)), model_columns)
        ax.grid(False)
        for i in range(len(model_columns)):
            for j in range(len(model_columns)):
                value = joint.iat[i, j]
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8, color=TEXT if value < 0.5 * joint.to_numpy().max() else "white")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="share of records with both missing")
        ax.set_title("Joint missingness of model attributes")
        figures["co_missingness"] = _save(fig, out_dir / "co_missingness.png")

    # 3. numeric distributions ------------------------------------------------------------------
    for column in numeric:
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        if values.empty:
            continue
        fig, ax = plt.subplots(figsize=(7.2, 3.6))
        bins = _bins_for(column, values, schema)
        ax.hist(values, bins=bins, color=OBSERVED, edgecolor=SURFACE, linewidth=0.6)
        ax.set_xlabel(column)
        ax.set_ylabel("records")
        ax.set_title(f"{column}: observed values (n={len(values):,})")
        median = float(values.median())
        ax.axvline(median, color=TEXT, linewidth=1.2, linestyle="--")
        ax.text(median, ax.get_ylim()[1] * 0.95, f" median {median:,.1f}", color=TEXT, fontsize=8.5, va="top")
        figures[f"distribution_{column}"] = _save(fig, out_dir / f"distribution_{column}.png")

    # 4. categorical frequencies -----------------------------------------------------------------
    if categorical:
        cols = min(3, len(categorical))
        rows = int(np.ceil(len(categorical) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 2.9 * rows), squeeze=False)
        for ax, column in zip(axes.flat, categorical):
            counts = frame[column].dropna().value_counts()
            spec = schema.column(column)
            order = [v for v in (spec.allowed or counts.index) if v in counts.index] + [v for v in counts.index if spec.allowed and v not in spec.allowed]
            _bar(ax, [str(v) for v in order], [counts[v] for v in order], OBSERVED)
            ax.set_title(f"{column} (n={int(counts.sum()):,})", fontsize=10)
            ax.set_ylabel("records")
        for ax in axes.flat[len(categorical) :]:
            ax.axis("off")
        fig.suptitle("Categorical attributes: observed frequencies", fontsize=11, fontweight="bold")
        fig.tight_layout()
        figures["categorical_frequencies"] = _save(fig, out_dir / "categorical_frequencies.png")

    # 5. numeric by category (box plots) --------------------------------------------------------------
    for num_col in numeric:
        if not categorical:
            break
        cols = min(3, len(categorical))
        rows = int(np.ceil(len(categorical) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(3.8 * cols, 3.0 * rows), squeeze=False)
        for ax, cat_col in zip(axes.flat, categorical):
            both = frame[[num_col, cat_col]].dropna()
            spec = schema.column(cat_col)
            groups = [v for v in (spec.allowed or sorted(both[cat_col].unique())) if v in set(both[cat_col])]
            data = [pd.to_numeric(both.loc[both[cat_col] == g, num_col]).to_numpy() for g in groups]
            if not data:
                ax.axis("off")
                continue
            box = ax.boxplot(data, tick_labels=[str(g) for g in groups], widths=0.55, patch_artist=True, showfliers=False)
            for patch in box["boxes"]:
                patch.set(facecolor="#d9e6f7", edgecolor=OBSERVED, linewidth=1.0)
            for key in ("whiskers", "caps"):
                for line in box[key]:
                    line.set(color=OBSERVED, linewidth=1.0)
            for line in box["medians"]:
                line.set(color=TEXT, linewidth=1.3)
            ax.set_title(f"{num_col} by {cat_col}", fontsize=10)
            ax.tick_params(axis="x", rotation=20)
            ax.grid(axis="x", visible=False)
        for ax in axes.flat[len(categorical) :]:
            ax.axis("off")
        fig.tight_layout()
        figures[f"{num_col}_by_category"] = _save(fig, out_dir / f"{num_col}_by_category.png")

    # 6. target missingness by attribute (MAR evidence) --------------------------------------------------
    for target in [t for t in schema.targets if t in frame.columns]:
        target_missing = frame[target].isna()
        overall = float(target_missing.mean()) if n else 0.0
        labels, ratios, sizes = [], [], []
        for column in categorical:
            if column == target:
                continue
            spec = schema.column(column)
            for value in (spec.allowed or sorted(frame[column].dropna().unique())):
                idx = frame[column] == value
                if idx.sum() == 0:
                    continue
                labels.append(f"{column} = {value}")
                ratios.append(float(target_missing[idx].mean()) / overall if overall else np.nan)
                sizes.append(int(idx.sum()))
        if not labels:
            continue
        fig, ax = plt.subplots(figsize=(7.2, 0.3 * len(labels) + 1.6))
        norm = TwoSlopeNorm(vcenter=1.0, vmin=min(0.0, np.nanmin(ratios)), vmax=max(2.0, np.nanmax(ratios)))
        colors = [DIVERGING(norm(r)) if np.isfinite(r) else "#b9b8b2" for r in ratios]
        positions = np.arange(len(labels))
        ax.barh(positions, ratios, color=colors, height=0.62)
        ax.axvline(1.0, color=TEXT, linewidth=1.0)
        ax.set_yticks(positions, labels)
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        ax.set_xlabel(f"missing rate of {target} within group / overall missing rate ({overall:.1%})")
        ax.set_title(f"Does {target} go missing more often for some records? (1.0 = no dependence)")
        for pos, size in zip(positions, sizes):
            ax.text(ax.get_xlim()[1], pos, f"n={size:,}", va="center", ha="right", color=MUTED, fontsize=8)
        figures[f"missingness_by_attribute_{target}"] = _save(fig, out_dir / f"missingness_by_attribute_{target}.png")

    return figures


def _bins_for(column: str, values: pd.Series, schema: Any) -> np.ndarray | int:
    spec = schema.column(column)
    if spec.dtype == "integer" and values.max() - values.min() <= 200:
        lo, hi = int(values.min()), int(values.max())
        return np.arange(lo, hi + 2) - 0.5
    return 40


# ---------------------------------------------------------------------------
# Validation figures: observed vs imputed
# ---------------------------------------------------------------------------


def render_imputation_figures(
    original: pd.DataFrame,
    imputed: pd.DataFrame,
    schema: Any,
    out_dir: str | Path,
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
) -> dict[str, Path]:
    """Compare the distribution of observed cells with the imputed cells per column."""
    out_dir = Path(out_dir)
    figures: dict[str, Path] = {}

    for column in numeric_columns:
        was_missing = original[column].isna()
        observed = pd.to_numeric(original.loc[~was_missing, column], errors="coerce").dropna()
        filled = pd.to_numeric(imputed.loc[was_missing, column], errors="coerce").dropna()
        if observed.empty or filled.empty:
            continue
        fig, ax = plt.subplots(figsize=(7.2, 3.8))
        bins = _bins_for(column, pd.concat([observed, filled]), schema)
        ax.hist(observed, bins=bins, density=True, color=OBSERVED, alpha=0.85, label=f"observed (n={len(observed):,})", edgecolor=SURFACE, linewidth=0.4)
        ax.hist(filled, bins=bins, density=True, color=IMPUTED, alpha=0.65, label=f"imputed (n={len(filled):,})", edgecolor=SURFACE, linewidth=0.4)
        ax.set_xlabel(column)
        ax.set_ylabel("density")
        ax.set_title(f"{column}: observed vs imputed cells")
        ax.legend(loc="upper left")
        figures[f"imputed_vs_observed_{column}"] = _save(fig, out_dir / f"imputed_vs_observed_{column}.png")

    for column in categorical_columns:
        was_missing = original[column].isna()
        observed = original.loc[~was_missing, column].dropna().astype(str)
        filled = imputed.loc[was_missing, column].dropna().astype(str)
        if observed.empty or filled.empty:
            continue
        spec = schema.column(column) if column in schema.columns else None
        classes = list(spec.allowed) if spec is not None and spec.allowed else sorted(set(observed) | set(filled))
        classes += [c for c in sorted(set(observed) | set(filled)) if c not in classes]
        obs_share = observed.value_counts(normalize=True).reindex(classes, fill_value=0.0)
        imp_share = filled.value_counts(normalize=True).reindex(classes, fill_value=0.0)
        fig, ax = plt.subplots(figsize=(7.2, 3.6))
        positions = np.arange(len(classes))
        width = 0.36
        ax.bar(positions - width / 2, obs_share.to_numpy(), width, color=OBSERVED, label=f"observed (n={len(observed):,})")
        ax.bar(positions + width / 2, imp_share.to_numpy(), width, color=IMPUTED, label=f"imputed (n={len(filled):,})")
        ax.set_xticks(positions, classes, rotation=20, ha="right")
        ax.set_ylabel("share")
        ax.set_ylim(0, 1)
        ax.grid(axis="x", visible=False)
        ax.set_title(f"{column}: class shares, observed vs imputed cells")
        ax.legend(loc="upper right")
        figures[f"imputed_vs_observed_{column}"] = _save(fig, out_dir / f"imputed_vs_observed_{column}.png")

    return figures


def render_reconstruction_figure(metrics: pd.DataFrame, out_dir: str | Path, title: str | None = None, filename: str = "reconstruction_metrics.png") -> Path | None:
    """Bar chart of reconstruction metrics per attribute and method.

    ``metrics`` is the long table written by the validation stage with columns
    ``method, variable, variable_type, mae, macro_f1``. Numeric attributes show
    MAE (lower is better), categorical attributes show macro-F1 (higher is better);
    the two live on separate panels so no axis is shared between unlike measures.
    """
    out_dir = Path(out_dir)
    if metrics.empty:
        return None
    methods = list(dict.fromkeys(metrics["method"]))
    palette = {m: CATEGORICAL[i % len(CATEGORICAL)] for i, m in enumerate(methods)}
    numeric = metrics[metrics["variable_type"] == "numeric"]
    categorical = metrics[metrics["variable_type"] == "categorical"]
    panels = [p for p in (("MAE (lower is better)", numeric, "mae"), ("macro-F1 (higher is better)", categorical, "macro_f1")) if not p[1].empty]
    if not panels:
        return None
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 3.8), squeeze=False)
    for ax, (title, table, metric) in zip(axes.flat, panels):
        variables = list(dict.fromkeys(table["variable"]))
        positions = np.arange(len(variables))
        width = 0.8 / max(len(methods), 1)
        for i, method in enumerate(methods):
            sub = table[table["method"] == method].set_index("variable")
            values = [float(sub.loc[v, metric]) if v in sub.index else np.nan for v in variables]
            ax.bar(positions + (i - (len(methods) - 1) / 2) * width, values, width * 0.92, color=palette[method], label=method)
        ax.set_xticks(positions, variables, rotation=20, ha="right")
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
        if metric == "macro_f1":
            ax.set_ylim(0, 1)
    axes.flat[0].legend(loc="upper left")
    fig.suptitle(title or "Reconstruction of deliberately hidden known values", fontsize=11, fontweight="bold")
    fig.tight_layout()
    return _save(fig, out_dir / filename)
