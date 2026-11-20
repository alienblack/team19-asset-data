"""Present a saved demo as one table and three figures with a short speaking guide."""
from argparse import ArgumentParser
import base64
import hashlib
from html import escape
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


NAMES = {"mixture": "Gaussian EM", "forest": "Predictive MissForest"}
COLORS = {"mixture": "#087D75", "forest": "#243F5B"}
FIELDS = {
    "x": "x", "y": "y", "install_date": "Installation day", "altitude": "Altitude",
    "measured_humid": "Humidity", "measured_salt": "Salt", "measured_overgrowth": "Overgrowth",
    "life": "Life", "measured_gas_a": "Gas A", "measured_gas_b": "Gas B",
    "measured_gas_c": "Gas C", "total_gas": "Total gas", "leakage": "Leakage",
}


def read_table(folder, name):
    path = folder / f"{name}.csv"
    if not path.exists():
        path = folder / f"{name}.csv.gz"
    return pd.read_csv(path, float_precision="round_trip")


def save_figure(fig, path):
    fig.savefig(path, dpi=225, facecolor="white")
    plt.close(fig)


def build_report(folder):
    folder = Path(folder).resolve()
    verification = json.loads((folder / "verification.json").read_text())
    if verification["status"] != "PASS":
        raise ValueError("The demonstration must pass verification before presenting its results.")
    settings = verification["settings"]
    summary = read_table(folder, "comparison")
    metrics = read_table(folder, "field_metrics")
    ledger = read_table(folder, "hidden_predictions")
    measured = read_table(folder, "measured_test")
    selected = summary.loc[summary.method.isin(NAMES)].copy()
    selected["condition"] = pd.Categorical(selected.condition, ["Scattered", "Grouped"], ordered=True)
    selected["method"] = pd.Categorical(selected.method, list(NAMES), ordered=True)
    selected = selected.sort_values(["condition", "method"])
    if len(selected) != 4:
        raise ValueError("Expected Gaussian EM and MissForest scores for both gap conditions.")
    output = folder / "presentation"
    output.mkdir(exist_ok=True)
    selected.to_csv(output / "results_table.csv", index=False)
    field_data = metrics.loc[metrics.method.isin(NAMES) & metrics.attribute.isin(FIELDS)].copy()
    field_data.to_csv(output / "field_comparison.csv", index=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 17,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.axisbelow": True, "text.color": "#18334C"})

    # Compare the two methods on the same assets under each missingness design.
    fig, ax = plt.subplots(figsize=(16, 9))
    for offset, method in [(-0.18, "mixture"), (0.18, "forest")]:
        values = selected.loc[selected.method.eq(method)].set_index("condition").loc[
            ["Scattered", "Grouped"], "mean_numeric_skill"
        ]
        bars = ax.bar(np.arange(2) + offset, values, width=0.32, color=COLORS[method], label=NAMES[method])
        ax.bar_label(bars, fmt="%.3f", padding=8, fontsize=23, fontweight="bold")
    ax.set(xticks=[0, 1], xticklabels=["Scattered gaps", "Related fields missing together"],
           ylabel="Mean numeric skill", ylim=(0, max(0.75, selected.mean_numeric_skill.max() + 0.15)))
    ax.set_title("Grouped gaps separate the methods in this example", loc="left", fontsize=25, pad=24)
    ax.legend(frameon=False, loc="upper right", fontsize=17)
    ax.grid(axis="y", alpha=0.18)
    fig.text(0.10, 0.035, "Zero matches the median floor. Higher values indicate greater error reduction. "
             "Single split, fixed settings.", fontsize=13)
    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.15, top=0.86)
    save_figure(fig, output / "01_gap_pattern_comparison.png")

    # Keep field-level results visible so the mean cannot conceal individual failures.
    fig, axes = plt.subplots(1, 2, figsize=(16, 9), sharey=True)
    minimum = min(-0.1, field_data.skill.min() - 0.08)
    for ax, condition in zip(axes, ["Scattered", "Grouped"]):
        fields = field_data.loc[field_data.condition.eq(condition)].pivot(
            index="attribute", columns="method", values="skill",
        ).reindex(FIELDS)
        y = np.arange(len(fields))
        ax.hlines(y, fields.mixture, fields.forest, color="#B9C8CD", linewidth=1.6)
        for method, marker in [("mixture", "o"), ("forest", "s")]:
            ax.scatter(fields[method], y, color=COLORS[method], marker=marker,
                       s=75, label=NAMES[method], zorder=3)
        ax.axvline(0, color="#87949D", linewidth=1.2)
        ax.set(xlim=(minimum, 1), title=f"{condition} gaps", xlabel="Field skill relative to its median floor")
        ax.grid(axis="x", alpha=0.18)
    axes[0].set(yticks=np.arange(len(FIELDS)), yticklabels=list(FIELDS.values()))
    axes[0].invert_yaxis()
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper right", bbox_to_anchor=(0.98, 0.97),
               ncol=2, frameon=False, fontsize=14)
    fig.suptitle("The average hides different strengths by field", x=0.13, ha="left", y=0.97, fontsize=23)
    fig.text(0.13, 0.035, "Values left of zero have more error than the median floor. "
             "Operational tolerance is assessed separately.", fontsize=13)
    fig.subplots_adjust(left=0.13, right=0.97, bottom=0.13, top=0.85, wspace=0.15)
    save_figure(fig, output / "02_field_comparison.png")

    # Use a fixed test order for the uncertainty illustration.
    order = measured.loc[measured.condition.eq("Scattered"), "asset_id"].tolist()
    dates = ledger.loc[ledger.condition.eq("Scattered") & ledger.method.eq("mixture")
                       & ledger.attribute.eq("install_date")].set_index("asset_id")
    sample = dates.reindex([asset for asset in order if asset in dates.index]).head(12)
    sample.to_csv(output / "date_interval_examples.csv", index_label="asset_id")
    fig, ax = plt.subplots(figsize=(16, 9))
    positions = np.arange(len(sample))
    halves = (sample.upper - sample.lower) / 2
    ax.errorbar(positions, sample["fill"], yerr=halves, fmt="o", markersize=9,
                color=COLORS["mixture"], capsize=5, linewidth=2,
                label="Gaussian EM estimate and calibrated interval")
    ax.scatter(positions, sample.truth, marker="x", color="#A76B25", s=100,
               linewidths=2.5, label="Clean reference", zorder=4)
    ax.set(xticks=positions, xticklabels=[str(asset) for asset in sample.index],
           xlabel="Asset ID", ylabel="Installation day")
    lower = min(sample.lower.min(), sample.truth.min())
    upper = max(sample.upper.max(), sample.truth.max())
    span = max(upper - lower, 1)
    ax.set_ylim(lower - 0.05 * span, upper + 0.22 * span)
    ax.set_title("A close point estimate can still have a wide interval", loc="left", fontsize=25, pad=25)
    ax.legend(frameon=False, fontsize=16, loc="upper left")
    ax.grid(axis="y", alpha=0.18)
    fig.text(0.10, 0.035, "First 12 hidden dates in fixed test order. "
             "Intervals target 90% coverage. Tight date tolerance is ±7 days and loose is ±30 days.", fontsize=13)
    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.15, top=0.86)
    save_figure(fig, output / "03_date_intervals.png")

    scores = selected.set_index(["condition", "method"])
    gmm_scattered = scores.loc[("Scattered", "mixture"), "mean_numeric_skill"]
    forest_scattered = scores.loc[("Scattered", "forest"), "mean_numeric_skill"]
    gmm_grouped = scores.loc[("Grouped", "mixture"), "mean_numeric_skill"]
    forest_grouped = scores.loc[("Grouped", "forest"), "mean_numeric_skill"]
    altitude = field_data.loc[field_data.condition.eq("Grouped") & field_data.attribute.eq("altitude")].set_index("method")
    closest = sample.loc[(sample["fill"] - sample.truth).abs().idxmin()]
    rows = []
    for row in selected.itertuples():
        rows.append(f"<tr><td>{row.condition}</td><td>{NAMES[row.method]}</td>"
                    f"<td>{row.mean_numeric_skill:.3f}</td><td>{row.mean_within_loose:.1%}</td>"
                    f"<td>{int(row.fields_breaking)} / 13</td></tr>")
    interpretations = [
        ("Same missing rate, different information", "01_gap_pattern_comparison.png",
         f"With scattered gaps, the mean skills are close, {gmm_scattered:.3f} for Gaussian EM and "
         f"{forest_scattered:.3f} for MissForest. With grouped gaps, they fall to {gmm_grouped:.3f} "
         f"and {forest_grouped:.3f}. Related inputs disappearing together changes recoverability. "
         "This single example supports a sensitivity comparison, with the full study providing the repeated evidence."),
        ("Which fields explain the difference", "02_field_comparison.png",
         f"Under grouped gaps, altitude skill is {altitude.loc['mixture', 'skill']:.3f} for Gaussian EM "
         f"and {altitude.loc['forest', 'skill']:.3f} for MissForest. Yet only "
         f"{altitude.loc['mixture', 'tolerance']:.1%} of Gaussian EM altitude fills meet the loose tolerance. "
         "Relative improvement and usable precision answer different questions. Both methods can still fail a field."),
        ("Why the system can abstain", "03_date_intervals.png",
         f"For the closest point estimate among these 12 examples, asset {closest.name}, the reference is "
         f"day {closest.truth:.1f} and the fill is {closest['fill']:.1f}. Its interval spans "
         f"{closest.lower:.1f}–{closest.upper:.1f} days. The decision is {closest.decision}. "
         "The estimate is close in this case, while the uncertainty remains too broad for the date tolerance. "
         "Coverage and precision are checked separately."),
    ]
    sections = []
    for title, filename, explanation in interpretations:
        encoded = base64.b64encode((output / filename).read_bytes()).decode()
        sections.append(f'<section><h2>{escape(title)}</h2><img src="data:image/png;base64,{encoded}" '
                        f'alt="{escape(title)}"><p class="interpretation">{escape(explanation)}</p></section>')
    scope = (f"{settings['train_rows']:,} training · {settings['calibration_rows']:,} calibration · "
             f"{settings['test_rows']:,} test assets · {settings['missing']:.0%} nominal gaps · "
             f"±{settings['noise']:.0%} measured noise · one spatial split · fixed settings")
    html = f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Gaussian EM and MissForest — small comparison</title>
<style>
*{{box-sizing:border-box}}body{{margin:0;background:#f2f5f6;color:#18334c;font-family:Inter,Arial,sans-serif}}
main{{max-width:1200px;margin:auto;padding:40px 30px}}header,section{{background:white;padding:30px;margin-bottom:24px;border-radius:10px}}
h1{{font-size:36px;line-height:1.15;margin:0 0 18px}}h2{{font-size:25px;margin:0 0 20px}}p{{font-size:18px;line-height:1.6}}
.scope,.definition{{color:#526777;font-size:15px}}table{{width:100%;border-collapse:collapse;font-size:19px;margin:22px 0}}
th,td{{padding:15px 12px;text-align:left;border-bottom:1px solid #dbe5e9}}th{{font-size:16px;background:#eef5f4}}
img{{width:100%;height:auto;display:block}}.interpretation{{border-left:4px solid #087d75;padding-left:18px}}
footer{{padding:8px 6px 30px;font-size:15px;color:#526777}}@media print{{body{{background:white}}main{{padding:0}}
header,section{{border-radius:0;break-inside:avoid}}section{{break-before:page}}}}
</style></head><body><main><header><h1>Gaussian EM and MissForest</h1>
<p>A small comparison of recovery under scattered and grouped missing values.</p>
<p class="scope">{escape(scope)}</p>
<table><thead><tr><th>Gap pattern</th><th>Method</th><th>Mean skill ↑</th>
<th>Mean within loose tolerance ↑</th><th>Fields failing the rule ↓</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<p class="definition">Skill is 1 − MAE / median-floor MAE. The means give 13 numeric fields equal weight.
A field fails when skill is below 0.10 or fewer than 50% of fills meet its loose tolerance.</p></header>
{''.join(sections)}
<footer>Source run — {escape(folder.name)}. Synthetic generator data, fixed settings and a single split limit generalisation.
The final report contains the tuned repeated study. All images are embedded in this file.</footer>
</main></body></html>'''
    report = output / "review_results.html"
    report.write_text(html)
    (output / "speaking_notes.md").write_text(
        "# Results walkthrough\n\n"
        "Opening — I am demonstrating a small comparison of my Gaussian EM implementation and Predictive MissForest. "
        "Both receive the same measured inputs and hidden test cells within each condition.\n\n"
        + "\n\n".join(title + " — " + explanation for title, _, explanation in interpretations)
        + "\n\nConclusion — The field, missingness pattern and required precision determine whether a fill is useful. "
        "These are single-split demonstration results. The repeated benchmark is documented separately.\n"
    )
    sources = [folder / "verification.json"]
    for name in ["comparison", "field_metrics", "hidden_predictions", "measured_test"]:
        path = folder / f"{name}.csv"
        sources.append(path if path.exists() else folder / f"{name}.csv.gz")
    provenance = {"source_files": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
                  "renderer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "source_run": str(folder), "model_fits_performed_by_renderer": 0}
    (output / "presentation_sources.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("output_directory", type=Path)
    arguments = parser.parse_args()
    print(build_report(arguments.output_directory))
