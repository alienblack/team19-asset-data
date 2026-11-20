"""Fit the existing imputers on a small synthetic example for the product review.

Run this file from VS Code or a terminal. Each ``# %%`` section is also a VS Code
Python cell. This demonstration uses fresh fits, one fixed spatial split and the
benchmark defaults. The final report contains the larger, tuned repeated study.
"""

# %% Imports and the existing project implementation
from argparse import ArgumentParser
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from subteam_a.data import Scenario, TRANSFORMER, measure, split_blocks
from subteam_a.evaluate import calibrate, cell_export, evaluate
from subteam_a.features import Features
from subteam_a.generator import generate
from subteam_a.mixture import MaskedMixture
from subteam_a.study import DEFAULT, create


METHODS = {
    "floor": "Median / mode",
    "ridge": "Chained ridge",
    "forest": "Predictive MissForest",
    "mixture": "Gaussian EM",
}
COLORS = {"floor": "#87949D", "ridge": "#A76B25", "forest": "#243F5B", "mixture": "#087D75"}
FIELDS = {
    "x": "x", "y": "y", "install_date": "Installation day", "altitude": "Altitude",
    "measured_humid": "Humidity", "measured_salt": "Salt", "measured_overgrowth": "Overgrowth",
    "life": "Life", "measured_gas_a": "Gas A", "measured_gas_b": "Gas B",
    "measured_gas_c": "Gas C", "total_gas": "Total gas", "leakage": "Leakage",
}


def arguments():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--missing", type=float, default=0.30, help="Nominal cell masking probability.")
    parser.add_argument("--noise", type=float, default=0.05, help="Relative jitter amplitude on measured fields.")
    parser.add_argument("--train-rows", type=int, default=1500)
    parser.add_argument("--calibration-rows", type=int, default=600)
    parser.add_argument("--test-rows", type=int, default=600)
    parser.add_argument("--seed", type=int, default=219)
    parser.add_argument("--split-seed", type=int, default=617)
    parser.add_argument("--no-gas-correction", action="store_true")
    parser.add_argument("--results-only", action="store_true",
                        help="Print a compact comparison and build a page with a table and three result graphs.")
    parser.add_argument("--out", type=Path, default=PROJECT / "outputs" / "product-review-demo",
                        help="Parent directory. Each execution creates a fresh run subdirectory.")
    args = parser.parse_args()
    if not 0 < args.missing < 1 or not 0 <= args.noise <= 1:
        parser.error("Use a missing probability between 0 and 1 and a noise amplitude from 0 to 1.")
    if min(args.train_rows, args.calibration_rows, args.test_rows) < 1:
        parser.error("Row counts must be positive.")
    return args


# %% Check the conditional Gaussian calculation against a known answer
def check_conditional_calculation():
    rows = np.array([[1.0, np.nan], [np.nan, np.nan]])
    means = np.zeros((1, 2))
    covariance = np.array([[1.0, 0.6], [0.6, 2.0]])
    prepared = MaskedMixture.preparation(rows)
    weights, conditional, uncertainty, _ = MaskedMixture.expect(
        prepared, means, covariance, np.ones(1),
    )
    np.testing.assert_allclose(conditional[:, 0], [[1.0, 0.6], [0.0, 0.0]])
    np.testing.assert_allclose(weights, 1.0)
    np.testing.assert_allclose(uncertainty[prepared[2][0], 1, 1], 1.64)
    return {"expected_missing_mean": 0.6, "expected_missing_variance": 1.64, "passed": True}


# %% Learn from measured training rows, then predict and calibrate on separate rows
def run_condition(label, truth, measured, calibration_view, partitions, args):
    train_ids, calibration_ids, test_ids = (
        partitions[name] for name in ("train", "calibration", "test")
    )
    training = measured[train_ids]
    test = measured[test_ids]
    calibration_input = calibration_view[calibration_ids]
    features = Features(TRANSFORMER, not args.no_gas_correction).fit(training)
    baseline = create("floor", DEFAULT["floor"], args.split_seed).fit(training, features).predict(test)
    no_calibration_rules = np.zeros_like(calibration_input, dtype=bool)
    no_test_rules = np.zeros_like(test, dtype=bool)
    metrics, ledgers, completions, runtimes = [], [], [], []
    predictions, confidence = {}, {}
    mixture_model = None
    observed = np.isfinite(test)
    untouched = measured.copy()

    for name, display_name in METHODS.items():
        started = perf_counter()
        model = create(name, DEFAULT[name].copy(), args.split_seed)
        model.fit(training, features)
        fitted = perf_counter()
        combined = model.predict(np.vstack([calibration_input, test]))
        calibration_prediction = combined.subset(slice(0, len(calibration_ids)))
        prediction = combined.subset(slice(len(calibration_ids), None))
        calibrated = calibrate(
            features, calibration_input, truth[calibration_ids], no_calibration_rules,
            calibration_prediction, test, no_test_rules, prediction,
        )
        cells, _, _, decisions = evaluate(
            test, truth[test_ids], prediction, baseline, calibrated, no_test_rules, features,
        )
        rows = pd.DataFrame(cells)
        rows = rows.loc[rows.group.eq("all")].copy()
        rows.insert(0, "method", name)
        rows.insert(0, "condition", label)
        metrics.append(rows)
        ledger = cell_export(
            TRANSFORMER, test_ids, test, truth[test_ids], prediction,
            calibrated, no_test_rules, decisions, baseline,
        )
        ledger.insert(0, "method", name)
        ledger.insert(0, "condition", label)
        ledgers.append(ledger)
        completed = pd.DataFrame(prediction.point, columns=TRANSFORMER.names)
        completed.insert(0, "asset_id", test_ids)
        completed.insert(0, "method", name)
        completed.insert(0, "condition", label)
        completions.append(completed)
        predictions[name], confidence[name] = prediction, calibrated
        if name == "mixture":
            mixture_model = model

        # These checks run for every model and missingness condition.
        np.testing.assert_array_equal(prediction.point[observed], test[observed])
        assert np.isfinite(prediction.point).all()
        assert np.isfinite(prediction.scale).all() and (prediction.scale > 0).all()
        probabilities = prediction.proba[TRANSFORMER.i("model_id")]
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
        np.testing.assert_array_equal(measured, untouched)
        seconds = perf_counter() - started
        runtimes.append({"condition": label, "method": name,
                         "fit_seconds": fitted - started, "total_seconds": seconds})
        numeric = rows.loc[rows.type.eq("numeric") & rows.attribute.ne("random_placebo")]
        if not args.results_only:
            print(f"  {display_name:23} skill {numeric.skill.mean(): .3f} | "
                  f"{seconds:.2f}s | observed values preserved", flush=True)

    return {
        "metrics": pd.concat(metrics, ignore_index=True),
        "ledger": pd.concat(ledgers, ignore_index=True),
        "completed": pd.concat(completions, ignore_index=True),
        "predictions": predictions, "confidence": confidence, "runtimes": runtimes,
        "mixture_model": mixture_model,
        "test": test, "test_ids": test_ids, "truth": truth[test_ids],
    }


# %% Recompute the displayed metrics from the saved per-gap predictions
def verify_saved_scores(folder):
    ledger = pd.read_csv(folder / "hidden_predictions.csv", float_precision="round_trip")
    metrics = pd.read_csv(folder / "field_metrics.csv", float_precision="round_trip")
    for row in metrics.itertuples():
        part = ledger.loc[
            ledger.condition.eq(row.condition) & ledger.method.eq(row.method)
            & ledger.attribute.eq(row.attribute)
        ]
        assert len(part) == row.n
        if row.type == "numeric":
            error = (part["fill"] - part.truth).abs().mean()
            floor_error = (part.floor - part.truth).abs().mean()
        else:
            error = part["fill"].ne(part.truth).mean()
            floor_error = part.floor.ne(part.truth).mean()
        np.testing.assert_allclose([error, floor_error], [row.error, row.floor_error], atol=1e-8)
    measured = pd.read_csv(folder / "measured_test.csv", float_precision="round_trip")
    completed = pd.read_csv(folder / "completed_test.csv", float_precision="round_trip")
    for (condition, method), frame in completed.groupby(["condition", "method"]):
        raw = measured.loc[measured.condition.eq(condition)].set_index("asset_id")
        filled = frame.set_index("asset_id").loc[raw.index, TRANSFORMER.names].to_numpy()
        visible = raw[list(TRANSFORMER.names)].to_numpy()
        keep = np.isfinite(visible)
        np.testing.assert_array_equal(filled[keep], visible[keep])
    return {"metric_rows_recomputed": len(metrics), "hidden_predictions_checked": len(ledger),
            "saved_observations_preserved": True}


# %% Generate readable comparison, fitting and uncertainty charts
def comparison_chart(metrics, summary, args, folder):
    fields = list(FIELDS)
    selected = metrics.loc[metrics.condition.eq("Scattered") & metrics.attribute.isin(fields)]
    scores = selected.pivot(index="attribute", columns="method", values="skill").reindex(fields)
    fig, axes = plt.subplots(1, 2, figsize=(16, 9), gridspec_kw={"width_ratios": [1.15, 1]})
    y = np.arange(len(scores))
    axes[0].hlines(y, scores.mixture, scores.forest, color="#B9C8CD")
    axes[0].scatter(scores.mixture, y, color=COLORS["mixture"], s=65, label="Gaussian EM", zorder=3)
    axes[0].scatter(scores.forest, y, color=COLORS["forest"], marker="s", s=55,
                    label="Predictive MissForest", zorder=3)
    axes[0].set(yticks=y, yticklabels=[FIELDS[name] for name in scores.index],
                xlabel="Skill relative to the median floor", title="Scattered gaps · field comparison")
    axes[0].invert_yaxis()
    axes[0].axvline(0, color="#87949D", linewidth=1)
    axes[0].legend(frameon=False, loc="upper left", fontsize=12)
    axes[0].grid(axis="x", alpha=0.2)
    for index, method in enumerate(METHODS):
        values = summary.loc[summary.method.eq(method)].set_index("condition").loc[
            ["Scattered", "Grouped"], "mean_numeric_skill"
        ]
        bars = axes[1].bar(np.arange(2) + (index - 1.5) * 0.19, values, width=0.18,
                           color=COLORS[method], label=METHODS[method])
        axes[1].bar_label(bars, fmt="%.3f", fontsize=11, padding=4)
    axes[1].set(xticks=[0, 1], xticklabels=["Scattered", "Grouped"],
                ylabel="Mean of 13 numeric field skills", title="Same assets · different gap patterns")
    axes[1].axhline(0, color="#87949D", linewidth=1)
    axes[1].grid(axis="y", alpha=0.2)
    axes[1].legend(frameon=False, fontsize=12, loc="upper right")
    fig.suptitle("Fresh demonstration · Gaussian EM and reference methods", fontsize=23, x=0.05, ha="left")
    fig.text(0.05, 0.035, f"One spatial split · {args.train_rows:,} training / {args.test_rows:,} test assets · "
             f"{args.missing:.0%} nominal gaps · ±{args.noise:.0%} noise · fixed demo settings", fontsize=12)
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.14, top=0.87, wspace=0.38)
    fig.savefig(folder / "live_comparison.png", dpi=200)
    plt.close(fig)


def fitting_chart(scattered, grouped, folder):
    fig, axes = plt.subplots(1, 2, figsize=(16, 9))
    for label, result, color in (("Scattered", scattered, "#087D75"), ("Grouped", grouped, "#A76B25")):
        trace = result["predictions"]["mixture"].trace
        axes[0].plot(np.arange(1, len(trace) + 1), trace, marker="o", label=label, color=color)
    axes[0].set(xlabel="EM iteration", ylabel="Average observed-data log likelihood",
                title="Fitting trace for each training condition")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.2)

    column = TRANSFORMER.i("install_date")
    hidden = np.flatnonzero(np.isnan(scattered["test"][:, column]))[:12]
    prediction = scattered["predictions"]["mixture"]
    confidence = scattered["confidence"]["mixture"]
    point = prediction.point[hidden, column]
    half = (confidence.upper[hidden, column] - confidence.lower[hidden, column]) / 2
    axes[1].errorbar(np.arange(len(hidden)), point, yerr=half, fmt="o", color=COLORS["mixture"],
                     capsize=4, label="Fill and calibrated band")
    axes[1].scatter(np.arange(len(hidden)), scattered["truth"][hidden, column], marker="x",
                    color="#A76B25", s=65, label="Clean reference")
    axes[1].set(xticks=np.arange(len(hidden)), xticklabels=scattered["test_ids"][hidden],
                xlabel="Asset ID", ylabel="Installation day", title="First 12 hidden dates in fixed test order")
    axes[1].tick_params(axis="x", rotation=60, labelsize=10)
    axes[1].legend(frameon=False, fontsize=12)
    axes[1].grid(axis="y", alpha=0.2)
    fig.suptitle("Gaussian EM · fitting and uncertainty", fontsize=23, x=0.05, ha="left")
    fig.text(0.05, 0.035, "Bands target 90% coverage using separate calibration rows. "
             "Coverage within every missing pattern requires further validation.", fontsize=12)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.18, top=0.86, wspace=0.3)
    fig.savefig(folder / "gmm_fitting_and_intervals.png", dpi=200)
    plt.close(fig)


def component_chart(scattered, folder):
    """Show an ambiguous component assignment, selected independently of prediction error."""
    column = TRANSFORMER.i("install_date")
    candidates = np.flatnonzero(np.isnan(scattered["test"][:, column]))
    model = scattered["mixture_model"]
    prepared = model.preparation(model.encode(scattered["test"][candidates]))
    weights, conditional, covariance, _ = model.expect(
        prepared, model.mu, model.cov, model.prior,
    )
    entropy = -(weights * np.log(np.maximum(weights, 1e-15))).sum(axis=1)
    selected = int(np.argmax(entropy))
    row = candidates[selected]
    posterior = weights[selected]
    numeric_column = TRANSFORMER.numeric.index(column)
    means = conditional[selected, :, numeric_column] * model.scale[numeric_column] + model.center[numeric_column]
    variance = covariance[prepared[2][selected], numeric_column, numeric_column] * model.scale[numeric_column] ** 2
    standard_deviation = np.sqrt(max(variance, 1e-12))
    weighted_mean = float(posterior @ means)
    reported = scattered["predictions"]["mixture"].point[row, column]
    np.testing.assert_allclose(max(0.0, weighted_mean), reported)
    asset_id = int(scattered["test_ids"][row])
    components = pd.DataFrame({
        "asset_id": asset_id, "target": "install_date", "component": np.arange(1, len(means) + 1),
        "posterior_weight": posterior, "conditional_mean_days": means,
        "conditional_sd_days": standard_deviation, "weighted_mean_contribution": posterior * means,
        "selection": "Maximum component-membership entropy among hidden dates",
    })
    components.to_csv(folder / "gmm_components.csv", index=False)
    before_after = pd.DataFrame({
        "asset_id": asset_id, "attribute": TRANSFORMER.names,
        "recorded_value": scattered["test"][row], "was_missing": np.isnan(scattered["test"][row]),
        "gmm_completed_value": scattered["predictions"]["mixture"].point[row],
        "clean_reference_for_scoring": scattered["truth"][row],
    })
    with np.errstate(invalid="ignore"):
        before_after.to_csv(folder / "gmm_example_record.csv", index=False)
    axis = np.linspace(means.min() - 4 * standard_deviation, means.max() + 4 * standard_deviation, 1200)
    densities = np.exp(-0.5 * ((axis[:, None] - means) / standard_deviation) ** 2)
    densities /= standard_deviation * np.sqrt(2 * np.pi)
    contributions = densities * posterior
    fig, ax = plt.subplots(figsize=(16, 9))
    for index in range(len(means)):
        ax.plot(axis, contributions[:, index], linewidth=1.8,
                label=f"Component {index + 1} · weight {posterior[index]:.1%}")
    ax.plot(axis, contributions.sum(axis=1), color=COLORS["mixture"], linewidth=3.5,
            label="Combined conditional density")
    ax.axvline(reported, color=COLORS["forest"], linestyle="--", linewidth=2,
               label=f"Point fill · day {reported:.1f}")
    ax.axvline(scattered["truth"][row, column], color="#A76B25", linestyle=":", linewidth=2,
               label=f"Clean reference · day {scattered['truth'][row, column]:.0f}")
    ax.set(xlabel="Possible installation day", ylabel="Conditional density",
           title=f"One missing date · asset {asset_id} · four component contributions")
    ax.grid(alpha=0.18)
    ax.legend(frameon=False, fontsize=13)
    fig.text(0.075, 0.035, "Example selected by component-membership uncertainty. "
             "The model density precedes separate interval calibration.", fontsize=12)
    fig.subplots_adjust(left=0.075, right=0.98, top=0.88, bottom=0.14)
    fig.savefig(folder / "gmm_components.png", dpi=200)
    plt.close(fig)


# %% Run the experiment and save an auditable demonstration
def main():
    args = arguments()
    plt.rcParams.update({"font.size": 14, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.axisbelow": True, "font.family": "DejaVu Sans"})
    started = perf_counter()
    print("PRODUCT REVIEW DEMO · fresh fits · single split · fixed settings", flush=True)
    conditional_check = check_conditional_calculation()
    if not args.results_only:
        print("PASS · conditional mean 0.6 and variance 1.64", flush=True)
    truth, noise_draws, mask_draws = (np.asarray(value, dtype=float) for value in generate(args.seed))
    partitions, blocks = split_blocks(truth, args.split_seed, TRANSFORMER)
    counts = {"train": args.train_rows, "calibration": args.calibration_rows, "test": args.test_rows}
    for name, count in counts.items():
        if count > len(partitions[name]):
            raise ValueError(f"Requested {count} {name} rows; this split has {len(partitions[name])}.")
        partitions[name] = partitions[name][:count]
    for first, second in (("train", "calibration"), ("train", "test"), ("calibration", "test")):
        assert not set(blocks[partitions[first]]).intersection(blocks[partitions[second]])
    calibration_draws = np.random.default_rng(
        np.random.SeedSequence([args.seed, 7203, 1])
    ).random(truth.shape)
    folder = args.out / datetime.now().astimezone().strftime("run_%Y%m%d_%H%M%S_%f")
    folder.mkdir(parents=True)
    results, measured_exports = {}, []
    for label, mechanism in (("Scattered", "MCAR"), ("Grouped", "GROUP")):
        scenario = Scenario(label, args.missing, args.noise, mechanism)
        measured = measure(truth, noise_draws, mask_draws, scenario)
        calibration_view = measure(truth, noise_draws, calibration_draws, scenario)
        if not args.results_only:
            print(f"\n{label} gaps · test cells hidden {np.isnan(measured[partitions['test']]).mean():.1%}", flush=True)
        with threadpool_limits(limits=1):
            results[label] = run_condition(label, truth, measured, calibration_view, partitions, args)
        exported = pd.DataFrame(measured[partitions["test"]], columns=TRANSFORMER.names)
        exported.insert(0, "asset_id", partitions["test"])
        exported.insert(0, "condition", label)
        measured_exports.append(exported)

    metrics = pd.concat([item["metrics"] for item in results.values()], ignore_index=True)
    ledger = pd.concat([item["ledger"] for item in results.values()], ignore_index=True)
    completed = pd.concat([item["completed"] for item in results.values()], ignore_index=True)
    numeric = metrics.loc[metrics.type.eq("numeric") & metrics.attribute.ne("random_placebo")].copy()
    numeric["breaks"] = numeric.skill.lt(0.10) | numeric.tolerance.lt(0.50)
    summary = numeric.groupby(["condition", "method"], sort=False).agg(
        mean_numeric_skill=("skill", "mean"), fields_breaking=("breaks", "sum"),
        mean_within_loose=("tolerance", "mean"), mean_numeric_coverage=("coverage", "mean"),
    ).reset_index()
    with np.errstate(invalid="ignore"):
        metrics.to_csv(folder / "field_metrics.csv", index=False)
        summary.to_csv(folder / "comparison.csv", index=False)
        ledger.to_csv(folder / "hidden_predictions.csv", index=False)
        completed.to_csv(folder / "completed_test.csv", index=False)
        pd.concat(measured_exports, ignore_index=True).to_csv(folder / "measured_test.csv", index=False)
    split_rows = [(int(asset), name, int(blocks[asset])) for name, ids in partitions.items() for asset in ids]
    pd.DataFrame(split_rows, columns=["asset_id", "partition", "spatial_block"]).to_csv(
        folder / "partitions.csv", index=False,
    )
    traces = [{"condition": label, "iteration": index, "observed_log_likelihood": value}
              for label, result in results.items()
              for index, value in enumerate(result["predictions"]["mixture"].trace, 1)]
    pd.DataFrame(traces).to_csv(folder / "gmm_fit_trace.csv", index=False)
    pd.DataFrame([row for result in results.values() for row in result["runtimes"]]).to_csv(
        folder / "runtime.csv", index=False,
    )
    examples = ledger.loc[ledger.condition.eq("Scattered") & ledger.method.eq("mixture")
                          & ledger.attribute.eq("install_date")].head(8)
    examples.to_csv(folder / "gmm_example_dates.csv", index=False)
    verified = verify_saved_scores(folder)
    comparison_chart(metrics, summary, args, folder)
    fitting_chart(results["Scattered"], results["Grouped"], folder)
    component_chart(results["Scattered"], folder)
    source_files = [Path(__file__), *sorted((PROJECT / "src" / "subteam_a").glob("*.py"))]
    verification = {
        "status": "PASS", "conditional_calculation": conditional_check,
        "spatial_partitions_disjoint": True, "fresh_model_fits": 8,
        "settings": {**vars(args), "out": str(folder)},
        "model_parameters": {method: DEFAULT[method] for method in METHODS},
        "source_sha256": {str(path.relative_to(PROJECT)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in source_files},
        **verified, "elapsed_seconds": perf_counter() - started,
        "scope": "Single-split subset demonstration with fixed defaults; see final report for tuned repeated results.",
    }
    (folder / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    if args.results_only:
        from present_product_review import build_report
        report = build_report(folder)
        table = summary.loc[summary.method.isin(["mixture", "forest"]), [
            "condition", "method", "mean_numeric_skill", "mean_within_loose", "fields_breaking",
        ]].copy()
        table["method"] = table.method.map(METHODS)
        print(table.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
        print(f"\nResults page · {report}\nVerification · PASS", flush=True)
    else:
        print("\nComparison of fresh fits", flush=True)
        print(summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
        print("\nGaussian EM examples · clean truth attached for scoring", flush=True)
        print(examples[["asset_id", "truth", "fill", "lower", "upper", "decision"]].round(2).to_string(index=False))
        print(f"\nPASS · {verified['metric_rows_recomputed']} metric rows recomputed from saved predictions", flush=True)
        print(f"PASS · completed exports preserve their observed inputs\nOutputs · {folder}", flush=True)
    print(f"Elapsed · {perf_counter() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
