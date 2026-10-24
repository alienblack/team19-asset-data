"""Paired clean/noisy evaluation of the overhead-conductor v3 pilot.

The clean pilot fixes partitions, masks and tuning choices. Each noise arm
changes observed records, refits on its noisy fitting partition, and scores
the same hidden test cells against the unchanged complete reference.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from .input_preparation import MODEL_FIELDS
from .noise import NoiseScenario, apply_noise, dataframe_hash, validate_output
from .v3_pilot import (
    MATERIALS,
    ROOT,
    ConditionalBaseline,
    PilotSettings,
    _sha256,
    assert_observed_preserved,
    evaluate,
    floor_predictions,
    forest_predictions,
    forest_predictions_from_model,
    evaluation_diagnostics,
    standardize_predictions,
    validate_predictions,
)
from team19_asset_data.imputation.predictive_missforest import PredictiveMissForest


@dataclass(frozen=True)
class NoiseArm:
    arm_id: str
    description: str
    steps: tuple[tuple[str, tuple[str, ...], float, int], ...]


ARMS = (
    NoiseArm("length20_negative_control", "20% of lengths altered; length is excluded from prediction",
             (("random_numeric", ("length_km",), .20, 0),)),
    NoiseArm("distance20", "20% of sea distances altered by up to 20%",
             (("random_numeric", ("distance_to_sea_km",), .20, 0),)),
    NoiseArm("voltage05", "5% of voltage labels replaced by another valid label",
             (("categorical_replacement", ("voltage_category",), .05, 0),)),
    NoiseArm("voltage10", "10% of voltage labels replaced by another valid label",
             (("categorical_replacement", ("voltage_category",), .10, 0),)),
    NoiseArm("voltage20", "20% of voltage labels replaced by another valid label",
             (("categorical_replacement", ("voltage_category",), .20, 0),)),
    NoiseArm("year10_step2", "10% of recorded years shifted by up to two years",
             (("installation_year_error", ("installation_year",), .10, 2),)),
    NoiseArm("year20_step2", "20% of recorded years shifted by up to two years",
             (("installation_year_error", ("installation_year",), .20, 2),)),
    NoiseArm("combined", "10% voltage and year errors plus 20% sea-distance errors",
             (("categorical_replacement", ("voltage_category",), .10, 0),
              ("installation_year_error", ("installation_year",), .10, 2),
              ("random_numeric", ("distance_to_sea_km",), .20, 0))),
)

ARM_LABELS = {
    "length20_negative_control": "Length 20%\ncontrol",
    "distance20": "Sea distance\n20%",
    "voltage05": "Voltage\n5%",
    "voltage10": "Voltage\n10%",
    "voltage20": "Voltage\n20%",
    "year10_step2": "Year +/-2\n10%",
    "year20_step2": "Year +/-2\n20%",
    "combined": "Combined",
}


def noisy_reference(clean: pd.DataFrame, arm: NoiseArm, seed: int) -> tuple[pd.DataFrame, list[dict]]:
    observed = clean.copy(deep=True)
    records = []
    for step_index, (strategy, columns, rate, year_step) in enumerate(arm.steps):
        scenario = NoiseScenario(
            scenario_id=f"{arm.arm_id}_{step_index}", name=arm.description,
            strategy=strategy, columns=columns, noise_level=rate,
            seed=int(np.random.SeedSequence([seed, 192, step_index]).generate_state(1)[0]),
            year_step=year_step or 1,
        )
        observed, metadata = apply_noise(observed, scenario)
        records.append(metadata)
    validate_output(clean, observed)
    return observed, records


def overlay_observed(clean_view: pd.DataFrame, noisy_complete: pd.DataFrame) -> pd.DataFrame:
    """Use noisy predictors while retaining the exact original target mask."""
    if clean_view["segment_id"].duplicated().any() or noisy_complete["segment_id"].duplicated().any():
        raise ValueError("segment IDs must be unique")
    lookup = noisy_complete.set_index("segment_id")
    if not clean_view["segment_id"].isin(lookup.index).all():
        raise ValueError("noisy reference does not contain every observed ID")
    observed = lookup.loc[clean_view["segment_id"], list(MODEL_FIELDS)].reset_index(drop=True)
    observed.insert(0, "segment_id", clean_view["segment_id"].to_numpy())
    for target in ("material", "installation_year"):
        missing = clean_view[target].isna().to_numpy()
        if missing.any():
            observed.loc[missing, target] = np.nan
        if not np.array_equal(observed[target].isna().to_numpy(), missing):
            raise AssertionError("noise changed the frozen target mask")
    for field in MODEL_FIELDS:
        if field not in ("installation_year", "distance_to_sea_km"):
            observed[field] = observed[field].map(lambda value: str(value) if pd.notna(value) else np.nan)
    return observed


def changed_input_counts(clean_view: pd.DataFrame, noisy_view: pd.DataFrame, partition: str) -> pd.DataFrame:
    rows = []
    for field in MODEL_FIELDS:
        available = clean_view[field].notna()
        changed = available & clean_view[field].ne(noisy_view[field])
        rows.append({"partition": partition, "field": field, "available_cells": int(available.sum()),
                     "changed_observed_cells": int(changed.sum()),
                     "changed_observed_rate": float(changed.sum() / available.sum()) if available.any() else 0.0})
    return pd.DataFrame(rows)


def _complete(test: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    filled = test.set_index("segment_id").copy()
    for target in ("material", "installation_year"):
        rows = candidates.loc[candidates["target"].eq(target)].set_index("segment_id")
        filled.loc[rows.index, target] = rows["point_prediction"].to_numpy()
    return filled


def material_by_decade(candidates: pd.DataFrame, test: pd.DataFrame, clean: pd.DataFrame) -> pd.DataFrame:
    truth = clean.set_index("segment_id").loc[test["segment_id"]]
    rows = []
    truth_decade = (truth["installation_year"].astype(int) // 10) * 10
    for method, predictions in candidates.groupby("method", sort=False):
        filled = _complete(test, predictions)
        filled_decade = (filled["installation_year"].astype(int) // 10) * 10
        for decade in sorted(set(truth_decade).union(filled_decade)):
            truth_n = int(truth_decade.eq(decade).sum())
            filled_n = int(filled_decade.eq(decade).sum())
            for material in MATERIALS:
                rows.append({"method": method, "decade": int(decade), "material": material,
                             "truth_decade_n": truth_n, "filled_decade_n": filled_n,
                             "truth_count": int((truth_decade.eq(decade) & truth["material"].eq(material)).sum()),
                             "filled_count": int((filled_decade.eq(decade) & filled["material"].eq(material)).sum())})
                rows[-1]["truth_share_within_decade"] = rows[-1]["truth_count"] / truth_n if truth_n else np.nan
                rows[-1]["filled_share_within_decade"] = rows[-1]["filled_count"] / filled_n if filled_n else np.nan
                rows[-1]["share_difference"] = rows[-1]["filled_share_within_decade"] - rows[-1]["truth_share_within_decade"]
    return pd.DataFrame(rows)


def review_cases(candidates: pd.DataFrame, clean: pd.DataFrame, limit: int = 3) -> pd.DataFrame:
    """Examples are evaluator-only; candidates remain free of clean answers."""
    truth = clean.set_index("segment_id")
    records = []
    for row in candidates.itertuples(index=False):
        actual = truth.at[row.segment_id, row.target]
        if row.target == "material":
            correct = str(row.point_prediction) == str(actual)
            evidence = float(max(row.p_steel, row.p_copper, row.p_aluminium))
            kind = "correct" if correct else "incorrect"
            uncertainty = 1 - evidence
        else:
            error = abs(float(row.point_prediction) - float(actual))
            kind = "near" if error <= 2 else "large_error" if error >= 10 else "middle"
            uncertainty = float(row.year_q90 - row.year_q10)
            evidence = float(error)
        records.append({"method": row.method, "segment_id": row.segment_id,
                        "target": row.target, "case_type": kind, "actual": actual,
                        "prediction": row.point_prediction, "diagnostic": evidence,
                        "uncertainty_rank": uncertainty, "fallback_reason": row.fallback_reason})
    frame = pd.DataFrame(records)
    selected = []
    for _, group in frame.groupby(["method", "target", "case_type"], sort=False):
        selected.append(group.nlargest(limit, "uncertainty_rank"))
    return pd.concat(selected, ignore_index=True) if selected else frame


def _paired(noisy: pd.DataFrame, clean: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    merged = noisy.merge(clean, on=keys, how="left", validate="many_to_one", suffixes=("_noisy", "_clean"))
    if merged["value_clean"].isna().any():
        raise AssertionError("no clean result for a noisy metric")
    merged["delta_noisy_minus_clean"] = merged["value_noisy"] - merged["value_clean"]
    return merged


def _plot_deltas(paired: pd.DataFrame, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for target, measure in (("material", "log_loss"), ("installation_year", "mae_years")):
        frame = paired.loc[paired["target"].eq(target) & paired["measure"].eq(measure)]
        arms = [arm.arm_id for arm in ARMS if arm.arm_id in set(frame["arm"])]
        fig, ax = plt.subplots(figsize=(10, 4.6), constrained_layout=True)
        for offset, method, color, label in (
            (-.10, "conditional_baseline", "#236B70", "Conditional baseline"),
            (.10, "predictive_missforest", "#C36B38", "Predictive MissForest"),
        ):
            groups = frame.loc[frame["method"].eq(method)].groupby("arm")["delta_noisy_minus_clean"].agg(
                ["mean", "min", "max"]
            ).reindex(arms)
            x = np.arange(len(arms)) + offset
            ax.errorbar(x, groups["mean"],
                        yerr=np.vstack((groups["mean"] - groups["min"], groups["max"] - groups["mean"])),
                        marker="o", capsize=3, color=color, label=label, linewidth=1.5)
        ax.axhline(0, color="black", linewidth=.8)
        ax.set_xticks(range(len(arms)), [ARM_LABELS[arm] for arm in arms], fontsize=8)
        ax.set_ylabel("Noisy minus clean (lower is better)")
        title = "material log loss" if target == "material" else "installation-year MAE"
        ax.set_title(f"Paired change in {title}; five-seed mean and range")
        ax.legend(loc="upper left")
        ax.grid(axis="y", alpha=.2)
        fig.savefig(output / f"paired_{measure}.png", dpi=150)
        plt.close(fig)


def _plot_population(paired: pd.DataFrame, output: Path) -> None:
    frame = paired.loc[paired["measure"].eq("aged_50_or_older_share_error") &
                       paired["method"].isin(("conditional_baseline", "predictive_missforest"))]
    arms = [arm.arm_id for arm in ARMS if arm.arm_id in set(frame["arm"])]
    fig, ax = plt.subplots(figsize=(10, 4.6), constrained_layout=True)
    for offset, method, color, label in (
        (-.10, "conditional_baseline", "#236B70", "Conditional baseline"),
        (.10, "predictive_missforest", "#C36B38", "Predictive MissForest"),
    ):
        grouped = frame.loc[frame["method"].eq(method)].groupby("arm")["delta_noisy_minus_clean"].mean().reindex(arms)
        ax.plot(np.arange(len(arms)) + offset, grouped * 100, marker="o", color=color, label=label)
    ax.axhline(0, color="black", linewidth=.8)
    ax.set_xticks(range(len(arms)), [ARM_LABELS[arm] for arm in arms], fontsize=8)
    ax.set_ylabel("Change in aged-50-or-older share error (percentage points)")
    ax.set_title("Population impact: noisy minus clean")
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=.2)
    fig.savefig(output / "paired_aged50_population.png", dpi=150)
    plt.close(fig)


def run_comparison(pilot_output: str | Path, output_dir: str | Path,
                   arms: tuple[NoiseArm, ...] = ARMS) -> Path:
    pilot = Path(pilot_output).resolve()
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite {output}")
    manifest = json.loads((pilot / "manifest.json").read_text(encoding="utf-8"))
    clean = pd.read_csv(pilot / "evaluator_only/clean_reference.csv")
    if _sha256(pilot / "evaluator_only/clean_reference.csv") != manifest["file_sha256"]["clean_reference"]:
        raise ValueError("clean reference hash differs from pilot manifest")
    output.mkdir(parents=True, exist_ok=True)
    clean_cell = pd.read_csv(pilot / "cell_metrics.csv")
    clean_pop = pd.read_csv(pilot / "population_metrics.csv")
    all_cell, all_pop, all_changes, all_decades, all_cases, all_confusion = [], [], [], [], [], []
    integrity = []
    for seed in manifest["seeds"]:
        prior = pilot / f"seed_{seed}"
        split = pd.read_csv(prior / "partitions.csv")
        masks = pd.read_csv(prior / "mask_manifest.csv")
        clean_views = {part: pd.read_csv(prior / f"observed_{part}.csv")
                       for part in ("fitting", "tuning", "calibration", "test")}
        for part, view in clean_views.items():
            expected_ids = split.loc[split["partition"].eq(part), "segment_id"].tolist()
            if view["segment_id"].tolist() != expected_ids:
                raise AssertionError(f"saved {part} IDs differ from partition manifest")
        test_masks = masks.loc[masks["partition"].eq("test")]
        baseline_tuning = json.loads((prior / "tuning.json").read_text(encoding="utf-8"))
        forest_tuning = json.loads((prior / "forest_tuning.json").read_text(encoding="utf-8"))
        clean_ids = clean_views["test"]["segment_id"].tolist()
        for arm in arms:
            noisy, steps = noisy_reference(clean, arm, int(seed))
            views = {part: overlay_observed(clean_views[part], noisy) for part in clean_views}
            for part, view in views.items():
                if view["segment_id"].tolist() != clean_views[part]["segment_id"].tolist():
                    raise AssertionError("partition IDs changed")
                change = changed_input_counts(clean_views[part], view, part)
                change.insert(0, "seed", seed)
                change.insert(1, "arm", arm.arm_id)
                all_changes.append(change)
            fitting, test = views["fitting"], views["test"]
            if test["segment_id"].tolist() != clean_ids or len(test_masks) != 2 * round(len(test) * manifest["mask_rate"]):
                raise AssertionError("frozen test IDs or mask count changed")
            baseline = ConditionalBaseline(baseline_tuning["material_alpha"],
                                           baseline_tuning["material_min_group"]).fit(fitting)
            year_baseline = ConditionalBaseline(baseline_tuning["material_alpha"],
                                                baseline_tuning["year_min_group"]).fit(fitting)
            conditional = baseline.predict(test, test_masks)
            year_cells = test_masks.loc[test_masks["target"].eq("installation_year")]
            year_rows = year_baseline.predict(test, year_cells)
            conditional.loc[conditional["target"].eq("installation_year"), :] = year_rows.to_numpy()
            floor = floor_predictions(fitting, test_masks)
            settings = PilotSettings(seeds=(int(seed),), mask_rate=manifest["mask_rate"],
                                     n_estimators=manifest["forest_trees"])
            forest_rows, forest = forest_predictions(fitting, test, test_masks, int(seed), settings, **forest_tuning)
            arm_dir = output / arm.arm_id / f"seed_{seed}"
            arm_dir.mkdir(parents=True, exist_ok=True)
            model_path = arm_dir / "forest.joblib"
            forest.save(model_path)
            replay = PredictiveMissForest.load(model_path)
            assert_observed_preserved(test, replay.transform(test.copy()))
            pd.testing.assert_frame_equal(forest_rows, forest_predictions_from_model(replay, test, test_masks), check_dtype=False)
            candidates = standardize_predictions(pd.concat((floor, conditional, forest_rows), ignore_index=True),
                                                 int(seed), manifest["mask_rate"])
            candidates["condition_id"] = arm.arm_id
            for _, group in candidates.groupby("method"):
                validate_predictions(group, test, test_masks)
            if arm.arm_id == "length20_negative_control":
                original = pd.read_csv(prior / "candidate_cells.csv")
                original["condition_id"] = arm.arm_id
                if not candidates["point_prediction"].astype(str).equals(original["point_prediction"].astype(str)):
                    raise AssertionError("excluded-length noise changed point predictions")
                comparison = candidates.copy()
                for column in ("fallback_reason", "evidence", "age_band_pred"):
                    comparison[column] = comparison[column].fillna("")
                    original[column] = original[column].fillna("")
                pd.testing.assert_frame_equal(comparison.drop(columns="point_prediction"),
                                              original.drop(columns="point_prediction"), check_dtype=False)
            test.to_csv(arm_dir / "observed_test.csv", index=False)
            candidates.to_csv(arm_dir / "candidate_cells.csv", index=False)
            (arm_dir / "noise_steps.json").write_text(json.dumps(steps, indent=2) + "\n", encoding="utf-8")
            # Clean answers enter only after candidates and integrity checks are saved.
            cell, population = evaluate(candidates, clean, test, test_masks)
            for table in (cell, population):
                table.insert(0, "arm", arm.arm_id)
                table.insert(0, "seed", seed)
            all_cell.append(cell)
            all_pop.append(population)
            decades = material_by_decade(candidates, test, clean)
            decades.insert(0, "arm", arm.arm_id)
            decades.insert(0, "seed", seed)
            all_decades.append(decades)
            cases = review_cases(candidates, clean)
            cases.insert(0, "arm", arm.arm_id)
            cases.insert(0, "seed", seed)
            all_cases.append(cases)
            confusion, _ = evaluation_diagnostics(candidates, clean, test_masks)
            confusion.insert(0, "arm", arm.arm_id)
            confusion.insert(0, "seed", seed)
            all_confusion.append(confusion)
            integrity.append({"seed": seed, "arm": arm.arm_id, "status": "passed",
                              "test_rows": len(test), "hidden_cells": len(test_masks),
                              "candidate_cells": len(candidates), "test_id_hash": dataframe_hash(test[["segment_id"]]),
                              "mask_hash": _sha256(prior / "mask_manifest.csv"),
                              "noisy_reference_hash": dataframe_hash(noisy)})
    cells = pd.concat(all_cell, ignore_index=True)
    population = pd.concat(all_pop, ignore_index=True)
    changes = pd.concat(all_changes, ignore_index=True)
    cells.to_csv(output / "cell_metrics.csv", index=False)
    population.to_csv(output / "population_metrics.csv", index=False)
    changes.to_csv(output / "observed_changes.csv", index=False)
    pd.concat(all_decades, ignore_index=True).to_csv(output / "material_by_decade.csv", index=False)
    pd.concat(all_confusion, ignore_index=True).to_csv(output / "material_confusion.csv", index=False)
    evaluator = output / "evaluator_only"
    evaluator.mkdir(exist_ok=True)
    pd.concat(all_cases, ignore_index=True).to_csv(evaluator / "review_cases.csv", index=False)
    clean_cell = clean_cell.loc[clean_cell["method"].ne("generator_reference")]
    paired_cell = _paired(cells, clean_cell, ["seed", "method", "target", "measure"])
    paired_pop = _paired(population, clean_pop, ["seed", "method", "measure"])
    paired_cell.to_csv(output / "paired_cell_metrics.csv", index=False)
    paired_pop.to_csv(output / "paired_population_metrics.csv", index=False)
    primary_cell = paired_cell.loc[
        (paired_cell["target"].eq("material") & paired_cell["measure"].isin(("log_loss", "macro_f1"))) |
        (paired_cell["target"].eq("installation_year") & paired_cell["measure"].eq("mae_years"))
    ].copy()
    primary_cell["metric"] = primary_cell["target"] + "_" + primary_cell["measure"]
    compact = primary_cell.pivot(index=["seed", "arm", "method"], columns="metric",
                                 values="delta_noisy_minus_clean")
    primary_pop = paired_pop.loc[paired_pop["measure"].isin(
        ("aged_50_or_older_share_error", "year_std_error", "material_expected_share_error_steel")
    ) & paired_pop["method"].ne("complete_case")]
    compact_pop = primary_pop.pivot(index=["seed", "arm", "method"], columns="measure",
                                    values="delta_noisy_minus_clean")
    compact.join(compact_pop, validate="one_to_one").reset_index().to_csv(
        output / "primary_paired_summary.csv", index=False
    )
    changes.groupby(["arm", "partition", "field"], as_index=False)[
        ["available_cells", "changed_observed_cells", "changed_observed_rate"]
    ].mean().to_csv(output / "observed_change_summary.csv", index=False)
    _plot_deltas(paired_cell, output / "plots")
    _plot_population(paired_pop, output / "plots")
    (output / "integrity_checks.json").write_text(json.dumps(integrity, indent=2) + "\n", encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps({
        "pilot_manifest_sha256": _sha256(pilot / "manifest.json"),
        "clean_truth_sha256": _sha256(pilot / "evaluator_only/clean_reference.csv"),
        "seeds": manifest["seeds"], "mask_rate": manifest["mask_rate"],
        "arms": [{"arm_id": arm.arm_id, "description": arm.description, "steps": arm.steps} for arm in arms],
        "protocol": "noise before original mask; same IDs/masks; noisy fitting refit; frozen clean-tuned settings; clean truth scored after candidate export",
        "reference_ceiling": "not used for noisy arms; clean MCAR generator reference is invalid here",
        "noise_source_sha256": _sha256(ROOT / "src/team19_asset_data/overhead_conductors/noise.py"),
        "comparison_source_sha256": _sha256(Path(__file__)),
    }, indent=2) + "\n", encoding="utf-8")
    return output
