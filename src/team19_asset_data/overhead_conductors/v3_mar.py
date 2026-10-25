"""Predeclared MAR year masks and paired age-distribution evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from team19_asset_data.imputation.predictive_missforest import PredictiveMissForest

from .v3_age_distributions import (
    AGE_BAND_BY_YEAR, AGE_BANDS, YEARS, calibration_rows, method_year_pmfs,
    population_age_metrics, validate_year_support,
)
from .v3_pilot import (
    SNAPSHOT_YEAR, ConditionalBaseline, PilotSettings, _sha256,
    forest_predictions, forest_predictions_from_model,
)


@dataclass(frozen=True)
class MarArm:
    arm_id: str
    driver: str
    probabilities: tuple[tuple[str, float], ...]
    unavailable_probability: float | None = None

    def probability_map(self) -> dict[str, float]:
        return dict(self.probabilities)


ARMS = (
    MarArm("voltage_mar", "voltage_category", (
        ("19 kV SWER", .35), ("11/7.6 kV", .27), ("LV", .12),
        ("33 kV", .55), ("66 kV", .35),
    )),
    MarArm("location_mar", "location_group", (
        ("Rural", .38), ("Regional", .18), ("Urban", .25),
    )),
    MarArm("covering_mar", "covering", (("Bare", .294), ("Covered", .80))),
    MarArm("material_mar", "material", (
        ("Steel", .19), ("Copper", .48), ("Aluminium", .75),
    ), unavailable_probability=.30),
)
PARTITIONS = ("fitting", "tuning", "calibration", "test")


def _driver_values(frame: pd.DataFrame, arm: MarArm) -> pd.Series:
    values = frame[arm.driver].astype("string").fillna("<unavailable>")
    allowed = set(arm.probability_map())
    if arm.unavailable_probability is not None:
        allowed.add("<unavailable>")
    unexpected = set(values.unique()) - allowed
    if unexpected:
        raise ValueError(f"Unexpected {arm.driver} values: {sorted(unexpected)}")
    return values


def make_mar_view(
    clean_truth: pd.DataFrame, saved_observed: pd.DataFrame,
    arm: MarArm, *, seed: int, arm_index: int, partition_index: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Replace only the saved year mask; draw from observed, non-year fields."""
    if saved_observed["segment_id"].duplicated().any():
        raise ValueError("Duplicate observed segment IDs")
    truth = clean_truth.set_index("segment_id")
    if not saved_observed["segment_id"].isin(truth.index).all():
        raise ValueError("Observed IDs are absent from clean truth")
    observed = saved_observed.copy().reset_index(drop=True)
    observed["installation_year"] = truth.loc[
        observed["segment_id"], "installation_year"
    ].to_numpy()
    driver = _driver_values(observed, arm)
    rate_map = arm.probability_map()
    if arm.unavailable_probability is not None:
        rate_map["<unavailable>"] = arm.unavailable_probability
    probabilities = driver.map(rate_map).to_numpy(dtype=float)
    if not np.isfinite(probabilities).all() or np.any((probabilities <= 0) | (probabilities >= 1)):
        raise ValueError("MAR probabilities must be finite and strictly between zero and one")
    rng = np.random.default_rng(np.random.SeedSequence([seed, 192, arm_index, partition_index]))
    hidden = rng.random(len(observed)) < probabilities
    observed.loc[hidden, "installation_year"] = np.nan
    if not saved_observed.drop(columns="installation_year").equals(
        observed.drop(columns="installation_year")
    ):
        raise AssertionError("MAR generation changed a non-year observed field")
    mask = pd.DataFrame({"segment_id": observed.loc[hidden, "segment_id"].to_numpy(),
                         "target": "installation_year"})
    counts = pd.DataFrame({"driver_value": driver, "hidden": hidden,
                           "probability": probabilities}).groupby("driver_value", sort=True).agg(
        n_rows=("hidden", "size"), n_hidden=("hidden", "sum"),
        assigned_probability=("probability", "first"),
    ).reset_index()
    counts["actual_rate"] = counts["n_hidden"] / counts["n_rows"]
    return observed, mask, counts


def complete_case_age_metrics(observed: pd.DataFrame, clean_truth: pd.DataFrame) -> list[dict]:
    known = observed["installation_year"].dropna().to_numpy(dtype=int)
    if not len(known):
        raise ValueError("No complete-case years")
    years = clean_truth.set_index("segment_id").loc[
        observed["segment_id"], "installation_year"
    ].to_numpy(dtype=int)
    rows = [{"measure": "aged_50_or_older_share_error",
             "value": float((known <= SNAPSHOT_YEAR - 50).mean() - (years <= SNAPSHOT_YEAR - 50).mean())},
            {"measure": "year_wasserstein",
             "value": float(wasserstein_distance(known, years))}]
    for band in AGE_BANDS:
        selected = YEARS[AGE_BAND_BY_YEAR == band]
        rows.append({"measure": f"age_band_{band}_share_error",
                     "value": float(np.isin(known, selected).mean() - np.isin(years, selected).mean())})
    return rows


def run_mar_evaluation(
    pilot_dir: str | Path, output_dir: str | Path,
    arms: tuple[MarArm, ...] = ARMS,
) -> Path:
    pilot = Path(pilot_dir).resolve()
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite {output}")
    if not arms or len({arm.arm_id for arm in arms}) != len(arms):
        raise ValueError("MAR arms must be nonempty and have unique IDs")
    manifest = json.loads((pilot / "manifest.json").read_text(encoding="utf-8"))
    clean_truth_path = pilot / "evaluator_only/clean_reference.csv"
    if _sha256(clean_truth_path) != manifest["file_sha256"]["clean_reference"]:
        raise ValueError("Clean reference hash differs from the frozen pilot")
    truth = pd.read_csv(clean_truth_path)
    support = validate_year_support(truth)
    output.mkdir(parents=True, exist_ok=True)
    rows, group_rows, mask_counts, calibration_details, calibration_bins, cell_rows, checks = [], [], [], [], [], [], []

    for seed in manifest["seeds"]:
        prior = pilot / f"seed_{seed}"
        partitions = pd.read_csv(prior / "partitions.csv")
        saved_masks = pd.read_csv(prior / "mask_manifest.csv")
        saved_views = {part: pd.read_csv(prior / f"observed_{part}.csv") for part in PARTITIONS}
        baseline_tuning = json.loads((prior / "tuning.json").read_text(encoding="utf-8"))
        forest_tuning = json.loads((prior / "forest_tuning.json").read_text(encoding="utf-8"))
        for part, view in saved_views.items():
            expected_ids = partitions.loc[partitions["partition"].eq(part), "segment_id"].tolist()
            if view["segment_id"].tolist() != expected_ids:
                raise ValueError(f"Saved {part} IDs differ from frozen partition manifest")

        for arm in arms:
            arm_index = next(i for i, declared in enumerate(ARMS) if declared.arm_id == arm.arm_id)
            arm_dir = output / arm.arm_id / f"seed_{seed}"
            arm_dir.mkdir(parents=True, exist_ok=True)
            views, masks = {}, {}
            for part_index, part in enumerate(PARTITIONS):
                view, mask, counts = make_mar_view(
                    truth, saved_views[part], arm,
                    seed=int(seed), arm_index=arm_index, partition_index=part_index,
                )
                views[part], masks[part] = view, mask
                counts.insert(0, "partition", part)
                counts.insert(0, "driver", arm.driver)
                counts.insert(0, "arm", arm.arm_id)
                counts.insert(0, "seed", seed)
                mask_counts.append(counts)
                view.to_csv(arm_dir / f"observed_{part}.csv", index=False)
                mask.to_csv(arm_dir / f"year_mask_{part}.csv", index=False)
            for part, view in views.items():
                original_material_ids = set(saved_masks.loc[
                    saved_masks["partition"].eq(part) & saved_masks["target"].eq("material"), "segment_id"
                ])
                if set(view.loc[view["material"].isna(), "segment_id"]) != original_material_ids:
                    raise AssertionError("Frozen material mask changed")

            fit_view, test_view = views["fitting"], views["test"]
            settings = PilotSettings(
                seeds=(int(seed),), mask_rate=float(manifest["mask_rate"]),
                n_estimators=int(manifest["forest_trees"]),
            )
            forest_candidates, forest = forest_predictions(
                fit_view, test_view, masks["test"], int(seed), settings, **forest_tuning,
            )
            model_path = arm_dir / "forest.joblib"
            forest.save(model_path)
            forest = PredictiveMissForest.load(model_path)
            pd.testing.assert_frame_equal(
                forest_candidates,
                forest_predictions_from_model(forest, test_view, masks["test"]),
                check_dtype=False,
            )
            for part in ("test", "calibration"):
                view = views[part]
                hidden_ids = masks[part]["segment_id"].reset_index(drop=True)
                pmfs = method_year_pmfs(
                    fit_view, view, hidden_ids, int(baseline_tuning["year_min_group"]),
                    forest, include_reference=True,
                )
                if part == "test":
                    lookup = view.set_index("segment_id")
                    floor_model = ConditionalBaseline(min_group=10**12).fit(fit_view)
                    baseline_model = ConditionalBaseline(
                        min_group=int(baseline_tuning["year_min_group"])
                    ).fit(fit_view)
                    candidates = []
                    for method, model in (("global_floor", floor_model),
                                          ("conditional_baseline", baseline_model)):
                        candidates.extend({
                            "method": method, "segment_id": segment_id,
                            "point_year": model.year(lookup.loc[segment_id])["prediction"],
                        } for segment_id in hidden_ids)
                    candidates.extend({
                        "method": "predictive_missforest", "segment_id": row.segment_id,
                        "point_year": int(row.point_prediction),
                    } for row in forest_candidates.itertuples(index=False))
                    candidate_table = pd.DataFrame(candidates)
                    if candidate_table.duplicated(["method", "segment_id"]).any():
                        raise AssertionError("Duplicate year candidate")
                    candidate_table.to_csv(arm_dir / "candidate_years_test.csv", index=False)
                    truth_by_id = truth.set_index("segment_id")["installation_year"]
                    for method, group in candidate_table.groupby("method", sort=False):
                        actual = truth_by_id.loc[group["segment_id"]].to_numpy(dtype=float)
                        predicted = group["point_year"].to_numpy(dtype=float)
                        cell_rows.append({
                            "seed": seed, "arm": arm.arm_id, "method": method,
                            "hidden_year_cells": len(group),
                            "year_mae": float(np.mean(np.abs(predicted - actual))),
                            "year_signed_bias": float(np.mean(predicted - actual)),
                        })
                    for metric in complete_case_age_metrics(view, truth):
                        rows.append({"seed": seed, "arm": arm.arm_id, "method": "complete_case", **metric})
                    for group_field in ("voltage_category", "location_group", "covering", "material"):
                        groups = view[group_field].astype("string").fillna("<unavailable>")
                        for value in sorted(groups.unique()):
                            subview = view.loc[groups.eq(value)].reset_index(drop=True)
                            n_hidden = int(subview["installation_year"].isna().sum())
                            if n_hidden == len(subview):
                                continue
                            group_rows.append({
                                "seed": seed, "arm": arm.arm_id, "method": "complete_case",
                                "group_field": group_field, "group_value": value,
                                "n_rows": len(subview), "n_hidden": n_hidden,
                                **next(metric for metric in complete_case_age_metrics(subview, truth)
                                       if metric["measure"] == "aged_50_or_older_share_error"),
                            })
                    saved_pmfs = []
                    for method, pmf in pmfs.items():
                        for metric in population_age_metrics(view, hidden_ids, pmf, truth):
                            rows.append({"seed": seed, "arm": arm.arm_id, "method": method, **metric})
                        if method != "generator_reference":
                            saved_pmfs.append(pd.DataFrame({
                                "method": method, "segment_id": hidden_ids.to_numpy(),
                                **{f"year_{year}": pmf[:, i] for i, year in enumerate(YEARS)},
                            }))
                        for group_field in ("voltage_category", "location_group", "covering", "material"):
                            groups = view[group_field].astype("string").fillna("<unavailable>")
                            for value in sorted(groups.unique()):
                                subview = view.loc[groups.eq(value)].reset_index(drop=True)
                                selected = hidden_ids.isin(subview["segment_id"]).to_numpy()
                                if not selected.any():
                                    continue
                                subset_ids = hidden_ids.loc[selected].reset_index(drop=True)
                                for metric in population_age_metrics(subview, subset_ids, pmf[selected], truth):
                                    if metric["measure"] == "aged_50_or_older_share_error":
                                        group_rows.append({"seed": seed, "arm": arm.arm_id, "method": method,
                                                           "group_field": group_field, "group_value": value,
                                                           "n_rows": len(subview), "n_hidden": len(subset_ids), **metric})
                    pd.concat(saved_pmfs, ignore_index=True).to_csv(
                        arm_dir / "predicted_year_pmfs_test.csv.gz", index=False,
                        compression="gzip", float_format="%.12g",
                    )
                else:
                    for method, pmf in pmfs.items():
                        details, bins = calibration_rows(hidden_ids, pmf, truth,
                                                         seed=int(seed), method=method)
                        for row in details:
                            row["arm"] = arm.arm_id
                        for row in bins:
                            row["arm"] = arm.arm_id
                        calibration_details.extend(details)
                        calibration_bins.extend(bins)
                checks.append({"seed": seed, "arm": arm.arm_id, "partition": part,
                               "rows": len(view), "hidden_years": len(hidden_ids),
                               "methods": sorted(pmfs), "status": "passed"})

    pd.DataFrame(rows).to_csv(output / "population_age_metrics.csv", index=False)
    pd.DataFrame(cell_rows).to_csv(output / "hidden_year_cell_metrics.csv", index=False)
    pd.DataFrame(group_rows).to_csv(output / "test_group_age_metrics.csv", index=False)
    pd.concat(mask_counts, ignore_index=True).to_csv(output / "mask_counts.csv", index=False)
    evaluator = output / "evaluator_only"
    evaluator.mkdir(exist_ok=True)
    pd.DataFrame(calibration_details).to_csv(evaluator / "calibration_cells.csv", index=False)
    pd.DataFrame(calibration_bins).to_csv(evaluator / "calibration_bins.csv", index=False)
    (output / "integrity_checks.json").write_text(json.dumps(checks, indent=2) + "\n", encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps({
        "pilot_manifest_sha256": _sha256(pilot / "manifest.json"),
        "clean_reference_sha256": _sha256(clean_truth_path),
        "year_support_check": support,
        "seeds": list(manifest["seeds"]),
        "arms": [{"arm_id": arm.arm_id, "driver": arm.driver,
                  "probabilities": arm.probability_map(),
                  "unavailable_probability": arm.unavailable_probability} for arm in arms],
        "partition_and_material_masks": "reused from frozen pilot",
        "forest_and_baseline_settings": "reused from clean tuning; refitted on MAR fitting views",
        "generator_reference_scope": "evaluator-only; MAR on recorded predictors",
    }, indent=2) + "\n", encoding="utf-8")
    return output
