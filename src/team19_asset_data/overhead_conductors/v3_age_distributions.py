"""Distribution-based age-population evaluation of the frozen v3 pilot."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from team19_asset_data.imputation.predictive_missforest import PredictiveMissForest

from .v3_noise_comparison import ARMS, noisy_reference, overlay_observed
from .v3_pilot import (
    SNAPSHOT_YEAR,
    ConditionalBaseline,
    _age_band,
    generator_year_pmf,
)


YEARS = np.arange(1930, 2015, dtype=int)
AGE_BANDS = ("<20", "20-34", "35-49", "50-64", "65+")
AGE_BAND_BY_YEAR = _age_band(pd.Series(YEARS)).astype(str).to_numpy()


def empirical_year_pmf(samples: np.ndarray) -> np.ndarray:
    values = np.asarray(samples, dtype=float)
    if not len(values) or not np.isfinite(values).all() or not np.all(values == np.rint(values)):
        raise ValueError("Year support must contain finite integer years")
    indices = values.astype(int) - YEARS[0]
    if np.any((indices < 0) | (indices >= len(YEARS))):
        raise ValueError("Year support lies outside the declared domain")
    counts = np.bincount(indices, minlength=len(YEARS)).astype(float)
    return counts / counts.sum()


def validate_year_support(clean_truth: pd.DataFrame) -> dict[str, int]:
    years = clean_truth["installation_year"].to_numpy(dtype=float)
    if not len(years) or not np.isfinite(years).all() or not np.all(years == np.rint(years)):
        raise ValueError("Clean truth must contain finite integer installation years")
    outside = int(((years < YEARS[0]) | (years > YEARS[-1])).sum())
    if outside:
        raise ValueError(f"Clean truth has {outside} installation years outside PMF support")
    return {"truth_min_year": int(years.min()), "truth_max_year": int(years.max()),
            "truth_outside_pmf_support": outside}


def baseline_year_pmfs(model: ConditionalBaseline, observed: pd.DataFrame, ids: pd.Series) -> np.ndarray:
    by_id = observed.set_index("segment_id")
    return np.stack([
        empirical_year_pmf(model.year_samples(by_id.loc[segment_id])[1])
        for segment_id in ids
    ])


def leaf_year_pmfs(
    model: PredictiveMissForest, fitting: pd.DataFrame,
    observed: pd.DataFrame, ids: pd.Series,
) -> np.ndarray:
    """Equal-tree mixture of empirical fitting-year distributions in matched leaves.

    A leaf contributes its observed fitting labels with total weight 1/T,
    divided equally among fitting examples in that leaf. No tree prediction
    percentile is used as a predictive interval.
    """
    target = "installation_year"
    if not fitting[target].isna().reset_index(drop=True).equals(
        model.fit_missing_mask_[target].reset_index(drop=True)
    ):
        raise ValueError("Fitting-year mask differs from the saved forest")
    frozen = model.fit_imputed_.reset_index(drop=True)
    fitting = fitting.reset_index(drop=True)
    if not frozen["segment_id"].equals(fitting["segment_id"]):
        raise ValueError("Fitting IDs differ from the saved forest")
    for column in fitting.columns.drop("segment_id"):
        known = fitting[column].notna().to_numpy()
        original = fitting.loc[known, column]
        saved = frozen.loc[known, column]
        if pd.api.types.is_numeric_dtype(original):
            same = np.allclose(original.to_numpy(dtype=float), saved.to_numpy(dtype=float), rtol=1e-9, atol=1e-9)
        else:
            same = original.astype(str).reset_index(drop=True).equals(saved.astype(str).reset_index(drop=True))
        if not same:
            raise ValueError(f"Fitting {column} differs from the saved forest")
    train_features = model.final_pass_features(fitting, target)
    query_features = model.final_pass_features(observed, target)
    train_rows = fitting[target].notna().to_numpy()
    query_lookup = pd.Series(np.arange(len(observed)), index=observed["segment_id"])
    positions = query_lookup.loc[ids].to_numpy(dtype=int)
    if observed.iloc[positions][target].notna().any():
        raise ValueError("A requested year cell is not hidden")

    pipeline = model.models_[(model.config.n_passes, target)]
    preprocessor = pipeline.named_steps["preprocessor"]
    forest = pipeline.named_steps["forest"]
    train_encoded = preprocessor.transform(train_features.loc[train_rows])
    query_encoded = preprocessor.transform(query_features.iloc[positions])
    labels = fitting.loc[train_rows, target].to_numpy(dtype=int)
    label_indices = labels - YEARS[0]
    if np.any((label_indices < 0) | (label_indices >= len(YEARS))):
        raise ValueError("Fitting labels outside the declared year domain")

    pmfs = np.zeros((len(positions), len(YEARS)), dtype=float)
    for tree in forest.estimators_:
        training_leaves = tree.apply(train_encoded)
        query_leaves = tree.apply(query_encoded)
        leaf_histograms = {}
        for leaf in np.unique(query_leaves):
            members = label_indices[training_leaves == leaf]
            if not len(members):
                raise AssertionError("Query entered a leaf with no fitting support")
            hist = np.bincount(members, minlength=len(YEARS)).astype(float)
            leaf_histograms[leaf] = hist / hist.sum()
        pmfs += np.stack([leaf_histograms[leaf] for leaf in query_leaves]) / len(forest.estimators_)
    if not np.isfinite(pmfs).all() or not np.allclose(pmfs.sum(axis=1), 1):
        raise AssertionError("Forest leaf distributions are not valid probabilities")
    return pmfs


def method_year_pmfs(
    fitting: pd.DataFrame, observed: pd.DataFrame, ids: pd.Series,
    year_min_group: int, forest: PredictiveMissForest, *, include_reference: bool,
) -> dict[str, np.ndarray]:
    floor = ConditionalBaseline(min_group=10**12).fit(fitting)
    baseline = ConditionalBaseline(min_group=year_min_group).fit(fitting)
    result = {
        "global_floor": baseline_year_pmfs(floor, observed, ids),
        "conditional_baseline": baseline_year_pmfs(baseline, observed, ids),
        "predictive_missforest": leaf_year_pmfs(forest, fitting, observed, ids),
    }
    if include_reference:
        by_id = observed.set_index("segment_id")
        result["generator_reference"] = np.stack([
            generator_year_pmf(by_id.loc[segment_id]) for segment_id in ids
        ])
    return result


def population_age_metrics(
    observed: pd.DataFrame, ids: pd.Series, pmfs: np.ndarray,
    clean_truth: pd.DataFrame,
) -> list[dict[str, float | str]]:
    if len(ids) != len(pmfs):
        raise ValueError("Distribution rows must match masked-year IDs")
    if not np.allclose(pmfs.sum(axis=1), 1):
        raise ValueError("Year distributions must sum to one")
    by_id = observed.set_index("segment_id")
    hidden = by_id.index.isin(ids)
    if by_id.loc[hidden, "installation_year"].notna().any():
        raise ValueError("Hidden-year IDs contain an observed year")
    observed_years = by_id.loc[~hidden, "installation_year"].to_numpy(dtype=int)
    observed_counts = empirical_year_pmf(observed_years) * len(observed_years)
    expected = (observed_counts + pmfs.sum(axis=0)) / len(by_id)
    actual_years = clean_truth.set_index("segment_id").loc[by_id.index, "installation_year"].to_numpy(dtype=int)
    truth = empirical_year_pmf(actual_years)
    rows: list[dict[str, float | str]] = [
        {"measure": "aged_50_or_older_share_error", "value": float(expected[YEARS <= SNAPSHOT_YEAR - 50].sum() - truth[YEARS <= SNAPSHOT_YEAR - 50].sum())},
        {"measure": "observed_aged_50_or_older_count", "value": float((observed_years <= SNAPSHOT_YEAR - 50).sum())},
        {"measure": "expected_hidden_aged_50_or_older_count", "value": float(pmfs[:, YEARS <= SNAPSHOT_YEAR - 50].sum())},
        {"measure": "year_wasserstein", "value": float(wasserstein_distance(YEARS, YEARS, u_weights=expected, v_weights=truth))},
    ]
    for band in AGE_BANDS:
        selected = AGE_BAND_BY_YEAR == band
        rows.append({"measure": f"age_band_{band}_share_error", "value": float(expected[selected].sum() - truth[selected].sum())})
    return rows


def calibration_rows(
    ids: pd.Series, pmfs: np.ndarray, clean_truth: pd.DataFrame,
    *, seed: int, method: str,
) -> tuple[list[dict], list[dict]]:
    probabilities = pmfs[:, YEARS <= SNAPSHOT_YEAR - 50].sum(axis=1)
    truth = clean_truth.set_index("segment_id").loc[ids, "installation_year"].to_numpy(dtype=int)
    outcomes = (truth <= SNAPSHOT_YEAR - 50).astype(int)
    details = [
        {"seed": seed, "method": method, "segment_id": segment_id,
         "p_aged_50_or_older": float(probability), "actual_aged_50_or_older": int(actual)}
        for segment_id, probability, actual in zip(ids, probabilities, outcomes)
    ]
    summary = []
    for lower, upper in zip(np.linspace(0, 1, 6)[:-1], np.linspace(0, 1, 6)[1:]):
        selected = (probabilities >= lower) & (probabilities < upper if upper < 1 else probabilities <= upper)
        if not selected.any():
            continue
        summary.append({"seed": seed, "method": method, "bin_lower": lower, "bin_upper": upper,
                        "n": int(selected.sum()), "mean_predicted": float(probabilities[selected].mean()),
                        "observed_frequency": float(outcomes[selected].mean()),
                        "brier": float(np.mean((probabilities[selected] - outcomes[selected]) ** 2))})
    return details, summary


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_age_distributions(
    pilot_dir: str | Path, output_dir: str | Path, noise_dir: str | Path | None = None,
    arm_ids: tuple[str, ...] | None = None,
) -> Path:
    pilot = Path(pilot_dir).resolve()
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite {output}")
    manifest = json.loads((pilot / "manifest.json").read_text(encoding="utf-8"))
    clean_truth = pd.read_csv(pilot / "evaluator_only/clean_reference.csv")
    year_support = validate_year_support(clean_truth)
    seeds = tuple(int(value) for value in manifest["seeds"])
    selected_arms = tuple(arm for arm in ARMS if arm_ids is None or arm.arm_id in arm_ids)
    if noise_dir is not None and not selected_arms:
        raise ValueError("No matching noise arms")
    output.mkdir(parents=True, exist_ok=True)

    population_rows, calibration_details, calibration_bins = [], [], []
    checks = []
    for seed in seeds:
        run = pilot / f"seed_{seed}"
        fitting = pd.read_csv(run / "observed_fitting.csv")
        calibration = pd.read_csv(run / "observed_calibration.csv")
        test = pd.read_csv(run / "observed_test.csv")
        masks = pd.read_csv(run / "mask_manifest.csv")
        year_min_group = json.loads((run / "tuning.json").read_text(encoding="utf-8"))["year_min_group"]
        conditions = [("clean", fitting, calibration, test, PredictiveMissForest.load(run / "forest.joblib"), True)]
        if noise_dir is not None:
            for arm in selected_arms:
                noisy_complete, _ = noisy_reference(clean_truth, arm, seed)
                noisy_fitting = overlay_observed(fitting, noisy_complete)
                noisy_calibration = overlay_observed(calibration, noisy_complete)
                noisy_test = overlay_observed(test, noisy_complete)
                saved = Path(noise_dir) / arm.arm_id / f"seed_{seed}"
                pd.testing.assert_frame_equal(
                    noisy_test, pd.read_csv(saved / "observed_test.csv"), check_dtype=False,
                )
                conditions.append((arm.arm_id, noisy_fitting, noisy_calibration, noisy_test,
                                   PredictiveMissForest.load(saved / "forest.joblib"), False))

        for condition, fit_view, calibration_view, test_view, forest, include_reference in conditions:
            for partition, observed_view in (("test", test_view), ("calibration", calibration_view)):
                ids = masks.loc[
                    masks["partition"].eq(partition) & masks["target"].eq("installation_year"),
                    "segment_id",
                ].reset_index(drop=True)
                if len(ids) != int(round(len(observed_view) * manifest["mask_rate"])):
                    raise AssertionError("Frozen year-mask count changed")
                distributions = method_year_pmfs(
                    fit_view, observed_view, ids, year_min_group, forest,
                    include_reference=include_reference,
                )
                saved_pmfs = []
                for method, pmfs in distributions.items():
                    if partition == "test":
                        saved_pmfs.append(pd.DataFrame({
                            "seed": seed, "condition": condition, "method": method,
                            "segment_id": ids.to_numpy(),
                            **{f"year_{year}": pmfs[:, index] for index, year in enumerate(YEARS)},
                        }))
                        for metric in population_age_metrics(observed_view, ids, pmfs, clean_truth):
                            population_rows.append({"seed": seed, "condition": condition,
                                                    "method": method, **metric})
                    else:
                        details, bins = calibration_rows(ids, pmfs, clean_truth, seed=seed, method=method)
                        for row in details:
                            row["condition"] = condition
                        for row in bins:
                            row["condition"] = condition
                        calibration_details.extend(details)
                        calibration_bins.extend(bins)
                if saved_pmfs:
                    destination = output / condition / f"seed_{seed}"
                    destination.mkdir(parents=True, exist_ok=True)
                    pd.concat(saved_pmfs, ignore_index=True).to_csv(
                        destination / "predicted_year_pmfs_test.csv.gz", index=False,
                        compression="gzip", float_format="%.12g",
                    )
                checks.append({"seed": seed, "condition": condition, "partition": partition,
                               "hidden_year_cells": len(ids), "methods": sorted(distributions),
                               "pmf_rows_valid": True})

    pd.DataFrame(population_rows).to_csv(output / "population_age_metrics.csv", index=False)
    evaluator_only = output / "evaluator_only"
    evaluator_only.mkdir(exist_ok=True)
    pd.DataFrame(calibration_details).to_csv(evaluator_only / "calibration_cells.csv", index=False)
    pd.DataFrame(calibration_bins).to_csv(evaluator_only / "calibration_bins.csv", index=False)
    (output / "integrity_checks.json").write_text(json.dumps(checks, indent=2) + "\n", encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps({
        "pilot_manifest_sha256": _sha256(pilot / "manifest.json"),
        "noise_manifest_sha256": _sha256(Path(noise_dir) / "manifest.json") if noise_dir else None,
        "seeds": seeds, "conditions": ["clean"] + ([arm.arm_id for arm in selected_arms] if noise_dir else []),
        "year_domain": [int(YEARS[0]), int(YEARS[-1])],
        "year_support_check": year_support,
        "point_predictions_changed": False,
        "generator_reference_scope": "clean MCAR only; evaluator-only conditional law",
    }, indent=2) + "\n", encoding="utf-8")
    return output
