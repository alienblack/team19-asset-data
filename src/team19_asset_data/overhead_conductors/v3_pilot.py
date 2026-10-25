"""Phase 1 overhead-conductor pilot from the revised methodology (v3).

The models receive observed seven-field views. Clean answers enter only the
scorer, after candidate tables and integrity checks have been written.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from matplotlib import pyplot as plt
from scipy.stats import wasserstein_distance
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, mean_absolute_error, recall_score
from sklearn.model_selection import train_test_split

from team19_asset_data.imputation.predictive_missforest import (
    PredictiveMissForest,
    PredictiveMissForestConfig,
)

from .generate_synthetic_data import Config, StrengthConfig, calibrate_route_length_marginals, generate_dataset
from .input_preparation import MODEL_FIELDS, audit_reference


MATERIALS = ("Steel", "Copper", "Aluminium")
TARGETS = ("material", "installation_year")
SNAPSHOT_YEAR = 2014
ROOT = Path(__file__).resolve().parents[3]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class PilotSettings:
    seeds: tuple[int, ...] = (101, 102, 103, 104, 105)
    mask_rate: float = 0.30
    n_estimators: int = 24
    n_passes: int = 2

    def validate(self) -> None:
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be a nonempty list of distinct integers")
        if not 0 < self.mask_rate < 1:
            raise ValueError("mask_rate must be between zero and one")
        if self.n_estimators < 1 or self.n_passes < 1:
            raise ValueError("forest settings must be positive")


def make_reference(n_rows: int = 10_000) -> pd.DataFrame:
    """Generate the documented seed-219, strength-100 complete population."""
    config = Config.from_yaml(ROOT / "config/overhead_conductors/generation.yaml")
    if n_rows != config.n_rows:
        from dataclasses import replace
        config = replace(config, n_rows=n_rows)
    strengths = StrengthConfig.from_yaml(
        ROOT / "config/overhead_conductors/relationship_strength/strength_100.yaml"
    )
    reference = calibrate_route_length_marginals(generate_dataset(config, strengths), config)
    audit_reference(reference)
    return reference


def _age_band(year: pd.Series) -> pd.Series:
    age = SNAPSHOT_YEAR - pd.to_numeric(year)
    return pd.cut(age, [-1, 19, 34, 49, 64, 200], labels=["<20", "20-34", "35-49", "50-64", "65+"])


def freeze_split(reference: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Stratify by material and age band before any learned transformation."""
    strata = reference["material"].astype(str) + "|" + _age_band(reference["installation_year"]).astype(str)
    ids = reference["segment_id"].astype(str).to_numpy()
    fit, rest = train_test_split(ids, train_size=0.60, random_state=seed, stratify=strata)
    strata_by_id = pd.Series(strata.to_numpy(), index=ids)
    material_by_id = pd.Series(reference["material"].to_numpy(), index=ids)
    rest_strata = strata_by_id.loc[rest]
    if rest_strata.value_counts().min() < 2:
        rest_strata = material_by_id.loc[rest]
    tuning, remaining = train_test_split(rest, train_size=0.25, random_state=seed + 1, stratify=rest_strata)
    rem_strata = strata_by_id.loc[remaining]
    if rem_strata.value_counts().min() < 2:
        rem_strata = material_by_id.loc[remaining]
    calibration, test = train_test_split(
        remaining, train_size=1 / 3, random_state=seed + 2, stratify=rem_strata
    )
    lookup = {str(segment): part for part, group in (
        ("fitting", fit), ("tuning", tuning), ("calibration", calibration), ("test", test)
    ) for segment in group}
    split = pd.DataFrame({"segment_id": ids, "partition": [lookup[item] for item in ids]})
    if split["partition"].value_counts().to_dict() != {
        "fitting": int(len(ids) * .60), "tuning": int(len(ids) * .10),
        "calibration": int(len(ids) * .10), "test": int(len(ids) * .20),
    }:
        raise AssertionError("unexpected split sizes")
    return split


def mask_targets(reference: pd.DataFrame, split: pd.DataFrame, seed: int, rate: float) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Hide only the two targets, independently within each partition."""
    merged = reference.merge(split, on="segment_id", validate="one_to_one")
    views: dict[str, pd.DataFrame] = {}
    manifest: list[dict[str, object]] = []
    for part_index, part in enumerate(("fitting", "tuning", "calibration", "test")):
        rows = merged.loc[merged["partition"].eq(part), ["segment_id", *MODEL_FIELDS]].copy().reset_index(drop=True)
        for target_index, target in enumerate(TARGETS):
            rng = np.random.default_rng(np.random.SeedSequence([seed, part_index, target_index]))
            chosen = rng.choice(len(rows), size=round(len(rows) * rate), replace=False)
            for pos in chosen:
                manifest.append({"segment_id": rows.at[pos, "segment_id"], "partition": part, "target": target})
            rows.loc[chosen, target] = np.nan
        views[part] = rows
    masks = pd.DataFrame(manifest)
    if masks.duplicated(["segment_id", "target"]).any():
        raise AssertionError("duplicate hidden-cell key")
    return views, masks


def _material_probabilities(rows: pd.DataFrame) -> np.ndarray:
    counts = rows["material"].value_counts().reindex(MATERIALS, fill_value=0).to_numpy(dtype=float)
    if counts.sum() == 0:
        raise ValueError("no observed fitting material")
    return counts / counts.sum()


def material_log_loss(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """Score probabilities in the declared Steel/Copper/Aluminium order."""
    indices = np.array([MATERIALS.index(str(label)) for label in labels], dtype=int)
    if len(indices) != len(probabilities) or probabilities.shape[1] != len(MATERIALS):
        raise ValueError("material labels and probability rows do not match")
    return float(-np.mean(np.log(np.clip(probabilities[np.arange(len(indices)), indices], 1e-12, 1))))


def configured_material_probabilities(year: int, voltage: str, corrosion: str, config: Config, strength: StrengthConfig) -> np.ndarray:
    """Exact material law used by sample_material, including its clipping."""
    material_era = "pre_1950" if year <= 1949 else "from_1980" if year >= 1980 else "1950_1979"
    expected = dict(config.material_target)
    for effects, key, strength_key in (
        (config.material.get("voltage_effects", {}), voltage, "voltage_to_material"),
        (config.material.get("era_effects", {}), material_era, "installation_year_to_material"),
        (config.material.get("corrosion_effects", {}), corrosion, "corrosion_to_material"),
    ):
        for label, delta in effects.get(key, {}).items():
            expected[label] += float(delta) * strength.get(strength_key)
    probabilities = np.clip([expected[label] for label in MATERIALS], .02, None)
    return probabilities / probabilities.sum()


def exact_year_prior() -> tuple[np.ndarray, np.ndarray]:
    """Integrate the branches of sample_installation_year analytically."""
    years = np.arange(1930, 2015)
    probabilities = np.zeros(len(years), dtype=float)
    for year, weight in zip((1955, 1956, 1958), (.36, .29, .35)):
        probabilities[year - 1930] += .59 * .48 * weight
    probabilities[(years >= 1948) & (years <= 1963)] += .59 * .52 * .78 / 16
    probabilities[(years >= 1930) & (years <= 1947)] += .59 * .52 * .22 / 18
    probabilities[1966 - 1930] += .39 * .28
    probabilities[(years >= 1964) & (years <= 1994)] += .39 * .72 / 31
    probabilities[(years >= 1995) & (years <= 2014)] += .02 / 20
    if not np.isclose(probabilities.sum(), 1):
        raise AssertionError("year prior does not sum to one")
    return years, probabilities


def _voltage_era(year: int) -> str:
    if year <= 1949:
        return "pre_1950"
    if year <= 1977:
        return "1950_1977"
    if year <= 1994:
        return "1978_1994"
    return "from_1995"


def _covered_probability(year: int) -> float:
    return .08 if year >= 1995 else .035 if year >= 1980 else .008


def generator_reference_predictions(observed: pd.DataFrame, masks: pd.DataFrame) -> pd.DataFrame:
    """Exact Phase 1 conditional reference, computed only by the evaluator.

    Given voltage, location and corrosion, location and sea distance add no
    information about year under this generator. Conductor type and length are
    excluded as descendants of material. MCAR masking adds no selection term.
    """
    config = Config.from_yaml(ROOT / "config/overhead_conductors/generation.yaml")
    strength = StrengthConfig.from_yaml(
        ROOT / "config/overhead_conductors/relationship_strength/strength_100.yaml"
    )
    years, prior = exact_year_prior()
    by_id = observed.set_index("segment_id")
    records = []
    for cell in masks.itertuples(index=False):
        row = by_id.loc[cell.segment_id]
        voltage, corrosion, covering = row["voltage_category"], row["corrosion_zone"], row["covering"]
        voltage_likelihood = np.array([
            config.voltage_distribution_for_era(_voltage_era(int(year)))[voltage]
            for year in years
        ], dtype=float)
        cover_likelihood = np.array([
            _covered_probability(int(year)) if covering == "Covered" else 1 - _covered_probability(int(year))
            for year in years
        ], dtype=float)
        posterior = prior * voltage_likelihood * cover_likelihood
        material_likelihoods = np.stack([
            configured_material_probabilities(int(year), voltage, corrosion, config, strength)
            for year in years
        ])
        record = {"segment_id": cell.segment_id, "target": cell.target,
                  "method": "generator_reference", "evidence": "configured_generator_conditional",
                  "fallback_reason": "", "p_steel": np.nan, "p_copper": np.nan,
                  "p_aluminium": np.nan, "year_q10": np.nan, "year_q50": np.nan,
                  "year_q90": np.nan, "max_vote_share": np.nan, "tree_std": np.nan}
        if cell.target == "material":
            if pd.isna(row["installation_year"]):
                posterior /= posterior.sum()
                probabilities = posterior @ material_likelihoods
            else:
                probabilities = material_likelihoods[int(row["installation_year"]) - 1930]
            record["point_prediction"] = MATERIALS[int(np.argmax(probabilities))]
            record.update(zip(("p_steel", "p_copper", "p_aluminium"), probabilities))
        else:
            if pd.notna(row["material"]):
                posterior *= material_likelihoods[:, MATERIALS.index(str(row["material"]))]
            posterior /= posterior.sum()
            cdf = np.cumsum(posterior)
            quantiles = [int(years[np.searchsorted(cdf, quantile)]) for quantile in (.10, .50, .90)]
            record["point_prediction"] = quantiles[1]
            record.update(zip(("year_q10", "year_q50", "year_q90"), quantiles))
        records.append(record)
    return pd.DataFrame(records)


def material_generator_check(reference: pd.DataFrame) -> pd.DataFrame:
    """Compare finite-sample groups to the configured material probabilities."""
    config = Config.from_yaml(ROOT / "config/overhead_conductors/generation.yaml")
    strength = StrengthConfig.from_yaml(
        ROOT / "config/overhead_conductors/relationship_strength/strength_100.yaml"
    )
    era = np.select(
        [reference["installation_year"].le(1949), reference["installation_year"].ge(1980)],
        ["pre_1950", "from_1980"], default="1950_1979",
    )
    grouped = reference.assign(material_era=era).groupby(
        ["voltage_category", "material_era", "corrosion_zone"], observed=True
    )
    rows = []
    for (voltage, material_era, corrosion), group in grouped:
        representative_year = 1940 if material_era == "pre_1950" else 1990 if material_era == "from_1980" else 1960
        probabilities = configured_material_probabilities(representative_year, voltage, corrosion, config, strength)
        for label, probability in zip(MATERIALS, probabilities):
            observed_share = group["material"].eq(label).mean()
            standard_error = np.sqrt(probability * (1 - probability) / len(group))
            rows.append({"voltage_category": voltage, "material_era": material_era,
                         "corrosion_zone": corrosion, "material": label, "n": len(group),
                         "configured_probability": float(probability), "observed_share": float(observed_share),
                         "sampling_standard_error": float(standard_error),
                         "standardized_deviation": float((observed_share - probability) / standard_error) if standard_error else np.nan})
    return pd.DataFrame(rows)


class ConditionalBaseline:
    """Smoothed material groups and hierarchical year-distribution backoff."""

    def __init__(self, alpha: float = 10.0, min_group: int = 20):
        self.alpha = float(alpha)
        self.min_group = int(min_group)
        if self.alpha < 0 or self.min_group < 1:
            raise ValueError("invalid smoothing or group size")

    def fit(self, fitting_observed: pd.DataFrame) -> "ConditionalBaseline":
        frame = fitting_observed.loc[:, MODEL_FIELDS].copy()
        observed_years = frame["installation_year"].dropna().astype(float)
        if observed_years.empty:
            raise ValueError("no fitting years")
        self.era_edges = np.unique(np.quantile(observed_years, [1 / 3, 2 / 3]))
        frame["learned_era"] = self._era(frame["installation_year"])
        self.frame = frame
        self.global_material = _material_probabilities(frame)
        self.global_years = observed_years.to_numpy()
        self.material_levels = (
            ("voltage_category",),
            ("voltage_category", "learned_era"),
            ("voltage_category", "learned_era", "corrosion_zone"),
        )
        self.year_levels = (
            ("voltage_category",),
            ("voltage_category", "covering"),
            ("voltage_category", "covering", "material"),
        )
        self.material_groups = []
        for keys in self.material_levels:
            groups = {}
            group_key_columns = keys[0] if len(keys) == 1 else list(keys)
            for group_key, group in frame.dropna(subset=["material", *keys]).groupby(group_key_columns):
                key = group_key if isinstance(group_key, tuple) else (group_key,)
                counts = group["material"].value_counts().reindex(MATERIALS, fill_value=0).to_numpy(dtype=float)
                groups[key] = (len(group), counts)
            self.material_groups.append(groups)
        self.year_groups = []
        for keys in self.year_levels:
            groups = {}
            group_key_columns = keys[0] if len(keys) == 1 else list(keys)
            for group_key, group in frame.dropna(subset=["installation_year", *keys]).groupby(group_key_columns):
                key = group_key if isinstance(group_key, tuple) else (group_key,)
                groups[key] = group["installation_year"].to_numpy(dtype=float)
            self.year_groups.append(groups)
        return self

    def _era(self, years: pd.Series) -> pd.Series:
        values = pd.to_numeric(years, errors="coerce")
        result = pd.Series(np.nan, index=years.index, dtype=object)
        valid = values.notna()
        result.loc[valid] = np.searchsorted(self.era_edges, values.loc[valid], side="right").astype(str)
        return result

    def material(self, row: pd.Series) -> dict[str, object]:
        parent = self.global_material.copy()
        selected = ("global", int(self.frame["material"].notna().sum()), parent.copy())
        fallback_reason = ""
        for keys, groups in zip(self.material_levels, self.material_groups):
            if any(pd.isna(row[field]) for field in keys):
                fallback_reason = "grouping_field_missing"
                break
            key = tuple(row[field] for field in keys)
            n, counts = groups.get(key, (0, None))
            if n < self.min_group:
                fallback_reason = "group_below_minimum_support" if n else "group_unavailable"
                break
            parent = (counts + self.alpha * parent) / (n + self.alpha)
            selected = ("x".join(keys), n, parent.copy())
        level, n, probs = selected
        return {"prediction": MATERIALS[int(np.argmax(probs))], "probabilities": probs,
                "evidence": f"{level};n={n}", "fallback_reason": fallback_reason}

    def year(self, row: pd.Series) -> dict[str, object]:
        selected = ("global", self.global_years)
        fallback_reason = ""
        for keys, groups in zip(self.year_levels, self.year_groups):
            if any(pd.isna(row[field]) for field in keys):
                fallback_reason = "grouping_field_missing"
                break
            key = tuple(row[field] for field in keys)
            years = groups.get(key, np.empty(0))
            if len(years) < self.min_group:
                fallback_reason = "group_below_minimum_support" if len(years) else "group_unavailable"
                break
            selected = ("x".join(keys), years)
        level, years = selected
        point = int(np.rint(np.median(years)))
        return {"prediction": point, "q10": float(np.quantile(years, .10)),
                "q50": float(np.median(years)), "q90": float(np.quantile(years, .90)),
                "evidence": f"{level};n={len(years)}", "fallback_reason": fallback_reason}

    def predict(self, observed: pd.DataFrame, masks: pd.DataFrame, method: str = "conditional_baseline") -> pd.DataFrame:
        frame = observed.copy()
        frame["learned_era"] = self._era(frame["installation_year"])
        by_id = frame.set_index("segment_id")
        records = []
        for cell in masks.itertuples(index=False):
            row = by_id.loc[cell.segment_id]
            result = self.material(row) if cell.target == "material" else self.year(row)
            record = {"segment_id": cell.segment_id, "target": cell.target, "method": method,
                      "point_prediction": result["prediction"], "evidence": result["evidence"],
                      "fallback_reason": result["fallback_reason"], "year_q10": np.nan,
                      "year_q50": np.nan, "year_q90": np.nan, "p_steel": np.nan,
                      "p_copper": np.nan, "p_aluminium": np.nan,
                      "max_vote_share": np.nan, "tree_std": np.nan}
            if cell.target == "material":
                record.update(zip(("p_steel", "p_copper", "p_aluminium"), result["probabilities"]))
            else:
                record.update({"year_q10": result["q10"], "year_q50": result["q50"], "year_q90": result["q90"]})
            records.append(record)
        return pd.DataFrame(records)


def floor_predictions(fitting: pd.DataFrame, masks: pd.DataFrame) -> pd.DataFrame:
    floor = ConditionalBaseline(min_group=10**12).fit(fitting)
    dummy = pd.DataFrame({"segment_id": masks["segment_id"].unique()})
    for field in MODEL_FIELDS:
        dummy[field] = np.nan
    result = floor.predict(dummy, masks, method="global_floor")
    result["fallback_reason"] = "global_floor_by_design"
    return result


def tune_baseline(fitting: pd.DataFrame, tuning: pd.DataFrame, masks: pd.DataFrame, truth: pd.DataFrame) -> tuple[ConditionalBaseline, dict[str, object]]:
    """Choose grouping strength on tuning cells only; test truth is untouched."""
    truth_lookup = truth.set_index("segment_id")
    material_masks = masks.loc[masks["target"].eq("material")]
    year_masks = masks.loc[masks["target"].eq("installation_year")]
    best_material = (float("inf"), 10.0, 20)
    best_year = (float("inf"), 20)
    records = []
    for alpha in (1, 5, 10, 25, 50):
        for min_group in (5, 10, 20, 30, 50, 100):
            model = ConditionalBaseline(alpha, min_group).fit(fitting)
            material = model.predict(tuning, material_masks)
            year = model.predict(tuning, year_masks)
            y_material = truth_lookup.loc[material["segment_id"], "material"].to_numpy()
            probabilities = material[["p_steel", "p_copper", "p_aluminium"]].to_numpy(dtype=float)
            loss = material_log_loss(y_material, probabilities)
            y_year = truth_lookup.loc[year["segment_id"], "installation_year"].to_numpy(dtype=float)
            mae = float(mean_absolute_error(y_year, year["point_prediction"].to_numpy(dtype=float)))
            records.append({"alpha": alpha, "min_group": min_group, "material_log_loss": loss, "year_mae": mae})
            if loss < best_material[0]:
                best_material = (loss, float(alpha), min_group)
            if mae < best_year[0]:
                best_year = (mae, min_group)
    # The two targets may need different backoff sizes; the year size is recorded
    # separately and applied by a second baseline object at prediction time.
    return ConditionalBaseline(best_material[1], best_material[2]).fit(fitting), {
        "material_alpha": best_material[1], "material_min_group": best_material[2],
        "year_min_group": best_year[1], "tuning_grid": records,
    }


def forest_predictions(
    fitting: pd.DataFrame, observed: pd.DataFrame, masks: pd.DataFrame,
    seed: int, settings: PilotSettings, *, max_depth: int | None = 8,
    min_samples_leaf: int = 5, n_passes: int | None = None,
) -> tuple[pd.DataFrame, PredictiveMissForest]:
    config = PredictiveMissForestConfig(
        n_estimators=settings.n_estimators, n_passes=n_passes or settings.n_passes,
        random_state=seed, n_jobs=1, min_samples_leaf=min_samples_leaf,
        max_depth=max_depth,
    )
    forest = PredictiveMissForest(
        numeric_columns=("installation_year", "distance_to_sea_km"),
        categorical_columns=("voltage_category", "location_group", "corrosion_zone", "material", "covering"),
        config=config,
        constraints={"installation_year": {"data_type": "integer", "min": 1930, "max": 2014},
                     "distance_to_sea_km": {"data_type": "float", "min": 0.0, "max": 350.0}},
    ).fit(fitting.copy())
    return forest_predictions_from_model(forest, observed, masks), forest


def _forest_scores(predictions: pd.DataFrame, truth: pd.DataFrame) -> tuple[float, float]:
    by_id = truth.set_index("segment_id")
    material = predictions.loc[predictions["target"].eq("material")]
    year = predictions.loc[predictions["target"].eq("installation_year")]
    log_loss = material_log_loss(
        by_id.loc[material["segment_id"], "material"].to_numpy(),
        material[["p_steel", "p_copper", "p_aluminium"]].to_numpy(dtype=float),
    )
    mae = mean_absolute_error(
        by_id.loc[year["segment_id"], "installation_year"].to_numpy(dtype=float),
        year["point_prediction"].to_numpy(dtype=float),
    )
    return float(log_loss), float(mae)


def tune_forest(
    fitting: pd.DataFrame, tuning: pd.DataFrame, fitting_masks: pd.DataFrame,
    tuning_masks: pd.DataFrame, fitting_truth: pd.DataFrame, tuning_truth: pd.DataFrame,
    seed: int, settings: PilotSettings,
) -> tuple[dict[str, int | None], pd.DataFrame, pd.DataFrame]:
    """One-factor curves; select settings using tuning cells, never test truth.

    The joint objective gives equal weight to each target's loss relative to
    its global-floor tuning loss. Fitting scores are diagnostic only.
    """
    floor_loss, floor_mae = _forest_scores(floor_predictions(fitting, tuning_masks), tuning_truth)
    selected: dict[str, int | None] = {
        "max_depth": 8, "min_samples_leaf": 5, "n_passes": settings.n_passes,
    }
    grid = {
        "max_depth": (4, 6, 8, 12, None),
        "min_samples_leaf": (1, 5, 10, 20, 50),
        "n_passes": (1, 2, 3, 4, 5),
    }
    curve_rows = []
    for parameter, values in grid.items():
        best = (float("inf"), selected[parameter])
        for value in values:
            trial = {**selected, parameter: value}
            fitting_predictions, model = forest_predictions(
                fitting, fitting, fitting_masks, seed, settings, **trial,
            )
            tuning_predictions = forest_predictions_from_model(model, tuning, tuning_masks)
            fit_loss, fit_mae = _forest_scores(fitting_predictions, fitting_truth)
            tune_loss, tune_mae = _forest_scores(tuning_predictions, tuning_truth)
            objective = tune_loss / floor_loss + tune_mae / floor_mae
            curve_rows.append({
                "seed": seed, "parameter": parameter, "value": str(value),
                "fitting_material_log_loss": fit_loss, "tuning_material_log_loss": tune_loss,
                "fitting_year_mae": fit_mae, "tuning_year_mae": tune_mae,
                "tuning_relative_loss_sum": objective,
            })
            if objective < best[0]:
                best = (objective, value)
        selected[parameter] = best[1]
    learning_rows = []
    for fraction in (.10, .25, .50, 1.0):
        if fraction == 1.0:
            subset = fitting
        else:
            rng = np.random.default_rng(np.random.SeedSequence([seed, 7, round(fraction * 100)]))
            indices = np.sort(rng.choice(len(fitting), round(len(fitting) * fraction), replace=False))
            subset = fitting.iloc[indices]
        _, model = forest_predictions(
            subset, subset, fitting_masks.loc[fitting_masks["segment_id"].isin(subset["segment_id"])],
            seed, settings, **selected,
        )
        predicted = forest_predictions_from_model(model, tuning, tuning_masks)
        tune_loss, tune_mae = _forest_scores(predicted, tuning_truth)
        learning_rows.append({
            "seed": seed, "fitting_fraction": fraction, "fitting_rows": len(subset),
            "tuning_material_log_loss": tune_loss, "tuning_year_mae": tune_mae,
        })
    return selected, pd.DataFrame(curve_rows), pd.DataFrame(learning_rows)


def forest_predictions_from_model(
    model: PredictiveMissForest, observed: pd.DataFrame, masks: pd.DataFrame,
) -> pd.DataFrame:
    """Extract standard candidate rows from a frozen forest without refitting."""
    result = model.transform_detailed(observed.copy(), with_scores=True)
    if result.scores is None:
        raise AssertionError("forest did not return diagnostics")
    lookup = {(str(observed.iloc[int(item.position)]["segment_id"]), str(item.column)): item
              for item in result.scores.itertuples(index=False)}
    rows = []
    for cell in masks.itertuples(index=False):
        item = lookup[(str(cell.segment_id), str(cell.target))]
        record = {
            "segment_id": cell.segment_id, "target": cell.target,
            "method": "predictive_missforest", "point_prediction": item.imputed,
            "evidence": str(item.prediction_source),
            "fallback_reason": item.prediction_source if item.prediction_source == "training_statistic_fallback" else "",
            "p_steel": np.nan, "p_copper": np.nan, "p_aluminium": np.nan,
            "year_q10": np.nan, "year_q50": np.nan, "year_q90": np.nan,
            "max_vote_share": item.max_proba, "tree_std": item.tree_std,
        }
        if cell.target == "material":
            probs = item.material_probabilities
            if probs is None:
                raise ValueError("forest material probabilities missing")
            record.update({"p_steel": probs.get("Steel", 0.0), "p_copper": probs.get("Copper", 0.0),
                           "p_aluminium": probs.get("Aluminium", 0.0)})
        else:
            record.update({"year_q10": item.tree_p10, "year_q50": item.tree_p50,
                           "year_q90": item.tree_p90})
        rows.append(record)
    return pd.DataFrame(rows)


def validate_predictions(predictions: pd.DataFrame, observed: pd.DataFrame, masks: pd.DataFrame) -> None:
    expected = set(zip(masks["segment_id"], masks["target"]))
    actual = set(zip(predictions["segment_id"], predictions["target"]))
    if len(predictions) != len(expected) or actual != expected:
        raise ValueError("candidate cells differ from the frozen mask")
    by_id = observed.set_index("segment_id")
    for item in masks.itertuples(index=False):
        if not pd.isna(by_id.at[item.segment_id, item.target]):
            raise ValueError("prediction targets a nonmissing observed value")
    material = predictions.loc[predictions["target"].eq("material")]
    if not material["point_prediction"].isin(MATERIALS).all():
        raise ValueError("predicted material is not in the declared domain")
    probabilities = material[["p_steel", "p_copper", "p_aluminium"]].to_numpy(dtype=float)
    if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(axis=1), 1):
        raise ValueError("material probabilities must be finite and sum to one")
    year = predictions.loc[predictions["target"].eq("installation_year"), "point_prediction"].astype(float)
    if not year.between(1930, 2014).all():
        raise ValueError("predicted years outside the declared generator range")
    year_rows = predictions.loc[predictions["target"].eq("installation_year")]
    quantiles = year_rows[["year_q10", "year_q50", "year_q90"]].to_numpy(dtype=float)
    if not np.isfinite(quantiles).all() or not np.all(
        (quantiles[:, 0] <= quantiles[:, 1]) & (quantiles[:, 1] <= quantiles[:, 2])
    ):
        raise ValueError("year quantiles must be finite and ordered")


def standardize_predictions(predictions: pd.DataFrame, seed: int, mask_rate: float) -> pd.DataFrame:
    standardized = predictions.copy()
    standardized.insert(0, "condition_id", f"strength100_MCAR{round(mask_rate * 100)}_clean")
    standardized.insert(1, "seed", seed)
    standardized["age_band_pred"] = np.nan
    year_rows = standardized["target"].eq("installation_year")
    standardized.loc[year_rows, "age_band_pred"] = _age_band(
        pd.to_numeric(standardized.loc[year_rows, "point_prediction"])
    ).astype(str).to_numpy()
    return standardized


def evaluate(predictions: pd.DataFrame, reference: pd.DataFrame, observed: pd.DataFrame, masks: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score frozen outputs against evaluator-only answers and population truth."""
    truth = reference.set_index("segment_id")
    rows = []
    population = []
    for method, values in predictions.groupby("method", sort=False):
        mat = values.loc[values["target"].eq("material")]
        years = values.loc[values["target"].eq("installation_year")]
        actual_m = truth.loc[mat["segment_id"], "material"].to_numpy()
        probs = mat[["p_steel", "p_copper", "p_aluminium"]].to_numpy(dtype=float)
        predicted_m = mat["point_prediction"].astype(str).to_numpy()
        rows.extend([
            {"method": method, "target": "material", "measure": "log_loss", "value": material_log_loss(actual_m, probs)},
            {"method": method, "target": "material", "measure": "brier", "value": float(np.mean(np.sum((probs - (actual_m[:, None] == np.array(MATERIALS))) ** 2, axis=1)))},
            {"method": method, "target": "material", "measure": "accuracy", "value": float(accuracy_score(actual_m, predicted_m))},
            {"method": method, "target": "material", "measure": "balanced_accuracy", "value": float(balanced_accuracy_score(actual_m, predicted_m))},
            {"method": method, "target": "material", "measure": "macro_f1", "value": float(f1_score(actual_m, predicted_m, labels=MATERIALS, average="macro", zero_division=0))},
            {"method": method, "target": "material", "measure": "fill_coverage", "value": float(np.mean(pd.notna(predicted_m)))},
        ])
        recalls = recall_score(actual_m, predicted_m, labels=MATERIALS, average=None, zero_division=0)
        rows.extend({"method": method, "target": "material", "measure": f"recall_{label.lower()}", "value": float(value)} for label, value in zip(MATERIALS, recalls))
        actual_y = truth.loc[years["segment_id"], "installation_year"].to_numpy(dtype=float)
        predicted_y = years["point_prediction"].to_numpy(dtype=float)
        q10 = years["year_q10"].to_numpy(dtype=float)
        q90 = years["year_q90"].to_numpy(dtype=float)
        rows.extend([
            {"method": method, "target": "installation_year", "measure": "mae_years", "value": float(mean_absolute_error(actual_y, predicted_y))},
            {"method": method, "target": "installation_year", "measure": "fill_coverage", "value": float(np.mean(np.isfinite(predicted_y)))},
            {"method": method, "target": "installation_year", "measure": "signed_bias_years", "value": float(np.mean(predicted_y - actual_y))},
            {"method": method, "target": "installation_year", "measure": "age_band_accuracy", "value": float(np.mean(_age_band(pd.Series(actual_y)).to_numpy() == _age_band(pd.Series(predicted_y)).to_numpy()))},
            {"method": method, "target": "installation_year", "measure": "raw_q10_q90_empirical_coverage", "value": float(np.mean((q10 <= actual_y) & (actual_y <= q90)))},
            {"method": method, "target": "installation_year", "measure": "raw_q10_q90_mean_width_years", "value": float(np.mean(q90 - q10))},
        ])
        for band in ("<20", "20-34", "35-49", "50-64", "65+"):
            subset = _age_band(pd.Series(actual_y)).astype(str).eq(band).to_numpy()
            if subset.any():
                rows.append({"method": method, "target": "installation_year", "measure": f"bias_{band}", "value": float(np.mean(predicted_y[subset] - actual_y[subset]))})
        filled = observed.set_index("segment_id").copy()
        filled.loc[mat["segment_id"], "material"] = predicted_m
        filled.loc[years["segment_id"], "installation_year"] = predicted_y
        truth_test = truth.loc[filled.index]
        weights = truth_test["length_km"].to_numpy(dtype=float)
        for label in MATERIALS:
            population.append({"method": method, "measure": f"material_share_error_{label.lower()}",
                               "value": float(filled["material"].eq(label).mean() - truth_test["material"].eq(label).mean())})
            population.append({"method": method, "measure": f"length_weighted_material_share_error_{label.lower()}",
                               "value": float(np.average(filled["material"].eq(label), weights=weights) - np.average(truth_test["material"].eq(label), weights=weights))})
            known_count = observed["material"].eq(label).sum()
            predicted_mass = mat[f"p_{label.lower()}"].to_numpy(dtype=float).sum()
            expected_share = (known_count + predicted_mass) / len(observed)
            population.append({"method": method, "measure": f"material_expected_share_error_{label.lower()}",
                               "value": float(expected_share - truth_test["material"].eq(label).mean())})
            observed_material = observed.set_index("segment_id")["material"]
            known_weight = weights[(observed_material.notna()).to_numpy() &
                                   (observed_material.eq(label)).to_numpy()].sum()
            hidden_weight = truth_test.loc[mat["segment_id"], "length_km"].to_numpy(dtype=float)
            weighted_expected = (known_weight + np.dot(hidden_weight, mat[f"p_{label.lower()}"].to_numpy(dtype=float))) / weights.sum()
            population.append({"method": method, "measure": f"length_weighted_material_expected_share_error_{label.lower()}",
                               "value": float(weighted_expected - np.average(truth_test["material"].eq(label), weights=weights))})
        population.extend([
            {"method": method, "measure": "aged_50_or_older_share_error", "value": float((SNAPSHOT_YEAR - filled["installation_year"].astype(float)).ge(50).mean() - (SNAPSHOT_YEAR - truth_test["installation_year"].astype(float)).ge(50).mean())},
            {"method": method, "measure": "year_std_error", "value": float(filled["installation_year"].astype(float).std(ddof=0) - truth_test["installation_year"].astype(float).std(ddof=0))},
            {"method": method, "measure": "year_wasserstein", "value": float(wasserstein_distance(filled["installation_year"].astype(float), truth_test["installation_year"].astype(float)))},
            {"method": method, "measure": "length_weighted_aged_50_or_older_share_error", "value": float(np.average((SNAPSHOT_YEAR - filled["installation_year"].astype(float)).ge(50), weights=weights) - np.average((SNAPSHOT_YEAR - truth_test["installation_year"].astype(float)).ge(50), weights=weights))},
            {"method": method, "measure": "length_weighted_year_wasserstein", "value": float(wasserstein_distance(filled["installation_year"].astype(float), truth_test["installation_year"].astype(float), u_weights=weights, v_weights=weights))},
        ])
    complete = observed.dropna(subset=list(TARGETS))
    truth_test = truth.loc[observed["segment_id"]]
    population.extend([
        {"method": "complete_case", "measure": f"material_share_error_{label.lower()}",
         "value": float(complete["material"].eq(label).mean() - truth_test["material"].eq(label).mean())}
        for label in MATERIALS
    ])
    complete_weights = truth.loc[complete["segment_id"], "length_km"].to_numpy(dtype=float)
    test_weights = truth_test["length_km"].to_numpy(dtype=float)
    population.extend([
        {"method": "complete_case", "measure": f"length_weighted_material_share_error_{label.lower()}",
         "value": float(np.average(complete["material"].eq(label), weights=complete_weights) - np.average(truth_test["material"].eq(label), weights=test_weights))}
        for label in MATERIALS
    ])
    population.append({"method": "complete_case", "measure": "aged_50_or_older_share_error",
                       "value": float((SNAPSHOT_YEAR - complete["installation_year"].astype(float)).ge(50).mean() - (SNAPSHOT_YEAR - truth_test["installation_year"].astype(float)).ge(50).mean())})
    population.append({"method": "complete_case", "measure": "year_std_error",
                       "value": float(complete["installation_year"].astype(float).std(ddof=0) - truth_test["installation_year"].astype(float).std(ddof=0))})
    population.append({"method": "complete_case", "measure": "year_wasserstein",
                       "value": float(wasserstein_distance(complete["installation_year"].astype(float), truth_test["installation_year"].astype(float)))})
    population.append({"method": "complete_case", "measure": "length_weighted_aged_50_or_older_share_error",
                       "value": float(np.average((SNAPSHOT_YEAR - complete["installation_year"].astype(float)).ge(50), weights=complete_weights) - np.average((SNAPSHOT_YEAR - truth_test["installation_year"].astype(float)).ge(50), weights=test_weights))})
    population.append({"method": "complete_case", "measure": "length_weighted_year_wasserstein",
                       "value": float(wasserstein_distance(complete["installation_year"].astype(float), truth_test["installation_year"].astype(float), u_weights=complete_weights, v_weights=test_weights))})
    return pd.DataFrame(rows), pd.DataFrame(population)


def evaluation_diagnostics(predictions: pd.DataFrame, reference: pd.DataFrame, masks: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Confusion counts and high-error examples stay with evaluator outputs."""
    truth = reference.set_index("segment_id")
    both_hidden = set(masks.loc[masks["target"].eq("material"), "segment_id"]) & set(
        masks.loc[masks["target"].eq("installation_year"), "segment_id"]
    )
    confusion = []
    cases = []
    for method, group in predictions.groupby("method", sort=False):
        material = group.loc[group["target"].eq("material")]
        for row in material.itertuples(index=False):
            actual = str(truth.at[row.segment_id, "material"])
            confusion.append({"method": method, "actual": actual, "predicted": str(row.point_prediction)})
            if actual != str(row.point_prediction):
                probability = getattr(row, f"p_{actual.lower()}")
                cases.append({"method": method, "segment_id": row.segment_id, "target": "material",
                              "actual": actual, "predicted": row.point_prediction,
                              "error": float(1 - probability), "both_targets_hidden": row.segment_id in both_hidden})
        years = group.loc[group["target"].eq("installation_year")]
        for row in years.itertuples(index=False):
            actual = int(truth.at[row.segment_id, "installation_year"])
            cases.append({"method": method, "segment_id": row.segment_id, "target": "installation_year",
                          "actual": actual, "predicted": row.point_prediction,
                          "error": float(abs(float(row.point_prediction) - actual)),
                          "both_targets_hidden": row.segment_id in both_hidden})
    confusion_frame = pd.DataFrame(confusion).groupby(["method", "actual", "predicted"], as_index=False).size()
    case_frame = pd.DataFrame(cases).sort_values(["method", "target", "error"], ascending=[True, True, False])
    case_frame = case_frame.groupby(["method", "target"], as_index=False).head(20)
    return confusion_frame, case_frame


def subgroup_diagnostics(
    predictions: pd.DataFrame, reference: pd.DataFrame, masks: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Show errors by configured era, voltage, location and joint missingness."""
    truth = reference.set_index("segment_id")
    both_hidden = set(masks.loc[masks["target"].eq("material"), "segment_id"]) & set(
        masks.loc[masks["target"].eq("installation_year"), "segment_id"]
    )
    rows = []
    for item in predictions.itertuples(index=False):
        actual_year = int(truth.at[item.segment_id, "installation_year"])
        actual = truth.at[item.segment_id, item.target]
        if item.target == "material":
            correct = str(item.point_prediction) == str(actual)
            error = float(not correct)
            measure = "material_error_rate"
        else:
            error = abs(float(item.point_prediction) - float(actual))
            measure = "year_mae"
        for group, label in (
            ("era", _voltage_era(actual_year)),
            ("voltage", str(truth.at[item.segment_id, "voltage_category"])),
            ("location", str(truth.at[item.segment_id, "location_group"])),
            ("both_targets_hidden", str(item.segment_id in both_hidden)),
        ):
            rows.append({"method": item.method, "target": item.target, "measure": measure,
                         "group": group, "level": label, "error": error,
                         "signed_error": (float(item.point_prediction) - actual_year) if item.target == "installation_year" else np.nan})
    detail = pd.DataFrame(rows)
    summary = detail.groupby(["method", "target", "measure", "group", "level"], as_index=False).agg(
        n=("error", "size"), mean_error=("error", "mean"), mean_signed_year_error=("signed_error", "mean")
    )
    backoff = predictions.groupby(["method", "target", "evidence", "fallback_reason"], dropna=False).size().reset_index(name="n")
    return summary, backoff


def subgroup_confusion(predictions: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Material confusion counts by configured era, voltage and location."""
    truth = reference.set_index("segment_id")
    rows = []
    for item in predictions.loc[predictions["target"].eq("material")].itertuples(index=False):
        year = int(truth.at[item.segment_id, "installation_year"])
        for group, level in (
            ("era", _voltage_era(year)),
            ("voltage", str(truth.at[item.segment_id, "voltage_category"])),
            ("location", str(truth.at[item.segment_id, "location_group"])),
        ):
            rows.append({"method": item.method, "group": group, "level": level,
                         "actual": str(truth.at[item.segment_id, "material"]),
                         "predicted": str(item.point_prediction)})
    return pd.DataFrame(rows).groupby(["method", "group", "level", "actual", "predicted"], as_index=False).size()


def assert_observed_preserved(original: pd.DataFrame, completed: pd.DataFrame) -> None:
    """Reject any forest replay that changes a nonmissing input cell."""
    for field in MODEL_FIELDS:
        known = original[field].notna()
        pd.testing.assert_series_equal(
            completed.loc[known, field].reset_index(drop=True),
            original.loc[known, field].reset_index(drop=True),
            check_names=False, check_dtype=False,
        )


def save_diagnostic_plots(curves: pd.DataFrame, learning: pd.DataFrame, scores: pd.DataFrame, output: Path) -> None:
    """Create compact pilot figures from saved numeric evidence."""
    plot_dir = output / "plots"
    plot_dir.mkdir(exist_ok=True)
    for parameter, group in curves.groupby("parameter", sort=False):
        values = list(dict.fromkeys(group["value"]))
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)
        for ax, fit_col, tune_col, title in (
            (axes[0], "fitting_material_log_loss", "tuning_material_log_loss", "Material log loss"),
            (axes[1], "fitting_year_mae", "tuning_year_mae", "Year MAE"),
        ):
            grouped = group.groupby("value")[[fit_col, tune_col]].mean().reindex(values)
            ax.plot(values, grouped[fit_col], marker="o", label="Fitting")
            ax.plot(values, grouped[tune_col], marker="o", label="Tuning")
            ax.set_title(title)
            ax.set_xlabel(parameter)
            ax.grid(alpha=.25)
            ax.legend()
        fig.savefig(plot_dir / f"validation_{parameter}.png", dpi=150)
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)
    grouped = learning.groupby("fitting_fraction")[["tuning_material_log_loss", "tuning_year_mae"]].mean()
    for ax, measure, title in (
        (axes[0], "tuning_material_log_loss", "Material log loss"),
        (axes[1], "tuning_year_mae", "Year MAE"),
    ):
        ax.plot(grouped.index * 100, grouped[measure], marker="o")
        ax.set_xlabel("Fitting records used (%)")
        ax.set_title(title)
        ax.grid(alpha=.25)
    fig.savefig(plot_dir / "learning_curve.png", dpi=150)
    plt.close(fig)
    for target, measure, title in (
        ("material", "log_loss", "Material log loss"),
        ("installation_year", "mae_years", "Installation-year MAE"),
    ):
        view = scores.loc[scores["target"].eq(target) & scores["measure"].eq(measure)]
        grouped = view.groupby("method")["value"].agg(["mean", "min", "max"])
        fig, ax = plt.subplots(figsize=(7, 3.5), constrained_layout=True)
        ax.bar(grouped.index, grouped["mean"], color="#426C86")
        ax.errorbar(range(len(grouped)), grouped["mean"],
                    yerr=np.vstack((grouped["mean"] - grouped["min"], grouped["max"] - grouped["mean"])),
                    fmt="none", ecolor="#292929", capsize=4)
        ax.tick_params(axis="x", rotation=18)
        ax.set_title(title + " (five-seed mean and range)")
        fig.savefig(plot_dir / f"pilot_{measure}.png", dpi=150)
        plt.close(fig)


def summarize_scores(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Descriptive five-seed comparisons, with no significance claims."""
    summary = (
        scores.groupby(["method", "target", "measure"])["value"]
        .agg(["mean", "min", "max"])
        .reset_index()
        .rename(columns={"min": "minimum", "max": "maximum"})
    )
    wide = scores.pivot(index=["seed", "target", "measure"], columns="method", values="value")
    paired = []
    for (seed, target, measure), row in wide.iterrows():
        for method, reference in (("conditional_baseline", "global_floor"),
                                  ("predictive_missforest", "conditional_baseline")):
            paired.append({"seed": seed, "target": target, "measure": measure,
                           "comparison": f"{method}_minus_{reference}",
                           "difference": float(row[method] - row[reference])})
    return summary, pd.DataFrame(paired)


def gap_closed(scores: pd.DataFrame) -> pd.DataFrame:
    """Compute only meaningful loss reductions against the exact reference."""
    supported = {("material", "log_loss"), ("material", "brier"),
                 ("installation_year", "mae_years")}
    wide = scores.pivot(index=["seed", "target", "measure"], columns="method", values="value")
    records = []
    for (target, measure), group in wide.groupby(level=["target", "measure"]):
        if (target, measure) not in supported:
            continue
        denominators = group["global_floor"] - group["generator_reference"]
        mean_gap = float(denominators.mean())
        seed_spread = float(denominators.std(ddof=1)) if len(denominators) > 1 else np.nan
        meaningful = mean_gap > 0 and (len(denominators) == 1 or mean_gap > seed_spread)
        for (seed, _, _), row in group.iterrows():
            for method in ("conditional_baseline", "predictive_missforest"):
                denominator = float(row["global_floor"] - row["generator_reference"])
                value = ((float(row["global_floor"] - row[method]) / denominator)
                         if meaningful and denominator > 0 else np.nan)
                records.append({"seed": seed, "target": target, "measure": measure,
                                "method": method, "gap_closed": value,
                                "floor_reference_mean_gap": mean_gap,
                                "seed_to_seed_gap_sd": seed_spread,
                                "status": "reported" if np.isfinite(value) else "not_applicable_small_or_nonpositive_gap"})
    return pd.DataFrame(records)


def run_pilot(output_dir: str | Path, settings: PilotSettings = PilotSettings(), n_rows: int = 10_000) -> Path:
    """Run the shared five-seed MCAR comparison; never overwrite a prior run."""
    settings.validate()
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True, exist_ok=True)
    reference = make_reference(n_rows)
    (output / "evaluator_only").mkdir()
    reference.to_csv(output / "evaluator_only/clean_reference.csv", index=False)
    material_generator_check(reference).to_csv(output / "evaluator_only/material_generator_check.csv", index=False)
    all_metrics = []
    all_population = []
    all_curves = []
    all_learning = []
    all_subgroups = []
    all_backoff = []
    all_subgroup_confusion = []
    integrity = []
    for seed in settings.seeds:
        run = output / f"seed_{seed}"
        run.mkdir()
        split = freeze_split(reference, seed)
        views, masks = mask_targets(reference, split, seed, settings.mask_rate)
        split.to_csv(run / "partitions.csv", index=False)
        masks.to_csv(run / "mask_manifest.csv", index=False)
        for part, view in views.items():
            view.to_csv(run / f"observed_{part}.csv", index=False)
            flags = pd.DataFrame({
                "segment_id": view["segment_id"],
                "material_missing": view["material"].isna().astype(int),
                "installation_year_missing": view["installation_year"].isna().astype(int),
            })
            flags["target_missing_count"] = flags[["material_missing", "installation_year_missing"]].sum(axis=1)
            flags.to_csv(run / f"missingness_flags_{part}.csv", index=False)
        fitting, tuning, test = views["fitting"], views["tuning"], views["test"]
        tuning_masks = masks.loc[masks["partition"].eq("tuning")]
        fitting_masks = masks.loc[masks["partition"].eq("fitting")]
        test_masks = masks.loc[masks["partition"].eq("test")]
        fitting_ids = split.loc[split["partition"].eq("fitting"), "segment_id"]
        tuning_ids = split.loc[split["partition"].eq("tuning"), "segment_id"]
        baseline, tuning_result = tune_baseline(
            fitting, tuning, tuning_masks, reference.loc[reference["segment_id"].isin(tuning_ids)]
        )
        year_baseline = ConditionalBaseline(tuning_result["material_alpha"], tuning_result["year_min_group"]).fit(fitting)
        conditional = baseline.predict(test, test_masks)
        year_conditional = year_baseline.predict(test, test_masks.loc[test_masks["target"].eq("installation_year")])
        conditional.loc[conditional["target"].eq("installation_year"), :] = year_conditional.to_numpy()
        floor = floor_predictions(fitting, test_masks)
        forest_settings, curves, learning = tune_forest(
            fitting, tuning, fitting_masks, tuning_masks,
            reference.loc[reference["segment_id"].isin(fitting_ids)],
            reference.loc[reference["segment_id"].isin(tuning_ids)],
            seed, settings,
        )
        all_curves.append(curves)
        all_learning.append(learning)
        forest_table, forest = forest_predictions(fitting, test, test_masks, seed, settings, **forest_settings)
        model_path = run / "forest.joblib"
        forest.save(model_path)
        replay_model = PredictiveMissForest.load(model_path)
        assert_observed_preserved(test, replay_model.transform(test.copy()))
        replay = forest_predictions_from_model(replay_model, test, test_masks)
        pd.testing.assert_frame_equal(replay, forest_table, check_dtype=False)
        candidates = standardize_predictions(pd.concat((floor, conditional, forest_table), ignore_index=True), seed, settings.mask_rate)
        for method, table in candidates.groupby("method"):
            validate_predictions(table, test, test_masks)
        if not test["segment_id"].is_unique or len(candidates) != 3 * len(test_masks):
            raise AssertionError("candidate row count or test IDs are inconsistent")
        integrity.append({"seed": seed, "status": "passed", "test_rows": len(test),
                          "hidden_cells": len(test_masks), "candidate_rows": len(candidates),
                          "same_hidden_keys": True, "observed_cells_preserved": True,
                          "forest_reload_matches": True})
        candidates.to_csv(run / "candidate_cells.csv", index=False)
        (run / "tuning.json").write_text(json.dumps(tuning_result, indent=2) + "\n", encoding="utf-8")
        (run / "forest_tuning.json").write_text(json.dumps(forest_settings, indent=2) + "\n", encoding="utf-8")
        oracle = standardize_predictions(generator_reference_predictions(test, test_masks), seed, settings.mask_rate)
        validate_predictions(oracle, test, test_masks)
        oracle.to_csv(output / "evaluator_only" / f"generator_reference_seed_{seed}.csv", index=False)
        # Truth is joined only after all learned candidate output and integrity checks.
        metrics, population = evaluate(pd.concat((candidates, oracle), ignore_index=True), reference, test, test_masks)
        confusion, cases = evaluation_diagnostics(
            pd.concat((candidates, oracle), ignore_index=True), reference, test_masks
        )
        confusion.to_csv(output / "evaluator_only" / f"confusion_seed_{seed}.csv", index=False)
        cases.to_csv(output / "evaluator_only" / f"review_cases_seed_{seed}.csv", index=False)
        subgroup, backoff = subgroup_diagnostics(
            pd.concat((candidates, oracle), ignore_index=True), reference, test_masks,
        )
        subgroup.insert(0, "seed", seed)
        backoff.insert(0, "seed", seed)
        all_subgroups.append(subgroup)
        all_backoff.append(backoff)
        grouped_confusion = subgroup_confusion(pd.concat((candidates, oracle), ignore_index=True), reference)
        grouped_confusion.insert(0, "seed", seed)
        all_subgroup_confusion.append(grouped_confusion)
        metrics.insert(0, "seed", seed)
        population.insert(0, "seed", seed)
        all_metrics.append(metrics)
        all_population.append(population)
    scores = pd.concat(all_metrics, ignore_index=True)
    population = pd.concat(all_population, ignore_index=True)
    scores.to_csv(output / "cell_metrics.csv", index=False)
    population.to_csv(output / "population_metrics.csv", index=False)
    curves = pd.concat(all_curves, ignore_index=True)
    learning = pd.concat(all_learning, ignore_index=True)
    curves.to_csv(output / "forest_validation_curve.csv", index=False)
    learning.to_csv(output / "forest_learning_curve.csv", index=False)
    pd.concat(all_subgroups, ignore_index=True).to_csv(output / "subgroup_diagnostics.csv", index=False)
    pd.concat(all_backoff, ignore_index=True).to_csv(output / "evidence_backoff_counts.csv", index=False)
    pd.concat(all_subgroup_confusion, ignore_index=True).to_csv(output / "subgroup_confusion.csv", index=False)
    (output / "integrity_checks.json").write_text(json.dumps(integrity, indent=2) + "\n", encoding="utf-8")
    summary, paired = summarize_scores(scores)
    summary.to_csv(output / "cell_metrics_summary.csv", index=False)
    paired.to_csv(output / "paired_differences.csv", index=False)
    gap_closed(scores).to_csv(output / "gap_closed.csv", index=False)
    population.groupby(["method", "measure"])["value"].agg(["mean", "min", "max"]).reset_index().to_csv(
        output / "population_metrics_summary.csv", index=False
    )
    save_diagnostic_plots(curves, learning, scores, output)
    (output / "manifest.json").write_text(json.dumps({
        "data_seed": 219, "strength": 100, "seeds": settings.seeds, "mask_rate": settings.mask_rate,
        "population_rows": n_rows, "methods": ["global_floor", "conditional_baseline", "predictive_missforest"],
        "forest_trees": settings.n_estimators,
        "forest_tuning": "per-seed, one-factor held-out tuning; joint relative log-loss and year-MAE objective",
        "file_sha256": {
            "clean_reference": _sha256(output / "evaluator_only/clean_reference.csv"),
            "pilot_source": _sha256(Path(__file__)),
            "forest_source": _sha256(ROOT / "src/team19_asset_data/imputation/predictive_missforest.py"),
            "generator_source": _sha256(ROOT / "src/team19_asset_data/overhead_conductors/generate_synthetic_data.py"),
            "generator_config": _sha256(ROOT / "config/overhead_conductors/generation.yaml"),
            "strength_config": _sha256(ROOT / "config/overhead_conductors/relationship_strength/strength_100.yaml"),
        },
        "library_versions": {"numpy": np.__version__, "pandas": pd.__version__,
                             "scikit_learn": sklearn.__version__},
        "reference_ceiling_status": "exact generator-conditional comparator for Phase 1 MCAR only; evaluator access only",
        "confidence_status": "raw diagnostics only; no acceptance threshold",
        "scope": "synthetic MCAR pilot; not SAPN records or PoF validation",
    }, indent=2) + "\n", encoding="utf-8")
    return output
