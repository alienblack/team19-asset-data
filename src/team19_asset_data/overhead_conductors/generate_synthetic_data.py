"""
Synthetic Overhead Conductor Dataset Generator
==============================================

Purpose
-------
Generate a controlled, SAPN-inspired synthetic overhead-conductor dataset for
material-imputation experiments.

Important
---------
This script DOES NOT claim to reproduce SA Power Networks' real asset register.
It combines published aggregate evidence with explicit synthetic assumptions.

Configuration
-------------
Generation settings are loaded from:

    config/overhead_conductors/generation.yaml

A custom configuration can also be supplied with:

    python generate_synthetic_overhead_conductors.py --config path/to/generation.yaml

The configuration is loaded and validated before any synthetic data is created.

Snapshot year
-------------
The current generation logic is calibrated to the 2014 overhead-conductor AMP.
Therefore ``snapshot_year`` must currently remain 2014.

Target leakage warning
----------------------
``conductor_type`` is retained in the complete synthetic asset register because
it is realistic, but it is excluded from the material-model-ready output because
codes such as AAC, AAAC, SC/AC and SC/GZ reveal material directly.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml


LOGGER = logging.getLogger(__name__)

CONFIG_RELATIVE_DIR = Path("config") / "overhead_conductors"
CONFIG_FILENAMES = ("generation.yaml", "generation.yml")

EXPECTED_VOLTAGE_CATEGORIES = {
    "66 kV",
    "33 kV",
    "19 kV SWER",
    "11/7.6 kV",
    "LV",
}
EXPECTED_MATERIAL_CLASSES = {"Steel", "Copper", "Aluminium"}
EXPECTED_CORROSION_ZONES = {"Low", "Severe", "Very Severe"}
EXPECTED_ERA_KEYS = {
    "pre_1950",
    "1950_1977",
    "1978_1994",
    "from_1995",
}

EXPECTED_LOCATION_GROUPS = {"Rural", "Regional", "Urban"}
EXPECTED_STRENGTH_KEYS = {
    "installation_year_to_voltage",
    "voltage_to_location",
    "voltage_to_corrosion",
    "location_to_corrosion",
    "installation_year_to_material",
    "voltage_to_material",
    "corrosion_to_material",
}
DEFAULT_STRENGTH_PROFILE = "strength_100"  # Default relationship-strength profile name 

ERA_KEY_ALIASES = {
    "pre_1950": "pre_1950",
    "1950_1977": "1950_1977",
    "19501977": "1950_1977",  # PyYAML may parse an unquoted 1950_1977 key as an int
    "1978_1994": "1978_1994",
    "19781994": "1978_1994",  # PyYAML may parse an unquoted 1978_1994 key as an int
    "from_1995": "from_1995",
}

REQUIRED_OUTPUT_COLUMNS = [
    "segment_id",
    "feeder_id",
    "installation_year",
    "asset_age",
    "voltage_category",
    "location_group",
    "corrosion_zone",
    "distance_to_sea_km",
    "material",
    "conductor_type",
    "covering",
    "length_km",
    "data_status",
    "record_version",
]


# ---------------------------------------------------------------------------
# 1. Configuration discovery and validation
# ---------------------------------------------------------------------------


def find_default_config_path() -> Path:
    """Locate generation.yaml/yml by searching upward from CWD and this file."""
    starts = [Path.cwd().resolve(), Path(__file__).resolve().parent]
    visited: set[Path] = set()

    for start in starts:
        for root in (start, *start.parents):
            if root in visited:
                continue
            visited.add(root)

            for filename in CONFIG_FILENAMES:
                candidate = root / CONFIG_RELATIVE_DIR / filename
                if candidate.is_file():
                    return candidate.resolve()

    expected = CONFIG_RELATIVE_DIR / CONFIG_FILENAMES[0]
    raise FileNotFoundError(
        "Could not locate the generation configuration. "
        f"Expected something like '{expected}'. "
        "Run from inside the project repository or pass --config explicitly."
    )


def infer_project_root(config_path: Path) -> Path:
    """Infer project root from the standard config/.../generation.yaml layout."""
    config_path = config_path.resolve()

    # Standard repository layout:
    # <project>/config/overhead_conductors/generation.yaml
    if (
        config_path.parent.name == "overhead_conductors"
        and config_path.parent.parent.name == "config"
    ):
        return config_path.parent.parent.parent

    # Fall back to a conventional repository marker if a custom config path is
    # supplied.
    for candidate in (config_path.parent, *config_path.parents):
        if (candidate / ".git").exists() or (candidate / "pyproject.toml").exists():
            return candidate

    # Last resort: resolve relative output paths from the config's directory.
    return config_path.parent


def require_mapping(parent: Mapping[str, Any], key: str, context: str) -> dict[str, Any]:
    """Return a required mapping section with a clear validation error."""
    if key not in parent:
        raise ValueError(f"Missing required configuration section: {context}.{key}")

    value = parent[key]
    if not isinstance(value, dict):
        raise ValueError(f"'{context}.{key}' must be a YAML mapping/object.")

    return value


def require_key(parent: Mapping[str, Any], key: str, context: str) -> Any:
    """Return a required configuration value with a clear validation error."""
    if key not in parent:
        raise ValueError(f"Missing required configuration value: {context}.{key}")
    return parent[key]


def normalize_distribution(
    distribution: Mapping[str, Any],
    name: str,
) -> dict[str, float]:
    """Validate and normalize a probability/share distribution."""
    if not isinstance(distribution, Mapping) or not distribution:
        raise ValueError(f"'{name}' must be a non-empty mapping.")

    converted: dict[str, float] = {}
    for label, raw_value in distribution.items():
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"'{name}.{label}' must be numeric; received {raw_value!r}."
            ) from exc

        if not np.isfinite(value):
            raise ValueError(f"'{name}.{label}' must be finite.")
        if value < 0:
            raise ValueError(f"'{name}.{label}' cannot be negative.")

        converted[str(label)] = value

    total = sum(converted.values())
    if total <= 0:
        raise ValueError(f"'{name}' must have a positive total probability/share.")

    return {label: value / total for label, value in converted.items()}




def normalize_era_mapping_keys(raw_mapping: Mapping[Any, Any]) -> dict[str, Any]:
    """Normalize YAML era keys, including numeric keys produced by PyYAML."""
    normalized: dict[str, Any] = {}

    for raw_key, value in raw_mapping.items():
        text_key = str(raw_key)
        normalized_key = ERA_KEY_ALIASES.get(text_key, text_key)

        if normalized_key in normalized:
            raise ValueError(
                "Duplicate voltage installation-era configuration after key "
                f"normalization: {normalized_key!r}"
            )

        normalized[normalized_key] = value

    return normalized


def validate_effects(
    effects: Any,
    allowed_outputs: set[str],
    name: str,
    allowed_groups: set[str] | None = None,
) -> None:
    """Validate additive-effect mappings used by the generator."""
    if effects is None:
        return
    if not isinstance(effects, Mapping):
        raise ValueError(f"'{name}' must be a mapping.")

    if allowed_groups is not None:
        unknown_groups = {str(group) for group in effects} - allowed_groups
        if unknown_groups:
            raise ValueError(
                f"'{name}' contains unknown group(s): {sorted(unknown_groups)}"
            )

    for group, adjustments in effects.items():
        if not isinstance(adjustments, Mapping):
            raise ValueError(f"'{name}.{group}' must be a mapping.")

        unknown = set(adjustments) - allowed_outputs
        if unknown:
            raise ValueError(
                f"'{name}.{group}' contains unknown output class(es): "
                f"{sorted(unknown)}"
            )

        for output, raw_value in adjustments.items():
            try:
                value = float(raw_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"'{name}.{group}.{output}' must be numeric."
                ) from exc

            if not np.isfinite(value):
                raise ValueError(f"'{name}.{group}.{output}' must be finite.")


@dataclass(frozen=True)
class StrengthConfig:
    """Validated relationship-strength profile."""

    profile_name: str
    description: str
    relationship_strengths: dict[str, float]
    config_path: Path

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "StrengthConfig":
        path = Path(config_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Relationship-strength config not found: {path}")

        try:
            with path.open("r", encoding="utf-8") as file:
                raw = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid YAML syntax in {path}: {exc}") from exc

        if not isinstance(raw, Mapping):
            raise ValueError("Relationship-strength configuration must be a YAML mapping.")

        profile_name = str(require_key(raw, "profile_name", "root")).strip()
        if not profile_name:
            raise ValueError("'profile_name' must be non-empty.")

        description = str(raw.get("description", "")).strip()
        strengths_raw = require_mapping(raw, "relationship_strengths", "root")

        missing = EXPECTED_STRENGTH_KEYS - set(strengths_raw)
        extra = set(strengths_raw) - EXPECTED_STRENGTH_KEYS
        if missing:
            raise ValueError(
                "relationship_strengths is missing required key(s): "
                f"{sorted(missing)}"
            )
        if extra:
            raise ValueError(
                "relationship_strengths contains unknown key(s): "
                f"{sorted(extra)}"
            )

        strengths: dict[str, float] = {}
        for name, raw_value in strengths_raw.items():
            try:
                value = float(raw_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"relationship_strengths.{name} must be numeric."
                ) from exc
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"relationship_strengths.{name} must be between 0.0 and 1.0; "
                    f"received {raw_value!r}."
                )
            strengths[name] = value

        return cls(
            profile_name=profile_name,
            description=description,
            relationship_strengths=strengths,
            config_path=path,
        )

    def get(self, relationship: str) -> float:
        return self.relationship_strengths[relationship]


@dataclass(frozen=True)
class Config:
    """Validated generation configuration."""

    n_rows: int
    seed: int
    snapshot_year: int
    output_dir: Path
    voltage: dict[str, Any]
    location: dict[str, Any]
    material: dict[str, Any]
    corrosion: dict[str, Any]
    config_path: Path

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "Config":
        """Load, resolve and validate generator settings from YAML."""
        path = Path(config_path).expanduser().resolve()

        if not path.is_file():
            raise FileNotFoundError(f"Generation config file not found: {path}")

        try:
            with path.open("r", encoding="utf-8") as file:
                raw_config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid YAML syntax in {path}: {exc}") from exc

        if not isinstance(raw_config, dict):
            raise ValueError(
                f"The generation configuration must be a YAML mapping/object: {path}"
            )

        dataset = require_mapping(raw_config, "dataset", "root")
        paths = require_mapping(raw_config, "paths", "root")
        voltage = require_mapping(raw_config, "voltage", "root")
        location = require_mapping(raw_config, "location", "root")
        material = require_mapping(raw_config, "material", "root")
        corrosion = require_mapping(raw_config, "corrosion", "root")

        try:
            n_rows = int(require_key(dataset, "n_rows", "dataset"))
            seed = int(require_key(dataset, "seed", "dataset"))
            snapshot_year = int(require_key(dataset, "snapshot_year", "dataset"))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "dataset.n_rows, dataset.seed and dataset.snapshot_year must be integers."
            ) from exc

        raw_output_dir = require_key(paths, "output_dir", "paths")
        if not isinstance(raw_output_dir, str) or not raw_output_dir.strip():
            raise ValueError("'paths.output_dir' must be a non-empty path string.")

        project_root = infer_project_root(path)
        output_dir = Path(raw_output_dir).expanduser()
        if not output_dir.is_absolute():
            output_dir = project_root / output_dir
        output_dir = output_dir.resolve()

        voltage_config = dict(voltage)
        raw_by_era = voltage_config.get("by_installation_era")
        if isinstance(raw_by_era, Mapping):
            voltage_config["by_installation_era"] = normalize_era_mapping_keys(raw_by_era)

        config = cls(
            n_rows=n_rows,
            seed=seed,
            snapshot_year=snapshot_year,
            output_dir=output_dir,
            voltage=voltage_config,
            location=dict(location),
            material=dict(material),
            corrosion=dict(corrosion),
            config_path=path,
        )
        config.validate()
        return config

    @property
    def voltage_target(self) -> dict[str, float]:
        return normalize_distribution(
            self.voltage["neutral_distribution"],
            "voltage.neutral_distribution",
        )

    @property
    def location_target(self) -> dict[str, float]:
        return normalize_distribution(
            self.location["neutral_distribution"],
            "location.neutral_distribution",
        )

    @property
    def corrosion_target(self) -> dict[str, float]:
        return normalize_distribution(
            self.corrosion["neutral_distribution"],
            "corrosion.neutral_distribution",
        )

    @property
    def material_target(self) -> dict[str, float]:
        return normalize_distribution(
            self.material["base_distribution"],
            "material.base_distribution",
        )

    def voltage_distribution_for_era(self, era: str) -> dict[str, float]:
        return normalize_distribution(
            self.voltage["by_installation_era"][era],
            f"voltage.by_installation_era.{era}",
        )

    def location_distribution_for_voltage(self, voltage: str) -> dict[str, float]:
        return normalize_distribution(
            self.location["by_voltage"][voltage],
            f"location.by_voltage.{voltage}",
        )

    def validate(self) -> None:
        """Validate all settings needed by the generation logic."""
        if self.n_rows <= 0:
            raise ValueError("'dataset.n_rows' must be greater than 0.")
        if self.seed < 0:
            raise ValueError("'dataset.seed' must be zero or greater.")
        if self.snapshot_year != 2014:
            raise ValueError(
                "'dataset.snapshot_year' must currently be 2014 because the "
                "installation-year and age-profile logic is calibrated to the "
                "2014 SAPN Overhead Conductor AMP."
            )

        # Voltage
        if "neutral_distribution" not in self.voltage:
            raise ValueError("Missing 'voltage.neutral_distribution'.")
        if "by_installation_era" not in self.voltage:
            raise ValueError("Missing 'voltage.by_installation_era'.")
        if not isinstance(self.voltage["by_installation_era"], Mapping):
            raise ValueError("'voltage.by_installation_era' must be a mapping.")
        if set(self.voltage_target) != EXPECTED_VOLTAGE_CATEGORIES:
            raise ValueError(
                "voltage.neutral_distribution must contain exactly these categories: "
                f"{sorted(EXPECTED_VOLTAGE_CATEGORIES)}"
            )
        by_era = self.voltage["by_installation_era"]
        missing_eras = EXPECTED_ERA_KEYS - set(by_era)
        if missing_eras:
            raise ValueError(
                "voltage.by_installation_era is missing required era(s): "
                f"{sorted(missing_eras)}"
            )
        for era in EXPECTED_ERA_KEYS:
            if set(self.voltage_distribution_for_era(era)) != EXPECTED_VOLTAGE_CATEGORIES:
                raise ValueError(
                    f"voltage.by_installation_era.{era} must contain exactly "
                    f"these categories: {sorted(EXPECTED_VOLTAGE_CATEGORIES)}"
                )

        # Location
        if "neutral_distribution" not in self.location:
            raise ValueError("Missing 'location.neutral_distribution'.")
        if "by_voltage" not in self.location or not isinstance(self.location["by_voltage"], Mapping):
            raise ValueError("Missing or invalid 'location.by_voltage'.")
        if set(self.location_target) != EXPECTED_LOCATION_GROUPS:
            raise ValueError(
                "location.neutral_distribution must contain exactly these groups: "
                f"{sorted(EXPECTED_LOCATION_GROUPS)}"
            )
        missing_voltage_location = EXPECTED_VOLTAGE_CATEGORIES - set(self.location["by_voltage"])
        if missing_voltage_location:
            raise ValueError(
                "location.by_voltage is missing voltage category/categories: "
                f"{sorted(missing_voltage_location)}"
            )
        for voltage in EXPECTED_VOLTAGE_CATEGORIES:
            if set(self.location_distribution_for_voltage(voltage)) != EXPECTED_LOCATION_GROUPS:
                raise ValueError(
                    f"location.by_voltage.{voltage} must contain exactly these groups: "
                    f"{sorted(EXPECTED_LOCATION_GROUPS)}"
                )

        # Corrosion
        if "neutral_distribution" not in self.corrosion:
            raise ValueError("Missing 'corrosion.neutral_distribution'.")
        if set(self.corrosion_target) != EXPECTED_CORROSION_ZONES:
            raise ValueError(
                "corrosion.neutral_distribution must contain exactly these zones: "
                f"{sorted(EXPECTED_CORROSION_ZONES)}"
            )
        validate_effects(
            self.corrosion.get("location_effects", {}),
            EXPECTED_CORROSION_ZONES,
            "corrosion.location_effects",
            allowed_groups=EXPECTED_LOCATION_GROUPS,
        )
        validate_effects(
            self.corrosion.get("voltage_effects", {}),
            EXPECTED_CORROSION_ZONES,
            "corrosion.voltage_effects",
            allowed_groups=EXPECTED_VOLTAGE_CATEGORIES,
        )

        # Material
        if "base_distribution" not in self.material:
            raise ValueError("Missing 'material.base_distribution'.")
        if set(self.material_target) != EXPECTED_MATERIAL_CLASSES:
            raise ValueError(
                "material.base_distribution must contain exactly these classes: "
                f"{sorted(EXPECTED_MATERIAL_CLASSES)}"
            )
        validate_effects(
            self.material.get("voltage_effects", {}),
            EXPECTED_MATERIAL_CLASSES,
            "material.voltage_effects",
            allowed_groups=EXPECTED_VOLTAGE_CATEGORIES,
        )
        validate_effects(
            self.material.get("era_effects", {}),
            EXPECTED_MATERIAL_CLASSES,
            "material.era_effects",
            allowed_groups={"pre_1950", "from_1980"},
        )
        validate_effects(
            self.material.get("corrosion_effects", {}),
            EXPECTED_MATERIAL_CLASSES,
            "material.corrosion_effects",
            allowed_groups=EXPECTED_CORROSION_ZONES,
        )


def resolve_strength_config_path(
    generation_config: Config,
    profile_name: str | None = None,
    explicit_path: str | Path | None = None,
) -> Path:
    """Resolve a strength-profile YAML path next to generation.yaml."""
    if explicit_path is not None:
        return Path(explicit_path).expanduser().resolve()

    profile = profile_name or DEFAULT_STRENGTH_PROFILE
    path = (
        generation_config.config_path.parent
        / "relationship_strength"
        / f"{profile}.yaml"
    )
    return path.resolve()


# ---------------------------------------------------------------------------
# 2. Reusable helpers
# ---------------------------------------------------------------------------


def weighted_choice(
    rng: np.random.Generator,
    labels: Sequence[Any],
    probs: Sequence[float],
    size: int | None = None,
):
    """Choose values using validated, normalized probabilities."""
    labels = list(labels)
    probabilities = np.asarray(probs, dtype=float)

    if not labels:
        raise ValueError("weighted_choice requires at least one label.")
    if len(labels) != len(probabilities):
        raise ValueError("weighted_choice labels/probabilities length mismatch.")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("weighted_choice probabilities must be finite.")
    if np.any(probabilities < 0):
        raise ValueError("weighted_choice probabilities cannot be negative.")

    total = probabilities.sum()
    if total <= 0:
        raise ValueError("weighted_choice probabilities must sum to > 0.")

    probabilities = probabilities / total
    return rng.choice(labels, size=size, p=probabilities)


def blend_probabilities(
    neutral_distribution: Mapping[str, float],
    conditional_distribution: Mapping[str, float],
    strength: float,
) -> dict[str, float]:
    """Interpolate from a neutral distribution to a conditional distribution."""
    if not 0.0 <= strength <= 1.0:
        raise ValueError("Relationship strength must be between 0.0 and 1.0.")
    if set(neutral_distribution) != set(conditional_distribution):
        raise ValueError(
            "Neutral and conditional distributions must contain identical labels."
        )

    blended = {
        label: (
            float(neutral_distribution[label])
            + strength
            * (float(conditional_distribution[label]) - float(neutral_distribution[label]))
        )
        for label in neutral_distribution
    }
    return normalize_distribution(blended, "blended_distribution")


def apply_scaled_effects(
    probabilities: dict[str, float],
    effects: Mapping[str, Any],
    strength: float,
) -> None:
    """Apply additive configured effects multiplied by a relationship strength."""
    for label, raw_effect in effects.items():
        probabilities[label] += float(raw_effect) * strength


def clipped_lognormal(
    rng: np.random.Generator,
    median: float,
    sigma: float,
    low: float,
    high: float,
) -> float:
    """Generate a positive right-skewed value inside a defined range."""
    value = rng.lognormal(mean=np.log(median), sigma=sigma)
    return float(np.clip(value, low, high))


# ---------------------------------------------------------------------------
# 3. Installation year / age
# ---------------------------------------------------------------------------


def sample_installation_year(rng: np.random.Generator) -> int:
    """
    Generate installation year from three age bands calibrated to the 2014 AMP.

    Published 2014 constraints:
    - about 59% of overhead conductors were older than 50 years
    - about 2% were younger than 20 years
    - strong historical installation peaks in 1955, 1956, 1958 and 1966

    Exact year-level probabilities within those broad constraints are synthetic.
    """
    age_band = weighted_choice(
        rng,
        labels=["older_than_50", "age_20_to_50", "younger_than_20"],
        probs=[0.59, 0.39, 0.02],
    )

    if age_band == "older_than_50":
        if rng.random() < 0.48:
            return int(
                weighted_choice(
                    rng,
                    labels=[1955, 1956, 1958],
                    probs=[0.36, 0.29, 0.35],
                )
            )

        if rng.random() < 0.78:
            return int(rng.integers(1948, 1964))
        return int(rng.integers(1930, 1948))

    if age_band == "age_20_to_50":
        if rng.random() < 0.28:
            return 1966
        return int(rng.integers(1964, 1995))

    return int(rng.integers(1995, 2015))


# ---------------------------------------------------------------------------
# 4. Voltage
# ---------------------------------------------------------------------------


def sample_voltage(
    rng: np.random.Generator,
    installation_year: int,
    config: Config,
    strengths: StrengthConfig,
) -> str:
    """Generate voltage using controlled installation-year relationship strength."""
    if installation_year <= 1949:
        era = "pre_1950"
    elif installation_year <= 1977:
        era = "1950_1977"
    elif installation_year <= 1994:
        era = "1978_1994"
    else:
        era = "from_1995"

    distribution = blend_probabilities(
        config.voltage_target,
        config.voltage_distribution_for_era(era),
        strengths.get("installation_year_to_voltage"),
    )
    return str(
        weighted_choice(
            rng,
            labels=list(distribution),
            probs=list(distribution.values()),
        )
    )


# ---------------------------------------------------------------------------
# 5. Location, corrosion and feeder
# ---------------------------------------------------------------------------


def sample_location_group(
    rng: np.random.Generator,
    voltage: str,
    config: Config,
    strengths: StrengthConfig,
) -> str:
    """Generate location by blending neutral and voltage-specific distributions."""
    distribution = blend_probabilities(
        config.location_target,
        config.location_distribution_for_voltage(voltage),
        strengths.get("voltage_to_location"),
    )
    return str(
        weighted_choice(
            rng,
            labels=list(distribution),
            probs=list(distribution.values()),
        )
    )


def sample_corrosion_zone(
    rng: np.random.Generator,
    voltage: str,
    location_group: str,
    config: Config,
    strengths: StrengthConfig,
) -> str:
    """Generate corrosion using separately controlled location and voltage effects."""
    probabilities = dict(config.corrosion_target)

    location_effect = config.corrosion.get("location_effects", {}).get(
        location_group, {}
    )
    apply_scaled_effects(
        probabilities,
        location_effect,
        strengths.get("location_to_corrosion"),
    )

    voltage_effect = config.corrosion.get("voltage_effects", {}).get(voltage, {})
    apply_scaled_effects(
        probabilities,
        voltage_effect,
        strengths.get("voltage_to_corrosion"),
    )

    labels = list(config.corrosion_target)
    probs = np.array([probabilities[label] for label in labels], dtype=float)
    probs = np.clip(probs, 0.01, None)
    probs = probs / probs.sum()

    return str(weighted_choice(rng, labels, probs))


def make_feeder_map() -> dict[str, list[str]]:
    """Create synthetic feeder IDs grouped by location."""
    return {
        "Urban": [f"U_FDR_{i:03d}" for i in range(1, 31)],
        "Regional": [f"G_FDR_{i:03d}" for i in range(1, 26)],
        "Rural": [f"R_FDR_{i:03d}" for i in range(1, 46)],
    }


def sample_feeder(
    rng: np.random.Generator,
    location_group: str,
    feeder_map: Mapping[str, Sequence[str]],
) -> str:
    return str(rng.choice(feeder_map[location_group]))


# ---------------------------------------------------------------------------
# 6. Distance to sea
# ---------------------------------------------------------------------------

def sample_distance_to_sea(
    rng: np.random.Generator,
    location_group: str,
    corrosion_zone: str,
) -> float:
    """
    Generate a plausible synthetic enrichment variable.

    The AMP supports a coastal/saline corrosion relationship but does not
    provide exact distance thresholds; these ranges are therefore assumptions.
    """
    zone_ranges: dict[str, tuple[float, float]] = {
        "Very Severe": (0.2, 15.0),
        "Severe": (2.0, 70.0),
        "Low": (20.0, 300.0),
    }

    low, high = zone_ranges[corrosion_zone]

    if location_group == "Urban":
        high = min(high, 100.0)
    elif location_group == "Rural" and corrosion_zone == "Low":
        high = 350.0

    x = rng.beta(1.6, 2.8)
    return round(low + x * (high - low), 2)


# ---------------------------------------------------------------------------
# 7. Material
# ---------------------------------------------------------------------------


def sample_material(
    rng: np.random.Generator,
    installation_year: int,
    voltage: str,
    corrosion_zone: str,
    config: Config,
    strengths: StrengthConfig,
) -> str:
    """Generate material using independently controlled direct relationships."""
    probabilities = dict(config.material_target)

    apply_scaled_effects(
        probabilities,
        config.material.get("voltage_effects", {}).get(voltage, {}),
        strengths.get("voltage_to_material"),
    )

    era: str | None = None
    if installation_year <= 1949:
        era = "pre_1950"
    elif installation_year >= 1980:
        era = "from_1980"

    if era is not None:
        apply_scaled_effects(
            probabilities,
            config.material.get("era_effects", {}).get(era, {}),
            strengths.get("installation_year_to_material"),
        )

    apply_scaled_effects(
        probabilities,
        config.material.get("corrosion_effects", {}).get(corrosion_zone, {}),
        strengths.get("corrosion_to_material"),
    )

    labels = list(config.material_target)
    probs = np.array([probabilities[label] for label in labels], dtype=float)
    probs = np.clip(probs, 0.02, None)
    probs = probs / probs.sum()

    return str(weighted_choice(rng, labels, probs))


# ---------------------------------------------------------------------------
# 8. Conductor type
# ---------------------------------------------------------------------------

def sample_conductor_type(
    rng: np.random.Generator,
    material: str,
) -> str:
    """Generate conductor type conditional on the synthetic material class."""
    if material == "Aluminium":
        return str(
            weighted_choice(
                rng,
                ["AAC", "AAAC", "Aluminium"],
                [1409, 367, 3069],
            )
        )

    if material == "Steel":
        return str(
            weighted_choice(
                rng,
                ["SC/AC", "SC/GZ"],
                [7587, 27769],
            )
        )

    if material == "Copper":
        return "Copper"

    raise ValueError(f"Unexpected material class: {material!r}")


# ---------------------------------------------------------------------------
# 9. Covering
# ---------------------------------------------------------------------------

def sample_covering(
    rng: np.random.Generator,
    installation_year: int,
) -> str:
    """Generate Bare/Covered with a deliberately weak age relationship."""
    if installation_year >= 1995:
        p_covered = 0.08
    elif installation_year >= 1980:
        p_covered = 0.035
    else:
        p_covered = 0.008

    return "Covered" if rng.random() < p_covered else "Bare"


# ---------------------------------------------------------------------------
# 10. Segment length
# ---------------------------------------------------------------------------

def sample_length_km(
    rng: np.random.Generator,
    location_group: str,
    voltage: str,
) -> float:
    """Generate a positive, right-skewed synthetic segment length."""
    if location_group == "Urban":
        median = 0.18
    elif location_group == "Regional":
        median = 0.35
    else:
        median = 0.55

    if voltage == "19 kV SWER":
        median *= 1.35
    elif voltage == "LV":
        median *= 0.70

    return round(
        clipped_lognormal(
            rng,
            median=median,
            sigma=0.65,
            low=0.03,
            high=5.0,
        ),
        3,
    )


# ---------------------------------------------------------------------------
# 11. Generate complete reference dataset
# ---------------------------------------------------------------------------

def generate_dataset(config: Config, strengths: StrengthConfig) -> pd.DataFrame:
    rng = np.random.default_rng(config.seed)
    feeder_map = make_feeder_map()
    rows: list[dict[str, Any]] = []

    for i in range(1, config.n_rows + 1):
        installation_year = sample_installation_year(rng)
        asset_age = config.snapshot_year - installation_year

        voltage = sample_voltage(rng, installation_year, config, strengths)
        location = sample_location_group(rng, voltage, config, strengths)
        corrosion = sample_corrosion_zone(
            rng, voltage, location, config, strengths
        )
        feeder = sample_feeder(rng, location, feeder_map)

        distance_to_sea = sample_distance_to_sea(
            rng,
            location_group=location,
            corrosion_zone=corrosion,
        )

        material = sample_material(
            rng,
            installation_year=installation_year,
            voltage=voltage,
            corrosion_zone=corrosion,
            config=config,
            strengths=strengths,
        )

        conductor_type = sample_conductor_type(rng, material)
        covering = sample_covering(rng, installation_year)
        length_km = sample_length_km(rng, location, voltage)

        rows.append(
            {
                "segment_id": f"SEG_{i:06d}",
                "feeder_id": feeder,
                "installation_year": installation_year,
                "asset_age": asset_age,
                "voltage_category": voltage,
                "location_group": location,
                "corrosion_zone": corrosion,
                "distance_to_sea_km": distance_to_sea,
                "material": material,
                "conductor_type": conductor_type,
                "covering": covering,
                "length_km": length_km,
                "data_status": "Synthetic",
                "record_version": "Complete reference",
            }
        )

    return pd.DataFrame(rows, columns=REQUIRED_OUTPUT_COLUMNS)


# ---------------------------------------------------------------------------
# 12. Route-length calibration
# ---------------------------------------------------------------------------


def calibrate_route_length_marginals(
    df: pd.DataFrame,
    config: Config,
    iterations: int = 12,
) -> pd.DataFrame:
    """
    Iteratively calibrate segment lengths to configured route-length marginals.

    Category membership is not changed; only generated segment lengths are
    rescaled. The aggregate total route length is preserved.
    """
    if iterations <= 0:
        raise ValueError("Calibration iterations must be greater than 0.")

    out = df.copy()

    targets = [
        ("voltage_category", config.voltage_target),
        ("corrosion_zone", config.corrosion_target),
        ("material", config.material_target),
    ]

    for column, target in targets:
        observed_categories = set(out[column].dropna().astype(str).unique())
        missing_categories = set(target) - observed_categories
        if missing_categories:
            raise ValueError(
                f"Cannot calibrate '{column}': generated data does not contain "
                f"target categories {sorted(missing_categories)}. "
                "Increase n_rows or review generation probabilities."
            )

    for _ in range(iterations):
        for column, target in targets:
            total_length = float(out["length_km"].sum())
            if total_length <= 0:
                raise ValueError("Total route length must be positive.")

            current = (
                out.groupby(column, observed=False)["length_km"].sum()
                / total_length
            )

            factors: dict[str, float] = {}
            for category, desired_share in target.items():
                current_share = float(current.get(category, 0.0))
                if current_share <= 0:
                    raise ValueError(
                        f"Cannot calibrate '{column}.{category}': current route "
                        "length share is zero."
                    )
                factors[category] = desired_share / current_share

            out["length_km"] = (
                out["length_km"]
                * out[column].map(factors).fillna(1.0)
            )

            # Preserve the total synthetic route length after each adjustment.
            new_total = float(out["length_km"].sum())
            if new_total <= 0:
                raise ValueError("Calibration produced a non-positive route length.")
            out["length_km"] *= total_length / new_total

    out["length_km"] = out["length_km"].clip(lower=0.01).round(4)
    return out


# ---------------------------------------------------------------------------
# 13. Validation helpers
# ---------------------------------------------------------------------------


def row_share(df: pd.DataFrame, column: str) -> dict[str, float]:
    return (
        df[column]
        .value_counts(normalize=True)
        .sort_index()
        .round(4)
        .to_dict()
    )


def route_length_share(df: pd.DataFrame, column: str) -> dict[str, float]:
    grouped = df.groupby(column, observed=False)["length_km"].sum()
    shares = grouped / grouped.sum()
    return shares.sort_index().round(6).to_dict()


def distribution_deviation(
    observed: Mapping[str, float],
    target: Mapping[str, float],
) -> dict[str, float]:
    """Absolute share difference by category."""
    return {
        category: round(abs(float(observed.get(category, 0.0)) - expected), 6)
        for category, expected in target.items()
    }


def validate_generated_dataset(
    df: pd.DataFrame,
    config: Config,
    route_share_tolerance: float = 0.01,
) -> None:
    """Fail fast if the generated reference dataset violates core invariants."""
    errors: list[str] = []

    missing_columns = set(REQUIRED_OUTPUT_COLUMNS) - set(df.columns)
    if missing_columns:
        errors.append(f"missing columns: {sorted(missing_columns)}")

    if len(df) != config.n_rows:
        errors.append(
            f"row count {len(df)} does not match configured n_rows={config.n_rows}"
        )

    if df.isna().any().any():
        errors.append(f"dataset contains {int(df.isna().sum().sum())} missing cells")

    if "segment_id" in df and df["segment_id"].duplicated().any():
        errors.append(
            f"dataset contains {int(df['segment_id'].duplicated().sum())} duplicate segment_id values"
        )

    if {"asset_age", "installation_year"}.issubset(df.columns):
        expected_age = config.snapshot_year - df["installation_year"]
        age_mismatch = (df["asset_age"] != expected_age).sum()
        if age_mismatch:
            errors.append(f"{int(age_mismatch)} asset_age values are inconsistent")

    if "installation_year" in df:
        invalid_years = (
            (df["installation_year"] < 1900)
            | (df["installation_year"] > config.snapshot_year)
        ).sum()
        if invalid_years:
            errors.append(f"{int(invalid_years)} installation years are invalid")

    if "length_km" in df:
        invalid_lengths = (~np.isfinite(df["length_km"]) | (df["length_km"] <= 0)).sum()
        if invalid_lengths:
            errors.append(f"{int(invalid_lengths)} segment lengths are invalid")

    category_checks = [
        ("voltage_category", EXPECTED_VOLTAGE_CATEGORIES),
        ("material", EXPECTED_MATERIAL_CLASSES),
        ("corrosion_zone", EXPECTED_CORROSION_ZONES),
        ("location_group", {"Rural", "Regional", "Urban"}),
        ("covering", {"Bare", "Covered"}),
    ]

    for column, allowed in category_checks:
        if column not in df:
            continue
        unexpected = set(df[column].dropna().astype(str).unique()) - allowed
        if unexpected:
            errors.append(f"{column} has unexpected values: {sorted(unexpected)}")

    route_targets = [
        ("voltage_category", config.voltage_target),
        ("corrosion_zone", config.corrosion_target),
        ("material", config.material_target),
    ]

    for column, target in route_targets:
        if column not in df or "length_km" not in df:
            continue
        observed = route_length_share(df, column)
        deviations = distribution_deviation(observed, target)
        too_large = {
            category: deviation
            for category, deviation in deviations.items()
            if deviation > route_share_tolerance
        }
        if too_large:
            errors.append(
                f"{column} route-length share deviation exceeds "
                f"{route_share_tolerance:.3f}: {too_large}"
            )

    if errors:
        formatted = "\n - ".join(errors)
        raise ValueError(f"Generated dataset validation failed:\n - {formatted}")


def validate_dataset(
    df: pd.DataFrame,
    config: Config,
    strengths: StrengthConfig,
) -> dict[str, Any]:
    """Create transparent validation statistics for reporting and audit."""
    voltage_route = route_length_share(df, "voltage_category")
    corrosion_route = route_length_share(df, "corrosion_zone")
    material_route = route_length_share(df, "material")

    return {
        "config_file": str(config.config_path),
        "strength_config_file": str(strengths.config_path),
        "strength_profile": strengths.profile_name,
        "relationship_strengths": dict(strengths.relationship_strengths),
        "seed": config.seed,
        "rows": len(df),
        "snapshot_year": config.snapshot_year,
        "missing_cells": int(df.isna().sum().sum()),
        "duplicate_segment_ids": int(df["segment_id"].duplicated().sum()),
        "mean_age": round(float(df["asset_age"].mean()), 2),
        "share_age_over_50": round(float((df["asset_age"] > 50).mean()), 4),
        "share_age_under_20": round(float((df["asset_age"] < 20).mean()), 4),
        "voltage_row_share": row_share(df, "voltage_category"),
        "voltage_route_length_share": voltage_route,
        "voltage_route_length_target": config.voltage_target,
        "voltage_route_length_deviation": distribution_deviation(
            voltage_route, config.voltage_target
        ),
        "corrosion_row_share": row_share(df, "corrosion_zone"),
        "corrosion_route_length_share": corrosion_route,
        "corrosion_route_length_target": config.corrosion_target,
        "corrosion_route_length_deviation": distribution_deviation(
            corrosion_route, config.corrosion_target
        ),
        "material_row_share": row_share(df, "material"),
        "material_route_length_share": material_route,
        "material_route_length_target": config.material_target,
        "material_route_length_deviation": distribution_deviation(
            material_route, config.material_target
        ),
        "covering_row_share": row_share(df, "covering"),
        "total_route_length_km_in_synthetic_sample": round(
            float(df["length_km"].sum()), 2
        ),
    }


# ---------------------------------------------------------------------------
# 14. Assumptions / provenance log
# ---------------------------------------------------------------------------


def format_distribution(distribution: Mapping[str, float]) -> str:
    return ", ".join(
        f"{label} {share * 100:.1f}%"
        for label, share in distribution.items()
    )


def assumptions_table(config: Config, strengths: StrengthConfig) -> pd.DataFrame:
    """Return a concise provenance table for generated-data assumptions."""
    rows = [
        {
            "item": "generation_config",
            "value": str(config.config_path),
            "basis": "External YAML configuration",
            "status": "Reproducibility control",
        },
        {
            "item": "relationship_strength_profile",
            "value": strengths.profile_name,
            "basis": str(strengths.config_path),
            "status": "Controlled synthetic experiment parameter",
        },
        {
            "item": "relationship_strengths",
            "value": ", ".join(
                f"{name}={value:.2f}"
                for name, value in strengths.relationship_strengths.items()
            ),
            "basis": "External relationship-strength YAML configuration",
            "status": "Controlled synthetic experiment parameter",
        },
        {
            "item": "snapshot_year",
            "value": str(config.snapshot_year),
            "basis": "SAPN AMP publication/network snapshot",
            "status": "SAPN grounded",
        },
        {
            "item": "installation_year_profile",
            "value": "Strong 1955-1977 expansion; extra peaks 1955, 1956, 1958, 1966",
            "basis": "SAPN AMP age profile and text",
            "status": "SAPN grounded shape; exact probabilities synthetic",
        },
        {
            "item": "asset_age",
            "value": f"{config.snapshot_year} - installation_year",
            "basis": "Deterministic derivation",
            "status": "Derived",
        },
        {
            "item": "voltage_categories",
            "value": ", ".join(config.voltage_target.keys()),
            "basis": "SAPN AMP Table 1",
            "status": "SAPN grounded",
        },
        {
            "item": "voltage_route_length_target",
            "value": format_distribution(config.voltage_target),
            "basis": "generation configuration / SAPN aggregate evidence",
            "status": "Configuration controlled",
        },
        {
            "item": "voltage_by_era",
            "value": "Older pre-1950 biased to 33 kV; 1955-1977 biased to SWER/11 kV",
            "basis": "SAPN AMP text",
            "status": "SAPN grounded direction; exact probabilities synthetic",
        },
        {
            "item": "corrosion_route_length_target",
            "value": format_distribution(config.corrosion_target),
            "basis": "generation configuration / SAPN corrosion-zone evidence",
            "status": "Configuration controlled",
        },
        {
            "item": "material_target",
            "value": ", ".join(config.material_target.keys()),
            "basis": "Project pilot scope + SAPN conductor groups",
            "status": "Project scope",
        },
        {
            "item": "material_route_length_reference",
            "value": format_distribution(config.material_target),
            "basis": "Normalized known non-ACSR route lengths from SAPN Table 2",
            "status": "SAPN-derived reference, not a direct material-field distribution",
        },
        {
            "item": "ACSR",
            "value": "Excluded from current 3-class target",
            "basis": "Composite conductor group; cannot cleanly map to one material class",
            "status": "Project modelling decision",
        },
        {
            "item": "conductor_type",
            "value": "AAC, AAAC, Aluminium, Copper, SC/AC, SC/GZ",
            "basis": "SAPN AMP Table 2",
            "status": "SAPN grounded",
        },
        {
            "item": "conductor_type_as_predictor",
            "value": "Excluded from material-model-ready output",
            "basis": "Would reveal the target and create leakage",
            "status": "Modelling safeguard",
        },
        {
            "item": "covering",
            "value": "Bare / Covered; low overall covered share",
            "basis": "SAPN AMP says covered conductor is a small share of the network",
            "status": "SAPN grounded marginal; exact age relationship synthetic",
        },
        {
            "item": "location_group",
            "value": "Urban / Regional / Rural",
            "basis": "AMP uses rural/urban but does not publish a full population split",
            "status": "Synthetic assumption",
        },
        {
            "item": "distance_to_sea_km",
            "value": "Conditional on corrosion zone and location group",
            "basis": "Coastal/saline corrosion relationship; exact km thresholds not published",
            "status": "Synthetic enrichment assumption",
        },
        {
            "item": "segment_length",
            "value": "Context-dependent lognormal distribution",
            "basis": "AMP gives total route length but no segment-length distribution",
            "status": "Synthetic assumption",
        },
        {
            "item": "route_length_calibration",
            "value": "Iterative calibration to voltage, corrosion and material route-length marginals",
            "basis": "Published percentages are route-length shares rather than row shares",
            "status": "Synthetic calibration step",
        },
        {
            "item": "missing_values",
            "value": "None in the complete reference dataset",
            "basis": "Controlled masking is a separate experimental step",
            "status": "Project design",
        },
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 15. Save outputs
# ---------------------------------------------------------------------------


def save_outputs(
    df: pd.DataFrame,
    summary: Mapping[str, Any],
    config: Config,
    strengths: StrengthConfig,
) -> Path:
    # Keep outputs for each strength profile separate so experiments do not overwrite.
    output_dir = config.output_dir / strengths.profile_name
    output_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(
        output_dir / "overhead_conductors_complete_reference.csv",
        index=False,
        encoding="utf-8",
    )

    assumptions_table(config, strengths).to_csv(
        output_dir / "generation_assumptions.csv",
        index=False,
        encoding="utf-8",
    )

    with (output_dir / "generation_summary.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=2)

    # Remove the direct target-leakage field. Identifiers/non-feature metadata
    # can be removed later by the modelling pipeline as appropriate.
    model_df = df.drop(columns=["conductor_type"]).copy()
    model_df.to_csv(
        output_dir / "material_model_ready_data.csv",
        index=False,
        encoding="utf-8",
    )

    return output_dir


# ---------------------------------------------------------------------------
# 16. CLI and main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a validated synthetic overhead-conductor dataset."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "Path to generation.yaml/yml. If omitted, the script searches "
            "upward for config/overhead_conductors/generation.yaml or .yml."
        ),
    )
    parser.add_argument(
        "--strength-profile",
        choices=[
            "strength_100", "strength_90", 
            "strength_80","strength_70","strength_60",
            "strength_50","strength_40","strength_30",
            "strength_20", "strength_10","strength_0","your_custom_profile"],
        default=DEFAULT_STRENGTH_PROFILE,
        help=(
            "Relationship-strength profile name. The file is loaded from "
            "config/overhead_conductors/relationship_strength/<profile>.yaml. "
            f"Default: {DEFAULT_STRENGTH_PROFILE}."
        ),
    )
    parser.add_argument(
        "--strength-config",
        type=Path,
        default=None,
        help=(
            "Optional explicit path to a relationship-strength YAML file. "
            "When supplied, this overrides --strength-profile."
        ),
    )
    return parser.parse_args()


def print_summary(summary: Mapping[str, Any], output_dir: Path) -> None:
    print("\nSynthetic dataset generated successfully.\n")
    print(f"Rows: {summary['rows']:,}")
    print(f"Missing cells: {summary['missing_cells']}")
    print(f"Duplicate segment IDs: {summary['duplicate_segment_ids']}")
    print(f"Mean age: {summary['mean_age']}")
    print(f"Share age > 50: {summary['share_age_over_50']}")
    print(f"Share age < 20: {summary['share_age_under_20']}")

    print("\nMaterial row shares:")
    for key, value in summary["material_row_share"].items():
        print(f"  {key}: {value:.4f}")

    print("\nVoltage route-length shares:")
    for key, value in summary["voltage_route_length_share"].items():
        print(f"  {key}: {value:.4f}")

    print("\nCorrosion route-length shares:")
    for key, value in summary["corrosion_route_length_share"].items():
        print(f"  {key}: {value:.4f}")

    print("\nFiles written to:")
    print(f"  {output_dir}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    try:
        config_path = args.config or find_default_config_path()
        LOGGER.info("Loading generation configuration: %s", config_path)

        config = Config.from_yaml(config_path)
        LOGGER.info("Generation configuration validated successfully.")

        strength_path = resolve_strength_config_path(
            config,
            profile_name=args.strength_profile,
            explicit_path=args.strength_config,
        )
        LOGGER.info("Loading relationship-strength configuration: %s", strength_path)
        strengths = StrengthConfig.from_yaml(strength_path)
        LOGGER.info(
            "Relationship-strength profile '%s' validated successfully.",
            strengths.profile_name,
        )

        LOGGER.info(
            "Configured run: n_rows=%s, seed=%s, snapshot_year=%s, profile=%s",
            f"{config.n_rows:,}",
            config.seed,
            config.snapshot_year,
            strengths.profile_name,
        )

        LOGGER.info("Generating complete synthetic reference dataset...")
        df = generate_dataset(config, strengths)

        LOGGER.info("Calibrating route-length marginals...")
        df = calibrate_route_length_marginals(df, config)

        LOGGER.info("Validating generated dataset...")
        validate_generated_dataset(df, config)
        summary = validate_dataset(df, config, strengths)

        LOGGER.info("Saving outputs...")
        output_dir = save_outputs(df, summary, config, strengths)

        print_summary(summary, output_dir)

    except (FileNotFoundError, OSError, ValueError) as exc:
        LOGGER.error("Generation failed: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()