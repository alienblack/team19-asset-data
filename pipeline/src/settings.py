"""Asset-specific settings loaded from ``config/<asset>/pipeline.yaml``.

Every value has a default so a partial file is valid; the command line can
override the common ones. Profile file paths are resolved relative to that
asset's config folder unless absolute.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

def _find_project_root() -> Path:
    """The directory holding ``config/`` and the package: ``pipeline/`` in the
    flat layout, the repository root in a ``src/`` layout."""
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "config").is_dir() and (candidate / "pyproject.toml").is_file():
            return candidate
    return here.parents[1]


REPO_ROOT = _find_project_root()
DEFAULT_ASSET = "overhead_conductors"
DEFAULT_CONFIG_DIR = REPO_ROOT / "config" / DEFAULT_ASSET


class SettingsError(ValueError):
    """Raised when pipeline.yaml holds an invalid value."""


@dataclass(frozen=True)
class ReconstructionSettings:
    enabled: bool = True
    mask_rate: float = 0.10
    mask_seed: int = 319
    repeats: int = 1
    stratify_categorical: bool = True
    include_reference: bool = True


@dataclass(frozen=True)
class EvaluationModeSettings:
    """``evaluation:`` block of pipeline.yaml (ground-truth evaluation mode)."""

    fractions: dict[str, float] = field(default_factory=lambda: {"fit": 0.60, "validation": 0.15, "test": 0.25})
    split_seed: int = 219
    strategy: str = "group"
    group_column: str | None = "feeder_id"
    mask_rate: float = 0.20
    mask_seed: int = 419
    repeats: int = 1
    protocols: tuple[str, ...] = ("unseen_records", "transductive")
    methods: tuple[str, ...] | None = None


@dataclass(frozen=True)
class PipelineSettings:
    seed: int = 219
    data_schema: Path = DEFAULT_CONFIG_DIR / "data_schema.yaml"
    cleaning_rules: Path = DEFAULT_CONFIG_DIR / "cleaning_rules.yaml"
    imputation: Path = DEFAULT_CONFIG_DIR / "imputation.yaml"
    stages: dict[str, bool] = field(
        default_factory=lambda: {
            "quality": True,
            "profiling": True,
            "cleaning": True,
            "imputation": True,
            "validation": True,
            "reporting": True,
        }
    )
    quality_fail_on_error: bool = True
    method: str = "predictive_missforest"
    reference_method: str | None = "median_mode"
    save_model: bool = True
    overrides: dict[str, Any] = field(default_factory=dict)
    reconstruction: ReconstructionSettings = field(default_factory=ReconstructionSettings)
    distribution_enabled: bool = True
    year_bin_width: int = 10
    low_confidence_fraction: float = 0.10
    low_confidence_min_rows: int = 20
    report_title: str = "Overhead Conductors - Data Quality and Imputation Report"
    embed_figures: bool = True
    evaluation: EvaluationModeSettings = field(default_factory=EvaluationModeSettings)
    #: ``release`` refuses to run stages on invalid or unapproved configuration;
    #: ``development`` records the state and continues.
    execution_mode: str = "release"
    source_path: Path | None = None

    @property
    def asset(self) -> str:
        return self.config_dir.name

    @property
    def config_dir(self) -> Path:
        return self.source_path.parent if self.source_path is not None else self.data_schema.parent

    @property
    def approvals_path(self) -> Path:
        return self.config_dir / "approvals.json"

    @property
    def config_files(self) -> dict[str, Path]:
        """The four configuration files by their canonical names."""
        files = {
            "data_schema.yaml": self.data_schema,
            "cleaning_rules.yaml": self.cleaning_rules,
            "imputation.yaml": self.imputation,
        }
        if self.source_path is not None:
            files["pipeline.yaml"] = self.source_path
        return files

    def with_updates(self, **changes: Any) -> "PipelineSettings":
        from dataclasses import replace

        return replace(self, **{k: v for k, v in changes.items() if v is not None})


def _resolve(path_value: Any, base: Path) -> Path:
    path = Path(str(path_value))
    return path if path.is_absolute() else (base / path).resolve()


def _bool(section: Mapping[str, Any], key: str, default: bool) -> bool:
    value = section.get(key, default)
    if type(value) is not bool:
        raise SettingsError(f"{key} must be true or false.")
    return value


def parse_settings(raw: Mapping[str, Any] | None, base: Path = DEFAULT_CONFIG_DIR, source: Path | None = None) -> PipelineSettings:
    raw = raw or {}
    if not isinstance(raw, Mapping):
        raise SettingsError("pipeline.yaml must contain a mapping.")
    paths = raw.get("paths") or {}
    stages = dict(PipelineSettings().stages)
    for key, value in (raw.get("stages") or {}).items():
        if key not in stages:
            raise SettingsError(f"Unknown stage '{key}'.")
        if type(value) is not bool:
            raise SettingsError(f"stages.{key} must be true or false.")
        stages[key] = value
    imputation = raw.get("imputation") or {}
    validation = raw.get("validation") or {}
    reconstruction = validation.get("reconstruction") or {}
    distribution = validation.get("distribution") or {}
    confidence = raw.get("confidence") or {}
    report = raw.get("report") or {}
    quality = raw.get("quality") or {}
    execution = raw.get("execution") or {}
    if not isinstance(execution, Mapping):
        raise SettingsError("'execution' must be a mapping.")
    evaluation = raw.get("evaluation") or {}
    if not isinstance(evaluation, Mapping):
        raise SettingsError("'evaluation' must be a mapping.")
    split = evaluation.get("split") or {}
    fractions = {k: float(split.get(k, d)) for k, d in (("fit", 0.60), ("validation", 0.15), ("test", 0.25))}
    if any(v < 0 for v in fractions.values()) or abs(sum(fractions.values()) - 1.0) > 1e-9:
        raise SettingsError("evaluation.split fit/validation/test must be non-negative and sum to one.")
    strategy = str(split.get("strategy", "group"))
    if strategy not in ("group", "row"):
        raise SettingsError("evaluation.split.strategy must be 'group' or 'row'.")
    eval_mask = evaluation.get("mask") or {}
    eval_rate = eval_mask.get("rate", 0.20)
    if isinstance(eval_rate, bool) or not isinstance(eval_rate, (int, float)) or not 0 < eval_rate < 1:
        raise SettingsError("evaluation.mask.rate must be between zero and one.")
    eval_repeats = eval_mask.get("repeats", 1)
    if type(eval_repeats) is not int or eval_repeats < 1:
        raise SettingsError("evaluation.mask.repeats must be a positive integer.")
    protocols = tuple(evaluation.get("protocols") or ("unseen_records", "transductive"))
    if any(p not in ("unseen_records", "transductive") for p in protocols):
        raise SettingsError("evaluation.protocols may contain only 'unseen_records' and 'transductive'.")
    eval_methods = evaluation.get("methods")
    if eval_methods is not None and not (isinstance(eval_methods, list) and all(isinstance(m, str) for m in eval_methods)):
        raise SettingsError("evaluation.methods must be a list of method names or null.")
    evaluation_settings = EvaluationModeSettings(
        fractions=fractions,
        split_seed=int(split.get("seed", 219)),
        strategy=strategy,
        group_column=split.get("group_column", "feeder_id"),
        mask_rate=float(eval_rate),
        mask_seed=int(eval_mask.get("seed", 419)),
        repeats=eval_repeats,
        protocols=protocols,
        methods=tuple(eval_methods) if eval_methods else None,
    )
    execution_mode = str(execution.get("mode", "release"))
    if execution_mode not in ("release", "development"):
        raise SettingsError("execution.mode must be 'release' or 'development'.")

    seed = raw.get("seed", 219)
    if type(seed) is not int or seed < 0:
        raise SettingsError("seed must be a non-negative integer.")
    rate = reconstruction.get("mask_rate", 0.10)
    if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not 0 < rate < 1:
        raise SettingsError("validation.reconstruction.mask_rate must be between zero and one.")
    repeats = reconstruction.get("repeats", 1)
    if type(repeats) is not int or repeats < 1:
        raise SettingsError("validation.reconstruction.repeats must be a positive integer.")
    fraction = confidence.get("low_confidence_fraction", 0.10)
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not 0 < fraction < 1:
        raise SettingsError("confidence.low_confidence_fraction must be between zero and one.")
    reference = imputation.get("reference_method", "median_mode")
    if reference is not None and not isinstance(reference, str):
        raise SettingsError("imputation.reference_method must be a method name or null.")
    overrides = imputation.get("overrides") or {}
    if not isinstance(overrides, Mapping):
        raise SettingsError("imputation.overrides must be a mapping.")

    return PipelineSettings(
        seed=seed,
        data_schema=_resolve(paths.get("data_schema", "data_schema.yaml"), base),
        cleaning_rules=_resolve(paths.get("cleaning_rules", "cleaning_rules.yaml"), base),
        imputation=_resolve(paths.get("imputation", "imputation.yaml"), base),
        stages=stages,
        quality_fail_on_error=_bool(quality, "fail_on_error", True),
        method=str(imputation.get("method", "predictive_missforest")),
        reference_method=reference,
        save_model=_bool(imputation, "save_model", True),
        overrides=dict(overrides),
        reconstruction=ReconstructionSettings(
            enabled=_bool(reconstruction, "enabled", True),
            mask_rate=float(rate),
            mask_seed=int(reconstruction.get("mask_seed", 319)),
            repeats=repeats,
            stratify_categorical=_bool(reconstruction, "stratify_categorical", True),
            include_reference=_bool(reconstruction, "include_reference", True),
        ),
        distribution_enabled=_bool(distribution, "enabled", True),
        year_bin_width=int(distribution.get("year_bin_width", 10)),
        low_confidence_fraction=float(fraction),
        low_confidence_min_rows=int(confidence.get("min_rows", 20)),
        report_title=str(report.get("title", PipelineSettings().report_title)),
        embed_figures=_bool(report, "embed_figures", True),
        evaluation=evaluation_settings,
        execution_mode=execution_mode,
        source_path=source,
    )


def load_settings(path: str | Path | None = None, *, asset: str = DEFAULT_ASSET) -> PipelineSettings:
    """Load a profile by asset name, or a specific ``pipeline.yaml`` path."""
    if path is None and (not asset or "/" in asset or "\\" in asset or asset in (".", "..")):
        raise SettingsError("Asset must be a directory name inside config/.")
    path = Path(path) if path is not None else REPO_ROOT / "config" / asset / "pipeline.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Pipeline settings not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    base = path.resolve().parent
    return parse_settings(raw, base=base, source=path.resolve())
