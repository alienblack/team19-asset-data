"""Experiment plan and parallel execution with resumable per-task checkpoints.

Truth values are never read here. Predictions are written to disk per task, so an
interrupted run continues from the tasks that have not finished.
"""

from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from functools import lru_cache
from pathlib import Path
import hashlib
import json
import os
import time
import warnings

from joblib import Parallel, delayed, parallel_config
import numpy as np
import pandas as pd
import psutil
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

from .columns import CATEGORY_TARGET, GAS_READINGS, NUMERIC_TARGETS, TARGETS, finalise
from .data import discover, file_hash, read_measured
from .models import (
    CATEGORY_METHODS, CHAINED_CLASSES, CLASS_COUNT, NUMERIC_METHODS,
    CalibratedForest, ChainedRidgeImputer, make_model,
)
from .models_external import EXTERNAL_CLASSES

PROBABILITY_COLUMNS = [f"prob_{i}" for i in range(CLASS_COUNT)]
MODEL_SOURCES = ("columns.py", "data.py", "models.py", "runner.py", "models_external.py", "miceforest_compat.py",
                 "vendor/subteam_a/data.py", "vendor/subteam_a/features.py", "vendor/subteam_a/mixture.py",
                 "vendor/subteam_a/models.py", "vendor/pipeline/comparison_models.py",
                 "vendor/pipeline/predictive_missforest.py", "vendor/pipeline/postprocess.py",
                 "vendor/pipeline/simple_imputer.py", "vendor/pipeline/mice_pmm_logistic.py")
# Every model that fills all columns of a fold in one fit: the study's chains and the other branches' models.
ALL_CHAINS = {**CHAINED_CLASSES, **EXTERNAL_CLASSES}
CHAIN_SETTINGS = ("passes", "linear_alpha", "curve_alpha")
ALL_NUMERIC = "all numeric"
ALL_COLUMNS = "all columns"
ITERATIVE = "IterativeImputer"
STRUCTURE_CHAINS = ("KNN imputer", "Random Forest", "HistGradientBoosting", "CatBoost")  # those with the ridge base
ABLATION_CHAINS = STRUCTURE_CHAINS + tuple(EXTERNAL_CLASSES)
SPATIAL = "spatial"  # folds made of whole blocks of the grid
NO_BASE = "no ridge base"
IMPORTANCE_METHODS = ("Random Forest", "CatBoost")


def default_parameters():
    # Every setting was chosen by the fixed rule on cells that no scenario hides.
    # The records are in outputs/imputation-study/optimisation/clean_selection.
    return {
        "KNN imputer": {"passes": 6, "linear_alpha": 30.0, "curve_alpha": 300.0, "n_neighbors": 80,
                        "weights": "distance"},
        ITERATIVE: {"max_iter": 10, "tol": 1e-3, "initial_strategy": "median",
                    "linear_alpha": 30.0, "curve_alpha": 1000.0},
        "Random Forest": {"passes": 4, "linear_alpha": 30.0, "curve_alpha": 300.0, "n_estimators": 100,
                          "min_samples_leaf": 2, "max_samples": 0.5},
        "HistGradientBoosting": {"passes": 6, "linear_alpha": 30.0, "curve_alpha": 300.0, "max_iter": 400,
                                 "classifier_rounds": 200, "early_stopping": True, "n_iter_no_change": 20,
                                 "validation_fraction": 0.15, "learning_rate": 0.05, "max_leaf_nodes": 31,
                                 "min_samples_leaf": 20, "l2_regularization": 0.0},
        "CatBoost": {"passes": 6, "linear_alpha": 30.0, "curve_alpha": 300.0, "iterations": 800, "depth": 4,
                     "learning_rate": 0.1, "patience": 30, "classifier_rounds": 500},
        "miceforest direct": {"iterations": 3, "rounds": 200, "learning_rate": 0.05, "num_leaves": 31,
                              "min_data_in_leaf": 20, "feature_fraction": 0.7},
        # Subteam A's models. Their package does not record the settings their tuning selected, so these
        # were chosen by the same fixed rule from their own search space: optimisation/merge_selection.
        "Gaussian EM": {"clusters": 6, "steps": 8, "reg": 0.001, "class_weight": 1.8},
        "Masked MLP": {"width": 256, "epochs": 60, "dropout": 0.0, "lr": 0.0015, "decay": 0.0001277, "remask": 0.4},
        # The team pipeline's models, with the settings its guide and run manifest record.
        "LightGBM chained": {"passes": 3, "lgbm_estimators": 100, "lgbm_learning_rate": 0.07, "lgbm_num_leaves": 31},
        "Predictive MissForest": {"n_estimators": 100, "n_passes": 5, "min_samples_leaf": 10,
                                  "max_features": "sqrt", "max_depth": None},
        # The team's MICE with predictive mean matching, with the settings its methodology document records.
        "MICE-PMM": {"chains": 5, "sweeps": 10, "donors": 5, "ridge": 1e-4, "logistic_C": 1.0, "logistic_max_iter": 500},
    }


def default_explanation():
    # Single-target models for importance and calibration. A chain covers every column
    # at once, so a column's influence on one target cannot be read from it.
    return {
        "Random Forest": {"n_estimators": 200, "min_samples_leaf": 2},
        "CatBoost": {"iterations": 500, "depth": 6, "learning_rate": 0.1},
    }


def default_blocks():
    # Predictor groups defined by what the columns measure.
    return {
        "without total_gas and leakage": ("total_gas", "leakage"),
        "without gas columns": GAS_READINGS + ("total_gas",),
        "without life": ("life",),
        "without location": ("x", "y", "altitude"),
        "without install_date": ("install_date",),
    }


@dataclass
class StudyConfig:
    data_directory: str = "data/datasets"
    mechanism_directory: str = "data/datasets_mechanisms"
    mechanisms: tuple = ("age", "value", "group", "errors")  # see mechanisms.py
    sparse_completions: tuple = (0.5, 0.3)  # heavier random missingness than the generator files
    output_directory: str = "outputs/imputation-study/runs"
    representative: str = "completion0.8-noise0.05"
    noise_series: tuple = ("completion0.8-noise0.0", "completion0.8-noise0.05", "completion0.8-noise0.1")
    scenarios: tuple | None = None  # None selects every scenario in the data directory
    seed: int = 219
    stability_seeds: tuple = (220, 221)
    folds: int = 5
    rows: int | None = None  # a smaller sample, for quick checks only
    iterative_budgets: tuple = (5, 30)
    bootstrap_draws: int = 1000
    small_group: int = 30
    material_gap: float = 0.05  # a smaller relative MAE gap is reported as practically tied
    parameters: dict = field(default_factory=default_parameters)
    explanation: dict = field(default_factory=default_explanation)
    blocks: dict = field(default_factory=default_blocks)


def quick_config(**changes):
    """Every stage on a small sample with small models. Results are not findings."""
    config = StudyConfig(
        rows=600, bootstrap_draws=50,
        scenarios=("completion0.7-noise0.05", "completion0.8-noise0.0", "completion0.8-noise0.05",
                   "completion0.8-noise0.1", "completion0.9-noise0.05"),
    )
    config.parameters["KNN imputer"]["passes"] = 2
    config.parameters["Random Forest"].update(n_estimators=10, passes=2)
    config.explanation["Random Forest"]["n_estimators"] = 10
    config.parameters["HistGradientBoosting"].update(max_iter=10, classifier_rounds=10, passes=2)
    config.parameters["CatBoost"].update(iterations=10, classifier_rounds=10, passes=2)
    config.explanation["CatBoost"]["iterations"] = 10
    config.parameters["miceforest direct"].update(rounds=10, iterations=2)
    config.parameters[ITERATIVE]["max_iter"] = 3
    config.parameters["Masked MLP"].update(epochs=3, width=32)
    config.parameters["LightGBM chained"].update(lgbm_estimators=10, passes=2)
    config.parameters["Predictive MissForest"].update(n_estimators=10, n_passes=1)
    config.parameters["MICE-PMM"].update(chains=2, sweeps=2)
    return replace(config, **changes)


@dataclass(frozen=True)
class Task:
    scenario: str
    experiment: str
    variant: str
    seed: int
    fold: int
    target: str
    method: str

    @property
    def task_id(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:20]


def build_plan(names, config, mechanism_names=()):
    """Every planned fit. The folds of a seed are shared by all methods and scenarios."""
    plan = []
    folds = range(config.folds)

    def add(scenario, experiment, variant, seed, fold, target, method):
        plan.append(Task(scenario, experiment, variant, seed, fold, target, method))

    def add_full_set(scenario, experiment, seed, fold):
        for target in NUMERIC_TARGETS:
            for method in NUMERIC_METHODS:
                if method != ITERATIVE and method not in ALL_CHAINS:
                    add(scenario, experiment, "all predictors", seed, fold, target, method)
        for method in CATEGORY_METHODS:
            if method not in ALL_CHAINS:
                add(scenario, experiment, "all predictors", seed, fold, CATEGORY_TARGET, method)
        add(scenario, experiment, "all predictors", seed, fold, ALL_NUMERIC, ITERATIVE)
        for method in ALL_CHAINS:
            add(scenario, experiment, "all predictors", seed, fold, ALL_COLUMNS, method)

    representative = config.representative
    for name in names:
        for fold in folds:
            add_full_set(name, "main", config.seed, fold)
    for seed in config.stability_seeds:
        for fold in folds:
            add_full_set(representative, "stability", seed, fold)
    for fold in folds:
        add_full_set(representative, SPATIAL, config.seed, fold)
    for fold in folds:
        for budget in config.iterative_budgets:
            add(representative, "budget", str(budget), config.seed, fold, ALL_NUMERIC, f"{ITERATIVE} ({budget})")
        for method in STRUCTURE_CHAINS:
            add(representative, "structure", NO_BASE, config.seed, fold, ALL_COLUMNS, method)
        for variant in ("all predictors", "without gas columns"):
            add(representative, "calibration", variant, config.seed, fold, CATEGORY_TARGET,
                "Random Forest calibrated")
    for name in config.noise_series:
        for variant in config.blocks:
            for fold in folds:
                add(name, "ablation", variant, config.seed, fold, ALL_NUMERIC, ITERATIVE)
                for method in ABLATION_CHAINS:
                    add(name, "ablation", variant, config.seed, fold, ALL_COLUMNS, method)
        for target in TARGETS:
            for method in IMPORTANCE_METHODS:
                for fold in folds:
                    add(name, "importance", "all predictors", config.seed, fold, target, method)
    for name in mechanism_names:
        for fold in folds:
            add_full_set(name, "mechanism", config.seed, fold)
    return plan


def plan_table(plan):
    frame = pd.DataFrame([asdict(task) for task in plan])
    return frame.groupby("experiment", sort=False).agg(
        tasks=("method", "size"), scenarios=("scenario", "nunique"), seeds=("seed", "nunique"),
        methods=("method", "nunique"),
    ).reset_index()


def tile_fold_ids(row_ids, folds, seed, side=100, tile=10):
    """Folds made of whole 10 x 10 blocks of the grid, so a held-out row has no training row beside it.

    The generator writes rows in grid order (row = 100 x + y), so a row's position gives its block.
    The position decides only the split; no model receives it.
    """
    rows = np.asarray(row_ids)
    per_side = side // tile
    blocks = (rows // side // tile) * per_side + (rows % side) // tile
    order = np.random.default_rng(seed).permutation(per_side * per_side)
    block_fold = np.empty(per_side * per_side, dtype=int)
    block_fold[order] = np.arange(per_side * per_side) % folds
    return block_fold[blocks]


def fold_ids(rows, folds, seed):
    assignment = np.empty(rows, dtype=int)
    for fold, (_, test) in enumerate(KFold(folds, shuffle=True, random_state=seed).split(np.arange(rows))):
        assignment[test] = fold
    return assignment


def data_folders(root, config):
    return Path(root) / config.data_directory, Path(root) / config.mechanism_directory


@lru_cache(maxsize=4)
def _measured(folders, name, rows, sample_seed):
    frame = read_measured(discover(*folders)[name])
    if rows is not None:
        chosen = np.random.default_rng(sample_seed).choice(len(frame), rows, replace=False)
        frame = frame.iloc[np.sort(chosen)]
    return frame


def load_measured(root, name, config):
    return _measured(tuple(str(f) for f in data_folders(root, config)), name, config.rows, config.seed)


def _rows(target, row_ids, raw, prediction=None, probabilities=None):
    result = pd.DataFrame({
        "row_id": row_ids, "target": target, "raw": np.asarray(raw, dtype=float),
        "prediction": finalise(target, raw) if prediction is None else prediction,
    })
    for index, name in enumerate(PROBABILITY_COLUMNS):
        result[name] = np.nan if probabilities is None else probabilities[:, index]
    return result


def _full_probabilities(classes, values):
    """Align class probabilities to all four model_id levels."""
    result = np.zeros((len(values), CLASS_COUNT))
    result[:, np.asarray(classes, dtype=int)] = values
    return result


def _point_task(task, config, train, test, drop):
    """Fit one target on one fold; returns predictions for its originally missing test cells."""
    model = make_model(task.method, task.target, config, {}, task.seed + task.fold, drop)
    hidden = test[task.target].isna()
    started = time.perf_counter()
    model.fit(train)
    fitted = time.perf_counter()
    if task.target == CATEGORY_TARGET:
        probabilities = _full_probabilities(model.classes_, model.predict_proba(test.loc[hidden]))
        raw = probabilities.argmax(axis=1).astype(float)
        frame = _rows(task.target, test.index[hidden], raw, raw, probabilities)
    else:
        raw = model.predict(test.loc[hidden])
        frame = _rows(task.target, test.index[hidden], raw)
    return frame, fitted - started, time.perf_counter() - fitted, {}


def _iterative_task(task, config, train, test, drop):
    """One chained fit supplies every numeric target for this fold."""
    budget = int(task.variant) if task.experiment == "budget" else None
    model = ChainedRidgeImputer(config.parameters[ITERATIVE], task.seed + task.fold, drop, budget)
    started = time.perf_counter()
    model.fit(train)
    fitted = time.perf_counter()
    filled = model.predict(test)
    frames = []
    for target in filled.columns:
        hidden = test[target].isna()
        frames.append(_rows(target, test.index[hidden], filled.loc[hidden, target]))
    frame = pd.concat(frames, ignore_index=True)
    return frame, fitted - started, time.perf_counter() - fitted, {"n_iter": int(model.n_iter_)}


def _chained_task(task, config, train, test, drop):
    """One chained fit supplies every column for this fold."""
    settings = dict(config.parameters[task.method])
    if task.variant == NO_BASE:
        settings["base"] = "none"
    model = ALL_CHAINS[task.method](settings, task.seed + task.fold, drop)
    started = time.perf_counter()
    model.fit(train)
    fitted = time.perf_counter()
    filled, probabilities = model.predict(test)
    frames = []
    for target in filled.columns:
        hidden = test[target].isna()
        frames.append(_rows(target, test.index[hidden], filled.loc[hidden, target]))
    hidden = test[CATEGORY_TARGET].isna().to_numpy()
    shares = _full_probabilities(model.classes_, probabilities[hidden])
    labels = shares.argmax(axis=1).astype(float)
    frames.append(_rows(CATEGORY_TARGET, test.index[hidden], labels, labels, shares))
    return pd.concat(frames, ignore_index=True), fitted - started, time.perf_counter() - fitted, {}


def _calibration_task(task, config, train, test, drop):
    # The same forest as the raw model_id result, so the comparison shows calibration alone.
    forest = {k: v for k, v in config.parameters["Random Forest"].items() if k not in CHAIN_SETTINGS}
    model = CalibratedForest(forest, task.seed + task.fold, drop)
    hidden = test[CATEGORY_TARGET].isna()
    started = time.perf_counter()
    model.fit(train)
    fitted = time.perf_counter()
    probabilities = _full_probabilities(model.classes_, model.predict_proba(test.loc[hidden]))
    raw = probabilities.argmax(axis=1).astype(float)
    frame = _rows(CATEGORY_TARGET, test.index[hidden], raw, raw, probabilities)
    return frame, fitted - started, time.perf_counter() - fitted, {}


def _importance_task(task, config, train, test):
    """Permutation importance on held-out observed labels of the measured data."""
    model = make_model(task.method, task.target, config, config.explanation[task.method], task.seed + task.fold)
    started = time.perf_counter()
    model.fit(train)
    fitted = time.perf_counter()
    rows = test[test[task.target].notna()]
    labels = rows[task.target].to_numpy()

    def loss(frame):
        predicted = model.predict(frame)
        return float(np.mean(predicted != labels)) if model.categorical else float(np.mean(np.abs(predicted - labels)))

    # The constant-guess loss puts every target on one scale: 1.0 is all signal removed.
    constant = float(np.mean(labels != pd.Series(labels).mode().iloc[0])) if model.categorical \
        else float(np.mean(np.abs(labels - np.median(labels))))
    base = loss(rows)
    generator = np.random.default_rng(task.seed)
    entries = []
    for predictor in model.predictors:
        shuffled = rows.copy()
        shuffled[predictor] = generator.permutation(shuffled[predictor].to_numpy())
        entries.append({"target": task.target, "predictor": predictor,
                        "importance": (loss(shuffled) - base) / constant if constant else np.nan})
    return pd.DataFrame(entries), fitted - started, time.perf_counter() - fitted, {}


def execute_task(task, config, root, run_directory):
    """Run one task in a worker. The JSON marker is written last and commits the result."""
    folder = Path(run_directory) / "tasks"
    marker = folder / f"{task.task_id}.json"
    if marker.exists():
        return json.loads(marker.read_text(encoding="utf-8"))
    measured = load_measured(root, task.scenario, config)
    if task.experiment == SPATIAL:
        assignment = tile_fold_ids(measured.index, config.folds, task.seed)
    else:
        assignment = fold_ids(len(measured), config.folds, task.seed)
    train, test = measured[assignment != task.fold], measured[assignment == task.fold]
    drop = config.blocks.get(task.variant, ())
    status, error, fit_seconds, predict_seconds, extra = "success", "", np.nan, np.nan, {}
    frame = None
    with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            if task.method.startswith(ITERATIVE):
                outcome = _iterative_task(task, config, train, test, drop)
            elif task.target == ALL_COLUMNS:
                outcome = _chained_task(task, config, train, test, drop)
            elif task.experiment == "calibration":
                outcome = _calibration_task(task, config, train, test, drop)
            elif task.experiment == "importance":
                outcome = _importance_task(task, config, train, test)
            else:
                outcome = _point_task(task, config, train, test, drop)
            frame, fit_seconds, predict_seconds, extra = outcome
        except Exception as exception:
            status, error = "failed", f"{type(exception).__name__}: {exception}"
    if frame is not None:
        frame.to_parquet(folder / f"{task.task_id}.parquet", index=False)
    meta = {
        **asdict(task), "task_id": task.task_id, "status": status, "error": error,
        "fit_seconds": fit_seconds, "predict_seconds": predict_seconds, "warnings": len(caught),
        "warning_types": ", ".join(sorted({w.category.__name__ for w in caught})),
        "warning_example": str(caught[0].message)[:200] if caught else "",
        "train_rows": len(train), **extra,
    }
    temporary = marker.with_suffix(".tmp")
    temporary.write_text(json.dumps(meta, default=float), encoding="utf-8")
    temporary.replace(marker)
    return meta


def run_contract(root, config, names):
    """Everything that must match to reuse a run's predictions.

    Only the code that produces predictions is included, so evaluation code and
    figures can change without invalidating a finished run.
    """
    source = Path(__file__).parent
    scenarios = discover(*data_folders(root, config))
    return {
        "config": json.loads(json.dumps(asdict(config), default=str)),
        "inputs": {s: [file_hash(scenarios[s].measured), file_hash(scenarios[s].truth)] for s in names},
        "source": {name: file_hash(source / name) for name in MODEL_SOURCES},
    }


def open_run(root, config, names, resume_run_id=None):
    """Create a run folder, or reopen one after confirming nothing has changed."""
    base = Path(root) / config.output_directory
    contract = run_contract(root, config, names)
    if resume_run_id:
        folder = base / resume_run_id
        saved = json.loads((folder / "contract.json").read_text(encoding="utf-8"))
        changed = [key for key in contract if saved[key] != contract[key]]
        if changed:
            raise ValueError(f"Cannot reuse {resume_run_id}: changed {changed}")
        return folder
    folder = base / datetime.now().strftime("%Y%m%d-%H%M%S")
    (folder / "tasks").mkdir(parents=True)
    (folder / "contract.json").write_text(json.dumps(contract, indent=1), encoding="utf-8")
    return folder


def default_workers():
    # Leaves two cores free. Results do not depend on the number of workers.
    return max(1, (psutil.cpu_count(logical=False) or 2) - 2)


def run_plan(plan, config, root, run_directory, workers=None, log_name="task_log"):
    """Execute every unfinished task across worker processes."""
    workers = workers or default_workers()
    # Spawned workers need the package path to unpickle these functions.
    source = str(Path(root) / "src")
    current = os.environ.get("PYTHONPATH", "")
    if source not in current.split(os.pathsep):
        os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, [source, current]))
    # Slowest kinds of work go first so that no long task is left running alone at the end.
    weight = {"importance": 2, "calibration": 3}
    ordered = sorted(plan, key=lambda t: (weight.get(t.experiment, 4), not t.method.startswith(ITERATIVE),
                                          t.method != "Random Forest"))
    started = time.perf_counter()
    results, step = [], max(1, len(plan) // 10)
    with parallel_config(backend="loky", inner_max_num_threads=1):
        stream = Parallel(n_jobs=workers, return_as="generator_unordered")(
            delayed(execute_task)(task, config, str(root), str(run_directory)) for task in ordered
        )
        for done, meta in enumerate(stream, 1):
            results.append(meta)
            if done % step == 0 or done == len(plan):
                print(f"{done}/{len(plan)} tasks, {time.perf_counter() - started:.0f} s", flush=True)
    log = pd.DataFrame(results)
    log.to_csv(Path(run_directory) / f"{log_name}.csv", index=False)
    return log


def load_outputs(log, run_directory, suffix=".parquet", experiments=None):
    """Read committed task files and attach each task's identity."""
    keys = ["scenario", "experiment", "variant", "seed", "fold", "method"]
    selected = log[log.status.eq("success")]
    if experiments is not None:
        selected = selected[selected.experiment.isin(experiments)]
    frames = []
    for row in selected.itertuples():
        path = Path(run_directory) / "tasks" / f"{row.task_id}{suffix}"
        if path.exists():
            frame = pd.read_parquet(path)
            for key in keys:
                if key not in frame:
                    frame[key] = getattr(row, key)
            frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
