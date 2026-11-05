"""Experiment plan and parallel execution with resumable per-task checkpoints.

Truth is never read here. Predictions are written to disk per task, so an
interrupted run continues from the tasks that have not finished.
"""

from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from functools import lru_cache
from itertools import product
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
    CATEGORY_METHODS, CLASS_COUNT, NUMERIC_METHODS,
    CalibratedForest, ChainedLinearImputer, JointModel, MiceforestModel, make_model,
)

PROBABILITY_COLUMNS = [f"prob_{i}" for i in range(CLASS_COUNT)]
ALL_NUMERIC = "all numeric"
ITERATIVE = "IterativeImputer"
ABLATION_METHODS = ("KNN imputer", "Random Forest", "CatBoost")
ABLATION_CATEGORY_METHODS = ("KNN", "Random Forest", "CatBoost")
IMPORTANCE_METHODS = ("Random Forest", "CatBoost")
TUNED_METHODS = ("KNN imputer", "KNN", "Random Forest", "HistGradientBoosting", "CatBoost", "miceforest direct")


def default_parameters():
    # The settings of the earlier study, kept as the untuned reference.
    return {
        "KNN imputer": {"n_neighbors": 5, "weights": "distance"},
        "KNN": {"n_neighbors": 5, "weights": "distance"},
        # Stops by itself once converged. Heavy missingness can need over a thousand rounds.
        ITERATIVE: {"max_iter": 2000, "tol": 1e-3, "initial_strategy": "median"},
        "Random Forest": {"n_estimators": 200, "min_samples_leaf": 2},
        "HistGradientBoosting": {"max_iter": 200, "learning_rate": 0.1, "l2_regularization": 1.0},
        "CatBoost": {"iterations": 250, "depth": 6, "learning_rate": 0.05},
        "miceforest direct": {"rounds": 100, "num_leaves": 31, "min_data_in_leaf": 20, "learning_rate": 0.1},
    }


def default_search():
    # Values to sample from. IterativeImputer has no entry: its one setting that
    # matters is the iteration limit, which is decided by convergence.
    neighbours = {"n_neighbors": [3, 5, 10, 20, 40], "weights": ["uniform", "distance"]}
    return {
        "KNN imputer": neighbours,
        "KNN": neighbours,
        "Random Forest": {"min_samples_leaf": [1, 2, 5, 10, 20], "max_features": [1.0, 0.7, 0.5, 0.33]},
        "HistGradientBoosting": {
            "max_iter": [100, 200, 400], "learning_rate": [0.03, 0.05, 0.1, 0.2],
            "max_leaf_nodes": [15, 31, 63], "min_samples_leaf": [10, 20, 50],
            "l2_regularization": [0.0, 1.0, 10.0],
        },
        "CatBoost": {
            "iterations": [250, 500, 1000], "depth": [4, 6, 8],
            "learning_rate": [0.03, 0.05, 0.1], "l2_leaf_reg": [1, 3, 10],
        },
        "miceforest direct": {
            "rounds": [100, 200, 400], "num_leaves": [15, 31, 63],
            "min_data_in_leaf": [10, 20, 50], "learning_rate": [0.03, 0.05, 0.1],
        },
    }


def default_blocks():
    # Predictor groups defined by what the columns measure.
    return {
        "without total_gas and leakage": ("total_gas", "leakage"),
        "without gas columns": GAS_READINGS + ("total_gas",),
        "without life": ("life",),
        "without location": ("x", "y", "altitude"),
    }


@dataclass
class StudyConfig:
    data_directory: str = "data/datasets"
    output_directory: str = "outputs/imputation-study/runs"
    representative: str = "completion0.8-noise0.05"
    noise_series: tuple = ("completion0.8-noise0.0", "completion0.8-noise0.05", "completion0.8-noise0.1")
    scenarios: tuple | None = None  # None selects every scenario in the data directory
    seed: int = 219
    stability_seeds: tuple = (220, 221)
    folds: int = 5
    rows: int | None = None  # a smaller sample, for quick checks only
    tuning_candidates: int = 12
    tuning_folds: int = 3
    iterative_budgets: tuple = (10, 30)
    mice_iterations: int = 5
    pmm_draws: int = 20
    pmm_candidates: int = 5
    interval: tuple = (0.1, 0.9)
    bootstrap_draws: int = 1000
    small_group: int = 30
    material_gap: float = 0.05  # a smaller relative MAE gap is reported as practically tied
    parameters: dict = field(default_factory=default_parameters)
    search: dict = field(default_factory=default_search)
    blocks: dict = field(default_factory=default_blocks)
    tuned: dict = field(default_factory=dict)  # "method|target" -> chosen settings


def quick_config(**changes):
    """Every stage on a small sample with small models. Results are not findings."""
    config = StudyConfig(
        rows=600, tuning_candidates=2, mice_iterations=2, pmm_draws=3, bootstrap_draws=50,
        scenarios=("completion0.7-noise0.05", "completion0.8-noise0.0", "completion0.8-noise0.05",
                   "completion0.8-noise0.1", "completion0.9-noise0.05"),
    )
    config.parameters["Random Forest"]["n_estimators"] = 10
    config.parameters["HistGradientBoosting"]["max_iter"] = 10
    config.parameters["CatBoost"]["iterations"] = 10
    config.parameters["miceforest direct"]["rounds"] = 10
    config.parameters[ITERATIVE]["max_iter"] = 15
    config.search["HistGradientBoosting"]["max_iter"] = [10, 20]
    config.search["CatBoost"]["iterations"] = [10, 20]
    config.search["miceforest direct"]["rounds"] = [10, 20]
    return replace(config, **changes)


def candidates(method, config):
    """The default settings plus a seeded random sample of the search space."""
    space = config.search[method]
    combos = [dict(zip(space, values)) for values in product(*space.values())]
    generator = np.random.default_rng(config.seed)
    chosen = generator.choice(len(combos), min(config.tuning_candidates, len(combos)), replace=False)
    return [{}] + [combos[int(i)] for i in chosen]


def settings_for(method, target, config, tuned=True):
    settings = dict(config.parameters[method])
    if tuned:
        settings.update(config.tuned.get(f"{method}|{target}", {}))
    return settings


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


def build_tuning_plan(config):
    plan = []
    for method in TUNED_METHODS:
        if method == "KNN imputer":
            targets = NUMERIC_TARGETS
        elif method == "KNN":
            targets = (CATEGORY_TARGET,)
        else:
            targets = TARGETS
        for target in targets:
            for index in range(len(candidates(method, config))):
                plan.append(Task(config.representative, "tuning", str(index), config.seed, 0, target, method))
    return plan


def build_plan(names, config):
    """Every planned fit. The folds of a seed are shared by all methods and scenarios."""
    plan = []
    folds = range(config.folds)

    def add(scenario, experiment, variant, seed, fold, target, method):
        plan.append(Task(scenario, experiment, variant, seed, fold, target, method))

    def add_full_set(scenario, experiment, seed, fold):
        for target in NUMERIC_TARGETS:
            for method in NUMERIC_METHODS:
                if method != ITERATIVE:
                    add(scenario, experiment, "all predictors", seed, fold, target, method)
        for method in CATEGORY_METHODS:
            add(scenario, experiment, "all predictors", seed, fold, CATEGORY_TARGET, method)
        add(scenario, experiment, "all predictors", seed, fold, ALL_NUMERIC, ITERATIVE)

    representative = config.representative
    for name in names:
        for fold in folds:
            add_full_set(name, "main", config.seed, fold)
    for seed in config.stability_seeds:
        for fold in folds:
            add_full_set(representative, "stability", seed, fold)
    for fold in folds:
        for target in TARGETS:
            for method in CATEGORY_METHODS if target == CATEGORY_TARGET else NUMERIC_METHODS:
                if method in TUNED_METHODS:
                    add(representative, "defaults", "default settings", config.seed, fold, target, method)
        for budget in config.iterative_budgets:
            add(representative, "budget", str(budget), config.seed, fold, ALL_NUMERIC, f"{ITERATIVE} ({budget})")
        for target in NUMERIC_TARGETS:
            add(representative, "pmm", "all predictors", config.seed, fold, target, "miceforest PMM")
        add(representative, "joint", "all predictors", config.seed, fold, "all", "Joint miceforest PMM")
        for variant in ("all predictors", "without gas columns"):
            add(representative, "calibration", variant, config.seed, fold, CATEGORY_TARGET,
                "Random Forest calibrated")
    for name in config.noise_series:
        for variant, dropped in config.blocks.items():
            for fold in folds:
                add(name, "ablation", variant, config.seed, fold, ALL_NUMERIC, ITERATIVE)
                for target in TARGETS:
                    if target in dropped:
                        continue
                    methods = ABLATION_CATEGORY_METHODS if target == CATEGORY_TARGET else ABLATION_METHODS
                    for method in methods:
                        add(name, "ablation", variant, config.seed, fold, target, method)
        for target in TARGETS:
            for method in IMPORTANCE_METHODS:
                add(name, "importance", "all predictors", config.seed, 0, target, method)
    return plan


def plan_table(plan):
    frame = pd.DataFrame([asdict(task) for task in plan])
    return frame.groupby("experiment", sort=False).agg(
        tasks=("method", "size"), scenarios=("scenario", "nunique"), seeds=("seed", "nunique"),
        methods=("method", "nunique"),
    ).reset_index()


def fold_ids(rows, folds, seed):
    assignment = np.empty(rows, dtype=int)
    for fold, (_, test) in enumerate(KFold(folds, shuffle=True, random_state=seed).split(np.arange(rows))):
        assignment[test] = fold
    return assignment


@lru_cache(maxsize=4)
def _measured(directory, name, rows, sample_seed):
    frame = read_measured(discover(directory)[name])
    if rows is not None:
        chosen = np.random.default_rng(sample_seed).choice(len(frame), rows, replace=False)
        frame = frame.iloc[np.sort(chosen)]
    return frame


def load_measured(root, name, config):
    return _measured(str(Path(root) / config.data_directory), name, config.rows, config.seed)


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
    settings = settings_for(task.method, task.target, config, tuned=task.experiment != "defaults") \
        if task.method in config.parameters else {}
    model = make_model(task.method, task.target, config, settings, task.seed + task.fold, drop)
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
    return frame, None, fitted - started, time.perf_counter() - fitted, {}


def _iterative_task(task, config, train, test, drop):
    """One chained fit supplies every numeric target for this fold."""
    budget = int(task.variant) if task.experiment == "budget" else None
    model = ChainedLinearImputer(config.parameters[ITERATIVE], task.seed + task.fold, drop, budget)
    started = time.perf_counter()
    model.fit(train)
    fitted = time.perf_counter()
    filled = model.predict(test)
    frames = []
    for target in filled.columns:
        hidden = test[target].isna()
        frames.append(_rows(target, test.index[hidden], filled.loc[hidden, target]))
    frame = pd.concat(frames, ignore_index=True)
    return frame, None, fitted - started, time.perf_counter() - fitted, {"n_iter": int(model.n_iter_)}


def _pmm_task(task, config, train, test):
    settings = settings_for("miceforest direct", task.target, config)
    model = MiceforestModel(task.target, config, settings, task.seed + task.fold, draws=config.pmm_draws)
    hidden = test[task.target].isna()
    started = time.perf_counter()
    model.fit(train)
    fitted = time.perf_counter()
    draws = model.draw(test.loc[hidden])
    frame = _rows(task.target, test.index[hidden], draws.mean(axis=1))
    long = pd.DataFrame({
        "row_id": np.repeat(test.index[hidden], draws.shape[1]), "target": task.target,
        "draw": np.tile(np.arange(draws.shape[1]), draws.shape[0]), "value": draws.ravel(),
    })
    return frame, long, fitted - started, time.perf_counter() - fitted, {}


def _joint_task(task, config, train, test):
    # One shared setting: a joint model has no per-target settings to tune.
    model = JointModel(config, config.parameters["miceforest direct"], task.seed + task.fold)
    started = time.perf_counter()
    model.fit(train)
    fitted = time.perf_counter()
    copies = model.predict(test)
    frames, long = [], []
    for target in TARGETS:
        hidden = test[target].isna()
        values = np.column_stack([copy.loc[hidden, target] for copy in copies])
        if target == CATEGORY_TARGET:
            votes = pd.DataFrame(values).mode(axis=1).iloc[:, 0].to_numpy()
            frames.append(_rows(target, test.index[hidden], votes, votes))
        else:
            frames.append(_rows(target, test.index[hidden], values.mean(axis=1)))
        long.append(pd.DataFrame({
            "row_id": np.repeat(test.index[hidden], values.shape[1]), "target": target,
            "draw": np.tile(np.arange(values.shape[1]), values.shape[0]), "value": values.ravel(),
        }))
    return (pd.concat(frames, ignore_index=True), pd.concat(long, ignore_index=True),
            fitted - started, time.perf_counter() - fitted, {})


def _calibration_task(task, config, train, test, drop):
    settings = settings_for("Random Forest", CATEGORY_TARGET, config)
    model = CalibratedForest(settings, task.seed + task.fold, drop)
    hidden = test[CATEGORY_TARGET].isna()
    started = time.perf_counter()
    model.fit(train)
    fitted = time.perf_counter()
    probabilities = _full_probabilities(model.classes_, model.predict_proba(test.loc[hidden]))
    raw = probabilities.argmax(axis=1).astype(float)
    frame = _rows(CATEGORY_TARGET, test.index[hidden], raw, raw, probabilities)
    return frame, None, fitted - started, time.perf_counter() - fitted, {}


def _importance_task(task, config, train, test):
    """Permutation importance on held-out observed labels of the measured data."""
    settings = settings_for(task.method, task.target, config)
    model = make_model(task.method, task.target, config, settings, task.seed + task.fold)
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
    return pd.DataFrame(entries), None, fitted - started, time.perf_counter() - fitted, {}


def _tuning_task(task, config, measured):
    """Cross-validate one candidate on observed labels only; truth is not involved."""
    index = int(task.variant)
    settings = {**config.parameters[task.method], **candidates(task.method, config)[index]}
    observed = np.flatnonzero(measured[task.target].notna().to_numpy())
    losses = []
    started = time.perf_counter()
    splitter = KFold(config.tuning_folds, shuffle=True, random_state=config.seed)
    for fold, (_, held) in enumerate(splitter.split(observed)):
        validation = measured.iloc[observed[held]]
        train = measured.drop(index=validation.index)
        model = make_model(task.method, task.target, config, settings, config.seed + fold).fit(train)
        labels = validation[task.target].to_numpy()
        if model.categorical:
            probabilities = _full_probabilities(model.classes_, model.predict_proba(validation))
            chosen = probabilities[np.arange(len(labels)), labels.astype(int)]
            losses.append(float(-np.log(np.maximum(chosen, 1e-12)).mean()))
        else:
            losses.append(float(np.mean(np.abs(model.predict(validation) - labels))))
    frame = pd.DataFrame([{
        "method": task.method, "target": task.target, "candidate": index,
        "loss": float(np.mean(losses)), "settings": json.dumps(settings, sort_keys=True),
    }])
    return frame, None, time.perf_counter() - started, 0.0, {}


def execute_task(task, config, root, run_directory):
    """Run one task in a worker. The JSON marker is written last and commits the result."""
    folder = Path(run_directory) / "tasks"
    marker = folder / f"{task.task_id}.json"
    if marker.exists():
        return json.loads(marker.read_text(encoding="utf-8"))
    measured = load_measured(root, task.scenario, config)
    assignment = fold_ids(len(measured), config.folds, task.seed)
    train, test = measured[assignment != task.fold], measured[assignment == task.fold]
    drop = config.blocks.get(task.variant, ())
    status, error, fit_seconds, predict_seconds, extra = "success", "", np.nan, np.nan, {}
    frame = draws = None
    with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            if task.experiment == "tuning":
                outcome = _tuning_task(task, config, measured)
            elif task.method.startswith(ITERATIVE):
                outcome = _iterative_task(task, config, train, test, drop)
            elif task.experiment == "pmm":
                outcome = _pmm_task(task, config, train, test)
            elif task.experiment == "joint":
                outcome = _joint_task(task, config, train, test)
            elif task.experiment == "calibration":
                outcome = _calibration_task(task, config, train, test, drop)
            elif task.experiment == "importance":
                outcome = _importance_task(task, config, train, test)
            else:
                outcome = _point_task(task, config, train, test, drop)
            frame, draws, fit_seconds, predict_seconds, extra = outcome
        except Exception as exception:
            status, error = "failed", f"{type(exception).__name__}: {exception}"
    if frame is not None:
        frame.to_parquet(folder / f"{task.task_id}.parquet", index=False)
    if draws is not None:
        draws.to_parquet(folder / f"{task.task_id}.draws.parquet", index=False)
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
    """Everything that must match for an interrupted run to be resumed."""
    source = Path(__file__).parent
    scenarios = discover(Path(root) / config.data_directory)
    settings = {k: v for k, v in asdict(config).items() if k != "tuned"}
    return {
        "config": json.loads(json.dumps(settings, default=str)),
        "inputs": {s: [file_hash(scenarios[s].measured), file_hash(scenarios[s].truth)] for s in names},
        "source": {p.name: file_hash(p) for p in sorted(source.glob("*.py"))},
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
            raise ValueError(f"Cannot resume {resume_run_id}: changed {changed}")
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
    weight = {"joint": 0, "pmm": 1, "importance": 2, "calibration": 3}
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


def select_parameters(tuning_log, run_directory, config):
    """Pick the candidate with the lowest cross-validated loss; returns settings and the record."""
    results = load_outputs(tuning_log, run_directory)
    best = results.loc[results.groupby(["method", "target"]).loss.idxmin()]
    default = results[results.candidate.eq(0)].set_index(["method", "target"]).loss.rename("default loss")
    table = best.set_index(["method", "target"]).join(default).reset_index()
    table["loss change"] = table.loss / table["default loss"] - 1
    tuned = {f"{row.method}|{row.target}": json.loads(row.settings) for row in table.itertuples()}
    columns = ["method", "target", "candidate", "default loss", "loss", "loss change", "settings"]
    return tuned, table[columns].rename(columns={"loss": "tuned loss"})
