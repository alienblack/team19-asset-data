"""Runs the study one model at a time and collects the results for the notebook.

Each model section of the notebook calls ``run_model`` and then ``show_model``. Fitting happens
in ``runner``, which never reads truth; truth is read here, only after a model's tasks have finished.

Results are kept in the run folder at two levels:

- Each task's predictions are saved when the task finishes. A task with saved predictions is read
  back and not fitted again.
- The evaluation tables built from those predictions are saved in ``evaluation_cache``. They are
  reused while the run's finished tasks and the evaluation code are unchanged, and rebuilt otherwise.

With ``evaluation_run_id`` the model sections fit nothing and read the run's task logs instead.
"""

import hashlib
import json
import pickle
from pathlib import Path

from IPython.display import display
import pandas as pd

from . import evaluation as ev
from . import failure as fl
from . import figures as fg
from . import relationships as rel
from .columns import NUMERIC_TARGETS
from .data import read_truth
from .mechanisms import write_mechanism_files
from .names import shown
from .runner import (
    build_plan, load_measured, load_outputs, open_run, run_plan,
)

# The notebook section of each model, in running order, with the names its tasks are fitted under.
MODEL_FAMILIES = {
    "Median and Mode": ("Median", "Mode"),
    "KNN": ("KNN imputer",),
    "IterativeImputer": ("IterativeImputer", "IterativeImputer (5)", "IterativeImputer (30)"),
    "Random Forest": ("Random Forest", "Random Forest calibrated"),
    "HistGradientBoosting": ("HistGradientBoosting",),
    "CatBoost": ("CatBoost",),
    "miceforest direct": ("miceforest direct",),
    "GMM": ("Gaussian EM",),
    "Masked MLP": ("Masked MLP",),
    "LightGBM chained": ("LightGBM chained",),
    "Predictive MissForest": ("Predictive MissForest",),
    "MICE-PMM": ("MICE-PMM",),
}
POINT_EXPERIMENTS = ("main", "stability", "budget", "ablation", "calibration", "structure", "mechanism", "spatial")
SHOWN_EXPERIMENTS = ("main",)
RESULT_COLUMNS = ["target", "MAE", "RMSE", "R2", "bias", "skill", "within tight", "within loose", "exact"]
# The code that turns predictions into evaluation tables. A change to any of these files rebuilds the saved tables.
EVALUATION_SOURCES = ("study.py", "evaluation.py", "failure.py", "relationships.py", "names.py", "columns.py")


def _slug(family):
    return family.lower().replace(" ", "_")


def _by_model(tables):
    """One table per main model, stacked with a model column, for saving as one file."""
    return pd.concat([(table.reset_index() if table.index.name else table).assign(model=model)
                      for model, table in tables.items()], ignore_index=True)


class SavedTables:
    """Evaluation tables kept in the run folder, each stored with the key it was built under."""

    def __init__(self, directory):
        self.folder = Path(directory) / "evaluation_cache"
        digest = hashlib.sha256((Path(directory) / "contract.json").read_bytes())
        for name in EVALUATION_SOURCES:
            digest.update((Path(__file__).parent / name).read_bytes())
        self.code = digest.hexdigest()

    def key(self, log):
        """The code and the finished tasks a table is built from."""
        tasks = "\n".join(sorted(log.task_id.astype(str) + ":" + log.status.astype(str)))
        return hashlib.sha256((self.code + tasks).encode("utf-8")).hexdigest()

    def get(self, name, key, build):
        """The saved value, or a newly built one that is then saved. Also returns whether it was saved."""
        path = self.folder / f"{name}.pkl"
        if path.exists():
            with path.open("rb") as handle:
                saved_key, value = pickle.load(handle)
            if saved_key == key:
                return value, True
        value = build()
        self.folder.mkdir(exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump((key, value), handle, protocol=pickle.HIGHEST_PROTOCOL)
        return value, False


class Study:
    """One study run: the task plan, the run folder, and the results of every model and of the evaluation."""

    def __init__(self, root, config, scenarios, workers, resume_run_id=None, evaluation_run_id=None):
        if resume_run_id and evaluation_run_id:
            raise ValueError("Set RUN_ID or EVALUATION_RUN_ID, not both")
        self.root, self.config, self.workers = root, config, workers
        self.names = list(scenarios)
        # Scenarios with other missingness mechanisms are built from the generator's files.
        self.mechanisms = write_mechanism_files(root, config)
        self.scenarios = {**scenarios, **self.mechanisms}
        self.saved = evaluation_run_id is not None
        self.directory = open_run(root, config, list(self.scenarios), evaluation_run_id or resume_run_id)
        self.figures = Path(self.directory) / "figures"
        self.plan = build_plan(self.names, config, list(self.mechanisms))
        self.measured = {name: load_measured(root, name, config) for name in self.scenarios}
        self.store = SavedTables(self.directory)
        self.truth = None
        self.logs, self.tables = {}, {}
        self.scored, self.importance = {}, {}

    # ------------------------------------------------------------------ models

    def _task_log(self, family):
        """Every task of one model: saved tasks are read back, the others are fitted."""
        slug = _slug(family)
        path = Path(self.directory) / f"task_log_{slug}.csv"
        if self.saved:
            if not path.exists():
                raise FileNotFoundError(f"{family} has no saved results in {self.directory}: finish that run first")
            log = pd.read_csv(path)
            print(f"{family}: {len(log)} saved tasks loaded")
            return log
        tasks = [t for t in self.plan if t.method in MODEL_FAMILIES[family]]
        folder = Path(self.directory) / "tasks"
        done = [t for t in tasks if (folder / f"{t.task_id}.json").exists()]
        pending = [t for t in tasks if not (folder / f"{t.task_id}.json").exists()]
        print(f"Tasks: {len(tasks)} | saved, skipped: {len(done)} | fitted now: {len(pending)}")
        log = pd.DataFrame([json.loads((folder / f"{t.task_id}.json").read_text(encoding="utf-8")) for t in done])
        if pending:
            fitted = run_plan(pending, self.config, self.root, self.directory, self.workers, f"task_log_{slug}")
            log = pd.concat([log, fitted], ignore_index=True)
        if pending or not path.exists():
            log.to_csv(path, index=False)
        return log

    def run_model(self, family):
        """Run all of this model's experiments, then score them against the baseline."""
        log = self._task_log(family)
        log["method"] = log.method.map(shown)
        self.logs[family] = log
        failed = log[log.status.ne("success")]
        if len(failed):
            display(failed[["scenario", "experiment", "target", "method", "fold", "error"]].head(20))
        baseline = self.logs.get("Median and Mode")
        if baseline is None:
            raise RuntimeError("Run the Median and Mode section first: skill is measured against it")
        both = log if family == "Median and Mode" else pd.concat([baseline, log], ignore_index=True)
        self.tables[family], _ = self.store.get(f"model_{_slug(family)}", self.store.key(both),
                                                lambda: self._model_tables(family))

    def _scored(self, family):
        """Predictions of one model with the truth attached. Read only when a table must be built."""
        if family not in self.scored:
            if self.truth is None:
                self.truth = {n: read_truth(s).loc[self.measured[n].index] for n, s in self.scenarios.items()}
            log = self.logs[family]
            outputs = load_outputs(log, self.directory, experiments=list(POINT_EXPERIMENTS))
            if len(outputs):
                outputs["method"] = outputs.method.map(shown)
            self.scored[family] = ev.attach_truth(outputs, self.truth)
            importance = load_outputs(log, self.directory, experiments=["importance"])
            if len(importance):
                importance["method"] = importance.method.map(shown)
            self.importance[family] = importance
        return self.scored[family]

    def _model_tables(self, family):
        """Scores of one model, with the baseline added so that skill can be computed."""
        rows = [self._scored(family)]
        if family != "Median and Mode":
            rows.append(self._scored("Median and Mode"))
        numeric, category = ev.score_tables(pd.concat(rows, ignore_index=True))
        numeric = ev.add_scenario_levels(numeric, self.scenarios)
        category = ev.add_scenario_levels(category, self.scenarios)
        methods = [shown(m) for m in MODEL_FAMILIES[family]]
        tables = {"numeric": numeric[numeric.method.isin(methods)], "category": category[category.method.isin(methods)]}
        if family == "Random Forest":
            tables["probability"] = ev.probability_table(self._scored(family), self.config)
        if family == "IterativeImputer":
            tables["passes"] = ev.budget_table(tables["numeric"], self.logs[family], self.config)
        return tables

    def scores_for(self, family):
        """Numeric and model_id scores of one model section, with the baseline's rows."""
        return self.tables[family]["numeric"], self.tables[family]["category"]

    def show_model(self, family, section):
        """Results of one model section under numbered captions.

        Tables are labelled ``Table <section>a``, ``b``, ... in the order shown: numeric results and ``model_id``
        results in the representative scenario, ``model_id`` accuracy in every scenario and runtime. The figure,
        skill in every scenario, is ``Figure <section>``. A later cell of the same section continues the lettering.
        """
        numeric, category = self.scores_for(family)
        config, representative = self.config, self.config.representative
        name = shown(MODEL_FAMILIES[family][0])
        letters = iter("abcdefghij")

        def show(text, frame):
            fg.table(f"Table {section}{next(letters)}", text, frame)

        primary = numeric[numeric.experiment.isin(SHOWN_EXPERIMENTS) & numeric.scenario.eq(representative)
                          & numeric.seed.eq(config.seed)]
        if len(primary):
            joined = primary.set_index("target")
            joined = joined.reindex([t for t in NUMERIC_TARGETS if t in joined.index]).reset_index()
            for method, part in joined.groupby("method", sort=False):
                columns = [c for c in RESULT_COLUMNS if c in part and part[c].notna().any()]
                show(f"{method}: error and skill on each numeric column, `{representative}`.", part[columns].round(4))
        labels = category[category.experiment.isin(SHOWN_EXPERIMENTS) & category.scenario.eq(representative)
                          & category.seed.eq(config.seed)]
        if len(labels):
            show(f"{', '.join(labels.method.unique())}: accuracy, macro F1 and balanced accuracy on `model_id`, `{representative}`.",
                 labels[["method", "n", "accuracy", "macro F1", "balanced accuracy"]].round(4))
        across = numeric[numeric.experiment.eq("main") & numeric.method.ne("Median")]
        if across.scenario.nunique() > 1:
            matrix = across.pivot(index="target", columns="scenario", values="skill").reindex(list(NUMERIC_TARGETS))
            fg.plot_heatmap(matrix, f"{name}: skill in each scenario", self.figures,
                            f"model_{_slug(family)}", "Skill vs median", center=0.0, limit=1.0)
            fg.caption(f"Figure {section}", f"{name}: skill on each numeric column in each of the nine generator "
                       "scenarios (0 equals the baseline, 1 is exact recovery).")
        all_labels = category[category.experiment.eq("main")]
        if all_labels.scenario.nunique() > 1 and all_labels.method.ne("Mode").all():
            show(f"{name}: accuracy on `model_id` in each generator scenario.",
                 all_labels.pivot(index="method", columns="scenario", values="accuracy").round(4))
        log = self.logs[family]
        cost = ev.cost_table(log)
        main = cost[cost.experiment.isin(SHOWN_EXPERIMENTS)]
        show(f"{name}: tasks and fitting time in the main experiment.",
             main[["method", "tasks", "fit seconds", "predict seconds", "seconds per target and fold"]].round(3))

    # -------------------------------------------------------------- evaluation

    def finish(self):
        """Combine every model's results and build the tables of the evaluation chapters."""
        missing = [family for family in MODEL_FAMILIES if family not in self.logs]
        if missing:
            raise RuntimeError(f"These model sections have not run: {missing}")
        self.log = pd.concat(self.logs.values(), ignore_index=True)
        state, reused = self.store.get("evaluation", self.store.key(self.log), self._evaluate)
        for name, value in state.items():
            setattr(self, name, value)
        fl.LEAD_METHOD = self.lead
        print("Evaluation tables:", "read from the run folder" if reused else "built and saved in the run folder")
        return self

    def _evaluate(self):
        """Every table of chapters 5 to 9 and the appendix, built from the cell-level results."""
        config, measured, representative = self.config, self.measured, self.config.representative
        scored = pd.concat([self._scored(f) for f in MODEL_FAMILIES], ignore_index=True)
        importance = pd.concat(self.importance.values(), ignore_index=True)
        truth = self.truth
        t = {}
        numeric, category = ev.score_tables(scored)
        t["numeric"] = ev.add_scenario_levels(numeric, self.scenarios)
        t["category"] = ev.add_scenario_levels(category, self.scenarios)
        t["pairs"] = ev.paired_intervals(scored, config)
        t["effects"] = ev.scenario_effects(scored, self.scenarios, config)
        t["cost"] = ev.cost_table(self.log)
        t["targets"] = ev.target_table(t["numeric"], t["pairs"], t["effects"], config)
        t["relations"] = rel.relation_table(scored, measured, config)
        t["recoverable"] = rel.recoverable_table(truth[representative], t["numeric"], config)
        # One row per predicted cell, with the baseline's error on the same cell and the cell's conditions.
        cells = fl.cell_table(scored, measured, truth)
        # The two main models are compared with the baseline; views of a single model are built for each.
        main = t["main"] = list(fl.MAIN_METHODS)
        lead = t["lead"] = fl.LEAD_METHOD
        standard = fl.skill_by(fl.main_cells(cells, config, representative), ["target", "method"])
        t["standard"] = standard[standard.method.isin(["Median", *main]) & standard.target.ne("model_id")]
        t["overall"] = fl.overall_table(cells, config)
        t["by_scenario"] = ev.add_scenario_levels(fl.scenario_table(cells, config), self.scenarios)
        t["visibility"] = fl.visibility_table(cells, config)
        t["ablation"] = fl.ablation_scores(cells, config)
        maps = {m: fl.failure_map(cells, t["ablation"], config, m) for m in fl.SHOWN_METHODS}
        t["failure_maps"] = {m: maps[m] for m in main}
        t["model_groups"] = fl.group_table(maps)
        t["completion"] = fl.completion_table(cells, self.scenarios, config)
        t["splits"] = fl.split_table(cells, config)
        t["wrong_values"] = fl.wrong_value_table(cells, measured, config)
        t["bands"] = fl.band_table(cells, config)
        t["mechanism_scores"] = fl.mechanism_table(cells, self.scenarios, config)
        t["mechanism_groups"] = fl.mechanism_groups(cells, truth, self.scenarios, config)
        t["kinds"] = {m: fl.kind_table(cells, config, m) for m in main}
        t["shrinkage"] = fl.shrinkage_table(cells, config)
        t["spread"] = fl.overall_shrinkage(cells, config)
        t["ranges"] = fl.range_table(cells, config)
        t["decline"] = {m: fl.decline_table(t["by_scenario"], t["visibility"], config, m) for m in main}
        t["correlations"] = fl.correlation_table(cells, measured, truth, config)
        t["consistency"] = fl.consistency_table(cells, measured, config)
        t["confidence_by_seen"] = fl.confidence_table(cells, config)
        t["confidence_by_scenario"] = fl.confidence_table(cells, config, by="scenario")
        t["overlap"] = fl.error_overlap(cells, config)
        t["structure"] = fl.structure_table(cells, config)
        t["information"] = {m: fl.information_table(cells, config, m) for m in main}
        t["noise"] = {m: fl.noise_table(cells, measured, truth, t["by_scenario"], config, m)[0] for m in main}
        t["links"] = fl.link_table(importance)
        # Importance comes from single-target models; the first main model is shown when it is one of them.
        t["explainer"] = lead if lead in set(importance.method) else "CatBoost"
        t["unexplained"] = fl.unexplained_links(importance, config.noise_series[0], t["explainer"])
        t["importance_means"] = importance.groupby(["method", "scenario", "target", "predictor"],
                                                   sort=False).importance.mean().reset_index()
        used = importance[importance.predictor.eq("random_placebo") & importance.target.ne("random_placebo")]
        t["placebo_importance"] = used.groupby(["method", "scenario"]).importance.agg(["mean", "max"])
        t["integrity"] = fl.integrity_table(cells, measured, config)
        seeds = fl.skill_by(cells[cells.scenario.eq(representative) & cells.experiment.isin(["main", "stability"])
                                  & cells.target.ne("random_placebo")], ["method", "seed", "target"])
        t["seed_skill"] = seeds.groupby(["method", "seed"]).skill.mean().unstack().drop(index=["Median", "Mode"], errors="ignore")
        t["stability"] = ev.stability_table(t["numeric"], t["category"], config)
        t["downstream"] = fl.downstream_table(cells, measured, truth, config)
        # Mean absolute error of the summary values, for the representative scenario and each mechanism.
        by_scenario = {representative: t["downstream"]}
        by_scenario.update({name: fl.downstream_table(cells, measured, truth, config, name) for name in self.mechanisms})
        t["downstream_by_scenario"] = pd.DataFrame({
            name: table.assign(size=table["relative error"].abs()).groupby("source", sort=False)["size"].mean()
            for name, table in by_scenario.items()})
        t["completed"] = ev.export_completions(scored, measured[representative], representative,
                                               Path(self.directory) / "completed")
        files = {"numeric_scores": t["numeric"], "category_scores": t["category"], "paired_intervals": t["pairs"],
                 "scenario_effects": t["effects"], "cost": t["cost"], "targets": t["targets"],
                 "relations": t["relations"], "overall": t["overall"].reset_index(),
                 "skill_by_scenario": t["by_scenario"], "skill_by_visible_links": t["visibility"],
                 "ablation": t["ablation"], "failure_map": _by_model(t["failure_maps"]),
                 "value_bands": t["bands"], "mechanisms": t["mechanism_scores"], "relationship_kinds": _by_model(t["kinds"]),
                 "shrinkage": t["shrinkage"], "correlations": t["correlations"], "consistency": t["consistency"],
                 "structure": t["structure"].reset_index(), "information": _by_model(t["information"]),
                 "noise": _by_model(t["noise"]), "links": t["links"], "downstream": t["downstream"],
                 "downstream_by_scenario": t["downstream_by_scenario"].reset_index(),
                 "model_groups": t["model_groups"].reset_index(), "skill_by_completion": t["completion"],
                 "fold_design": t["splits"], "wrong_values": t["wrong_values"]}
        for name, table in files.items():
            table.to_csv(Path(self.directory) / f"{name}.csv", index=False)
        return t

    def planned_tasks(self):
        """Number of tasks in the plan."""
        return len(self.plan)
