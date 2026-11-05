"""Runs the study one model at a time and collects the results for the notebook.

Each model section of the notebook calls ``run_model`` and then ``show_model``.
Fitting happens in ``runner``, which never reads truth; truth is read here,
only after a model's tasks have finished.
"""

from dataclasses import replace
import json
from pathlib import Path

from IPython.display import Markdown, display
import pandas as pd

from . import evaluation as ev
from . import figures as fg
from .columns import CATEGORY_TARGET, NUMERIC_TARGETS, TARGETS
from .data import read_truth
from .runner import (
    build_plan, build_tuning_plan, load_measured, load_outputs, open_run, run_plan, select_parameters,
)

# The notebook section each method belongs to, in running order.
MODEL_FAMILIES = {
    "Median and Mode": ("Median", "Mode"),
    "KNN": ("KNN imputer", "KNN"),
    "IterativeImputer": ("IterativeImputer", "IterativeImputer (10)", "IterativeImputer (30)"),
    "Random Forest": ("Random Forest", "Random Forest calibrated"),
    "HistGradientBoosting": ("HistGradientBoosting",),
    "CatBoost": ("CatBoost",),
    "miceforest direct": ("miceforest direct",),
    "miceforest PMM": ("miceforest PMM",),
    "Joint miceforest PMM": ("Joint miceforest PMM",),
}
POINT_EXPERIMENTS = ("main", "stability", "defaults", "budget", "ablation", "pmm", "joint", "calibration")
SHOWN_EXPERIMENTS = ("main", "pmm", "joint")
RESULT_COLUMNS = ["target", "MAE", "RMSE", "R2", "bias", "skill", "within tight", "within loose", "exact",
                  "tuning change"]


class Study:
    def __init__(self, root, config, scenarios, workers, resume_run_id=None):
        self.root, self.config, self.scenarios, self.workers = root, config, scenarios, workers
        self.names = list(scenarios)
        self.directory = open_run(root, config, self.names, resume_run_id)
        self.figures = Path(self.directory) / "figures"
        self.plan = build_plan(self.names, config)
        self.tuning_plan = build_tuning_plan(config)
        self.measured = {name: load_measured(root, name, config) for name in self.names}
        self.truth = None
        self.logs, self.tuning_logs, self.tuning_tables = {}, {}, {}
        self.scored, self.draws, self.importance = {}, {}, {}

    def run_model(self, family):
        """Tune this model's settings if it has any, then run all of its experiments."""
        methods = MODEL_FAMILIES[family]
        slug = family.lower().replace(" ", "_")
        tuning = [t for t in self.tuning_plan if t.method in methods]
        if tuning:
            print(f"Tuning: {len(tuning)} candidates")
            log = run_plan(tuning, self.config, self.root, self.directory, self.workers, f"tuning_log_{slug}")
            chosen, table = select_parameters(log, self.directory, self.config)
            self.tuning_logs[family], self.tuning_tables[family] = log, table
            self.config = replace(self.config, tuned={**self.config.tuned, **chosen})
            path = Path(self.directory) / "tuned_parameters.json"
            path.write_text(json.dumps(self.config.tuned, indent=1), encoding="utf-8")
        tasks = [t for t in self.plan if t.method in methods]
        print(f"Experiments: {len(tasks)} tasks")
        self.logs[family] = run_plan(tasks, self.config, self.root, self.directory, self.workers, f"task_log_{slug}")
        self._collect(family)
        failed = self.logs[family][self.logs[family].status.ne("success")]
        if len(failed):
            display(failed[["scenario", "experiment", "target", "method", "fold", "error"]].head(20))

    def _collect(self, family):
        """The first point where truth is read: after this model's tasks have finished."""
        if self.truth is None:
            self.truth = {n: read_truth(self.scenarios[n]).loc[self.measured[n].index] for n in self.names}
        log = self.logs[family]
        outputs = load_outputs(log, self.directory, experiments=list(POINT_EXPERIMENTS))
        self.scored[family] = ev.attach_truth(outputs, self.truth)
        self.draws[family] = load_outputs(log, self.directory, ".draws.parquet")
        self.importance[family] = load_outputs(log, self.directory, experiments=["importance"])

    def _scores(self, family):
        """Scores for one model, with the median added so that skill can be computed."""
        baseline = "Median and Mode"
        if baseline not in self.scored:
            raise RuntimeError("Run the Median and Mode section first: skill is measured against it")
        rows = [self.scored[family]]
        if family != baseline:
            rows.append(self.scored[baseline])
        numeric, category = ev.score_tables(pd.concat(rows, ignore_index=True))
        numeric = ev.add_scenario_levels(numeric, self.scenarios)
        category = ev.add_scenario_levels(category, self.scenarios)
        methods = MODEL_FAMILIES[family]
        return numeric[numeric.method.isin(methods)], category[category.method.isin(methods)]

    def show_model(self, family):
        """Settings, results by target, results by scenario and run status for one model."""
        numeric, category = self._scores(family)
        config, representative = self.config, self.config.representative
        table = self.tuning_tables.get(family)
        if table is not None:
            summary = table.groupby("method").agg(
                **{"targets": ("target", "size"), "targets changed": ("candidate", lambda s: int((s > 0).sum())),
                   "largest loss reduction": ("loss change", "min")}).reset_index()
            display(Markdown("**Tuning** (cross-validated loss on observed labels)"))
            display(summary.round(4))
            with pd.option_context("display.max_colwidth", None):
                settings = table.set_index("target").reindex(list(TARGETS)).dropna(subset=["settings"]).reset_index()
                display(settings[["target", "method", "settings"]])
        primary = numeric[numeric.experiment.isin(SHOWN_EXPERIMENTS) & numeric.scenario.eq(representative)
                          & numeric.seed.eq(config.seed)]
        if len(primary):
            comparison = ev.default_comparison(numeric, category, config)
            change = comparison.set_index(["target", "method"])["relative change"].rename("tuning change")
            joined = primary.join(change, on=["target", "method"]).set_index("target")
            joined = joined.reindex([t for t in NUMERIC_TARGETS if t in joined.index]).reset_index()
            for method, part in joined.groupby("method", sort=False):
                columns = [c for c in RESULT_COLUMNS if c in part and part[c].notna().any()]
                display(Markdown(f"**{method} · numeric targets · {representative}**"))
                display(part[columns].round(4))
        labels = category[category.experiment.isin(SHOWN_EXPERIMENTS) & category.scenario.eq(representative)
                          & category.seed.eq(config.seed)]
        if len(labels):
            display(Markdown(f"**model_id · {representative}**"))
            display(labels[["method", "n", "accuracy", "macro F1", "balanced accuracy"]].round(4))
        across = numeric[numeric.experiment.eq("main") & numeric.method.ne("Median")]
        if across.scenario.nunique() > 1:
            matrix = across.pivot(index="target", columns="scenario", values="skill").reindex(list(NUMERIC_TARGETS))
            slug = family.lower().replace(" ", "_")
            fg.plot_heatmap(matrix, f"{family}: skill in each scenario", self.figures, f"model_{slug}",
                            "Skill vs median", center=0.0, limit=1.0)
        all_labels = category[category.experiment.eq("main")]
        if all_labels.scenario.nunique() > 1 and all_labels.method.ne("Mode").all():
            display(all_labels.pivot(index="method", columns="scenario", values="accuracy").round(4))
        log = self.logs[family]
        cost = ev.cost_table(log)
        main = cost[cost.experiment.isin(SHOWN_EXPERIMENTS)]
        display(main[["method", "tasks", "fit seconds", "predict seconds", "seconds per target and fold"]].round(3))
        print(f"Tasks: {len(log)} | failed: {int(log.status.ne('success').sum())} | with warnings: {int(log.warnings.gt(0).sum())}")

    def scores_for(self, family):
        return self._scores(family)

    def finish(self):
        """Combine every model's results for the comparison sections."""
        missing = [family for family in MODEL_FAMILIES if family not in self.logs]
        if missing:
            raise RuntimeError(f"These model sections have not run: {missing}")
        self.log = pd.concat(self.logs.values(), ignore_index=True)
        self.tuning_log = pd.concat(self.tuning_logs.values(), ignore_index=True)
        self.tuning = pd.concat(self.tuning_tables.values(), ignore_index=True)
        self.all_scored = pd.concat(self.scored.values(), ignore_index=True)
        self.all_draws = pd.concat(self.draws.values(), ignore_index=True)
        self.all_importance = pd.concat(self.importance.values(), ignore_index=True)
        numeric, category = ev.score_tables(self.all_scored)
        self.numeric = ev.add_scenario_levels(numeric, self.scenarios)
        self.category = ev.add_scenario_levels(category, self.scenarios)
        self.pairs = ev.paired_intervals(self.all_scored, self.config)
        self.effects = ev.scenario_effects(self.all_scored, self.scenarios, self.config)
        self.groups = ev.group_table(self.all_scored, self.measured, self.config)
        self.ablation, self.ablation_labels = ev.ablation_table(self.numeric, self.category)
        self.cost = ev.cost_table(self.log)
        tables = {"numeric_scores": self.numeric, "category_scores": self.category, "paired_intervals": self.pairs,
                  "scenario_effects": self.effects, "visible_input_groups": self.groups,
                  "ablation_numeric": self.ablation, "ablation_model_id": self.ablation_labels,
                  "cost": self.cost, "tuning_results": self.tuning}
        for name, table in tables.items():
            table.to_csv(Path(self.directory) / f"{name}.csv", index=False)
        return self

    def planned_tasks(self):
        return len(self.plan) + len(self.tuning_plan)
