"""Runs the study one model at a time and collects the results for the notebook.

Each model section of the notebook calls ``run_model`` and then ``show_model``.
Fitting happens in ``runner``, which never reads truth; truth is read here,
only after a model's tasks have finished.

With ``evaluation_run_id`` nothing is fitted: saved predictions from a finished
run are loaded, so evaluation can be repeated or started directly.
"""

from pathlib import Path

from IPython.display import Markdown, display
import pandas as pd

from . import evaluation as ev
from . import figures as fg
from . import relationships as rel
from .columns import NUMERIC_TARGETS
from .data import read_truth
from .runner import (
    build_plan, load_measured, load_outputs, open_run, run_plan,
)

# The notebook section each method belongs to, in running order.
MODEL_FAMILIES = {
    "Median and Mode": ("Median", "Mode"),
    "KNN": ("KNN imputer",),
    "IterativeImputer": ("IterativeImputer", "IterativeImputer (5)", "IterativeImputer (30)"),
    "Random Forest": ("Random Forest", "Random Forest calibrated"),
    "HistGradientBoosting": ("HistGradientBoosting",),
    "CatBoost": ("CatBoost",),
    "miceforest direct": ("miceforest direct",),
    "miceforest PMM": ("miceforest PMM",),
    "Joint miceforest PMM": ("Joint miceforest PMM",),
}
POINT_EXPERIMENTS = ("main", "stability", "budget", "ablation", "pmm", "joint", "calibration")
SHOWN_EXPERIMENTS = ("main", "pmm", "joint")
RESULT_COLUMNS = ["target", "MAE", "RMSE", "R2", "bias", "skill", "within tight", "within loose", "exact"]


def _slug(family):
    return family.lower().replace(" ", "_")


class Study:
    def __init__(self, root, config, scenarios, workers, resume_run_id=None, evaluation_run_id=None):
        if resume_run_id and evaluation_run_id:
            raise ValueError("Set RESUME_RUN_ID or EVALUATION_RUN_ID, not both")
        self.root, self.config, self.scenarios, self.workers = root, config, scenarios, workers
        self.names = list(scenarios)
        self.saved = evaluation_run_id is not None
        self.directory = open_run(root, config, self.names, evaluation_run_id or resume_run_id)
        self.figures = Path(self.directory) / "figures"
        self.plan = build_plan(self.names, config)
        self.measured = {name: load_measured(root, name, config) for name in self.names}
        self.truth = None
        self.logs = {}
        self.scored, self.draws, self.importance = {}, {}, {}

    def _load_saved(self, family):
        """Read a finished model's task log and predictions. Nothing is fitted."""
        path = Path(self.directory) / f"task_log_{_slug(family)}.csv"
        if not path.exists():
            raise FileNotFoundError(f"{family} has no saved results in {self.directory}: finish that run first")
        self.logs[family] = pd.read_csv(path)
        self._collect(family)
        print(f"{family}: loaded {len(self.logs[family])} saved tasks")

    def run_model(self, family):
        """Run all of this model's experiments, or load them when evaluating a finished run."""
        if self.saved:
            self._load_saved(family)
            return
        methods = MODEL_FAMILIES[family]
        slug = _slug(family)
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
        primary = numeric[numeric.experiment.isin(SHOWN_EXPERIMENTS) & numeric.scenario.eq(representative)
                          & numeric.seed.eq(config.seed)]
        if len(primary):
            joined = primary.set_index("target")
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
            fg.plot_heatmap(matrix, f"{family}: skill in each scenario", self.figures, f"model_{_slug(family)}",
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
        """Combine every model's results for the evaluation chapters."""
        missing = [family for family in MODEL_FAMILIES if family not in self.logs]
        if self.saved:
            for family in missing:
                self._load_saved(family)
            missing = []
        if missing:
            raise RuntimeError(f"These model sections have not run: {missing}")
        self.log = pd.concat(self.logs.values(), ignore_index=True)
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
        representative = self.config.representative
        self.relations = rel.relation_table(self.all_scored, self.measured, self.config)
        self.links = rel.link_table(self.all_importance)
        self.visibility = rel.visibility_table(self.all_scored, self.measured[representative], self.config)
        self.correlations = rel.correlation_table(self.all_scored, self.all_draws, self.measured[representative],
                                                  self.truth[representative], self.config)
        self.targets = ev.target_table(self.numeric, self.pairs, self.effects, self.config)
        tables = {"numeric_scores": self.numeric, "category_scores": self.category, "paired_intervals": self.pairs,
                  "scenario_effects": self.effects, "visible_input_groups": self.groups,
                  "ablation_numeric": self.ablation, "ablation_model_id": self.ablation_labels,
                  "cost": self.cost, "relations": self.relations, "links": self.links,
                  "visibility": self.visibility, "correlations": self.correlations, "targets": self.targets}
        for name, table in tables.items():
            table.to_csv(Path(self.directory) / f"{name}.csv", index=False)
        return self

    def planned_tasks(self):
        return len(self.plan)
