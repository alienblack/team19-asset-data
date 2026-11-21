"""Fits every model family of the study outside the notebook. Usage: fit_driver.py quick|full [resume_run_id]"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # the project folder
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")
from synthetic_imputation.data import discover
from synthetic_imputation.runner import StudyConfig, default_workers, plan_table, quick_config
from synthetic_imputation.study import MODEL_FAMILIES, Study

if __name__ == "__main__":
    quick = sys.argv[1] == "quick"
    resume = sys.argv[2] if len(sys.argv) > 2 else None
    config = quick_config() if quick else StudyConfig()
    found = discover(ROOT / config.data_directory)
    scenarios = {name: found[name] for name in (config.scenarios or sorted(found))}
    study = Study(ROOT, config, scenarios, 4 if quick else default_workers(), resume)
    print("run folder:", study.directory, flush=True)
    print(plan_table(study.plan).to_string(index=False), "\ntotal", len(study.plan), flush=True)
    started = time.time()
    for family in MODEL_FAMILIES:
        began = time.time()
        study.run_model(family)
        log = study.logs[family]
        print(f"== {family}: {len(log)} tasks, failed {int(log.status.ne('success').sum())}, {time.time() - began:.0f} s", flush=True)
    print(f"ALL DONE in {time.time() - started:.0f} s", flush=True)
