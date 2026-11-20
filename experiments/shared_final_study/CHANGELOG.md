# Handover update — version 5.1

The original eleven-model comparison, baseline, model implementations and selected full-study settings are retained. In particular, the GMM remains six shared-covariance components, at most eight EM steps, regularisation 0.001 and model-ID weight 1.8.

## Final-submission packaging — 20 November 2026

- Integrated this shared final study separately from the earlier Subteam A study.
- Added GitHub/Jira handover instructions, a local-output ignore file and Git
  attributes that preserve exact scientific source/data bytes across platforms.
- Restricted package metadata to the documented, tested Python 3.12 environment.
- Kept the client ZIP's folder name stable when built from the repository and
  excluded additional macOS/notebook cache files.
- Repeated unit tests, integrity/selection/report checks and a complete quick run
  from a fresh extraction; recorded the scope in `FINAL_SUBMISSION_AUDIT.json`.
- Preserved original model implementations, selected settings and archived results.

## Added

- One command for archived-result reporting, fresh quick/full execution and saved-prediction evaluation.
- Automatic offline HTML reports with every generated chart, source labels, model comparison, filters, field-level metrics, completed-data downloads, print styling and machine-readable summary/provenance.
- Automatic original-generator execution before fresh CLI runs, with its seed and draw order preserved.
- Failure status and execution logs; a resumed fitting run retries failed task checkpoints.
- Explicit configuration overrides, kept in the original run contract.
- A parameter-selection audit for the eight published tuned models. All eight supplied final choices are recovered by the declared rule.
- A replacement parameter-refitting command with explicit seeds and measured-data-only scoring. The original three selection scripts were not supplied. This command is new; it does not claim to reproduce their unrecorded historical remasking draws.
- An integrated copy of the supplied result exports from package 02, a ready-to-open full-study report and verification evidence.

## Updated

- The interactive notebook defaults to fresh fitting (`EVALUATION_RUN_ID = None`), with an automatic report step at the end.
- Original saved notebook outputs were cleared to prevent confusing them with a new execution. Published outputs remain available in the supplied result package and the generated report.
- Setup, run-mode and source documentation; references to unavailable tuning scripts are replaced with the supplied audit/refit commands.
- Explicit Jinja2 version for the report renderer; package version 5.1.0 and packaged HTML template.

## Verification scope

See `VERIFICATION.md` for tests and actual completed runs. Archived results and fresh quick-check results are kept distinct. A quick run is not a substitute for a newly completed full 3,580-task run. The shared study evaluates point predictions and does not validate per-asset GMM uncertainty intervals.
