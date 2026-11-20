# Verification of the automated handover

## Version 5.2.2 - client documentation and transformer walkthrough

- Added the supplied workbench/report screenshots, a transformer walkthrough,
  field tolerances and groups, noise/masking comparisons, report interpretation,
  sharing instructions and a client release/version-control checklist.
- Checked **26 local Markdown links/anchors** with no unresolved targets. Both
  supplied PNGs decode correctly and retain their exact source bytes.
- Validated `examples/transformer_workbench_config.json` against the supplied
  10,000-row, 15-column input/truth pair. Its modelling settings match the actual
  product demonstration after removing profiling-only column metadata.
- The preceding transformer product demonstration succeeded: **nine validation
  fits**, **three 10,000-row completion exports**, no model-stage failures and
  recorded values retained. Its example metadata is in
  `docs/transformer-example-run.json`; this is one diagnostic condition.
- All **200 protected backend/result files remain byte-identical**. No model,
  metric, UI or dependency-pin change was made in this documentation update.
  The 91 Python tests and 12 browser checks below refer to version 5.2.1; they
  have not been relabelled as fresh runs for a documentation-only change.

The documentation audit is `docs/audits/5.2.2-client-documentation.json`.
PR merge, release tagging and client publication remain distinct steps; a
built ZIP or pushed documentation commit does not imply they have occurred.

## Version 5.2.1 - documentation and workbench usability

The README now includes a first-run walkthrough, navigation and troubleshooting.
[docs/TESTING.md](docs/TESTING.md) provides manual checks and the optional
Playwright browser test. Advanced field controls, evaluation settings and extra
methods are collapsed until requested. Upload selection and validation states
were clarified; the modelling and scoring code is unchanged.

- **91 Python tests passed**, with the same one expected small-budget
  IterativeImputer convergence warning. `pip check` found no broken requirements.
- **12 browser checks passed**: example loading, uploads with aligned truth,
  scattered/grouped runs, selected-field noise, invalid uploads/settings, report
  filters, enlarged charts, downloads, print preparation and responsive layout.
- A separate run with the unmodified default UI settings completed successfully
  on the 240-row example. No uncaught browser JavaScript errors were observed.
- Desktop (1512 px) and mobile (390 px) screenshots were visually inspected.
  Tables scroll within their panels on small screens.
- **200 protected backend/result files retain their exact SHA-256 hashes**,
  including all model Python, archived results and saved example report outputs.
  Completed synthetic exports retain the supplied observed inputs.

`UI_REVIEW_AUDIT.json` records this follow-up. `FINAL_SUBMISSION_AUDIT.json` and
existing generated reports remain historical version-5.2.0 records; they have
not been relabelled as new scientific runs. The optional browser-test runtime is
excluded from the handover archive.

## Version 5.2 - guided datasets and report redesign

- **91 tests passed**, with one expected IterativeImputer convergence warning
  under a deliberately small test budget. There were no failed tests.
- The new checks cover arbitrary schemas, five categorical labels, all twelve
  model adapters, train/test isolation, absence of truth in fitting, paired masks,
  independent noise draws, exact year tolerances, date units, preservation of
  recorded values/identifier text, partial-model failures and local-server bounds.
- A complete new-schema run with all twelve methods succeeded under grouped gaps
  and added 10% measurement noise: 24 validation fits and twelve full-input exports,
  with zero model-stage failures. These are diagnostic settings, not published
  research findings.
- The browser also accepted the original 10,000-row, 15-field transformer CSV and
  matching truth, correctly identifying model ID as categorical. Desktop/mobile
  rendering produced no page-level overflow or JavaScript errors.
- Report figures decoded offline. Printing loaded all figures, included every
  result row and restored the selected screen filters afterwards. The example PDF
  was rendered and visually checked for pagination/clipping.
- The original 25 scientific modules, 30 input CSVs and 58 result CSVs retain
  their exact hashes. The original published scientific summary is identical.
  Dependency pins are unchanged; the existing Python 3.12 environment was reused.

The version-5.2 audit is `FINAL_SUBMISSION_AUDIT.json`. The earlier version-5.1 audit
is retained in `docs/audits/5.1-final-submission.json`; its code hashes refer to that
earlier release. New-dataset reports use their own schemas and generic settings.
They are not relabelled reproductions of the original benchmark. No new full
3,580-task original benchmark was claimed for this interface update.

## Earlier version 5.1 verification

The original numerical/model code and scientific inputs were kept intact. `ORIGINAL_SOURCES.json` records the hashes from the two supplied archives. The package builder checks all **25 original Python package modules**, **30 input CSVs** and **58 result CSVs** before writing a client archive.

## Completed checks

- **68 tests passed:** the original 62 model/study tests plus six new checks of reporting, provenance, configuration and selection.
- **Fresh quick-study workflow:** all **3,060 tasks succeeded**, with zero failed tasks. The quick profile uses 600 sampled rows and small models; these results are diagnostic, not final research findings. Model iteration warnings are retained in `execution.log` rather than interpreted as task failures.
- **Saved-prediction reevaluation:** the completed quick run was reevaluated and reported without refitting. A clean extracted archive also completed a fresh 3,060-task quick run. All twelve completed datasets agreed between the two seeded executions within numerical tolerance.
- **Selection audit:** all eight published tuned choices match the declared rule when reapplied to the supplied candidate scores.
- **Replacement tuning runner:** a two-candidate, three-fold, three-scenario GMM diagnostic completed on 600 rows. It writes its new selection separately. It uses measured data only and does not claim to reproduce unrecorded historical remasking draws.
- **Published report:** all 54 supplied charts decoded in browser testing. Method/field filters, full-size chart inspection, PNG downloads and local result-file links worked. No JavaScript page errors were observed.
- **Responsive and print layout:** checked at 1440-pixel desktop width and 390-pixel mobile width, with no page-level horizontal overflow. Charts remain visible in print styling.
- **Reference-result consistency, before the update:** all twelve completed datasets matched 179 target/model score groups. The 29,928 GMM-imputed reference cells matched an independent five-fold rerun; numeric skill was 0.7389234103 and model-ID accuracy 98.4224%.

The assembled archive is also verified from a fresh extraction. Machine-readable details are in `verification.json`.

## What these checks do not claim

The supplied full-study results are the original archived run **20261115-121714**, not a newly completed 3,580-task full refit. The automated full command is provided; the end-to-end execution test uses the quick profile. The new report distinguishes those sources prominently.

The original separate Subteam A repository's 32 tests also passed during the earlier review, but that repository is not needed to execute this integrated shared-study package. No model is certified for operational SAPN use by these synthetic checks.

## Final-submission check — 20 November 2026

The PGS219-265 integration was checked again before submission:

- All **68 tests passed** in 38.02 seconds using Python 3.12.11; `pip check`
  reported no broken requirements.
- All 25 original scientific modules, 30 input CSVs and 58 result CSVs retained
  their recorded SHA-256 hashes. Their staged Git blobs also matched, and
  `.gitattributes` now prevents checkout line-ending conversion on other platforms.
- The selection audit recovered all eight recorded choices without new tuning.
- A fresh extraction completed the **3,060-task quick workflow with zero failed
  tasks**, including CSVs, figures and the automatically generated report.
- The published report decoded all 54 embedded charts, used no network requests,
  produced no JavaScript errors and had 72 valid local report/download links.
- The client ZIP was rebuilt with the package builder to exclude macOS metadata,
  local environments, build/cache files and generated quick-run predictions.

The existing pinned dependency environment was checked and reused; no new
installation was claimed. No new 3,580-task full refit was performed. Package 03
was unavailable, so the original full run's raw-prediction reevaluation was not
tested in this final check. See `FINAL_SUBMISSION_AUDIT.json` for the new run ID,
warning counts and runtime file hashes.
