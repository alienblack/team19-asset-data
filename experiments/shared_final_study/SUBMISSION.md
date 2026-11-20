# Final code submission — PGS219-265

This folder is the combined Team 19 final study: the original code/input package
01, result/figure package 02, and the automated execution/reporting handover.
It is independent of the earlier Subteam A study elsewhere in this repository.
Start with [START_HERE.md](START_HERE.md).

## Which files to submit

| Destination | Submission |
| --- | --- |
| GitHub | This complete folder through the `PGS219-265-final-product-handover` branch/PR. Retain the earlier experiments and their history. |
| Jira PGS219-265 — Final Product | The clean handover ZIP, the actual PR/commit link and the validation summary. |
| Jira PGS219-266 — Final Report | The agreed final report/PDF, linked to the same code release. The generated HTML report is supporting result evidence. |
| Team/client handover | The same clean handover ZIP and its SHA-256 file; open `START_HERE.md` after extraction. |

Use the real PR URL after publication. A local branch or draft PR description is
not evidence that the changes have been pushed or merged. A GitHub source-code
ZIP is not automatically the same archive as the separately built client ZIP.

The client ZIP contains the code, tests, notebooks, exact synthetic inputs,
archived tables, twelve completed reference datasets, 54 figures and an offline
HTML report. It excludes virtual environments, builds, caches, macOS metadata
and generated quick-run predictions. Retain the containing folder when sharing
the HTML report: its figures are embedded, but CSV download links use nearby files.

## Package 03: when it is required

- **Read the original results:** not required; package 02 contains their exports.
- **Train fresh models:** not required; quick/full modes fit from the supplied inputs.
- **Reevaluate the original per-cell predictions without fitting:** required. Place
  its exact run at `outputs/imputation-study/runs/20261115-121714/` and use the
  documented evaluate command. Those original predictions are not in this release.

Do not describe report regeneration as a fresh model experiment. Fresh quick
experiments are installation diagnostics; their smaller models are not the
published research findings. The existing full-study results remain archived
run `20261115-121714`; a new full command has not been claimed as completed here.

## Keep the two study environments separate

| Study | Environment | Reference protocol |
| --- | --- | --- |
| This shared final study | Python 3.12; this folder's pinned requirements | Five row folds; reference 20% missing and 5% noise; GMM has six components; eleven imputers plus median/mode baseline. |
| Earlier `subteam_a_transformer_benchmark` | Python 3.11; that folder's pinned requirements | Separate spatial train/tune/calibration/test design; reference 30% missing; additional uncertainty and confidence analyses. |

Their results answer different experimental questions. Do not replace one
study's scores with the other's or install both pinned environments together.
This shared final study evaluates point predictions; it does not establish
calibrated per-asset uncertainty intervals for operational SAPN use.

## Release construction and checks

From this folder in the Python 3.12 environment:

```sh
python -m pip check
python -m pytest -q
python scripts/selection_audit.py
python run_study.py
python tools/build_handover.py --out /path/outside/project/Team19_Automated_Handover.zip
```

Use the Python builder rather than compressing the working folder in Finder.
It excludes local experiment artefacts and refreshes `PACKAGE_MANIFEST.json`.
Its integrity gate requires the 25 original scientific modules, 30 input CSVs
and 58 result CSVs to match `ORIGINAL_SOURCES.json`.

Extract the resulting ZIP to a new folder before testing the documented entry
point. Keep the archive, its checksum, source commit and test evidence together.
`VERIFICATION.md` records the earlier validation; the final-submission audit
records any additional checks and their actual scope.

## Suggested Jira comment after publishing

> Prepared the combined final study for submission, with code, synthetic inputs,
> archived results, completed datasets, figures and an automated offline report.
> Verified the original scientific source/data hashes and checked the runnable
> entry points. Removed generated caches and local quick-run predictions from
> the client archive. The archived full results and fresh diagnostic runs are
> clearly distinguished. PR: **[insert actual published URL]**. Release ZIP and
> validation evidence are attached. Original saved-prediction reevaluation needs
> the optional package 03; fresh fitting does not.

These files preserve the supplied team source and provenance records. No new
licence grant is asserted over other contributors' code or third-party libraries.
