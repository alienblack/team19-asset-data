# Final code submission — PGS219-265

This folder is the combined Team 19 final study: the original code/input package
01, result/figure package 02, and the automated execution/reporting handover.
Version 5.2.2 includes a local guided workflow for new CSV datasets, with a
separate configurable evaluation and report. It preserves the archived study.
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
HTML report, plus the new dataset workbench and its synthetic example. It excludes virtual environments, builds, caches, macOS metadata
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

## Branch and release workflow

Use the existing ticket branch for this handover:
`PGS219-265-final-product-handover`, reviewed through
[PR #28](https://github.com/alienblack/team19-asset-data/pull/28). Keep `main` as
the reviewed, working team version. A second permanent `dev` or `final` branch
would create another copy to synchronise; it is unnecessary for this submission.
Continue later fixes on short Jira branches and review their PRs into `main`.

1. Finish and test the ticket branch. Keep its PR scope and test evidence current.
2. Review/merge PR #28 into `main` using the team's agreed review process. Check
   the PR's actual state; a push to its branch is not a merge.
3. From a clean checkout of the merged `main`, verify the package version and
   required checks. Build the client ZIP with `tools/build_handover.py`.
4. Tag **the merged main commit** as `v5.2.2` for this package version, after
   checking that the tag does not already exist. Do not move an existing release
   tag; a later correction gets a new patch version and tag.
5. Publish a GitHub Release for that tag with short release notes, the clean
   `Team19_Automated_Handover.zip`, its `.sha256`, `release_receipt.json` and the
   agreed final report/PDF. Keep the exact source commit and checksum in the
   receipt. Attach this curated ZIP rather than relying on GitHub's automatic
   whole-repository source ZIP.
6. Add that same release/PR link, ZIP/checksum and evidence to Jira PGS219-265.
   Attach the agreed report to PGS219-266. Send the client the named release and
   the first-run instructions. Record what was actually submitted.

The tag/release step is intentionally done **after** the PR merge. The client
ZIP can be prepared and checked beforehand, but the final receipt should identify
the tagged merged source commit. Do not describe a proposed tag as already live.
[GitHub's release documentation](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases)
explains how releases use tags to identify a specific repository revision.

### Keep Git and the client archive clean

- Version source, tests, documentation/screenshots, pinned dependencies, synthetic
  examples/configs and intentionally curated reference evidence.
- Keep `.venv/`, caches, `node_modules/`, new uploaded datasets and scratch runs
  out of commits and the client ZIP. New generated runs remain in ignored
  `outputs/`; retain required curated reference results in their existing folders.
- Keep real client data outside the public repository. The supplied inputs are
  synthetic; their presence does not authorise publishing operational records.
- Put release ZIPs and final PDFs in release assets rather than committing a
  fresh large ZIP on each change. Preserve already agreed historical evidence.
- Use `git status` and review the staged diff before committing. Avoid a broad
  cleanup that deletes teammates' experiments or rewrites shared history.
- Give later fixes a new Jira branch, PR and patch release. Leave the client's
  previous tag and ZIP intact so its results remain traceable.

The integration PR is https://github.com/alienblack/team19-asset-data/pull/28.
It must be reviewed/merged before these additions are described as present on main.
