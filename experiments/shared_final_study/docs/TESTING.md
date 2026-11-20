# Test the dataset workbench

These checks use synthetic example data. They create new run folders and do not
replace the published results. Start with the first walkthrough; the remaining
checks are useful before sharing a release or changing the interface.

## 1. Start the app

After the Python 3.12 environment has been installed, open a terminal in the
project folder and activate it. Then run:

```sh
python run_study.py --mode guided
```

Open **http://127.0.0.1:8765**. Keep the terminal open. If the port is in use,
choose another, for example `--port 8766`. To stop the server, press Ctrl+C.
The macOS `Start_Workbench.command` and Windows `Start_Workbench.bat` launch the
same app after installation.

## 2. First run: the built-in example

1. Click **Use example data**. The summary should show **240 records**, **7
   columns** and **207 missing cells**. It is a synthetic demonstration.
2. Confirm `asset_id` is **Ignore / ID** and `material` is **Categorical**.
   Numeric fields should have numeric types. The example supplies several
   illustrative tolerances; these are not client-approved engineering limits.
3. Click **Advanced field settings** to reveal bounds, groups and rounding.
   `span_m` and `coastal_index` should share the group `location`. Click it again
   to return to the shorter table.
4. Keep Median / mode, Predictive MissForest and Gaussian EM selected. The
   baseline remains included. **More methods** reveals the other choices.
5. Keep **20% additional masking**, **Scattered gaps**, no added noise and the
   default evaluation/model settings. Click **Run study & generate report**.
6. Wait for **Report ready**, then select **Open report**. Expect three methods
   in the comparison if all fits succeeded. The report should explicitly show
   any method-stage errors; do not ignore them.
7. Download a completed CSV and `config.json`. The CSV should have **240 data
   rows and 7 columns**. Recorded values should be retained and modelled gaps
   should be filled for successful completion fits. IDs must remain intact.
8. Filter the field table by **Gaussian EM**, then by a field. Click a chart to
   enlarge it, close it, and try its **PNG** download.
9. Click **Print / save PDF**, choose **Save as PDF**, and inspect the preview.
   It should include the whole report, even if you had filtered the on-screen
   field table. Charts should load, and the original filter should return after
   the print action.

A successful run does not require a particular skill score. Weakly predictable
fields can score close to, or below, the baseline. Check that this is reported
honestly and that you can trace each result to the saved files.

## 3. Upload your own files, using the supplied example pair first

1. Under **Dataset with missing values**, select `examples/example_assets.csv`.
2. Open **Reference data (optional)** and select `examples/example_truth.csv`.
3. Click **Load dataset**. Look for **matching truth supplied** in the status.
4. Review the inferred types. The upload route does not import the example's
   illustrative tolerances or field groups automatically; enter these if needed.
5. Run the comparison. The report's scoring reference should identify the
   supplied truth. Truth helps score held-out gaps and is never fitted by models.
6. Choose a different file without loading it. The old field table should clear
   and the run button should be disabled. Click **Load dataset** again to proceed.

Matching truth must have the same columns, column order and row order. Do not
supply an unrelated dataset or assume the app joins records by ID. The ordinary
no-truth workflow remains useful when clean reference values do not exist.

## 4. Test grouped gaps and measurement noise

Use **Use example data** again. It clears file selections and restores the
example's field definitions. Your experiment controls remain as you set them.

- Choose **Grouped gaps**. Advanced field settings should open automatically.
  Verify `span_m` and `coastal_index` both have group `location`.
- Open **Optional measurement-noise stress test**, enter **5**, and select
  `span_m` and `coastal_index`. Avoid percentage noise on numeric calendar years.
- Run a new study. The report should show **grouped** hiding and **±5%** noise on
  the selected fields. Completed exports should still preserve recorded inputs:
  artificial masking and noise apply only to evaluation.
- Compare with a separate scattered/no-noise run. Keep the same seed, sample,
  folds, methods and model settings when investigating one changed condition.
  Each run represents one condition, not a complete robustness study.

To keep a development smoke check short, use **2 folds**, an **80-row evaluation
sample**, **8 trees**, **1 chained pass** and **5 EM steps**. These small settings
exercise the interface; they are not suitable evidence for a final model ranking.

## 5. Check failures are understandable

| Action | Expected response |
| --- | --- |
| Click Load dataset before choosing a CSV | A visible message asks you to choose a dataset; no run starts. |
| Upload a CSV with duplicate headers | A validation message appears; the run button stays disabled. |
| Supply a truth file with mismatched rows/columns | Loading fails with a clear validation message. |
| Select grouped gaps without a valid shared group | The run reports the configuration error; it does not invent groups. |
| Add noise without selecting numeric fields | The run reports the missing field selection. |
| Change a selected file after inspection | The schema clears and the run button is disabled until you load again. |
| Open on a narrow screen | The page fits; wide tables scroll inside their panels. |

Do not test failures by corrupting the original benchmark inputs. Use temporary
copies or small synthetic files.

## 6. Automated Python checks

From this project's folder, in its Python environment:

```sh
python -m pip check
python -m pytest -q
```

The Python suite covers models, masks, preprocessing isolation, truth exclusion,
metrics, preservation of recorded inputs and server validation. One
IterativeImputer convergence warning is expected under the deliberately small
iteration budget used by its test. A convergence warning does not by itself mean
a fit failed; inspect the context and saved warnings.

## 7. Optional automated browser checks

The app itself needs only Python. The optional browser checks also require a
supported Node.js installation and Playwright's Chromium browser. Install these
from the test folder:

```sh
cd tools/ui-tests
npm install
npx playwright install chromium
```

Keep the app running in another terminal. Then, from `tools/ui-tests`, run:

```sh
npm test
```

For a different port:

```sh
npm test -- --url http://127.0.0.1:8766
```

The script uses the synthetic example, performs three fresh comparisons, tests
uploads, progressive settings, report filters, chart/CSV/config downloads,
printing and desktop/mobile layout. It saves `ui-results.json`, screenshots,
downloads and a browser-generated PDF under `outputs/ui-checks/`. A failing check
exits with an error and saves a screenshot. These folders and `node_modules/`
are excluded from Git and the client ZIP.

This is a functional Chromium smoke test, not a certification of every browser,
assistive technology or large dataset. The manual PDF print-preview check is
still useful, particularly on the computer used for the presentation.
