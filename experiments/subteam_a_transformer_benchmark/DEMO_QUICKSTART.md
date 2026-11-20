# PGS219-259 live demonstration

Run the existing Gaussian EM, Predictive MissForest, chained ridge and median/mode implementations on a small synthetic transformer experiment.

From the shared repository root, use a Python 3.11 environment with the demo requirements installed.

```sh
python -m pip install -r experiments/subteam_a_transformer_benchmark/requirements-demo.txt
python experiments/subteam_a_transformer_benchmark/scripts/product_review_demo.py
```

The script performs eight fresh fits across scattered and grouped gaps. Its default partitions contain 1,500 training, 600 calibration and 600 test rows. The fixed settings and single split make this a quick functionality demonstration. Refer to the final report for the tuned repeated results.

The output directory is printed at completion. Open `gmm_example_record.csv`, `gmm_components.png`, `gmm_fitting_and_intervals.png`, `live_comparison.png` and `verification.json`.

```sh
python experiments/subteam_a_transformer_benchmark/scripts/product_review_demo.py --noise 0.10
```

PGS219-260 provides independent saved-output checks. PGS219-261 provides the longer walkthrough and archived notebook evidence. This entry point reuses the benchmark implementation imported by PGS219-258.

## Results-focused review

Use the compact mode to fit the models and generate one results table and three graphs on an offline HTML page.

```sh
python experiments/subteam_a_transformer_benchmark/scripts/product_review_demo.py --results-only
```

Open the printed `presentation/review_results.html` path in a browser. The page embeds its figures and can be shared as one file. It compares Gaussian EM and MissForest, explains their field-level differences and illustrates why a close prediction can still have too much uncertainty for automatic filling. The existing numerical exports and checks are also produced.

To present a saved execution, run this command with its output directory.

```sh
python experiments/subteam_a_transformer_benchmark/scripts/present_product_review.py PATH_TO_RUN_DIRECTORY
```

The presentation reader uses the saved evidence. Its `presentation_sources.json` records the input file hashes. The directory also includes exact chart data and a short speaking guide. The means give 13 numeric fields equal weight. These remain single-split demonstration results with fixed settings.
