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
