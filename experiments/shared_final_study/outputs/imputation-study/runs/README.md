# Study runs

Fresh quick/full execution writes timestamped run folders here. Each contains its contract,
task checkpoints, predictions, result tables, figures and completed reference datasets.

Package 03 is optional. To reevaluate the original full-study predictions, put its folder at
`outputs/imputation-study/runs/20261115-121714`, then run from the project root:

```sh
python run_study.py --mode evaluate --run-id 20261115-121714
```

The interactive notebook defaults to fresh fitting. To reevaluate through the notebook,
set `EVALUATION_RUN_ID` to the completed run folder name and use the matching quick/full
configuration. Merely copying a run folder does not change the selected execution mode.

To resume interrupted fitting, use `--mode quick --resume RUN_ID` or `--mode full --resume RUN_ID`
with the original configuration. Successful checkpoints are reused and failed ones are retried.
