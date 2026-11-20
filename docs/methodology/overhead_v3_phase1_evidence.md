# Phase 1 pilot evidence manifest

Most synthetic pilot output is Git-ignored. The two per-seed metric CSVs below are committed so reviewers can inspect the report's inputs without rerunning the pilot. These SHA-256 hashes identify the original Phase 1 run at commit `2d7e370` and its key summaries; later evaluation work adds metrics to the source. Re-run the command in `overhead_v3_phase1_runbook.md` from that commit in a new output directory, then compare the generated files. The five-seed test data and tuning-only tree-count check are separate experiments.

| File | SHA-256 |
| --- | --- |
| `src/team19_asset_data/overhead_conductors/v3_pilot.py` | `8337006301aec365590139c8f529c432bc770153e4d4f48c5bf82e88443b3f88` |
| `src/team19_asset_data/overhead_conductors/run_v3_tree_check.py` | `4dce61bbd33a6c1ce176b43cf72fb74bf74c4ae67efc43f3f7ccedbf210f9f5b` |
| `outputs/v3_phase1_age50_label_20261024/manifest.json` | `aed568d218335799819ebe5ae20c8b84a09a3ed677be05b712bfa6c2b895c0f0` |
| `outputs/v3_phase1_age50_label_20261024/cell_metrics.csv` | `7f7bf50f0f19b7da731356571beb52a6ab2297383d65aa47a4b2cf089d682f8d` |
| `outputs/v3_phase1_age50_label_20261024/population_metrics.csv` | `6e372216ade08c53c3792cf9a924d94148156c082227983f306b09ee98a512e2` |
| `outputs/v3_phase1_age50_label_20261024/cell_metrics_summary.csv` | `75a93db02678ea76452d877481b2446e093544d1698aa31adf23fd78e14cbb2f` |
| `outputs/v3_phase1_age50_label_20261024/population_metrics_summary.csv` | `a4f9508974575bef446e13ec99b129f63bedfbe9f2feadf873ba0bced6adcc84` |
| `outputs/v3_phase1_age50_label_20261024/gap_closed.csv` | `5a5324020306510c9615f5d6c820d69117eaf00bedf6b349b4a00e0d6f9383cc` |
| `outputs/v3_phase1_age50_label_20261024/integrity_checks.json` | `1444681edb382ce184f61b3b66cc9fcf37f531aed40e8cb1a7e7ec9716ea5caf` |
| `outputs/v3_tree_count_tuning_age50_20261024.csv` | `54ddf8f47a0464ac27d2d02ad7dd4a72978b1e058b88daec9f36d6a625f372cd` |

Apart from the two per-seed metric CSVs, this manifest preserves fingerprints, not the output files themselves. The code and documented commands are required to recreate the ignored outputs. No SAPN data is present in this experiment.
