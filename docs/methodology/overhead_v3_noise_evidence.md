# PGS219-192 paired-noise evidence

The full generated output remains Git-ignored. Copies of the compact `primary_paired_summary.csv` and `observed_change_summary.csv`, plus two plots, are under `docs/results/overhead_conductors/v3_noise/` and `docs/images/overhead_conductors/v3_noise/` for review. These SHA-256 hashes identify the verified local output, not SAPN records.

| File | SHA-256 |
| --- | --- |
| `outputs/v3_phase1_final_20261025/manifest.json` | `86677a23cfcd7a10b6b7df111d06f75c4f4488218f25c7683a86bb57722ec3cc` |
| `outputs/v3_noise_client_20261025/manifest.json` | `63ced4acc4522a770bc81724c812d7c8ccb76fc4343593d3a9c5eb1043c1ee4a` |
| `outputs/v3_noise_client_20261025/primary_paired_summary.csv` | `5c1f466755506495d9d21e065cf8d74e1ccfeed41f7d3c2003d304704024f442` |
| `outputs/v3_noise_client_20261025/observed_change_summary.csv` | `a672253d4b367a474e1332afc2a7234e7087c70c5c7d441538a76bf57548300d` |
| `outputs/v3_noise_client_20261025/paired_cell_metrics.csv` | `8f957bb245ef0228943b9456f349bc97f745527002b3c1a3d02a2757529d898e` |
| `outputs/v3_noise_client_20261025/paired_population_metrics.csv` | `850785581ea79eefe59dc655aaa9a21f7220316516224ef9291c1e4b249bd429` |
| `outputs/v3_noise_client_20261025/material_confusion.csv` | `81387d8954650798cf724d634012274b0b687e387ca4283736d0abdfd8a6cf96` |
| `outputs/v3_noise_client_20261025/material_by_decade.csv` | `ed5b38e56a8a91e5763f21da24b2008c5a8e3dcd02d3f62d8010ecfe9718` |
| `docs/images/overhead_conductors/v3_noise/paired_log_loss.png` | `4c9b908acc09251f26993fdb2a8d1c6a96ea567fa3e4787222e1107bb5a6064a` |
| `docs/images/overhead_conductors/v3_noise/paired_aged50_population.png` | `73b4a768519f8b9a9b857ab53c69f50b7e8162379ead4ee44daeb66d6a3904e4` |

The output is reproducible with `overhead_v3_noise_runbook.md`, using the saved clean-pilot settings and source versions recorded in the manifests. Results are descriptive for one synthetic generator and do not imply SAPN performance.
