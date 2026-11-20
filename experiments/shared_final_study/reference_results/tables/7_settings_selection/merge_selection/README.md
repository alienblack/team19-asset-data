# Settings for the models from the other branches

Written on 2026-11-15, when the main models of Subteam A's benchmark (SAPN_endtoend) and the team
pipeline were added to the study. `RULE.md` holds the rule and its two addenda; each was written
before the candidates it covers were scored.

| Model | Source | Settings | How they were set |
|---|---|---|---|
| Gaussian EM | Subteam A, `src/subteam_a/mixture.py` | 6 groups, 8 steps, penalty 0.001, `model_id` weight 1.8 | Rule of `RULE.md`, three rounds (`em.csv`, `em_extra.csv`, `em_final_round.csv`) |
| Masked MLP | Subteam A, `src/subteam_a/models.py` | width 256, 60 epochs, learning rate 0.0015, decay 0.0001277, 40% remasking, no dropout | Rule of `RULE.md`, three rounds (`mlp*.csv`) |
| LightGBM chained | pipeline, `experiment/transformer/scripts/comparison_models.py` | 3 passes, 100 trees, learning rate 0.07, 31 leaves | As written in the pipeline's guide and run manifest |
| Predictive MissForest | pipeline, `src/imputation/predictive_missforest` | 100 trees, 5 passes, leaf 10, `sqrt` features | As written in the pipeline's guide and run manifest |
| MICE-PMM | pipeline, `experiment/transformer/mice_pmm_logistic.py`, and the team's MICE notebook and methodology document (same code) | 5 chains, 10 sweeps, 5 donors, ridge 0.0001, logistic C = 1, 500 iterations | As written in the methodology document |

Subteam A's package did not include the settings their own tuning selected (`selected_settings.csv`
was left out), so their two models were chosen here from their own search space. The MLP was still
improving with more capacity in the last round, and selection stopped there as declared.

Parts of Subteam A's code that encode the generator are switched off in this study: the gas-decay
feature, the spatial installation-day feature, the exact formulas and the blanking of the placebo.
MICE-PMM was written to complete a whole table; here every model it fits sees training rows only.

The original `run.py`, `run2.py`, `run3.py` and their logs were not present in the supplied packages.
The updated handover supplies two replacement commands, from the project root:

- `python scripts/selection_audit.py`: reapply the declared rule to these recorded scores. This recovers the final choices without model fitting.
- `python scripts/refit_selection.py --model em` (or `--model mlp`): a new three-fold measured-data-only refit of the recorded candidates. Its remasking seed is explicit; the original unrecorded draws are not claimed to be reproduced. See the main README.

Use `--round 1`, `--round 2` or `--round 3` to include successive candidate rounds. Fresh selection outputs are separate and do not silently replace the published settings.
