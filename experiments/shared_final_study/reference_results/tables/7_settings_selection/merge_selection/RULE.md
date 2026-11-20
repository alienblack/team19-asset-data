# Selection rule for the models from the other branches

Written on 2026-11-15 before any candidate below was scored. The rule is the one in
`../clean_selection/RULE.md` and is not changed after results appear.

## Models and candidates
- **Gaussian EM** and **Masked MLP** (Subteam A). Their package records the default settings and the
  search space, but not the settings their tuning selected (`selected_settings.csv` was left out of
  the package). Candidates are their default plus seven draws from their own `candidate()` search
  space, drawn with `numpy.random.default_rng(219)`. Their gas-decay feature, spatial installation-day
  feature, exact formulas and placebo blanking are off, because they encode the generator.
- **LightGBM chained** and **Predictive MissForest** (team pipeline). Their settings are fully
  documented in `TRANSFORMER_EXPERIMENT_GUIDE.md` and `run_manifest.json`, so they are used as written
  and not tuned: LightGBM 3 passes, 100 estimators, learning rate 0.07, 31 leaves; Predictive
  MissForest 100 trees, 5 passes, minimum leaf 10, `max_features='sqrt'`, no depth limit.

## Data
The three scenarios at 0.7 completion. Masks are nested, so every cell visible here is visible in every
scenario, and no cell hidden in any scenario is read. Truth files are not read. Each of 3 row folds hides a
further 10% of observed cells in its test rows.

## Choice
1. Score = mean, over the 13 numeric targets without random_placebo, of MAE divided by the median's MAE on
   the same cells, averaged over scenarios and folds.
2. Drop every candidate whose mean random_placebo ratio is above 1.01.
3. Keep candidates within 0.002 of the best remaining score.
4. Choose the fastest of those.

## Addendum, written before any of these candidates was scored
Both chosen settings lie on an edge of Subteam A's search space: the Gaussian EM at the smallest penalty and
the MLP at the most epochs. Candidates beyond those edges are added, keeping the other chosen settings:
Gaussian EM penalties 0.001, 0.002 and 0.0035, 15 and 25 steps, and 6 components; MLP 30 and 40 epochs,
width 192, and 40 epochs at width 192. The rule above is unchanged and is applied to the earlier and the
new candidates together.

## Second addendum, written before any of these candidates was scored, and the last round
The second round again chose edge values: the EM at penalty 0.001 and the MLP at 40 epochs and width 192.
This final round adds Gaussian EM penalties 0.0003 and 0.0005, and 6 components at penalty 0.001; MLP 60 and
80 epochs at width 192, and width 256 at 60 epochs. The rule is unchanged. Selection stops after this round
whatever it chooses, and the result is recorded as it is.
