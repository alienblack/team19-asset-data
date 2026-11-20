# Selection rule, fixed before any result of this round was seen

Written on 2026-11-05 before the first fit of this round. No choice below may be changed after results appear.

## Data
- Scenarios: completion0.7-noise0.0, completion0.7-noise0.05, completion0.7-noise0.1.
- Masks are nested, so every cell visible here is visible in every scenario. No cell hidden in any scenario is read.
- Truth files are not read. Each of 3 row folds hides a further 10% of observed cells in its test rows.

## Point models
1. Score = mean, over the 13 numeric targets without random_placebo, of MAE divided by the median's MAE on the same cells, averaged over scenarios and folds.
2. Drop every candidate whose mean random_placebo ratio is above 1.01.
3. Keep candidates within 0.002 of the best remaining score.
4. Choose the fastest of those.

## Donor-matched models
1. Band gap = mean over the 14 numeric targets of |coverage - 0.72|. A 10% to 90% band from 20 draws covers about 72% of cells when calibrated.
2. Keep candidates within 0.01 of the smallest band gap.
3. Choose the lowest point score of those. Ties go to the fastest.

## Confirmation
Each chosen configuration is scored against truth once. The result is reported as it is, and no setting is changed because of it.

## Addendum, written before any of these candidates was scored
HistGradientBoosting was offered 2, 3, 4 and 6 passes, while the other chained models were offered fewer. The same range is added for them, keeping their other chosen settings: Random Forest and CatBoost 2, 3 and 6 passes, KNN 4 and 6, miceforest 6. The rule above is unchanged and is applied to the earlier and the new candidates together.
