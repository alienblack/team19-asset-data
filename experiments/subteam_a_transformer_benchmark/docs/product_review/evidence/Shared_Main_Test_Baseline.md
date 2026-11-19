# Shared-main test baseline

The shared repository at commit `94f9dcc7f999102c3b57dccbba959bc19ddd1663` has three existing failing tests. The full shared suite produced 150 passes and these three failures. All three failures were also reproduced from a separate untouched worktree at that main commit.

- `tests/test_noise.py::test_protected_target_cannot_be_used_as_noise_column`
- `tests/test_predictive_experiment_runner_review.py::test_runner_disjoint_fit_and_heldout_scoring`
- `tests/test_predictive_experiment_runner_review.py::test_runner_rejects_incomplete_reference_before_training`

The noise test expects a protected-target rejection. The runner fixtures omit execution/tuning/evaluation sections required by the current runner. These belong to the earlier overhead workflow and are disclosed separately from the new transformer demo checks.

The imported transformer benchmark passed all 32 of its tests using the prepared local synthetic input pairs. Its source snapshot matches SAPN_endtoend commit `6ae6621d40324b22dd5e415b2857b5d072ba2aca`.
