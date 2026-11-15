"""Project-owned fixes for miceforest 6.0.5, without editing site-packages.

The new-data loop follows miceforest's synchronous iteration schedule: every
field uses the completed state at the beginning of a round. All results from
that round, rather than initialization, feed the next round. The index, counter
and completion fixes are deliberately limited to the pinned upstream version.
Upstream: AnotherSamWilson/miceforest (MIT); see the installed package license.
"""

from importlib.metadata import version

import miceforest as mf
import numpy as np
import pandas as pd
from miceforest.imputed_data import ImputedData
from miceforest.utils import ensure_rng


COMPAT_REVISION = "mf605-index-and-new-data-v1"


def require_supported_version():
    actual = version("miceforest")
    if actual != "6.0.5":
        raise RuntimeError(
            f"This audited compatibility implementation requires miceforest==6.0.5; "
            f"found {actual}. Re-audit before changing the dependency."
        )


def nonmissing_positions(data, variable):
    """Use platform-sized row positions, independently of na_where's dtype."""
    available = np.ones(data.shape[0], dtype=bool)
    missing = np.asarray(data.na_where[variable], dtype=np.intp)
    if missing.size:
        assert missing.min() >= 0 and missing.max() < data.shape[0]
        assert np.unique(missing).size == missing.size
        available[missing] = False
    result = np.flatnonzero(available)
    assert result.size == data.shape[0] - missing.size
    assert np.unique(result).size == result.size
    return result


def assert_observed_unchanged(original, completed):
    assert original.index.equals(completed.index)
    assert original.columns.equals(completed.columns)
    for column in original:
        known = original[column].notna()
        pd.testing.assert_series_equal(
            original.loc[known, column], completed.loc[known, column],
            check_dtype=False, check_names=False,
        )


class CorrectedImputedData(ImputedData):
    def _get_nonmissing_index(self, variable):
        return nonmissing_positions(self, variable)

    def complete_data(self, dataset=0, iteration=-1, inplace=False, variables=None):
        result = self.working_data if inplace else self.working_data.copy()
        selected = self.imputed_variables if variables is None else variables
        assert set(selected).issubset(self.imputed_variables)
        for variable in selected:
            selected_iteration = (
                self.iteration_count(dataset=dataset, variable=variable)
                if iteration == -1 else iteration
            )
            result.loc[self.na_where[variable], variable] = self[
                variable, selected_iteration, dataset
            ]
        return None if inplace else result


class CorrectedKernel(mf.ImputationKernel):
    """Preserve upstream models/PMM; repair positional indices and new-data state."""

    def __init__(self, *args, **kwargs):
        require_supported_version()
        super().__init__(*args, **kwargs)
        self.training_index_audit = {}

    def _get_nonmissing_index(self, variable):
        return nonmissing_positions(self, variable)

    def _get_nonmissing_subset_index(self, variable, seed):
        indices = np.asarray(super()._get_nonmissing_subset_index(variable, seed), dtype=np.intp)
        expected = nonmissing_positions(self, variable)
        assert np.unique(indices).size == indices.size
        assert np.isin(indices, expected).all()
        wanted = self.data_subset[variable]
        if wanted == 0 or wanted >= expected.size:
            np.testing.assert_array_equal(indices, expected)
        else:
            assert indices.size == wanted
        self.training_index_audit[variable] = {
            "available_rows": int(expected.size), "selected_rows": int(indices.size),
            "unique_selected_rows": int(np.unique(indices).size),
            "maximum_position": int(indices.max()) if indices.size else None,
            "positions_over_255": int((indices > 255).sum()),
        }
        return indices

    complete_data = CorrectedImputedData.complete_data

    def impute_new_data(
        self, new_data, datasets=None, iterations=None, save_all_iterations_data=True,
        copy_data=True, random_state=None, random_seed_array=None, verbose=False,
    ):
        assert self.save_all_iterations_data, "Train with save_all_iterations_data=True."
        assert isinstance(new_data, pd.DataFrame)
        assert self.working_data.columns.equals(new_data.columns)
        assert all(self.working_data[c].dtype == new_data[c].dtype for c in self.column_names)
        original = new_data.copy(deep=True)
        datasets = list(self.datasets if datasets is None else datasets)
        assert set(datasets).issubset(self.datasets)
        trained_iterations = int(self.iteration_count())
        iterations = trained_iterations if iterations is None else int(iterations)
        assert iterations >= 1 and trained_iterations >= 1
        result = CorrectedImputedData(
            impute_data=new_data, datasets=datasets, variable_schema=self.variable_schema.copy(),
            save_all_iterations_data=save_all_iterations_data, copy_data=copy_data,
            random_seed_array=random_seed_array,
        )
        order = [v for v in self.model_training_order if v in result.vars_with_any_missing]
        if random_seed_array is not None and random_state is None:
            raise ValueError("random_seed_array requires explicit random_state")
        rng = self._random_state if random_state is None else ensure_rng(random_state)
        previous_rng = self._random_state
        self._random_state = rng  # PMM and initialization share the explicit new-data seed.
        result.iteration_diagnostics = []
        try:
            self._initialize_dataset(result, random_state=rng)
            for iteration in range(1, iterations + 1):
                model_iteration = min(iteration, trained_iterations)
                for dataset in datasets:
                    self.complete_data(dataset=dataset, inplace=True)
                    result.complete_data(dataset=dataset, inplace=True)
                    for variable in order:
                        previous = result.working_data.loc[result.na_where[variable], variable].copy()
                        model = self.get_model(variable, dataset=dataset, iteration=model_iteration)
                        values = self._mean_match_ind(
                            variable=variable, lgbmodel=model,
                            bachelor_features=result._get_bachelor_features(variable),
                            dataset=dataset, iteration=model_iteration,
                            hashed_seeds=result._get_hashed_seeds(variable),
                        )
                        values.index = result.na_where[variable]
                        assert values.shape == (result.na_counts[variable],)
                        assert not values.isna().any()
                        result[variable, iteration, dataset] = values
                        result.iteration_tab[variable, dataset] = iteration
                        categorical = isinstance(values.dtype, pd.CategoricalDtype)
                        change = (
                            np.mean(previous.astype(object).to_numpy() != values.astype(object).to_numpy())
                            if categorical else np.mean(np.abs(values.astype(float).to_numpy() - previous.astype(float).to_numpy()))
                        )
                        result.iteration_diagnostics.append({
                            "dataset": dataset, "iteration": iteration, "variable": variable,
                            "n_missing": len(values), "change": float(change),
                            "change_kind": "changed_fraction" if categorical else "mean_absolute_change",
                        })
                        if not save_all_iterations_data:
                            del result[variable, iteration - 1, dataset]
                    # Complete modeled fields without missing cells still advance the round.
                    for variable in result.modeled_variables:
                        result.iteration_tab[variable, dataset] = iteration
                    if verbose:
                        print(f"New data: iteration {iteration}, dataset {dataset}")
            assert int(result.iteration_count()) == iterations
            for dataset in datasets:
                completed = result.complete_data(dataset=dataset)
                assert_observed_unchanged(original, completed)
                pd.testing.assert_frame_equal(
                    completed, result.complete_data(dataset=dataset, iteration=iterations)
                )
            result._ampute_original_data()
            return result
        finally:
            self._ampute_original_data()
            self._random_state = previous_rng


def completed_draws(result, variables):
    """Read final-round draws; preserve record order and expose round diagnostics."""
    final = int(result.iteration_count())
    assert final > 0
    return {
        variable: np.column_stack([
            result.complete_data(dataset=d)[variable].to_numpy() for d in result.datasets
        ]) for variable in variables
    }


def preflight_checks(threads=1):
    """Small signal/index/state checks before long experiments; no public data fit."""
    from sklearn.metrics import r2_score
    from threadpoolctl import threadpool_limits

    rng = np.random.default_rng(219)
    x = rng.uniform(-4, 4, 576)
    y = 3 * x
    train = pd.DataFrame({"x": x[:512], "target": y[:512]})
    test = pd.DataFrame({"x": x[512:], "target": np.nan})
    test.loc[:3, "target"] = y[512:516]
    with threadpool_limits(limits=threads):
        kernel = CorrectedKernel(train, num_datasets=2, variable_schema={"target": ["x"]},
                                 mean_match_candidates=3, random_state=219)
        features, label = kernel._make_features_label("target", 219)
        np.testing.assert_array_equal(features.index, np.arange(512))
        np.testing.assert_array_equal(label.index, np.arange(512))
        kernel.mice(3, num_iterations=80, num_threads=threads, num_leaves=15,
                    min_data_in_leaf=5, learning_rate=.15, verbosity=-1)
        result = kernel.impute_new_data(test, iterations=3, random_state=1234)
        hidden = test.target.isna()
        final = result.complete_data()
        pd.testing.assert_frame_equal(final, result.complete_data(iteration=3))
        assert_observed_unchanged(test, final)
        initial = result.complete_data(iteration=0).loc[hidden, "target"].to_numpy()
        predicted = final.loc[hidden, "target"].to_numpy()
        truth = y[512:][hidden]
        final_mae = float(np.mean(np.abs(predicted - truth)))
        initial_mae = float(np.mean(np.abs(initial - truth)))
        assert int(result.iteration_count()) == 3
        assert r2_score(truth, predicted) > .95
        assert final_mae < .2 * initial_mae
        shuffled = test.iloc[::-1].reset_index(drop=True)
        direct = kernel.get_model("target", 0).predict(test[["x"]], num_threads=threads)
        reordered = kernel.get_model("target", 0).predict(shuffled[["x"]], num_threads=threads)
        np.testing.assert_allclose(direct, reordered[::-1])
    return pd.DataFrame([
        {"check": "512 distinct training positions", "status": "passed", "value": len(features)},
        {"check": "final round and observed values", "status": "passed", "value": int(result.iteration_count())},
        {"check": "known signal final MAE", "status": "passed", "value": final_mae},
        {"check": "initialization MAE", "status": "reference", "value": initial_mae},
        {"check": "row correspondence", "status": "passed", "value": len(test)},
    ])
