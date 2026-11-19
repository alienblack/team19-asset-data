"""
Independently check saved benchmark predictions, confidence decisions and screening scores
against their source rows.
"""
import argparse
from functools import cached_property
from pathlib import Path
import json
import numpy as np
import pandas as pd
from subteam_a.data import (
    ANCHOR_STRESS, CORE, FAMILY_NOISE, GROUP_REMASK, MNAR, OPTIONAL, OUTLIER_ANCHOR, OVERHEAD_GRID,
    SINGLE_TARGETS, STRESS, split_blocks,
)
from subteam_a.screening import METHODS as SCREEN_METHODS, TARGETS as SCREEN_TARGETS
from subteam_a.study import load


class RunTables:
    """Read the archived CSVs or equivalent records from task JSON in a fresh run."""

    def __init__(self, out):
        self.out = out

    @cached_property
    def _tasks(self):
        return [json.loads(p.read_text()) for p in sorted((self.out/'tasks').glob('*.json'))]

    def get(self, name, task_key=None):
        """Return a named CSV table or its original task payload from a run without a report."""
        path = self.out/'tables'/f'{name}.csv'
        if path.is_file():
            return pd.read_csv(path)
        return pd.DataFrame(row for task in self._tasks for row in task[task_key or name])

    def trials(self):
        """Return the saved tuning records from either the CSV or the tuning JSON files."""
        path = self.out/'tables'/'validation_trials.csv'
        if path.is_file():
            return pd.read_csv(path)
        return pd.DataFrame(
            row for p in sorted((self.out/'tuning').glob('*.json')) for row in json.loads(p.read_text())['trials']
        )


def verify_screening(data, out, contract):
    """
    Rebuild the test candidate lists and confirm recorded values, truth labels and flag
    cutoffs. Return the verified row count.
    """
    rows = 0
    for gs in contract['generator_seeds']:
        schema, truth, raw, _, groups = load(str(data.resolve()), OUTLIER_ANCHOR, gs)
        for ss in contract['split_seeds']:
            ids = split_blocks(truth, ss, schema, groups)[0]['test']
            d = pd.read_csv(out/'screening'/f'{gs}-{ss}.csv.gz', float_precision='round_trip')
            assert d.generator_seed.eq(gs).all() and d.split_seed.eq(ss).all()
            assert set(d.method) == set(SCREEN_METHODS) and set(d.attribute) == set(SCREEN_TARGETS)
            assert not d.duplicated(['asset_id', 'attribute', 'method']).any()
            for method in SCREEN_METHODS:
                for field in SCREEN_TARGETS:
                    j = schema.i(field)
                    wanted = ids[np.isfinite(raw[ids, j])]
                    batch = d[d.method.eq(method) & d.attribute.eq(field)]
                    np.testing.assert_array_equal(batch.asset_id, wanted)
                    np.testing.assert_array_equal(batch.recorded, raw[wanted, j])
                    np.testing.assert_array_equal(batch.truth, truth[wanted, j])
                    np.testing.assert_array_equal(batch.injected_error, raw[wanted, j] != truth[wanted, j])
                    assert batch.p_value.between(0, 1, inclusive='right').all()
                    assert batch.calibration_n.ge(contract['minimum_pattern_support']).all()
                    np.testing.assert_array_equal(batch.p_value <= contract['primary_flag_level'],
                                                  batch.score > batch.cutoff)
            rows += len(d)
    return rows


def verify(data, out):
    """
    Check a run against the input data and write the resulting verification counts to the run
    directory.
    """
    scenarios={s.name:s for s in (*CORE,MNAR,*STRESS,*OPTIONAL,*OVERHEAD_GRID,*ANCHOR_STRESS,
                                 *SINGLE_TARGETS,*FAMILY_NOISE,GROUP_REMASK)}
    contract=json.loads((out/'contract.json').read_text())
    status=json.loads((out/'run_status.json').read_text())
    tables = RunTables(out)
    score = tables.get('cells')
    groups=score[score.domain_features & (score.group=='all') & (score.split_seed==617)]
    life_tolerances = contract.get('life_tolerance_units')
    numeric_tolerances = contract.get('numeric_tolerances')
    snapshot_year = contract.get('snapshot_year')
    if snapshot_year is not None:
        minimum_error = contract['categorical_skill_min_floor_error']
        categorical = score[score.type == 'categorical']
        assert categorical.loc[categorical.floor_error < minimum_error, 'skill'].isna().all()
        defined = categorical[categorical.floor_error >= minimum_error]
        np.testing.assert_allclose(defined.skill, 1 - defined.error / defined.floor_error, atol=1e-12)
        if score.mechanism.eq('Overhead').any():
            age_drift = tables.get('overhead_age_drift', 'tables')
            if 'mechanism' in age_drift:
                age_drift = age_drift[age_drift.mechanism.eq('Overhead')]
            age_drift = age_drift[age_drift.split_seed == 617].set_index(['scenario', 'arm', 'method'])
    if life_tolerances and score.scenario.isin({'mcar30-n05', 'mnar30-n05'}).any():
        age_scores = tables.get('installation_age')
        age_scores = age_scores[age_scores.domain_features & (age_scores.generator_seed == 219) &
                                (age_scores.split_seed == 617)]
    n_files=n_cells=0
    overhead_year_exports = 0
    for p in sorted((out/'examples').iterdir()):
        sc=scenarios[p.name]
        gs=0 if sc.mechanism=='Overhead' else 219
        schema,truth,raw,_,_=load(str(data.resolve()),sc,gs)
        parts=pd.read_csv(p/'partitions.csv')
        test_ids=parts.loc[parts.partition=='test','asset_id'].to_numpy()
        measured=pd.read_csv(p/'measured_test.csv.gz',float_precision='round_trip')
        np.testing.assert_array_equal(measured.asset_id,test_ids)
        assert parts.asset_id.is_unique
        assert parts.groupby('split_block').partition.nunique().max()==1
        train_ids = parts.loc[parts.partition == 'train', 'asset_id'].to_numpy()
        np.testing.assert_allclose(measured[list(schema.names)].isna().to_numpy(),np.isnan(raw[test_ids]))
        for filled in sorted(p.glob('*-filled.csv.gz')):
            arm,method,_=filled.name.split('-')
            f=pd.read_csv(filled,float_precision='round_trip')
            np.testing.assert_array_equal(f.asset_id,test_ids)
            z=f[list(schema.names)].copy()
            for j,labels in schema.categories.items():
                key=schema.names[j]
                assert set(z[key])<=set(labels)
                z[key]=z[key].map({v:i for i,v in enumerate(labels)})
            z=z.to_numpy(float)
            vis=np.isfinite(raw[test_ids])
            np.testing.assert_array_equal(z[vis],raw[test_ids][vis])
            assert np.isfinite(z).all()
            if snapshot_year is not None:
                if schema.transformer:
                    bounds = {'leakage': (.8, 1.2), 'life': (-np.inf, 100_000), 'total_gas': (0, np.inf)}
                    for attribute, (lower, upper) in bounds.items():
                        column = schema.i(attribute)
                        fills = z[~vis[:, column], column]
                        assert ((fills >= lower) & (fills <= upper)).all(), (sc.name, method, attribute)
                else:
                    years = f.installation_year.to_numpy()
                    lower = np.nanmin(raw[train_ids, schema.i('installation_year')])
                    missing_year = ~vis[:, schema.i('installation_year')]
                    assert ((years[missing_year] >= lower) & (years[missing_year] <= snapshot_year)).all()
                    np.testing.assert_array_equal(years, np.rint(years))
                    ages = f[f'asset_age_at_{snapshot_year}'].to_numpy()
                    np.testing.assert_array_equal(ages, snapshot_year - years)
                    assert (ages >= 0).all()
                    row = age_drift.loc[(sc.name, arm, method)]
                    reference_years = truth[test_ids, schema.i('installation_year')]
                    np.testing.assert_allclose(row.aged_50_truth, np.mean(reference_years <= snapshot_year - 50))
                    np.testing.assert_allclose(row.aged_50_completed, np.mean(ages >= 50))
                    overhead_year_exports += 1
            ledger=pd.read_csv(p/f'{arm}-{method}-cells.csv.gz',float_precision='round_trip')
            assert len(ledger)==int((~vis).sum()) and not ledger.duplicated(['asset_id','attribute']).any()
            assert set(ledger.decision)<= {'auto','review','abstain'}
            for j,key in enumerate(schema.names):
                g=ledger[ledger.attribute==key]
                hidden=np.isnan(raw[test_ids,j])
                np.testing.assert_array_equal(g.asset_id.to_numpy(),test_ids[hidden])
                np.testing.assert_array_equal(g.truth.to_numpy(),truth[test_ids[hidden],j])
                np.testing.assert_array_equal(g.fill.to_numpy(),z[hidden,j])
                if not hidden.any():
                    continue
                if j in schema.categories:
                    pcols=[f'p_code_{k}' for k in range(len(schema.categories[j]))]
                    np.testing.assert_allclose(g[pcols].sum(1),1,rtol=2e-6)
                    err=(g.truth!=g.fill).mean()
                else:
                    assert (g.lower<=g.fill).all() and (g.fill<=g.upper).all()
                    r=g[g.rule]
                    np.testing.assert_allclose(r['fill'],r.truth,rtol=1e-10,atol=1e-10)
                    err=np.abs(g.truth-g.fill).mean()
                r=groups[(groups.scenario==sc.name)&(groups.generator_seed==gs)&(groups.arm==arm)&
                         (groups.method==method)&(groups.attribute==key)]
                assert len(r)==1 and int(r.n.iloc[0])==len(g), (sc.name,arm,method,key)
                np.testing.assert_allclose(err,r.error.iloc[0],atol=1e-10,rtol=1e-10)
                assert int(r.auto_n.iloc[0])==int((g.decision=='auto').sum())
                if numeric_tolerances and j not in schema.categories:
                    spec = numeric_tolerances[key]
                    error = (g.truth - g.fill).abs()
                    half_width = (g.upper - g.lower) / 2
                    smallest_truth = (g['fill'].abs() - half_width).clip(lower=0)
                    limits = {}
                    for level in ('tight', 'loose'):
                        value = spec[level]
                        if spec['relative']:
                            observed_train = raw[train_ids, j]
                            range_floor = spec['range_floor'] * (np.nanmax(observed_train) - np.nanmin(observed_train))
                            limits[level] = (
                                np.maximum(g.truth.abs() * value, range_floor),
                                np.maximum(smallest_truth * value, range_floor),
                            )
                        else:
                            limits[level] = (value, value)
                    tight_ok = error <= limits['tight'][0] + 1e-10
                    loose_ok = error <= limits['loose'][0] + 1e-10
                    tight_eligible = (half_width <= limits['tight'][1]) | g.rule
                    loose_eligible = half_width <= limits['loose'][1]
                    decision = np.where(tight_eligible, 'auto', np.where(loose_eligible, 'review', 'abstain'))
                    np.testing.assert_array_equal(g.decision, decision)
                    np.testing.assert_allclose(r.tolerance_tight.iloc[0], tight_ok.mean())
                    np.testing.assert_allclose(r.tolerance.iloc[0], loose_ok.mean())
                    assert int(r.auto_correct.iloc[0]) == (tight_eligible & tight_ok).sum()
                    assert int(r.loose_eligible_n.iloc[0]) == loose_eligible.sum()
                    assert int(r.loose_correct_n.iloc[0]) == (loose_eligible & loose_ok).sum()
                elif life_tolerances and key == 'life':
                    error = (g.truth - g.fill).abs()
                    half_width = (g.upper - g.lower) / 2
                    tight, loose = life_tolerances['tight'], life_tolerances['loose']
                    tight_ok, loose_ok = error <= tight + 1e-10, error <= loose + 1e-10
                    tight_eligible, loose_eligible = half_width <= tight, half_width <= loose
                    decision = np.where(tight_eligible, 'auto', np.where(loose_eligible, 'review', 'abstain'))
                    np.testing.assert_array_equal(g.decision, decision)
                    np.testing.assert_allclose(r.tolerance.iloc[0], tight_ok.mean())
                    np.testing.assert_allclose(r.tolerance_loose.iloc[0], loose_ok.mean())
                    assert int(r.auto_correct.iloc[0]) == (tight_eligible & tight_ok).sum()
                    assert int(r.loose_eligible_n.iloc[0]) == loose_eligible.sum()
                    assert int(r.loose_correct_n.iloc[0]) == (loose_eligible & loose_ok).sum()
                if life_tolerances and key == 'install_date' and sc.name in {'mcar30-n05', 'mnar30-n05'}:
                    date_truth = truth[test_ids, j]
                    age_edges = np.quantile(truth[:, j], [.25, .50, .75])
                    reference = np.digitize(date_truth, age_edges)
                    filled_band = np.digitize(z[:, j], age_edges)
                    by_band = age_scores[(age_scores.scenario == sc.name) & (age_scores.arm == arm) &
                                         (age_scores.method == method)].set_index('age_band')
                    bands = ('oldest', 'older_middle', 'younger_middle', 'youngest')
                    for band, band_name in enumerate(bands):
                        band_hidden = hidden & (reference == band)
                        row = by_band.loc[band_name]
                        assert row.hidden_n == band_hidden.sum()
                        if band_hidden.any():
                            np.testing.assert_allclose(row.signed_bias_days, np.mean(
                                z[band_hidden, j] - date_truth[band_hidden]))
                        np.testing.assert_allclose(row.age_share_error_pp, 100 * (
                            np.mean(filled_band == band) - np.mean(reference == band)), atol=1e-12)
            n_files+=1
            n_cells+=len(ledger)
    if life_tolerances and 'rules' in contract.get('arms', ('model', 'rules')):
        effects = tables.get('rule_effects')
        keys = ['scenario', 'generator_seed', 'split_seed', 'domain_features', 'method', 'attribute']
        point = score[score.group == 'all'].pivot(index=keys, columns='arm', values='skill')
        paired = effects.join(point, on=keys, validate='one_to_one')
        np.testing.assert_allclose(
            paired.direct_skill_delta + paired.remaining_skill_delta,
            paired.rules - paired.model, rtol=1e-10, atol=1e-12,
        )
    count=len(list((out/'tasks').glob('*.json')))
    assert count==status['tasks']
    if contract['profile'] == 'anchor-diagnostics':
        assert n_files == len(contract['scenarios'])*len(contract['methods'])
    tuning=tables.trials() if contract['trials_per_trainable_method'] else None
    if tuning is not None:
        assert (tuning.groupby(['generator_seed','split_seed','method']).size()==contract['trials_per_trainable_method']).all()
    result=dict(tasks=count,completed_exports_verified=n_files,hidden_cells_verified=n_cells,
                tuning_trials=0 if tuning is None else len(tuning))
    if contract['profile'] == 'failure-audit':
        assert n_files == 2
        assert len(list((out/'screening').glob('*.csv.gz'))) == status['screening_tasks']
        result['screening_rows_verified'] = verify_screening(data, out, contract)
    if snapshot_year is not None:
        result['overhead_year_exports_verified'] = overhead_year_exports
    (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result)


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--data',default=Path('data'),type=Path)
    cli.add_argument('--out',default=Path('outputs/final'),type=Path)
    args=cli.parse_args()
    verify(args.data,args.out)
