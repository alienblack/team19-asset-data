"""
Check source reproduction, data isolation, model contracts, calibration and field decision
rules.
"""
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest

from subteam_a.data import (
    ANCHOR, CLEAN_ANCHOR, CORE, DATE_ERROR_DAYS, DATE_ERROR_RATE, FAMILY_NOISE, FIELD_GROUPS, GROUP_ANCHOR,
    MODEL_ERROR_RATE, MNAR, NOISE_FAMILIES, OUTLIER_ANCHOR, OVERHEAD_SCHEMA, SINGLE_TARGETS, SNAPSHOT_YEAR,
    TRANSFORMER as S, share_group_draws,
    transformer_data, split_blocks, verify_transformer,
)
from subteam_a.features import Features, exact_fill
from subteam_a.evaluate import (
    Confidence, evaluate, installation_age_scores, qconformal, calibrate, relative_skill, table_checks,
    NUMERIC_TOLERANCES, tolerance_limit,
)
from subteam_a.mixture import MaskedMixture
from subteam_a.models import remask_training_input
from subteam_a.screening import conformity_cutoffs, screening_view
from subteam_a.study import create, DEFAULT, METHODS

ROOT = Path(__file__).resolve().parents[1]/'data'/'source'


@pytest.fixture(scope='module')
def source():
    """Load the seed-219 transformer fixture for tests using the supplied source pairs."""
    return transformer_data(ROOT,CORE[-2],219)


def test_subteam_c_pairs_reproduced_exactly():
    """Check that subteam c pairs reproduced exactly."""
    audit=verify_transformer(ROOT)
    assert len(audit)==9 and audit.truth_sha256.nunique()==1
    assert audit.measured_sha256.nunique()==9


@pytest.mark.parametrize('seed',[617,631,653])
def test_split_is_spatial_disjoint_and_covers_each_x_band(source,seed):
    """Check that split is spatial disjoint and covers each x band."""
    truth,_=source
    part,block=split_blocks(truth,seed,S)
    idx=list(part.values())
    assert sorted(np.concatenate(idx))==list(range(10000))
    for ids in idx:
        assert set(truth[ids,0]//10)==set(range(10))
        assert np.any(np.diff(ids)<0)
    for i,a in enumerate(idx):
        for b in idx[i+1:]:
            assert not set(block[a])&set(block[b])


def test_exact_rules_respect_measurement_noise_and_noninvertible_zero(source):
    """Check that exact rules respect measurement noise and noninvertible zero."""
    truth,_=source
    raw=truth[10:15].copy()
    j=S.i
    raw[0,j('x')]=np.nan
    raw[1,j('total_gas')]=np.nan
    raw[2,j('measured_overgrowth')]=np.nan
    raw[3,j('measured_overgrowth')]=0
    raw[3,j('measured_humid')]=np.nan
    raw[4,j('measured_gas_b')]=np.nan
    for noise in (0,.05):
        point,rule=exact_fill(raw,S,noise)
        assert np.array_equal(point[np.isfinite(raw)],raw[np.isfinite(raw)])
        assert rule[0,j('x')] and point[0,j('x')]==truth[10,j('x')]
        assert not rule[3,j('measured_humid')]
        assert rule[1,j('total_gas')]==(noise==0)
        assert rule[2,j('measured_overgrowth')]==(noise==0)
        assert rule[4,j('measured_gas_b')]==(noise==0)


def test_mnar_really_prefers_older_installations_and_recalibration_is_independent():
    """Check that mnar really prefers older installations and recalibration is independent."""
    truth,measured=transformer_data(ROOT,MNAR,220)
    _,cal=transformer_data(ROOT,MNAR,220,draw=1)
    j=S.i('install_date')
    older=truth[:,j]<np.median(truth[:,j])
    hide=np.isnan(measured[:,j])
    assert hide[older].mean()-hide[~older].mean()>.40
    assert not np.array_equal(np.isnan(cal),np.isnan(measured))
    both=np.isfinite(cal)&np.isfinite(measured)
    assert np.array_equal(cal[both],measured[both])


@pytest.mark.parametrize('draw', [0, 1])
def test_group_masks_cohide_fields_reuse_draws_and_stay_nested(draw):
    """Check that group masks cohide fields reuse draws and stay nested."""
    anchor = next(s for s in CORE if s.name == ANCHOR)
    _, independent = transformer_data(ROOT, anchor, 219, draw)
    _, group = transformer_data(ROOT, GROUP_ANCHOR, 219, draw)
    _, lighter = transformer_data(ROOT, replace(GROUP_ANCHOR, missing=.20), 219, draw)
    hidden = np.isnan(group)
    assert hidden.mean() == pytest.approx(.30, abs=.012)
    assert (hidden | ~np.isnan(lighter)).all()
    grouped = {field for members in FIELD_GROUPS.values() for field in members}
    for members in FIELD_GROUPS.values():
        first_mask = np.isnan(independent[:, S.i(members[0])])
        for member in members:
            np.testing.assert_array_equal(hidden[:, S.i(member)], first_mask)
    for name in set(S.names) - grouped:
        np.testing.assert_array_equal(group[:, S.i(name)], independent[:, S.i(name)])
    _, filled = exact_fill(group, S, GROUP_ANCHOR.noise)
    assert not filled.any()


def test_mlp_grouped_remasking_hides_location_partners_before_rules(source):
    """Check that mlp grouped remasking hides location partners before rules."""
    truth, _ = source
    raw = truth[:2].copy()
    raw[1, S.i('y')] = np.nan
    draws = np.ones(raw.shape)
    draws[:, S.i('x')] = 0.
    masked = np.isfinite(raw) & (share_group_draws(draws, S, FIELD_GROUPS.values()) < .3)
    x, targets = remask_training_input(raw, masked, S, .05, True)
    columns = [S.i(c) for c in FIELD_GROUPS['location']]
    assert np.isnan(x[:, columns]).all()
    np.testing.assert_array_equal(targets, masked)
    assert not targets[1, S.i('y')]
    assert draws[:, S.i('y')].min() == 1.


def test_screening_holds_out_candidate_and_uses_clean_calibration_tail():
    """Check that screening holds out candidate and uses clean calibration tail."""
    raw = np.array([[2., 3.], [4., np.nan], [8., 7.]])
    ids, view, recorded = screening_view(raw, np.arange(3), 1)
    np.testing.assert_array_equal(ids, [0, 2])
    np.testing.assert_array_equal(recorded, [3., 7.])
    assert np.isnan(view[:, 1]).all() and raw[0, 1] == 3.
    cal = np.arange(100.)
    p, cutoff, support, source = conformity_cutoffs(
        cal, np.repeat([0, 1], 50), np.array([100., 99., 99.]), np.array([1, 1, 2]), alpha=.05,
    )
    np.testing.assert_array_equal(p <= .05, [True, True, True])
    np.testing.assert_array_equal(support, [50, 50, 100])
    assert list(source) == ['key+burden', 'key+burden', 'pooled']
    np.testing.assert_array_equal(p <= .05, np.array([100., 99., 99.]) > cutoff)
    p, *_ = conformity_cutoffs(np.ones(50), np.zeros(50), np.ones(1), np.zeros(1))
    assert p.item() == 1.


@pytest.mark.parametrize('draw', [0, 1])
def test_single_target_reuses_anchor_target_mask_and_retains_visible_values(draw):
    """Check that single target reuses anchor target mask and retains visible values."""
    anchor = next(s for s in CORE if s.name == ANCHOR)
    _, scattered = transformer_data(ROOT, anchor, 219, draw)
    observed = np.isfinite(scattered)
    for scenario in SINGLE_TARGETS:
        _, single = transformer_data(ROOT, scenario, 219, draw)
        target = S.i(scenario.target)
        np.testing.assert_array_equal(single[:, target], scattered[:, target])
        np.testing.assert_array_equal(single[observed], scattered[observed])
        assert np.isnan(single).sum() == np.isnan(scattered[:, target]).sum()


@pytest.mark.parametrize('draw', [0, 1])
def test_family_noise_is_isolated_and_uses_the_same_mask_as_clean(draw):
    """Check that family noise is isolated and uses the same mask as clean."""
    clean = next(s for s in CORE if s.name == CLEAN_ANCHOR)
    truth, scattered = transformer_data(ROOT, clean, 219, draw)
    observed = np.isfinite(scattered)
    for scenario in FAMILY_NOISE:
        reference, noisy = transformer_data(ROOT, scenario, 219, draw)
        np.testing.assert_array_equal(reference, truth)
        np.testing.assert_array_equal(np.isfinite(noisy), observed)
        affected = set(NOISE_FAMILIES[scenario.noise_family])
        for j, field in enumerate(S.names):
            y = scattered[observed[:, j], j]
            error = np.abs(noisy[observed[:, j], j] - y)
            if field in affected:
                assert (error > 0).any()
                assert (error <= scenario.noise*np.abs(y) + 1e-10).all()
            else:
                assert (error == 0).all()


@pytest.mark.parametrize('draw', [0, 1])
def test_structural_outliers_keep_mcar_masks_jitter_and_valid_codes(draw):
    """Check that structural outliers keep mcar masks jitter and valid codes."""
    anchor = next(s for s in CORE if s.name == ANCHOR)
    truth, clean = transformer_data(ROOT, anchor, 219, draw)
    reference, noisy = transformer_data(ROOT, OUTLIER_ANCHOR, 219, draw)
    np.testing.assert_array_equal(truth, reference)
    np.testing.assert_array_equal(np.isnan(clean), np.isnan(noisy))
    date, model = S.i('install_date'), S.i('model_id')
    unchanged = [j for j in range(len(S.names)) if j not in (date, model)]
    np.testing.assert_array_equal(clean[:, unchanged], noisy[:, unchanged])
    for j, target in ((date, DATE_ERROR_RATE), (model, MODEL_ERROR_RATE)):
        seen = np.isfinite(noisy[:, j])
        assert (noisy[seen, j] != truth[seen, j]).mean() == pytest.approx(target, abs=.008)
    days = noisy[np.isfinite(noisy[:, date]), date]
    shifts = days - truth[np.isfinite(noisy[:, date]), date]
    assert set(shifts) == {-DATE_ERROR_DAYS, 0, DATE_ERROR_DAYS}
    assert (days >= 0).all()
    assert set(noisy[np.isfinite(noisy[:, model]), model]) == set(S.categories[model])


def test_mixture_marginal_and_conditional_moments():
    """Check that mixture marginal and conditional moments."""
    z=np.array([[1.,np.nan],[np.nan,np.nan],[1.,-1.]])
    mu=np.zeros((1,2))
    cov=np.array([[1.,.6],[.6,2.]])
    r,fill,var,_=MaskedMixture.expect(MaskedMixture.preparation(z),mu,cov,np.ones(1))
    np.testing.assert_allclose(fill[:,0,:],[[1,.6],[0,0],[1,-1]])
    np.testing.assert_allclose(r,np.ones((3,1)))
    _,_,row=MaskedMixture.preparation(z)
    assert var[row[0],1,1]==pytest.approx(2-.36)
    assert var[row[1],1,1]==pytest.approx(2)


def test_train_spatial_lookup_excludes_same_row_and_gas_features_declare_inputs(source):
    """Check that train spatial lookup excludes same row and gas features declare inputs."""
    truth,_=source
    f=Features(S,True).fit(truth[:300])
    row=truth[15:16].copy()
    v,deps=f.extra(row)
    row[:,S.i('install_date')]+=1000000
    v2,_=f.extra(row)
    assert np.array_equal(v[0],v2[0])
    for k,c in enumerate('abc',1):
        assert deps[k]=={S.i(n) for n in ('install_date','leakage','measured_gas_'+c)}
    assert len(Features(S,False).fit(truth[:300]).extra(row)[0])==1


@pytest.mark.parametrize('method',METHODS)
def test_each_model_preserves_observations_and_emits_valid_category_probabilities(source,method):
    """Check that each model preserves observations and emits valid category probabilities."""
    truth,x=source
    idx,_=split_blocks(truth,617,S)
    a,b=(x[idx[k][:300]].copy() for k in ('train','test'))
    b[0]=np.nan
    f=Features(S,True).fit(a)
    p=DEFAULT[method].copy()
    if method=='autoencoder':
        p['epochs']=3
    elif method=='forest':
        p['trees']=4
    elif method=='mixture':
        p['steps']=4
    result=create(method,p,17).fit(a,f).predict(b)
    keep=np.isfinite(b)
    assert np.array_equal(result.point[keep],b[keep])
    assert np.isfinite(result.point).all() and (result.scale>0).all()
    probs=result.proba[S.i('model_id')]
    assert probs.shape==(len(b),4)
    np.testing.assert_allclose(probs.sum(1),1,rtol=1e-6)
    assert set(np.unique(result.point[:,S.i('model_id')])) <= {0,1,2,3}


def test_conformal_quantile_and_deterministic_rule_intervals(source):
    """Check that conformal quantile and deterministic rule intervals."""
    assert qconformal(np.arange(19.))==17
    assert np.isinf(qconformal(np.arange(5.)))
    truth,raw=source
    data=raw[:300].copy()
    x,rule=exact_fill(data,S,.05,True)
    f=Features(S,True).fit(raw[500:1000])
    est=create('floor',{},617).fit(raw[500:1000],f)
    pred=est.predict(x)
    conf=calibrate(f,raw[2000:2500],truth[2000:2500],np.zeros((500,15),bool),est.predict(raw[2000:2500]),data,rule,pred)
    assert rule.sum()>0
    assert np.array_equal(conf.lower[rule],conf.upper[rule])
    assert (conf.source[rule]=='rule').all()
    mask=np.isnan(data)&~rule
    assert (conf.support[mask]>0).all()


def test_life_uses_absolute_tolerances_for_accuracy_and_decisions(source):
    """Check that life uses absolute tolerances for accuracy and decisions."""
    truth = source[0][:5].copy()
    life = S.i('life')
    truth[:, life] = 100_000
    raw = truth.copy()
    raw[:, life] = np.nan
    point = truth.copy()
    point[:, life] += [100, 101, 499, 501, 600]
    half_width = np.zeros_like(point)
    half_width[:, life] = [100, 99, 350, 500, 501]
    confidence = Confidence(
        point - half_width, point + half_width, half_width, half_width, {},
        np.full(point.shape, 'pooled'), np.zeros_like(point), {},
    )
    features = SimpleNamespace(schema=S, range=np.full(point.shape[1], 2_000), scale=np.ones(point.shape[1]))
    cells, _, _, decisions = evaluate(
        raw, truth, SimpleNamespace(point=point), SimpleNamespace(point=point), confidence,
        np.zeros(point.shape, dtype=bool), features,
    )
    overall = cells[0]
    assert overall['attribute'] == 'life' and overall['group'] == 'all'
    assert overall['tolerance_tight'] == pytest.approx(1/5)
    assert overall['tolerance'] == pytest.approx(3/5)
    assert overall['auto_n'] == 2 and overall['auto_correct'] == 1
    assert overall['loose_eligible_n'] == 4 and overall['loose_correct_n'] == 3
    assert list(decisions[:, life]) == ['auto', 'auto', 'review', 'review', 'abstain']


def test_installation_bias_conditions_on_true_age_and_counts_full_register_shares():
    """Check that installation bias conditions on true age and counts full register shares."""
    truth = np.array([10., 20., 40., 60., 90.])
    raw = np.array([np.nan, 20., np.nan, 60., 90.])
    fills = np.array([65., 20., 30., 60., 90.])
    rows = {row['age_band']: row for row in installation_age_scores(
        raw, truth, fills, np.full_like(truth, 55.), [25., 50., 75.],
    )}
    assert rows['oldest']['hidden_n'] == 1
    assert rows['oldest']['signed_bias_days'] == pytest.approx(55.)
    assert rows['older_middle']['signed_bias_days'] == pytest.approx(-10.)
    assert rows['older_half']['age_share_error_pp'] == pytest.approx(-20.)
    assert rows['all']['mae'] == pytest.approx(32.5)
    assert rows['all']['floor_mae'] == pytest.approx(30.)


def test_overhead_year_bounds_rounding_and_age_cutoff():
    """Check that overhead year bounds rounding and age cutoff."""
    schema = OVERHEAD_SCHEMA
    col = schema.i('installation_year')
    cutoff = SNAPSHOT_YEAR - 50
    oldest = SNAPSHOT_YEAR - 84
    train = np.zeros((2, len(schema.names)))
    train[:, col] = [oldest, SNAPSHOT_YEAR - 1]
    features = Features(schema, domain=False).fit(train)
    raw = np.zeros((5, len(schema.names)))
    raw[:, col] = [np.nan, np.nan, np.nan, np.nan, cutoff]
    proposals = raw.copy()
    proposals[:, col] = [SNAPSHOT_YEAR + 2, oldest - 3, cutoff + .4, cutoff + .6, SNAPSHOT_YEAR]
    result = features.finish(raw, proposals, np.ones_like(raw), {}).point
    np.testing.assert_array_equal(result[:, col], [SNAPSHOT_YEAR, oldest, cutoff, cutoff + 1, cutoff])

    # Population checks also handle a fractional proposal at the inclusive cutoff.
    fractional = result.copy()
    fractional[2:4, col] = [cutoff + .4, cutoff + .6]
    stats = table_checks(features, raw, result, fractional, 617)
    assert stats['aged_50_truth'] == pytest.approx(.6)
    assert stats['aged_50_completed'] == pytest.approx(.6)


def test_near_perfect_categorical_floor_suppresses_only_categorical_skill():
    """Check that near perfect categorical floor suppresses only categorical skill."""
    assert np.isnan(relative_skill(.09, .015, categorical=True))
    assert np.isnan(relative_skill(0, 0, categorical=True))
    assert relative_skill(.03, .05, categorical=True) == pytest.approx(.4)
    assert relative_skill(.009, .015, categorical=False) == pytest.approx(.4)


def test_transformer_domain_limits_leave_observed_values_intact(source):
    """Check that transformer domain limits leave observed values intact."""
    train = source[0][:40]
    features = Features(S, domain=False).fit(train)
    raw = train[:3].copy()
    columns = [S.i(name) for name in ('leakage', 'life', 'total_gas')]
    raw[:2, columns] = np.nan
    proposals = train[:3].copy()
    proposals[:, columns] = [[.7, 100_001, -2], [1.3, 100_002, -5], [1.5, 100_003, -6]]
    result = features.finish(raw, proposals, np.ones_like(raw), {}).point
    np.testing.assert_array_equal(result[:2, columns], [[.8, 100_000, 0], [1.2, 100_000, 0]])
    np.testing.assert_array_equal(result[2], raw[2])


def test_attribute_tolerances_follow_dataset_table():
    """Check that attribute tolerances follow dataset table."""
    expected = {
        'x': (1, 5), 'y': (1, 5), 'install_date': (7, 30), 'altitude': (1, 5),
        'life': (100, 500), 'installation_year': (5, 10),
        'measured_humid': (10, 15), 'measured_salt': (10, 15),
        'measured_overgrowth': (10, 15), 'random_placebo': (10, 15),
        'measured_gas_a': (5, 10), 'measured_gas_b': (5, 10), 'measured_gas_c': (5, 10),
        'leakage': (5, 10), 'total_gas': (1, 5), 'distance_to_sea_km': (8, 8), 'length_km': (8, 8),
    }
    assert set(NUMERIC_TOLERANCES) == set(expected) == {
        schema.names[index] for schema in (S, OVERHEAD_SCHEMA) for index in schema.numeric
    }
    for attribute, (tight, loose) in expected.items():
        assert tolerance_limit(attribute, -100., 400., tight=True) == pytest.approx(tight)
        assert tolerance_limit(attribute, -100., 400., tight=False) == pytest.approx(loose)
    assert tolerance_limit('installation_year', 1960, 84, tight=True) == 5
    assert tolerance_limit('measured_overgrowth', 0., 2., tight=False) == 0.


def test_rules_autoencoder_masks_before_building_derived_inputs(source):
    """Check that rules autoencoder masks before building derived inputs."""
    original = source[0][100:102].copy()
    x, y, altitude = (S.i(attribute) for attribute in ('x', 'y', 'altitude'))
    original[0, altitude] = np.nan
    hide = np.zeros_like(original, dtype=bool)
    hide[:, x] = True
    prepared, loss_mask = remask_training_input(original, hide, S, .05, True)

    # In the first row, altitude had never been measured. Hiding x must leave
    # both x and altitude unavailable, even though y is visible.
    assert np.isnan(prepared[0, x]) and np.isnan(prepared[0, altitude])
    assert prepared[0, y] == original[0, y] and loss_mask[0, x]
    # In the second row altitude is a genuine input. The exact rule fills x,
    # so the MLP does not train on a target that the rule already solved.
    assert prepared[1, x] == original[1, x] and not loss_mask[1, x]
