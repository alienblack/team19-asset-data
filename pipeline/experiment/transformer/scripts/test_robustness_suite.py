"""Tests for reproducible, non-leaking robustness experiment interventions."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from robustness_suite import (
    GROUPS, RobustnessSuite, _seeded_swap, co_mask_source_on_anchor,
    count_preserving_grouped_pair, reference_split, scaled_test_noise,
    paired_block_skill_interval,
)

@pytest.fixture(scope='module')
def anchor():
    return reference_split(quick=True)

def test_reference_partition_does_not_overlap(anchor):
    tr,te,truth,split=anchor
    assert len(tr)==1000 and len(te)==1000
    assert not set(tr.index)&set(te.index)
    assert not truth.isna().any().any()

def test_t4_frozen_comasking_keeps_scored_anchor_cells(anchor):
    _,test,_,_=anchor
    x,log=co_mask_source_on_anchor(test,'altitude',('x','y'))
    assert x.altitude.equals(test.altitude)
    assert len(log)>0
    assert x.isna().to_numpy()[test.isna().to_numpy()].all()
    assert (x.isna().sum()>=test.isna().sum()).all()

def test_t4_count_control_does_not_change_anchor_masks():
    from robustness_suite import EXPERIMENT
    from transformer_noise_masking import read_pair
    observed,truth=read_pair(EXPERIMENT/'dataset',.8,.05)
    a,b,log=count_preserving_grouped_pair(observed,truth,seed=99)
    assert a.isna().sum().equals(b.isna().sum())
    assert len(log)>0
    for anchor,_ in GROUPS.values():
        assert a[anchor].equals(b[anchor])
    overlap=a.notna()&b.notna()
    for c in a:
        assert a.loc[overlap[c],c].equals(b.loc[overlap[c],c])

def test_t5_swaps_only_observed_values(anchor):
    _,test,_,_=anchor
    other,log=_seeded_swap(test,'model_id',.25,seed=17)
    assert len(log)>0
    assert other.isna().equals(test.isna())
    assert other.drop(columns='model_id').equals(test.drop(columns='model_id'))
    assert (other.loc[log.row_position,'model_id'].to_numpy()!=test.loc[log.row_position,'model_id'].to_numpy()).all()

def test_t5_zero_prevalence_is_identity(anchor):
    _,test,_,_=anchor
    x,log=_seeded_swap(test,'install_date',0,seed=19)
    assert x.equals(test) and log.empty

def test_t6_zero_noise_exactly_matches_supplied_original():
    x=reference_split(quick=True,noise=0)[1]
    y=reference_split(quick=True,noise=.05)[1]
    pd.testing.assert_frame_equal(scaled_test_noise(x,y,0),x)
    pd.testing.assert_frame_equal(scaled_test_noise(x,y,.05),y,check_exact=False,rtol=0,atol=1e-10)

def test_t6_rejects_mask_changes(anchor):
    _,test,_,_=anchor
    changed=test.copy()
    changed.loc[changed.index[0],'x']=np.nan
    with pytest.raises(ValueError,match='Mask mismatch'):
        scaled_test_noise(test,changed,.20)

def test_paired_bootstrap_detects_same_scoring_cells():
    df=[]
    for method,errors in [('a',[1.,1.5,2.,1.5]),('b',[2.,2.,3.,3.])]:
        for i,e in enumerate(errors):
            df.append(dict(method=method,variable='life',row_position=i,x_tile=i//2,
                           y_tile=0,absolute_error=e,baseline_absolute_error=4.))
    result=paired_block_skill_interval(pd.DataFrame(df),'a','b','life',n_boot=30)
    assert result['delta_skill_a_minus_b']>0
    assert result['n_blocks']==2

def test_suite_pilot_saves_cell_level_results(tmp_path):
    suite=RobustnessSuite(output_dir=tmp_path/'pilot',methods=('median_mode',),quick=True,save_models=False)
    m=suite.t4(groups={'gas':GROUPS['gas']})
    assert len(m)==1 and m.variable.iloc[0]=='measured_gas_a'
    t5=suite.t5(predictors=('model_id',),prevalences=(0,.1))
    assert len(t5)>=15
    t6=suite.t6(train_noises=(.05,),apply_noises=(0.,.05))
    assert len(t6)==30
    all_m,all_c=suite.consolidate()
    assert not all_c.empty and (tmp_path/'pilot'/'run_manifest.json').exists()
    assert (tmp_path/'pilot'/'all_masked_cells.csv').exists()


def test_condition_intervention_bootstrap():
    from robustness_suite import paired_condition_delta_skill
    rows=[]
    for condition,offset in [('reference',0.),('intervention',1.)]:
        for i in range(6):
            rows.append({'method':'lightgbm_chained','variable':'life',
                         'condition':condition,'row_position':i,'x_tile':i//2,'y_tile':0,
                         'absolute_error':1.+offset, 'baseline_absolute_error':4.,'truth':10.})
    ci=paired_condition_delta_skill(pd.DataFrame(rows),method='lightgbm_chained',
         variable='life',condition_reference='reference',condition_intervention='intervention',n_boot=100)
    assert abs(ci['skill_lost']-.25)<1e-9 and ci['n_blocks']==3
