"""Checks of reporting provenance, aggregation and the public execution options."""
import base64
import json
from pathlib import Path

import pandas as pd
import pytest

from synthetic_imputation.execution import configured
from synthetic_imputation.reporting import build_report
from synthetic_imputation.selection import TARGETS, choose


@pytest.fixture
def exports(tmp_path):
    source=tmp_path/'results'
    source.mkdir()
    rows=[]
    for method,skills in [('Median',(0,0,0)),('Gaussian EM',(.5,.8,-.4)),('Random Forest',(.9,.7,-.2))]:
        for target,skill in zip(('x','life','random_placebo'),skills):
            rows.append({'scenario':'completion0.8-noise0.05','experiment':'main','variant':'all predictors',
                'seed':219,'target':target,'method':method,'n':100,'coverage':1.,'MAE':1-skill,
                'RMSE':1-skill,'median MAE':1.,'skill':skill})
    pd.DataFrame(rows).to_csv(source/'numeric_scores.csv',index=False)
    pd.DataFrame([{'scenario':'completion0.8-noise0.05','experiment':'main','variant':'all predictors',
        'seed':219,'target':'model_id','method':method,'n':100,'coverage':1.,'accuracy':accuracy}
        for method,accuracy in [('Mode',.25),('Gaussian EM',.98),('Random Forest',.94)]]
        ).to_csv(source/'category_scores.csv',index=False)
    pd.DataFrame([{'experiment':'main','method':m,'tasks':5,'seconds per target and fold':seconds}
        for m,seconds in [('Median',.01),('Gaussian EM',.1),('Random Forest',1.)]]
        ).to_csv(source/'cost.csv',index=False)
    (source/'figures').mkdir()
    # A valid one-pixel PNG is sufficient to test embedding and path escaping.
    (source/'figures'/'chart<script>.png').write_bytes(base64.b64decode(
        'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII='))
    return source


def test_report_separates_numeric_skill_category_accuracy_and_placebo(exports,tmp_path):
    out=tmp_path/'report'
    path=build_report(exports,out,{'run_id':'test','refitted':False})
    summary=json.loads((out/'summary.json').read_text())
    gmm=next(r for r in summary['leaderboard'] if r['method']=='Gaussian EM')
    assert gmm['skill']==pytest.approx(.65)  # neither placebo nor categorical accuracy enters this mean
    assert gmm['accuracy']==pytest.approx(.98)
    assert summary['numeric_fields_in_mean']==2
    assert summary['charts']==1
    html=path.read_text()
    assert 'data:image/png;base64,' in html
    assert 'chart<script>' not in html
    assert 'chart&lt;script&gt;' in html
    assert 'Published full-study results' in html
    assert 'notebook' not in summary['run_id']
    assert 'src="http' not in html and 'href="http' not in html


def test_quick_report_is_explicit_and_keeps_its_run_source(exports,tmp_path):
    out=tmp_path/'report'
    path=build_report(exports,out,{'run_id':'quick-123','quick':True,'refitted':True,
                                   'config':{'rows':600}})
    summary=json.loads((out/'summary.json').read_text())
    assert summary['quick'] is True and summary['refitted'] is True
    assert summary['population_rows']==600
    assert 'Quick validation — not research findings' in path.read_text()
    manifest=json.loads((out/'manifest.json').read_text())
    assert manifest['run_id']=='quick-123'
    assert 'numeric_scores.csv' in manifest['source_files']


def test_report_rejects_changed_skill_and_missing_evidence(exports,tmp_path):
    scores=pd.read_csv(exports/'numeric_scores.csv')
    scores.loc[0,'skill']=.123
    scores.to_csv(exports/'numeric_scores.csv',index=False)
    with pytest.raises(ValueError,match='inconsistent'):
        build_report(exports,tmp_path/'report')
    assert not (tmp_path/'report/report.html').exists()
    (exports/'category_scores.csv').unlink()
    with pytest.raises(ValueError,match='category_scores'):
        build_report(exports,tmp_path/'other-report')


def test_selection_obeys_placebo_gate_near_best_margin_and_speed():
    rows=[]
    for name,score,placebo,seconds in [('best',.30,1.,10),('fast',.301,1.,1),
                                    ('bad_placebo',.20,1.02,.1),('too_weak',.40,1.,.1)]:
        rows.append({'settings':json.dumps({'name':name}),'seconds':seconds,'random_placebo':placebo,
                     **{t:score for t in TARGETS}})
    choice,table=choose(pd.DataFrame(rows))
    assert choice=={'name':'fast'}
    assert not table.loc[table.settings.str.contains('bad_placebo'),'eligible'].item()
    assert not table.loc[table.settings.str.contains('too_weak'),'near_best'].item()


def test_configuration_override_preserves_other_models_and_rejects_typos():
    baseline=configured()
    custom=configured(overrides={'parameters':{'Gaussian EM':{'clusters':3}}})
    assert custom.parameters['Gaussian EM']['clusters']==3
    assert custom.parameters['Gaussian EM']['reg']==baseline.parameters['Gaussian EM']['reg']
    assert custom.parameters['Random Forest']==baseline.parameters['Random Forest']
    assert baseline.parameters['Gaussian EM']['clusters']==6
    with pytest.raises(ValueError,match='Unknown configuration'):
        configured(overrides={'foldss':3})


def test_cli_checks_saved_inputs_before_starting_and_records_failure(tmp_path):
    from synthetic_imputation.cli import main
    out=tmp_path/'report'
    assert main(['--mode','evaluate','--run-id','absent','--out',str(out)],project_root=tmp_path)==1
    status=json.loads((out/'run_status.json').read_text())
    assert status['status']=='failed'
    assert 'No saved contract' in status['error']
    assert not (out/'report.html').exists()
