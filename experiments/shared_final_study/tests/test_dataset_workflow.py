"""Checks of new-schema inference, train/test isolation and the guided entry point."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import pytest

from synthetic_imputation.dataset import (
    DatasetConfig, corrupt, field_scores, inspect_dataset, prepare, run_dataset,
)
from synthetic_imputation.dataset_models import METHODS, Baseline, make_model

ROOT=Path(__file__).resolve().parents[1]


def data(rows=100):
    rng=np.random.default_rng(8)
    x=rng.normal(1000,20,rows)
    frame=pd.DataFrame({'x':x,'score':2*x+rng.normal(0,3,rows),
        'material':np.array(['A','B','C','D','E'])[np.arange(rows)%5].astype(object)})
    frame.loc[::7,'x']=np.nan
    frame.loc[::9,'score']=np.nan
    frame.loc[::11,'material']=np.nan
    return frame


def small_config():
    config=DatasetConfig(trees=5,passes=1,clusters=2,em_steps=5,epochs=2,
                         neighbors=3,chains=2,folds=2,evaluation_rows=0)
    return asdict(config)


def test_archived_scientific_source_inputs_and_results_are_unchanged():
    expected=json.loads((ROOT/'ORIGINAL_SOURCES.json').read_text())
    for group in ('original_prediction_modules','original_input_files','original_result_csvs'):
        for name,h in expected[group].items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h,name


@pytest.mark.parametrize('method',list(METHODS))
def test_generic_adapters_handle_five_labels_and_preserve_test_observations(method):
    frame=data(100)
    train,test=frame.iloc[:80].copy(),frame.iloc[80:].copy()
    numeric,categorical=['x','score'],['material']
    config=small_config()
    fitted=make_model(method,numeric,categorical,config,31).fit(train)
    completed=fitted.transform(test)
    completed=completed.reindex(columns=test.columns,index=test.index)
    assert not completed.isna().any().any()
    for c in test:
        known=test[c].notna()
        assert test.loc[known,c].eq(completed.loc[known,c]).all()
    assert set(completed.material)<=set(train.material.dropna())
    assert (pd.to_numeric(completed.x)>99).all()  # no transformer x-bound imposed


def test_absolute_year_tolerance_and_relative_zero_denominator():
    predictions=pd.DataFrame([
        {'method':'median','target':'year','truth':1960.,'prediction':1968.,'baseline':1900.},
        {'method':'median','target':'ratio','truth':0.,'prediction':50.,'baseline':20.},
        {'method':'median','target':'ratio','truth':100.,'prediction':105.,'baseline':20.},
    ])
    specs={'year':{'type':'numeric','tolerance_kind':'absolute','tolerance':5},
           'ratio':{'type':'numeric','tolerance_kind':'relative','tolerance':.1}}
    rows={r['target']:r for r in field_scores(predictions,specs)}
    assert rows['year']['within_tolerance']==0
    assert rows['ratio']['tolerance_n']==1
    assert rows['ratio']['within_tolerance']==1


def test_noise_changes_neither_training_nor_test_masks_and_scales_same_draws():
    frame=data(80)
    masks=[];values=[]
    for amplitude in (0.,.05,.1):
        config=DatasetConfig(noise=amplitude,noise_columns=['x'])
        streams=[np.random.default_rng(np.random.SeedSequence([219,1,c])) for c in range(4)]
        _,a=corrupt(frame.iloc[:50],config,[],streams[0],training=True,noise_rng=streams[1])
        test,b=corrupt(frame.iloc[50:],config,[],streams[2],noise_rng=streams[3])
        masks.append((a,b));values.append(test)
    for a,b in masks[1:]:
        pd.testing.assert_frame_equal(a,masks[0][0]);pd.testing.assert_frame_equal(b,masks[0][1])
    observed=values[0].x.notna()
    e1=values[1].loc[observed,'x']/values[0].loc[observed,'x']-1
    e2=values[2].loc[observed,'x']/values[0].loc[observed,'x']-1
    np.testing.assert_allclose(e2,2*e1,atol=1e-12)


def test_identifier_and_unlearnable_columns_are_excluded_and_duplicate_headers_rejected(tmp_path):
    frame=data(20).assign(asset_id=np.arange(20),empty=np.nan)
    path=tmp_path/'data.csv';frame.to_csv(path,index=False)
    types={c['name']:c['type'] for c in inspect_dataset(path)['columns']}
    assert types['asset_id']==types['empty']=='ignore'
    bad=tmp_path/'bad.csv';bad.write_text('x,x\n1,2\n')
    with pytest.raises(ValueError,match='unique'):inspect_dataset(bad)


def test_truth_alignment_and_grouped_design_are_explicit(tmp_path):
    frame=data(30).assign(asset_id=np.arange(30))
    path=tmp_path/'data.csv';frame.to_csv(path,index=False)
    config=DatasetConfig.from_dict({'masking':'grouped'},path)
    with pytest.raises(ValueError,match='group'):prepare(path,config)
    truth=tmp_path/'truth.csv';frame.iloc[::-1].to_csv(truth,index=False)
    config.masking='scattered'
    with pytest.raises(ValueError,match='misaligned'):prepare(path,config,truth)


def test_iso_dates_use_day_units_and_restore_recorded_strings(tmp_path,monkeypatch):
    import synthetic_imputation.dataset_reporting as reports
    frame=data(40).assign(install_date=pd.date_range('1990-01-01',periods=40).strftime('%Y-%m-%d'))
    frame.loc[::7,'install_date']=np.nan
    path=tmp_path/'data.csv';frame.to_csv(path,index=False)
    config=DatasetConfig.from_dict({'methods':['median'],'folds':2},path)
    date=next(c for c in config.columns if c['name']=='install_date')
    assert date['type']=='datetime'
    date.update(tolerance_kind='absolute',tolerance=7)
    _,model,_,specs,_,_,_=prepare(path,config)
    assert model.install_date.dropna().between(7000,8000).all()
    assert specs['install_date']['source_type']=='datetime'
    def report(out,*args):
        p=out/'report.html';p.write_text('checked');return p
    monkeypatch.setattr(reports,'build_dataset_report',report)
    run_dataset(path,tmp_path/'run',asdict(config))
    completed=pd.read_csv(tmp_path/'run/completed_median.csv')
    known=frame.install_date.notna()
    assert frame.loc[known,'install_date'].eq(completed.loc[known,'install_date']).all()
    assert completed.install_date.str.match(r'^\d{4}-\d{2}-\d{2}$').all()


def test_identifier_text_and_numeric_category_truth_format_are_preserved(tmp_path,monkeypatch):
    import synthetic_imputation.dataset_reporting as reports
    frame=data(40).drop(columns='material')
    frame['asset_id']=[f'{i:05d}' for i in range(40)]
    frame['model_id']=(np.arange(40)%5).astype(float)
    truth=frame.copy();truth['model_id']=truth.model_id.astype(int)
    frame.loc[::7,'model_id']=np.nan
    path=tmp_path/'data.csv';truth_path=tmp_path/'truth.csv'
    frame.to_csv(path,index=False);truth.to_csv(truth_path,index=False)
    config=DatasetConfig.from_dict({'methods':['median'],'folds':2},path)
    _,model,reference,_,_,cats,_=prepare(path,config,truth_path)
    assert 'model_id' in cats
    known=model.model_id.notna()
    assert model.loc[known,'model_id'].eq(reference.loc[known,'model_id']).all()
    def report(out,*args):
        p=out/'report.html';p.write_text('checked');return p
    monkeypatch.setattr(reports,'build_dataset_report',report)
    run_dataset(path,tmp_path/'run',asdict(config),truth_path)
    completed=pd.read_csv(tmp_path/'run/completed_median.csv',dtype=str)
    assert completed.asset_id.tolist()==frame.asset_id.tolist()


def test_models_never_fit_holdout_rows_or_clean_truth(tmp_path,monkeypatch):
    import synthetic_imputation.dataset as module
    import synthetic_imputation.dataset_reporting as reports
    frame=data(40);path=tmp_path/'data.csv';frame.to_csv(path,index=False)
    truth=frame.copy();truth[['x','score']]=truth[['x','score']].fillna(1000)+1_000_000
    truth_path=tmp_path/'truth.csv';truth.to_csv(truth_path,index=False)
    calls=[]
    class Spy(Baseline):
        def fit(self,train):
            assert train.x.dropna().max()<100_000
            self.indices=set(train.index);self.final=len(train)==len(frame);calls.append(self.indices)
            return super().fit(train)
        def transform(self,test):
            if not self.final:assert not self.indices&set(test.index)
            return super().transform(test)
    monkeypatch.setattr(module,'make_model',lambda method,numeric,categorical,params,seed:Spy(numeric,categorical))
    def report(out,*args):
        p=out/'report.html';p.write_text('checked');return p
    monkeypatch.setattr(reports,'build_dataset_report',report)
    config=small_config();config.update(methods=['median','forest'])
    run_dataset(path,tmp_path/'run',config,truth_path)
    assert len(calls)==6  # two methods × two folds, then two full-input fits
    ledger=pd.read_csv(tmp_path/'run/validation_predictions.csv')
    keys=['target','repeat','fold','row_number']
    a=ledger[ledger.method=='median'][keys].reset_index(drop=True)
    b=ledger[ledger.method=='forest'][keys].reset_index(drop=True)
    pd.testing.assert_frame_equal(a,b)


def test_generic_end_to_end_report_and_exports(tmp_path):
    frame=data(60);path=tmp_path/'data.csv';frame.to_csv(path,index=False)
    config=small_config();config.update(methods=['median','forest','gaussian'])
    report=run_dataset(path,tmp_path/'run',config)
    summary=json.loads((report.parent/'summary.json').read_text())
    assert not summary['failures']
    assert summary['truth_supplied'] is False
    assert 'data:image/png;base64,' in report.read_text()
    assert 'Naturally missing values' not in summary['score_reference']
    for output in summary['completed']:
        completed=pd.read_csv(report.parent/output['file'])
        for c in frame:
            known=frame[c].notna()
            if c=='material':assert frame.loc[known,c].eq(completed.loc[known,c]).all()
            else:np.testing.assert_allclose(frame.loc[known,c],completed.loc[known,c])


def test_failed_validation_models_are_excluded_and_cli_signals_partial_results(tmp_path,monkeypatch):
    import synthetic_imputation.dataset as module
    import synthetic_imputation.dataset_reporting as reports
    from synthetic_imputation.cli import main
    path=tmp_path/'data.csv';data(40).to_csv(path,index=False)
    def create(method,numeric,categorical,params,seed):
        if method=='forest':raise ValueError('deliberate model-stage failure')
        return Baseline(numeric,categorical)
    def report(out,*args):
        p=out/'report.html';p.write_text('checked');return p
    monkeypatch.setattr(module,'make_model',create)
    monkeypatch.setattr(reports,'build_dataset_report',report)
    settings=tmp_path/'config.json';settings.write_text(json.dumps({'methods':['median','forest'],'folds':2}))
    result=main(['--mode','dataset','--dataset',str(path),'--config',str(settings),'--out',str(tmp_path/'run')])
    assert result==2
    summary=json.loads((tmp_path/'run/summary.json').read_text())
    assert [r['method'] for r in summary['leaderboard']]==['median']
    assert len(summary['failures'])==3


def test_local_workbench_requires_its_token_and_blocks_path_escape():
    from synthetic_imputation.dataset_app import create_server
    server=create_server(ROOT,0)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    base=f'http://127.0.0.1:{server.server_port}'
    try:
        with urlopen(base) as response:assert b'Fill missing asset data' in response.read()
        with pytest.raises(HTTPError) as error:
            urlopen(Request(base+'/api/run',data=b'{}',headers={'Content-Type':'application/json'}))
        assert error.value.code==403
        with pytest.raises(HTTPError) as error:
            urlopen(base+'/benchmark/reports/published/%2e%2e/%2e%2e/src/synthetic_imputation/cli.py')
        assert error.value.code==404
    finally:server.shutdown();server.server_close();thread.join(timeout=2)
