"""Schema-configured CSV experiments, separate from the fixed historical benchmark."""
from __future__ import annotations

import csv
from decimal import Decimal, InvalidOperation
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

from .dataset_models import METHODS, Baseline, make_model
from .cli import _environment


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_ready(value):
    if isinstance(value, dict): return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [json_ready(v) for v in value]
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (float, np.floating)): return float(value) if np.isfinite(value) else None
    if isinstance(value, np.bool_): return bool(value)
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(json_ready(value), indent=2, allow_nan=False) + '\n', encoding='utf-8')


def read_csv(path):
    path = Path(path)
    with path.open(encoding='utf-8-sig', newline='') as handle:
        header = next(csv.reader(handle), [])
    if not header or any(not c.strip() for c in header) or len(header) != len(set(header)):
        raise ValueError('CSV headers must be nonempty and unique')
    data = pd.read_csv(path, encoding='utf-8-sig', dtype=object)
    if len(data) < 12: raise ValueError('Use at least 12 records for held-out evaluation')
    if len(data) > 100_000 or len(data.columns) > 200:
        raise ValueError('This local interface supports up to 100,000 rows and 200 columns per CSV')
    data.index = pd.RangeIndex(len(data), name='row_number')
    return data


def date_days(series):
    dates=pd.to_datetime(series,format='ISO8601',errors='raise',utc=True)
    return (dates.astype('int64').astype(float)/86_400_000_000_000).where(dates.notna(),np.nan)


def inspect_dataset(path):
    data = read_csv(path)
    columns = []
    for c in data:
        seen, distinct = int(data[c].notna().sum()), int(data[c].nunique())
        numeric = bool(seen and pd.to_numeric(data[c].dropna(),errors='coerce').notna().all())
        id_like = c.lower() in ('id', 'index', 'row_id', 'asset_id', 'record_id', 'row_number') and distinct >= .95*max(seen, 1)
        code_like = (c.lower()=='id' or c.lower().endswith(('_id','_code','_type'))) and distinct<=200
        kind = 'ignore' if seen < 2 or id_like else 'numeric' if numeric and not code_like else 'categorical'
        if kind=='categorical' and any(word in c.lower() for word in ('date','timestamp')):
            try:
                date_days(data[c]);kind='datetime'
            except (ValueError,TypeError):pass
        note = 'ISO dates: modelled in days; recorded strings are retained.' if kind=='datetime' else 'Identifier: excluded from predictors and scoring.' if id_like else (
            'Fewer than two observed values: excluded.' if seen < 2 else
            'Many labels: check whether this is an identifier, free text or a date.' if not numeric and distinct > 50 else '')
        columns.append({'name':c, 'type':kind, 'missing':int(data[c].isna().sum()),
                        'observed':seen, 'distinct':distinct, 'note':note,
                        'group':'', 'tolerance_kind':'none', 'tolerance':None,
                        'lower':None, 'upper':None, 'integer':kind=='datetime',
                        'exclude_from_mean':c == 'random_placebo'})
    return {'filename':Path(path).name, 'rows':len(data), 'columns':columns,
            'missing':int(data.isna().sum().sum()), 'total_cells':int(data.size),
            'preview':json_ready(data.head(5).where(data.notna(), None).to_dict(orient='records'))}


def category_reference(reference, recorded):
    """Align numeric-code formatting for scoring without changing recorded labels."""
    result=reference.map(lambda v:str(v) if pd.notna(v) else np.nan)
    labels=recorded.dropna().astype(str).unique()
    try:
        keys=[Decimal(v) for v in labels]
        if not all(v.is_finite() for v in keys) or len(set(keys))!=len(labels):return result
        lookup=dict(zip(keys,labels))
        def align(value):
            if pd.isna(value):return np.nan
            try:return lookup.get(Decimal(str(value)),str(value))
            except InvalidOperation:return str(value)
        return reference.map(align)
    except InvalidOperation:return result


@dataclass
class DatasetConfig:
    title: str = 'Asset data imputation'
    columns: list = field(default_factory=list)
    methods: list = field(default_factory=lambda: ['median', 'forest', 'gaussian'])
    folds: int = 3
    repeats: int = 1
    seed: int = 219
    mask_rate: float = .2
    masking: str = 'scattered'
    noise: float = 0.
    noise_columns: list = field(default_factory=list)
    evaluation_rows: int = 1000
    trees: int = 50
    passes: int = 3
    clusters: int = 4
    em_steps: int = 12
    regularisation: float = .025
    epochs: int = 30
    neighbors: int = 15
    ridge_alpha: float = 10.
    chains: int = 3
    donors: int = 5

    @classmethod
    def from_dict(cls, value, path):
        unknown = set(value) - set(cls.__dataclass_fields__)
        if unknown: raise ValueError(f'Unknown dataset settings: {sorted(unknown)}')
        config = cls(**value)
        if not config.columns: config.columns = inspect_dataset(path)['columns']
        config.methods = ['median'] + list(dict.fromkeys(m for m in config.methods if m != 'median'))
        return config


def prepare(path, config, truth_path=None):
    source = read_csv(path)
    names = [s['name'] for s in config.columns]
    if len(names) != len(set(names)) or set(names) != set(source):
        raise ValueError('Column configuration must describe every CSV column exactly once')
    if not (2 <= config.folds <= 10 and 1 <= config.repeats <= 5):
        raise ValueError('Use 2-10 folds and 1-5 repetitions')
    if not (.01 <= config.mask_rate <= .7) or config.masking not in ('scattered', 'grouped'):
        raise ValueError('Additional hiding must be 1-70%, with scattered or grouped masks')
    if not (0 <= config.noise <= .5): raise ValueError('Noise must be between 0 and 50%')
    if not (0 <= config.seed <= 2**31-10_000): raise ValueError('Use a nonnegative seed below 2^31-10000')
    limits = {'trees':(1,1000),'passes':(1,30),'clusters':(1,20),'em_steps':(1,100),
              'epochs':(1,500),'neighbors':(1,200),'chains':(1,20),'donors':(1,50),
              'evaluation_rows':(0,100_000)}
    for name,(low,high) in limits.items():
        value = getattr(config,name)
        if not isinstance(value,int) or not low <= value <= high:
            raise ValueError(f'{name} must be an integer from {low} to {high}')
    if not np.isfinite(config.regularisation) or not np.isfinite(config.ridge_alpha) or config.regularisation <= 0 or config.ridge_alpha <= 0:
        raise ValueError('Regularisation and ridge alpha must be positive')
    if set(config.methods)-set(METHODS): raise ValueError('Unknown model selected')
    specs = {s['name']:dict(s) for s in config.columns}
    numeric, categorical = [], []
    data = source.copy()
    for c,spec in specs.items():
        allowed={'name','type','missing','observed','distinct','note','group','tolerance_kind',
                 'tolerance','lower','upper','integer','exclude_from_mean'}
        if set(spec)-allowed:raise ValueError(f'{c}: unknown field settings {sorted(set(spec)-allowed)}')
        kind = spec['type']
        if kind not in ('numeric','categorical','datetime','ignore'): raise ValueError(f'{c}: invalid column type')
        if kind == 'ignore': continue
        if data[c].notna().sum() < 2: raise ValueError(f'{c}: ignore columns with fewer than two observations')
        if kind in ('numeric','datetime'):
            data[c] = date_days(data[c]) if kind=='datetime' else pd.to_numeric(data[c],errors='raise').astype(float)
            if kind=='datetime':spec.update(type='numeric',source_type='datetime',integer=True)
            if not np.isfinite(data[c].dropna()).all(): raise ValueError(f'{c}: infinite values are not supported')
            numeric.append(c)
        else:
            data[c] = data[c].map(lambda v: str(v) if pd.notna(v) else np.nan).astype(object)
            if data[c].nunique() > 200: raise ValueError(f'{c}: more than 200 categories; exclude IDs or convert dates to a numeric time field')
            categorical.append(c)
        mode = spec.get('tolerance_kind','none')
        if mode not in ('none','absolute','relative'): raise ValueError(f'{c}: invalid tolerance type')
        if mode != 'none' and (spec.get('tolerance') is None or float(spec['tolerance']) <= 0):
            raise ValueError(f'{c}: supply a positive tolerance or choose none')
        if mode!='none' and not np.isfinite(float(spec['tolerance'])):
            raise ValueError(f'{c}: tolerance must be finite')
        if kind=='datetime' and mode=='relative':
            raise ValueError(f'{c}: use an absolute date tolerance in days')
        lo,hi = spec.get('lower'),spec.get('upper')
        if kind=='categorical' and (mode!='none' or lo is not None or hi is not None or spec.get('integer')):
            raise ValueError(f'{c}: categories use exact labels, not numeric tolerances, bounds or rounding')
        if any(v is not None and not np.isfinite(v) for v in (lo,hi)):
            raise ValueError(f'{c}: bounds must be finite or left blank')
        if lo is not None and hi is not None and lo > hi: raise ValueError(f'{c}: lower bound exceeds upper bound')
        if spec.get('integer') and lo is not None and hi is not None and np.ceil(lo)>np.floor(hi):
            raise ValueError(f'{c}: bounds contain no integer')
    active = numeric + categorical
    if not active: raise ValueError('Choose at least one numeric or categorical column')
    if config.noise and (not config.noise_columns or set(config.noise_columns)-set(numeric)
                         or any(specs[c].get('source_type')=='datetime' for c in config.noise_columns)):
        raise ValueError('For noise, explicitly choose the numeric fields to perturb')
    groups = {}
    for c in active:
        label=specs[c].get('group')
        group=('group',label) if label else ('field',c)
        groups.setdefault(group,[]).append(c)
    if config.masking == 'grouped' and not any(len(g)>1 for g in groups.values()):
        raise ValueError('Grouped masking requires at least one group containing two or more fields')
    truth = None
    if truth_path:
        truth = read_csv(truth_path)
        if len(truth)!=len(source) or list(truth)!=list(source):
            raise ValueError('Truth CSV must have the same rows, column names and column order')
        for c in source:
            if specs[c]['type']=='ignore' and source[c].is_unique and source[c].notna().all():
                if not source[c].equals(truth[c]): raise ValueError(f'Truth rows are misaligned at identifier {c}')
        for c in numeric:
            truth[c] = date_days(truth[c]) if specs[c].get('source_type')=='datetime' else pd.to_numeric(truth[c],errors='raise')
            if not np.isfinite(truth[c].dropna()).all():raise ValueError(f'{c}: truth contains infinite values')
        for c in categorical: truth[c] = category_reference(truth[c],source[c])
    return source,data[active],truth,specs,numeric,categorical,list(groups.values())


def corrupt(frame, config, groups, rng, training=False, *, noise_rng):
    draws = rng.random(frame.shape)
    if config.masking == 'grouped':
        for group in groups:
            positions = [frame.columns.get_loc(c) for c in group]
            draws[:,positions] = draws[:,positions[:1]]
    hidden = frame.notna() & (draws < config.mask_rate)
    if training:
        for c in frame:
            known = np.flatnonzero(frame[c].notna())
            if len(known) and (frame[c].notna() & ~hidden[c]).sum()<min(2,len(known)):
                hidden.iloc[rng.choice(known,min(2,len(known)),replace=False),frame.columns.get_loc(c)] = False
    result = frame.mask(hidden)
    if config.noise:
        noise_draws=noise_rng.uniform(-1,1,frame.shape)
        for c in config.noise_columns:
            result[c] = result[c]*(1+config.noise*noise_draws[:,frame.columns.get_loc(c)])
    return result,hidden


def finish(original, filled, specs):
    filled = filled.reindex(index=original.index,columns=original.columns).copy()
    for c in original:
        missing = original[c].isna()
        spec = specs[c]
        if spec['type']=='numeric':
            values = pd.to_numeric(filled.loc[missing,c],errors='coerce')
            lo,hi = spec.get('lower'),spec.get('upper')
            if spec.get('integer'):
                values = values.round()
                lo = np.ceil(lo) if lo is not None else None
                hi = np.floor(hi) if hi is not None else None
            filled.loc[missing,c] = values.clip(lower=lo,upper=hi)
        filled.loc[~missing,c] = original.loc[~missing,c]
        if spec['type']=='numeric' and not np.isfinite(pd.to_numeric(filled[c],errors='coerce')).all():
            raise ValueError(f'{c}: model returned missing or non-finite numeric values')
        if not original.loc[~missing,c].equals(filled.loc[~missing,c]):
            if not original.loc[~missing,c].eq(filled.loc[~missing,c]).all():
                raise AssertionError(f'Observed values changed in {c}')
    return filled


def field_scores(predictions, specs):
    records = []
    for (method,target),part in predictions.groupby(['method','target'],sort=False):
        spec = specs[target]
        numeric = spec['type']=='numeric'
        row = {'method':method,'label':METHODS[method],'target':target,'kind':spec['type'],
               'n':len(part),'coverage':float(part.prediction.notna().mean()),
               'mae':None,'rmse':None,'skill':None,'accuracy':None,'macro_f1':None,
               'within_tolerance':None,'tolerance_n':0,'below_floor':False}
        if numeric:
            actual = pd.to_numeric(part.truth).to_numpy(float)
            guess = pd.to_numeric(part.prediction,errors='coerce').to_numpy(float)
            baseline = pd.to_numeric(part.baseline).to_numpy(float)
            errors = np.abs(guess-actual)
            floor = float(np.mean(np.abs(baseline-actual)))
            row.update(mae=float(np.mean(errors)),rmse=float(np.sqrt(np.mean((guess-actual)**2))),
                       baseline_mae=floor,skill=1-float(np.mean(errors))/floor if floor>1e-12 else None)
            mode = spec.get('tolerance_kind','none')
            if mode!='none':
                keep = np.isfinite(actual) & ((actual!=0) if mode=='relative' else True)
                tolerance = np.abs(actual)*float(spec['tolerance']) if mode=='relative' else np.full(len(actual),float(spec['tolerance']))
                row.update(within_tolerance=float((errors[keep]<=tolerance[keep]).mean()) if keep.any() else None,
                           tolerance_n=int(keep.sum()))
            row['below_floor'] = row['skill'] is not None and row['skill']<0
        else:
            actual,guess = part.truth.astype(str),part.prediction.fillna('[unfilled]').astype(str)
            row.update(accuracy=float(accuracy_score(actual,guess)),
                       macro_f1=float(f1_score(actual,guess,average='macro',zero_division=0)))
        records.append(row)
    return records


def run_dataset(path, output, config=None, truth_path=None, progress=None):
    """Paired fold validation, then independent full-input fits for completed exports."""
    start = time.perf_counter()
    path,output = Path(path).resolve(),Path(output).resolve()
    if output.exists() and any(output.iterdir()): raise ValueError('Choose an empty output folder for a new dataset run')
    if output==path.parent or path.is_relative_to(output): raise ValueError('Output must not contain or replace the input CSV')
    config = DatasetConfig.from_dict(config or {},path)
    source,data,truth,specs,numeric,categorical,groups = prepare(path,config,truth_path)
    output.mkdir(parents=True,exist_ok=True)
    write_json(output/'config.json',asdict(config))
    write_json(output/'run_status.json',{'status':'running'})
    report_progress = progress or (lambda message, done, total: print(message,flush=True))
    total = config.repeats*config.folds*len(config.methods)+len(config.methods)+1
    done = 0
    profile = inspect_dataset(path)
    index = np.arange(len(data))
    if config.evaluation_rows and config.evaluation_rows<len(index):
        index = np.sort(np.random.default_rng(config.seed).choice(index,config.evaluation_rows,replace=False))
    if len(index)<config.folds*3: raise ValueError('Evaluation sample is too small for the selected fold count')
    predictions,timing,failures,warning_records = [],[],[],[]
    successful = {m:0 for m in config.methods}
    try:
        with threadpool_limits(limits=1):
            for repeat in range(config.repeats):
                seed = config.seed+repeat*1000
                splitter = KFold(config.folds,shuffle=True,random_state=seed)
                for fold,(train_index,test_index) in enumerate(splitter.split(index),1):
                    train_original,test_original = data.iloc[index[train_index]],data.iloc[index[test_index]]
                    streams=[np.random.default_rng(np.random.SeedSequence([seed,fold,channel])) for channel in range(4)]
                    train,_ = corrupt(train_original,config,groups,streams[0],training=True,noise_rng=streams[1])
                    test,hidden = corrupt(test_original,config,groups,streams[2],noise_rng=streams[3])
                    if any(train[c].notna().sum()==0 for c in train):
                        raise ValueError(f'Fold {fold} has a field with no training observations; revise folds, seed or excluded columns')
                    reference = truth.loc[test.index,list(data)] if truth is not None else test_original
                    score_mask = test.isna() & reference.notna()
                    baseline = finish(test,Baseline(numeric,categorical).fit(train).transform(test),specs)
                    for method in config.methods:
                        report_progress(f'{METHODS[method]} · validation {repeat+1}/{config.repeats}, fold {fold}/{config.folds}',done,total)
                        clock = time.perf_counter()
                        try:
                            with warnings.catch_warnings(record=True) as caught:
                                warnings.simplefilter('always')
                                model = make_model(method,numeric,categorical,asdict(config),seed).fit(train)
                                filled = finish(test,model.transform(test),specs)
                            if filled.isna().any().any(): raise ValueError('Model left gaps in modelled fields')
                            for w in caught:
                                warning_records.append({'method':method,'stage':'validation','type':w.category.__name__,'message':str(w.message)[:400]})
                            for c in data:
                                for position in np.flatnonzero(score_mask[c].to_numpy()):
                                    rid = test.index[position]
                                    predictions.append({'method':method,'target':c,'repeat':repeat+1,'fold':fold,
                                        'row_number':int(rid)+1,'truth':reference.at[rid,c],
                                        'prediction':filled.at[rid,c],'baseline':baseline.at[rid,c],
                                        'artificially_hidden':bool(hidden.at[rid,c]),
                                        'visible_inputs':int(test.loc[rid].notna().sum())})
                            successful[method]+=1
                        except Exception as error:
                            failures.append({'method':method,'stage':'validation','repeat':repeat+1,
                                             'fold':fold,'error':f'{type(error).__name__}: {error}'})
                        timing.append({'method':method,'stage':'validation','repeat':repeat+1,'fold':fold,'seconds':time.perf_counter()-clock})
                        done+=1
            complete_methods = [m for m in config.methods if successful[m]==config.repeats*config.folds]
            if not predictions or 'median' not in complete_methods: raise ValueError('No complete paired baseline evaluation was possible')
            prediction_table = pd.DataFrame(predictions)
            prediction_table.to_csv(output/'validation_predictions.csv',index=False)
            scores = field_scores(prediction_table[prediction_table.method.isin(complete_methods)],specs)
            completed = []
            for method in config.methods:
                report_progress(f'{METHODS[method]} · filling the supplied dataset',done,total)
                clock = time.perf_counter()
                try:
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter('always')
                        model = make_model(method,numeric,categorical,asdict(config),config.seed).fit(data)
                        filled = finish(data,model.transform(data),specs)
                    for w in caught:
                        warning_records.append({'method':method,'stage':'completion','type':w.category.__name__,'message':str(w.message)[:400]})
                    export = source.copy()
                    for c in data:
                        missing = source[c].isna()
                        export[c] = export[c].astype(object) if c in categorical else export[c]
                        values=filled.loc[missing,c]
                        if specs[c].get('source_type')=='datetime':
                            values=pd.to_datetime(values,unit='D',origin='unix',utc=True).dt.strftime('%Y-%m-%d')
                        export.loc[missing,c] = values
                        assert source.loc[~missing,c].eq(export.loc[~missing,c]).all()
                    filename = f'completed_{method}.csv'
                    export.to_csv(output/filename,index=False)
                    ledger=[]
                    for c in data:
                        for rid in data.index[data[c].isna()]:
                            ledger.append({'row_number':int(rid)+1,'attribute':c,'fill':export.at[rid,c],
                                'method':METHODS[method],'status':'filled' if pd.notna(export.at[rid,c]) else 'unfilled',
                                'individually_validated':False})
                    pd.DataFrame(ledger,columns=['row_number','attribute','fill','method','status','individually_validated']).to_csv(output/f'fills_{method}.csv',index=False)
                    completed.append({'method':method,'label':METHODS[method],'file':filename,
                        'rows':len(export),'missing':int(export[list(data)].isna().sum().sum()),
                        'observations_preserved':True})
                except Exception as error:
                    failures.append({'method':method,'stage':'completion','error':f'{type(error).__name__}: {error}'})
                timing.append({'method':method,'stage':'completion','seconds':time.perf_counter()-clock})
                done+=1
        pd.DataFrame(scores).to_csv(output/'field_scores.csv',index=False)
        pd.DataFrame(timing).to_csv(output/'runtime.csv',index=False)
        write_json(output/'failures.json',failures)
        write_json(output/'warnings.json',warning_records)
        ranking=[]
        for method in complete_methods:
            rows = [r for r in scores if r['method']==method]
            skills = [r['skill'] for r in rows if r['kind']=='numeric' and r['skill'] is not None and not specs[r['target']].get('exclude_from_mean')]
            f1 = [r['macro_f1'] for r in rows if r['macro_f1'] is not None]
            within = [r['within_tolerance'] for r in rows if r['within_tolerance'] is not None]
            ranking.append({'method':method,'label':METHODS[method],
                'skill':float(np.mean(skills)) if skills else None,'macro_f1':float(np.mean(f1)) if f1 else None,
                'within_tolerance':float(np.mean(within)) if within else None,
                'seconds':sum(r['seconds'] for r in timing if r['method']==method and r['stage']=='validation'),
                'below_floor':sum(r['below_floor'] for r in rows),
                'completion_available':any(c['method']==method for c in completed)})
        ranking.sort(key=lambda r:r['skill'] if r['skill'] is not None else r['macro_f1'] if r['macro_f1'] is not None else -np.inf,reverse=True)
        summary={'title':config.title,'mode':'dataset','run_id':output.name,
            'generated_utc':datetime.now(timezone.utc).isoformat(),'dataset':path.name,
            'rows':len(source),'evaluation_rows':len(index),'columns':len(source.columns),
            'numeric':numeric,'categorical':categorical,
            'date_fields':[c for c in numeric if specs[c].get('source_type')=='datetime'],
            'ignored':[c for c in source if c not in data],
            'natural_missing':int(data.isna().sum().sum()),'truth_supplied':truth is not None,
            'score_reference':'Supplied truth at held-out missing cells' if truth is not None else 'Recorded values temporarily hidden in held-out rows',
            'config':asdict(config),'profile':profile,'leaderboard':ranking,'field_scores':scores,
            'completed':completed,'failures':failures,'warnings_count':len(warning_records),
            'validation_prediction_rows':len(prediction_table),'elapsed_seconds':time.perf_counter()-start}
        write_json(output/'summary.json',summary)
        write_json(output/'manifest.json',{'schema_version':1,'mode':'dataset','input_sha256':digest(path),
            'truth_sha256':digest(truth_path) if truth_path else None,'config':asdict(config),
            'environment':_environment(),
            'code_sha256':{str(p.relative_to(Path(__file__).parent)):digest(p) for p in Path(__file__).parent.rglob('*')
                           if p.is_file() and p.suffix in ('.py','.html','.css')},
            'project_sha256':{name:digest(Path(__file__).resolve().parents[2]/name)
                for name in ('requirements.txt','pyproject.toml','run_study.py')},
            'note':'Generic schema adapters; archived benchmark settings/results are separate.'})
        report_progress('Building figures and the report',done,total)
        from .dataset_reporting import build_dataset_report
        report = build_dataset_report(output,source,prediction_table,data)
        write_json(output/'run_status.json',{'status':'completed_with_errors' if failures else 'success',
            'report':report.name,'validation_fits_succeeded':sum(successful.values()),
            'failures':len(failures),'elapsed_seconds':time.perf_counter()-start})
        report_progress('Report ready',total,total)
        return report
    except Exception as error:
        write_json(output/'run_status.json',{'status':'failed','error':f'{type(error).__name__}: {error}'})
        raise
