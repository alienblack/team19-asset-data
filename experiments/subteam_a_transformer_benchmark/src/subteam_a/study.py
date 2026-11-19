"""Fit, tune and evaluate methods on the benchmark partitions and write atomic task results."""
from functools import lru_cache
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd
from .data import (TRANSFORMER, OVERHEAD_SCHEMA, ANCHOR, CORE, transformer_data, overhead_data, split_blocks)
from .data import SNAPSHOT_YEAR
from .features import Features, exact_fill
from .models import Floor, Chained, Neighbours, Autoencoder
from .mixture import MaskedMixture
from .evaluate import (
    calibrate, evaluate, table_checks, consistency, downstream, cell_export,
    installation_age_scores, rule_effects, relative_skill,
)

METHODS = ('floor','ridge','knn','forest','mixture','autoencoder')
PRIMARY = ('forest','mixture','autoencoder')
DEFAULT = {
    'floor': {},
    'forest': dict(trees=16,depth=11,leaf=7,features=.75,passes=2),
    'ridge': dict(alpha=15.,passes=3),
    'knn': dict(k=17,donors=1000,power=1.,penalty=.3),
    'mixture': dict(clusters=4,steps=12,reg=.025,class_weight=1.8),
    'autoencoder': dict(width=96,epochs=16,dropout=.1,lr=.003,decay=.002,remask=.27),
}


def candidate(method, rng):
    """Sample one settings dictionary for a method from the saved tuning search space."""
    def draw(*args):
        return rng.choice(args).item()
    if method == 'forest':
        return dict(trees=draw(10,14,18,22),depth=draw(8,10,12,14),leaf=draw(4,6,10,16),
                    features=draw(.55,.75,1.),passes=draw(2,3))
    if method == 'ridge':
        return dict(alpha=round(float(10**rng.uniform(-1,2.4)),5),passes=draw(2,3,4,5))
    if method == 'knn':
        return dict(k=draw(7,11,17,25,37),donors=draw(700,1000,1300),power=draw(.5,1.,1.5,2.),
                    penalty=draw(.0,.15,.35,.7))
    if method == 'mixture':
        return dict(clusters=4,steps=draw(8,10,12,15),reg=round(float(10**rng.uniform(-2.3,-.85)),6),
                    class_weight=draw(1.1,1.5,1.8,2.2,2.6))
    return dict(width=draw(64,96,128),epochs=draw(10,14,18,22),dropout=draw(.0,.07,.15,.22),
                lr=draw(.0015,.0025,.004),decay=round(float(10**rng.uniform(-4,-1.4)),7),remask=draw(.18,.24,.32,.4))


def create(method,p,seed):
    """Construct the requested imputer from its settings and seed."""
    if method=='floor':
        return Floor()
    if method in ('ridge','forest'):
        return Chained(method,p,seed)
    if method=='knn':
        return Neighbours(p,seed)
    if method=='mixture':
        return MaskedMixture(p,seed)
    if method=='autoencoder':
        return Autoencoder(p,seed)
    raise ValueError(method)


def write_json(p, obj):
    """
    Write a JSON artifact via a temporary file so interrupted tasks do not leave a completed
    filename.
    """
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
    tmp.replace(p)


@lru_cache(maxsize=10)
def load(data, sc, gs):
    """
    Return schema, truth, measured rows, calibration view and any feeder groups for one
    scenario and seed.
    """
    if sc.mechanism=='Overhead':
        truth, raw, groups = overhead_data(Path(data)/'overhead',sc)
        return OVERHEAD_SCHEMA, truth, raw, raw, groups
    root = Path(data)/'source'
    truth, raw = transformer_data(root,sc,gs)
    _, recaled = transformer_data(root,sc,gs,draw=1)
    return TRANSFORMER, truth, raw, recaled, None


def validation_skill(schema, f, x, truth, pred, baseline):
    """
    Average hidden-cell skill over eligible attributes, excluding the placebo, to compare
    tuning candidates.
    """
    out=[]
    for j, c in enumerate(schema.names):
        if c=='random_placebo':
            continue
        keep=np.isnan(x[:,j])
        if not keep.any():
            continue
        y,p,b=truth[keep,j],pred[keep,j],baseline[keep,j]
        categorical = j in schema.categories
        if categorical:
            e, be = np.mean(y!=p), np.mean(y!=b)
        else:
            e, be = np.mean(np.abs(y-p)),np.mean(np.abs(y-b))
        if be > f.scale[j]*1e-10:
            score = relative_skill(e, be, categorical)
            if np.isfinite(score):
                out.append(score)
    return float(np.mean(out))


def tuning(data, out, gs, ss, trials, small=False):
    """
    Fit candidate settings on the anchor training sample and select the best on tuning blocks.
    Save the selections and trial scores.
    """
    dest=Path(out)/'tuning'/f'{gs}-{ss}.json'
    if dest.exists():
        return json.loads(dest.read_text())
    sc=next(s for s in CORE if s.name==ANCHOR)
    schema,truth,raw,_,groups=load(data,sc,gs)
    ids,_=split_blocks(truth,ss,schema,groups)
    # Keep x-band representation via a shuffled uniform training subset.
    ix=ids['train'][:(650 if small else 1500)]
    va=ids['tune'][:(300 if small else 750)]
    f=Features(schema,True).fit(raw[ix])
    base=Floor().fit(raw[ix],f).predict(raw[va]).point
    rows=[]
    chosen={'floor':{}}
    for m in METHODS[1:]:
        rng=np.random.default_rng(np.random.SeedSequence([gs,ss,METHODS.index(m)]))
        settings=[]
        if trials:
            settings.append(DEFAULT[m])
        while len(settings)<trials:
            c=candidate(m,rng)
            if c not in settings:
                settings.append(c)
        best=-np.inf
        chosen[m]=DEFAULT[m]
        for trial,p in enumerate(settings):
            t=perf_counter()
            est=create(m,p,ss).fit(raw[ix],f)
            s=validation_skill(schema,f,raw[va],truth[va],est.predict(raw[va]).point,base)
            rows.append(dict(generator_seed=gs,split_seed=ss,method=m,trial=trial,validation_skill=s,
                             seconds=perf_counter()-t,params=p,**p))
            if s>best:
                chosen[m],best=p,s
    result=dict(selected=chosen,trials=rows)
    write_json(dest,result)
    return result


def svm_demo(raw, truth, ids, f, seed):
    """
    Return the SVM regularisation sweep and the held-out score at the validation-selected
    value of C.
    """
    from sklearn.svm import LinearSVC
    j=f.schema.i('model_id')
    a, deps=f.design(raw)
    use=[j not in d and f.schema.i('random_placebo') not in d for d in deps]
    train=ids['train'][np.isfinite(raw[ids['train'],j])]
    tune=ids['tune'][np.isnan(raw[ids['tune'],j])]
    test=ids['test'][np.isnan(raw[ids['test'],j])]
    from sklearn.metrics import f1_score
    rows=[]
    for c in (.0001,.001,.01,.1,1.,10.,100.):
        est=LinearSVC(C=c,max_iter=10000,dual=False,random_state=seed).fit(a[train][:,use],raw[train,j])
        rows.append(dict(C=c,weight_norm=float(np.linalg.norm(est.coef_)),
                         tune_f1=float(f1_score(truth[tune,j],est.predict(a[tune][:,use]),average='macro'))))
    best=max(rows,key=lambda r:r['tune_f1'])
    est=LinearSVC(C=best['C'],max_iter=10000,dual=False,random_state=seed).fit(a[train][:,use],raw[train,j])
    best['test_f1']=float(f1_score(truth[test,j],est.predict(a[test][:,use]),average='macro'))
    return rows


def run_task(data,out,sc,gs,ss,settings,arms,methods,domain=True,export=False,small=False):
    """
    Fit the selected methods and arms for one scenario and split. Save scores and optional
    prediction ledgers, and return the task filename.
    """
    dest=Path(out)/'tasks'/f'{sc.name}-{gs}-{ss}-{"domain" if domain else "no-gas"}.json'
    if dest.exists():
        return dest.name
    schema,truth,raw,recal,groups=load(data,sc,gs)
    ids,block=split_blocks(truth,ss,schema,groups)
    if small:
        ids={k:v[:n] for (k,v),n in zip(ids.items(),(900,300,450,500))}
    tr,ca,te=(ids[k] for k in ('train','calibration','test'))
    f0=Features(schema,domain).fit(raw[tr])
    base=Floor().fit(raw[tr],f0).predict(raw[te])
    payload={k:[] for k in ('cells','patterns','distribution','tables','consistency','downstream','trace','runtime','data_audit','svm','installation_age','rule_effects')}
    meta=dict(scenario=sc.name,mechanism=sc.mechanism,nominal_missing=sc.missing,noise=sc.noise,
              generator_seed=gs,split_seed=ss,domain_features=domain)
    for part,ind in ids.items():
        for j,c in enumerate(schema.names):
            n_missing=np.isnan(raw[ind,j]).sum()
            payload['data_audit'].append(dict(**meta,partition=part,attribute=c,rows=len(ind),n_missing=int(n_missing),
                                              visible_noise=float((np.isfinite(raw[ind,j]) & (np.abs(raw[ind,j]-truth[ind,j])>1e-9)).mean())))
    if schema.transformer and sc.name==ANCHOR and gs==219 and ss==617 and domain:
        payload['svm']=[dict(**meta,**r) for r in svm_demo(raw,truth,ids,f0,ss)]
    if export:
        export_root=Path(out)/'examples'/sc.name
        export_root.mkdir(parents=True,exist_ok=True)
        audit=pd.DataFrame([(int(r),p,str(block[r])) for p,ind in ids.items() for r in ind],
                           columns=['asset_id','partition','split_block'])
        audit.to_csv(export_root/'partitions.csv',index=False)
        obs=schema.decode(raw[te]); obs.insert(0,'asset_id',te)
        obs.to_csv(export_root/'measured_test.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    compare_arms = schema.transformer and {'model', 'rules'}.issubset(arms)
    model_points = {}
    age_audit = schema.transformer and (sc.name == ANCHOR or sc.mechanism == 'MNAR')
    if age_audit:
        date_col = schema.i('install_date')
        age_edges = np.quantile(truth[:, date_col], [.25, .50, .75])
    for arm in arms:
        train,_=exact_fill(raw[tr],schema,sc.noise,arm=='rules')
        cal_x,cal_rule=exact_fill(recal[ca],schema,sc.noise,arm=='rules')
        test_x,test_rule=exact_fill(raw[te],schema,sc.noise,arm=='rules')
        f=Features(schema,domain).fit(train)
        # Downstream imputation never sees life on any row being transformed.
        if schema.transformer:
            held=np.concatenate([raw[tr],raw[te]]).copy()
            held[:,schema.i('life')]=np.nan
            held=exact_fill(held,schema,sc.noise,arm=='rules')[0]
        for m in methods:
            tick=perf_counter()
            est = create(m, settings[m], ss)
            if m == 'autoencoder' and arm == 'rules':
                est.fit(raw[tr], f, noise=sc.noise, rules=True)
            else:
                est.fit(train, f)
            fit_time=perf_counter()-tick
            predict_x=np.vstack([cal_x,test_x,held]) if schema.transformer else np.vstack([cal_x,test_x])
            pred=est.predict(predict_x)
            fill_time=perf_counter()-tick-fit_time
            cal=pred.subset(slice(0,len(ca)))
            test=pred.subset(slice(len(ca),len(ca)+len(te)))
            visible=np.isfinite(raw[te])
            if not np.array_equal(test.point[visible],raw[te][visible]):
                raise RuntimeError('Observed cells changed')
            conf=calibrate(f,recal[ca],truth[ca],cal_rule,cal,raw[te],test_rule,test)
            cell,pat,dist,decision=evaluate(raw[te],truth[te],test,base,conf,test_rule,f)
            results=dict(cells=cell,patterns=pat,distribution=dist,
                         tables=[table_checks(f,raw[te],truth[te],test.point,ss)],
                         consistency=consistency(f,raw[te],test.point,truth[te]),
                         trace=[dict(step=k+1,value=v) for k,v in enumerate(pred.trace)],
                         runtime=[dict(fit_seconds=fit_time,predict_seconds=fill_time)])
            if schema.transformer:
                fills=pred.point[len(ca)+len(te):]
                results['downstream']=downstream(f,truth[tr],raw[tr],fills[:len(tr)],truth[te],fills[len(tr):])
            if age_audit:
                results['installation_age'] = installation_age_scores(
                    raw[te, date_col], truth[te, date_col], test.point[:, date_col],
                    base.point[:, date_col], age_edges,
                )
            if compare_arms:
                if arm == 'model':
                    model_points[m] = test.point
                else:
                    results['rule_effects'] = rule_effects(
                        schema, raw[te], truth[te], model_points[m], test.point, test_rule, base.point, f0,
                    )
            for key,rows in results.items():
                payload[key].extend(dict(**meta,arm=arm,method=m,**r) for r in rows)
            if export:
                obs=schema.decode(test.point)
                obs.insert(0,'asset_id',te)
                if not schema.transformer:
                    # Year fills were rounded before calibration and scoring.
                    obs[f'asset_age_at_{SNAPSHOT_YEAR}'] = SNAPSHOT_YEAR - obs['installation_year']
                obs.to_csv(export_root/f'{arm}-{m}-filled.csv.gz',index=False,compression={'method':'gzip','mtime':0})
                cell_export(schema,te,raw[te],truth[te],test,conf,test_rule,decision,base).to_csv(
                    export_root/f'{arm}-{m}-cells.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    def serial(v):
        if isinstance(v,dict):
            return {k:serial(x) for k,x in v.items()}
        if isinstance(v,list):
            return [serial(x) for x in v]
        if isinstance(v,(float,np.floating)):
            return float(v) if np.isfinite(v) else None
        if isinstance(v,np.integer):
            return int(v)
        if isinstance(v,np.bool_):
            return bool(v)
        return v
    write_json(dest,serial(payload))
    return dest.name
