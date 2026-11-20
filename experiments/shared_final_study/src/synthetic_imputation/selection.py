"""Audit recorded selection, or perform a new measured-data-only selection run.

The original selection scripts were not supplied. This replacement preserves their
declared scoring rule and candidate settings; its explicit remasking seed is new.
It does not claim to reconstruct unrecorded historical remasking draws.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

MODELS = {'knn':'KNN imputer','ridge':'IterativeImputer','forest':'Random Forest',
          'hgb':'HistGradientBoosting','cat':'CatBoost','mice':'miceforest direct',
          'em':'Gaussian EM','mlp':'Masked MLP'}
TARGETS = ('x','y','install_date','altitude','measured_humid','measured_salt',
           'measured_overgrowth','life','measured_gas_a','measured_gas_b',
           'measured_gas_c','total_gas','leakage')


def records(root, model, round_number=3):
    if model not in MODELS:
        raise ValueError(f'Unknown selection model: {model}')
    folder = Path(root)/'reference_results/tables/7_settings_selection'/(
             'merge_selection' if model in ('em','mlp') else 'clean_selection')
    suffixes = ('','_extra','_final_round')[:round_number]
    paths = [folder/f'{model}{s}.csv' for s in suffixes if (folder/f'{model}{s}.csv').is_file()]
    if not paths:
        raise FileNotFoundError(f'No candidate records for {model} in {folder}')
    return pd.concat([pd.read_csv(p) for p in paths],ignore_index=True),folder


def choose(frame):
    required = {'settings','seconds','random_placebo',*TARGETS}
    if not required <= set(frame):
        raise ValueError(f'Selection records missing {sorted(required-set(frame))}')
    frame = frame.copy()
    if not np.isfinite(frame[[*TARGETS,'random_placebo','seconds']]).all().all():
        raise ValueError('Selection contains failed/non-finite candidate scores')
    frame['settings'] = frame.settings.map(lambda s:json.dumps(json.loads(s),sort_keys=True))
    frame['score'] = frame[list(TARGETS)].mean(axis=1)
    summary = frame.groupby('settings',sort=True).agg(score=('score','mean'),
              placebo_ratio=('random_placebo','mean'),seconds=('seconds','mean'),fold_scenarios=('score','size'))
    summary['eligible'] = summary.placebo_ratio <= 1.01
    eligible = summary[summary.eligible]
    if eligible.empty:
        raise ValueError('No candidate meets the placebo ratio limit of 1.01; report this outcome without changing the rule')
    summary['near_best'] = summary.eligible & (summary.score <= eligible.score.min()+.002)
    winner = summary[summary.near_best].sort_values(['seconds','score'],kind='stable').iloc[0]
    return json.loads(winner.name),summary.reset_index()


def audit(root, models=None, round_number=3):
    results = []
    for model in models or MODELS:
        frame,folder = records(root,model,round_number)
        selected,table = choose(frame)
        final = folder/f'{model}.final.json'
        chosen = folder/f'{model}.chosen.json'
        expected = json.loads((final if round_number==3 and final.exists() else chosen).read_text())
        # Intermediate round 2 did not ship its own chosen JSON.
        comparable = round_number in (1,3)
        results.append({'model':MODELS[model],'round':round_number,'selected':selected,
                        'recorded_final':expected if comparable else None,
                        'matches_recorded':selected==expected if comparable else None,
                        'candidate_settings':len(table),'fold_scenario_records':len(frame)})
    return {'operation':'Audit of supplied candidate scores; no models refitted',
            'rule':'Mean numeric MAE ratio; placebo <=1.01; within 0.002 of best; fastest',
            'models':results}


def refit(root, model, output, round_number=3, rows=None, seed=219, limit=None):
    """New three-fold remasking evaluation; reads measured files only, never truth files."""
    from sklearn.model_selection import KFold
    from threadpoolctl import threadpool_limits
    from .runner import ALL_CHAINS, default_parameters
    from .models import ChainedRidgeImputer
    root,output = Path(root),Path(output)
    output.mkdir(parents=True,exist_ok=True)
    if rows is not None and rows < 90:
        raise ValueError('Use at least 90 rows for a diagnostic selection check')
    if limit is not None and limit < 1:
        raise ValueError('--limit must be positive')
    original,_ = records(root,model,round_number)
    candidates = list(dict.fromkeys(json.dumps(json.loads(s),sort_keys=True) for s in original.settings))
    if limit:
        candidates = candidates[:limit]
    name = MODELS[model]
    data = []
    input_hashes = {}
    for noise in ('0.0','0.05','0.1'):
        scenario = f'completion0.7-noise{noise}'
        path = root/'data/datasets'/f'{scenario}-measured.csv'
        frame = pd.read_csv(path).astype(float)
        input_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        if rows is not None:
            ids = np.sort(np.random.default_rng(seed).choice(len(frame),min(rows,len(frame)),replace=False))
            frame = frame.iloc[ids]
        data.append((scenario,frame))
    saved = []
    for candidate_index,encoded in enumerate(candidates):
        settings = {**default_parameters()[name],**json.loads(encoded)}
        for scenario,frame in data:
            for fold,(train_ids,test_ids) in enumerate(KFold(3,shuffle=True,random_state=seed).split(frame)):
                train, observed = frame.iloc[train_ids],frame.iloc[test_ids]
                hidden = (np.random.default_rng(seed+1000+fold).random(observed.shape)<.10)&observed.notna()
                test = observed.mask(hidden)
                started = perf_counter()
                with threadpool_limits(limits=1):
                    if name=='IterativeImputer':
                        estimates = ChainedRidgeImputer(settings,seed+fold).fit(train).predict(test)
                        probabilities = None
                    else:
                        fitted = ALL_CHAINS[name](settings,seed+fold).fit(train)
                        estimates,probabilities = fitted.predict(test)
                entry = {'settings':encoded,'scenario':scenario,'fold':fold,'seconds':perf_counter()-started}
                for target in (*TARGETS,'random_placebo'):
                    use = hidden[target]
                    if not use.any():
                        raise ValueError(f'No remasked {target} values: increase diagnostic row count')
                    y = observed.loc[use,target].to_numpy()
                    error = np.mean(np.abs(estimates.loc[use,target].to_numpy()-y))
                    base = np.mean(np.abs(train[target].median()-y))
                    if base <= 0:
                        raise ValueError(f'Zero baseline error for {target}; increase diagnostic row count')
                    entry[target] = float(error/base)
                if probabilities is not None:
                    use = hidden.model_id.to_numpy()
                    labels = np.asarray(fitted.classes_)[np.asarray(probabilities).argmax(axis=1)]
                    entry['model_id error'] = float(np.mean(labels[use]!=observed.model_id.to_numpy()[use]))
                saved.append(entry)
                pd.DataFrame(saved).to_csv(output/'candidate_scores.csv',index=False)
        print(f'{name}: {candidate_index+1}/{len(candidates)} candidates complete',flush=True)
    table = pd.DataFrame(saved)
    chosen,summary = choose(table)
    summary.to_csv(output/'selection_summary.csv',index=False)
    (output/'selected.json').write_text(json.dumps(chosen,indent=2))
    manifest = {'operation':'New measured-data-only refit; historical remasking draws were not supplied',
        'model':name,'seed':seed,'rows':rows,'diagnostic_only':rows is not None or limit is not None,
        'round':round_number,'candidates':len(candidates),'truth_files_read':False,
        'extra_mask_rate':.10,'remask_seed_rule':'seed + 1000 + fold',
        'input_hashes':input_hashes,'selected':chosen,'generated_utc':datetime.now(timezone.utc).isoformat()}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    return manifest
