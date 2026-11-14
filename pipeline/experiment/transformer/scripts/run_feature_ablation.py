"""Transformer T3 predictor-set ablation, using the project's Predictive MissForest.

No generator changes, no ground truth leakage, no automatic hyperparameter tuning.
Exactly the same held-out, originally missing target cells are scored in every
retained target. Values are read from the benchmark's 20% MCAR/5% noise pair.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
import pandas as pd
import numpy as np

THIS = Path(__file__).resolve()
EXPERIMENT = THIS.parent.parent
PIPELINE = THIS.parents[3] # /pipeline (script, scripts, transformer, experiment, pipeline)
sys.path.insert(0, str(PIPELINE))
from transformer_noise_masking import (load_experiment_config, make_spatial_split, read_pair)
from src.imputation.predictive_missforest.predictive_missforest import PredictiveMissForest, PredictiveMissForestConfig
import importlib
pmf_module = importlib.import_module("src.imputation.predictive_missforest.predictive_missforest")
from src.evaluation.metrics import evaluate_imputation

ABLATIONS = {
    'all_attributes': (),
    'without_coordinates': ('x', 'y'),
    'without_install_date': ('install_date',),
    'without_model_id': ('model_id',),
    'without_total_gas': ('total_gas',),
    'without_altitude': ('altitude',),
    'without_placebo': ('random_placebo',),
}

def run_one(variant: str, output_dir: Path, *, quick: bool=False, split_seed: int=617):
    if variant not in ABLATIONS: raise ValueError(variant)
    config = load_experiment_config(EXPERIMENT / 'config.yaml')
    remove = set(ABLATIONS[variant])
    numeric = [c for c in config['numeric_columns'] if c not in remove]
    categorical = [c for c in config['categorical_columns'] if c not in remove]
    columns = numeric + categorical
    measured, truth = read_pair(EXPERIMENT / 'dataset', 0.8, 0.05)
    partition = make_spatial_split(truth, split_seed, False)
    train_idx = partition.index[partition.partition.eq('train')]
    test_idx = partition.index[partition.partition.eq('test')]
    fit = measured.loc[train_idx, columns].copy()
    test = measured.loc[test_idx, columns].copy()
    test_truth = truth.loc[test_idx, columns].copy()
    test_mask = test.isna()
    settings = dict(config['effective_methods']['predictive_missforest'])
    if quick:
        settings.update(n_estimators=15, n_passes=1)
    model_cfg = PredictiveMissForestConfig.from_mapping(settings)
    model = PredictiveMissForest(numeric, categorical, config=model_cfg)
    print(f'Running {variant}: {len(numeric)} numeric, {len(categorical)} categorical; {settings["n_estimators"]} trees x {settings["n_passes"]} passes',flush=True)
    # The standard model seeds a target as random_state + its column index.
    # Dropping a feature would otherwise alter the random forest seed for
    # every later target. In this experimental wrapper ONLY, preserve each
    # target's seed based on its index in the ORIGINAL 15-column model.
    # No production module is modified; the original constructor is restored.
    complete_order = list(config['numeric_columns'] + config['categorical_columns'])
    stable_seeds = {c: int(settings['random_state']) + complete_order.index(c) for c in columns}
    original_builder = pmf_module._build_pipeline
    def fixed_seed_builder(*args, **kwargs):
        target = kwargs['target'] if 'target' in kwargs else args[0]
        kwargs['seed'] = stable_seeds[target]
        return original_builder(*args, **kwargs)
    start = perf_counter()
    pmf_module._build_pipeline = fixed_seed_builder
    try:
        model.fit(fit)
    finally:
        pmf_module._build_pipeline = original_builder
    fit_time = perf_counter()-start
    start = perf_counter()
    predicted = model.transform(test)
    predict_time = perf_counter()-start
    assert list(predicted.columns)==list(test.columns)
    if predicted.isna().any().any(): raise AssertionError('Imputed table has missing values')
    for column in columns:
        orig = test[column]
        keep = orig.notna()
        if not predicted.loc[keep,column].equals(orig.loc[keep]):
            # Numeric dtype can change but recorded values must be the same
            if not np.array_equal(predicted.loc[keep,column].to_numpy(),orig.loc[keep].to_numpy()):
                raise AssertionError(f'Observed values modified for {column}')
    scores = evaluate_imputation(test_truth, predicted, test_mask, numeric, categorical)
    scores.insert(0, 'ablation',variant)
    scores.insert(1,'removed_features',','.join(ABLATIONS[variant]))
    scores['fit_seconds']=fit_time
    scores['transform_seconds']=predict_time
    scores['n_estimators']=settings['n_estimators']
    scores['n_passes']=settings['n_passes']
    # Verify each per-target n_masked is the original source mask. This prevents silent comparison drift.
    for _,r in scores.iterrows():
        assert int(r['n_masked']) == int(test_mask[r['variable']].sum())
    # Important for interpretation: record how often the removed predictors would have been observed
    availability=[]
    for target in columns:
        idx = test_mask[target]
        for source in ABLATIONS[variant]:
            availability.append({'ablation':variant,'target':target,'removed_source':source,
                'n_scored_target':int(idx.sum()),
                'fraction_source_observed_on_scored':float(test.loc[idx,source].notna().mean()) if source in test else float(measured.loc[test_idx,source].loc[idx].notna().mean())})
    output_dir.mkdir(parents=True,exist_ok=True)
    scores.to_csv(output_dir/f'{variant}_metrics.csv',index=False)
    pd.DataFrame(availability).to_csv(output_dir/f'{variant}_source_availability.csv',index=False)
    meta = {'variant':variant,'removed_predictors':list(ABLATIONS[variant]),'train_rows':len(fit),'test_rows':len(test),
        'completion':0.8,'noise_amplitude':0.05,'split_seed':split_seed,'training_excludes_test':True,
        'settings':settings,'seed_policy':'random_state + target index in original 15-column schema',
        'effective_target_seeds':stable_seeds,'fit_seconds':fit_time,'transform_seconds':predict_time,
        'generated_utc':datetime.now(timezone.utc).isoformat(),
        'model_module':str(Path(sys.modules[PredictiveMissForest.__module__].__file__).resolve()),
        'source_input_sha256':hashlib.sha256((EXPERIMENT/'dataset'/'completion0.8-noise0.05-measured.csv').read_bytes()).hexdigest(),
        'reference_truth_used_only_for_scoring':True,'scoring_cells':'held-out test, originally NaN',
        'note':'One generator and one split; differences not statistical confidence intervals.'}
    (output_dir/f'{variant}_manifest.json').write_text(json.dumps(meta,indent=2))
    print('Finished',variant,'fit %.1fs'%fit_time,'predict %.1fs'%predict_time, flush=True)
    return scores

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant',choices=list(ABLATIONS),required=True)
    p.add_argument('--output-dir',type=Path,default=EXPERIMENT/'outputs'/'feature_ablation_seed617')
    p.add_argument('--quick',action='store_true',help='Smoke test, 15 trees and one pass; not report-ready')
    args=p.parse_args()
    run_one(args.variant,args.output_dir,quick=args.quick)

if __name__=='__main__':main()
