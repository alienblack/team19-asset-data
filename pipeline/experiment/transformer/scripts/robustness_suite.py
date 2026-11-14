"""Unified transformer robustness experiments (T4, T5, T6), held-out protocol.

This module is intentionally isolated under experiment/transformer. It reuses
T2's 10x10 spatial split and original paired synthetic files. Ground truth is
passed only to scoring, except in the explicitly labelled count-preserving
T4 generator where it simulates newly exposed observed values. That test uses
a common reconstructed noisy input table, never truth as test predictors.

Protocol boundaries:
  - T4 primary: frozen fitted imputer; co-mask source fields on target-missing
    test rows. This changes predictor visibility, NOT the anchor target cells.
    It is not a fixed-overall-missingness comparison.
  - T4 secondary: count-preserving grouped masks (separate fits). Newly revealed
    original-missing cells are simulated from generator truth + synthetic noise.
  - T5: swap observed test predictors only; paired, fixed target cells/model.
  - T6: fit for each training noise, apply to multiple same-mask test noise
    levels; frozen fit is reused for all applications.

Methods: median_mode, chained_ridge, lightgbm_chained, predictive_missforest,
optional tabpfn_unsupervised. Original MissForest is strictly transductive and
is not presented as a held-out/frozen model. The original T2 joint-table
comparison remains available in transformer_noise_masking.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from time import perf_counter
import json
import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
PIPELINE = EXPERIMENT.parents[1]
for root in (str(HERE), str(PIPELINE)):
    if root not in sys.path:
        sys.path.insert(0, root)
from transformer_noise_masking import NUMERIC, CATEGORICAL, NOISY, read_pair, make_spatial_split, load_experiment_config
from comparison_models import ModelSpec, make_candidate, ChainedConditionalImputer, TabPFNExperimentalImputer
from run_model_comparison import _score, _validate_output

METHODS = ('median_mode', 'chained_ridge', 'lightgbm_chained', 'predictive_missforest', 'tabpfn_unsupervised')
GROUPS = {
    'gas': ('measured_gas_a', ('measured_gas_b', 'measured_gas_c')),
    'coordinates': ('altitude', ('x', 'y')),
    'environment': ('measured_humid', ('measured_salt', 'measured_overgrowth')),
}


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _schema() -> ModelSpec:
    return ModelSpec(numeric=NUMERIC, categorical=CATEGORICAL, seed=219)


def reference_split(*, quick: bool = False, split_seed: int = 617, noise: float = .05):
    noise=float(noise)
    measured, truth = read_pair(EXPERIMENT / 'dataset', .8, noise)
    split = make_spatial_split(truth, split_seed, quick)
    train_idx = split.index[split.partition.eq('train')]
    test_idx = split.index[split.partition.eq('test')]
    assert len(set(train_idx) & set(test_idx)) == 0
    return (measured.loc[train_idx].copy(), measured.loc[test_idx].copy(),
            truth.loc[test_idx].copy(), split)


def co_mask_source_on_anchor(test: pd.DataFrame, anchor: str, sources: tuple[str, ...]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hide sources where anchor already missing; never reveal any values."""
    if anchor in sources or not sources or any(c not in test for c in (anchor, *sources)):
        raise ValueError('Invalid anchor or source columns')
    output = test.copy(deep=True)
    anchor_missing = test[anchor].isna()
    changes = []
    for source in sources:
        add = anchor_missing & output[source].notna()
        for row in test.index[add]:
            changes.append({'row_position': int(row), 'column': source, 'action': 'additional_missing'})
        output.loc[add, source] = np.nan
    if not output[anchor].equals(test[anchor]):
        raise AssertionError('Anchor target changed')
    if not output.isna().to_numpy()[test.isna().to_numpy()].all():
        raise AssertionError('Previously missing cells were uncovered')
    return output, pd.DataFrame(changes, columns=['row_position','column','action'])


def _seeded_swap(frame: pd.DataFrame, column: str, prevalence: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replace selected observed values using different observed same-column donors."""
    if column not in frame or not 0 <= prevalence <= 1:
        raise ValueError('Unknown column or prevalence outside [0,1]')
    result = frame.copy(deep=True)
    visible = frame.index[frame[column].notna()].to_numpy()
    size = int(round(len(visible) * prevalence))
    rng = np.random.default_rng(seed)
    selected = rng.choice(visible, size=size, replace=False)
    pool = frame.loc[visible, column].to_numpy()
    log = []
    for row in selected:
        original = frame.at[row,column]
        valid = np.flatnonzero(pool != original)
        if not len(valid):
            continue
        candidate = pool[int(rng.choice(valid))]
        result.at[row,column] = candidate
        log.append({'row_position':int(row), 'column':column,
                    'original':str(original),'replacement':str(candidate),
                    'requested_prevalence':float(prevalence)})
    if not result[column].isna().equals(frame[column].isna()):
        raise AssertionError('Corruption changed missingness')
    return result, pd.DataFrame(log, columns=['row_position','column','original','replacement','requested_prevalence'])


def scaled_test_noise(zero: pd.DataFrame, noise05: pd.DataFrame, severity: float) -> pd.DataFrame:
    """Scale original observed ±5% noise draws to a new amplitude (no truth)."""
    if severity < 0 or severity > .99:
        raise ValueError('Noise amplitude must be in [0,0.99]')
    if not zero.isna().equals(noise05.isna()):
        raise ValueError('Mask mismatch between supplied scenarios')
    result = zero.copy(deep=True)
    for c in NOISY:
        visible = zero[c].notna()
        result.loc[visible,c] = (zero.loc[visible,c].astype(float) +
            (noise05.loc[visible,c].astype(float) - zero.loc[visible,c].astype(float)) * (severity/.05))
    return result


def _draw_reexposed_values(original: pd.DataFrame, truth: pd.DataFrame, column: str,
                           rows: pd.Index, *, noise: float, rng: np.random.Generator):
    """Synthetic observed-value simulation for newly revealed cells in T4 only."""
    values = truth.loc[rows, column].copy()
    if column in NOISY and noise:
        # The originally masked cells have no recorded noisy values. Draw fresh
        # multiplicative errors; use the same resulting measured pool in both
        # comparator conditions, and explicitly report this synthetic step.
        values = values.astype(float) * (1 + rng.uniform(-noise, noise, size=len(rows)))
    return values


def count_preserving_grouped_pair(measured: pd.DataFrame, truth: pd.DataFrame,
                                  *, groups: dict = GROUPS, seed: int = 219,
                                  noise: float = .05) -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    """Make paired MCAR/grouped conditions with EXACT per-column mask counts.

    The original MCAR condition is retained exactly. Newly exposed cells in the
    grouped condition require synthetic observed values: they are drawn from
    truth plus fresh documented measurement noise (not scored before masking).
    The two conditions share every observed value on their intersection.
    """
    if not measured.index.equals(truth.index) or not measured.columns.equals(truth.columns):
        raise ValueError('Measured/truth frame mismatch')
    rng = np.random.default_rng(seed)
    base_mask = measured.isna()
    grouped_mask = base_mask.copy(deep=True)
    if len({c for g in groups.values() for c in (g[0],*g[1])}) != sum(1+len(g[1]) for g in groups.values()):
        raise ValueError('T4 grouped columns must be disjoint')
    for group_name,(anchor,sources) in groups.items():
        anchor_positions = measured.index[base_mask[anchor]].to_numpy()
        for c in sources:
            count = int(base_mask[c].sum())
            take_anchor = rng.choice(anchor_positions, size=min(count,len(anchor_positions)),replace=False)
            count_left = count-len(take_anchor)
            outside = measured.index.difference(pd.Index(anchor_positions)).to_numpy()
            extra = rng.choice(outside, size=count_left,replace=False) if count_left else np.array([],dtype=int)
            grouped_mask[c] = measured.index.isin(np.concatenate([take_anchor,extra]))
    if not grouped_mask.sum().equals(base_mask.sum()):
        raise AssertionError('Per-column missing counts must be exactly preserved')
    grouped = measured.copy(deep=True)
    reexposed = []
    for col in measured.columns:
        expose = base_mask[col] & ~grouped_mask[col]
        hide = ~base_mask[col] & grouped_mask[col]
        if expose.any():
            ix = measured.index[expose]
            grouped.loc[ix,col] = _draw_reexposed_values(measured,truth,col,ix,noise=noise,rng=rng)
            reexposed += [{'row_position':int(i), 'column':col, 'source':'synthetic_reexposure'} for i in ix]
        grouped.loc[hide,col] = np.nan
    if not grouped.isna().equals(grouped_mask):
        raise AssertionError('Grouped data mask integrity failed')
    for anchor,_ in groups.values():
        if not grouped[anchor].equals(measured[anchor]):
            raise AssertionError(f'Anchor {anchor} must be identical')
    return measured.copy(deep=True), grouped, pd.DataFrame(reexposed, columns=['row_position','column','source'])


def _make_imputer(method: str, *, quick: bool, comparison_config: dict, seed: int):
    if method not in METHODS:
        raise ValueError(f'Unsupported held-out method: {method}')
    spec = ModelSpec(numeric=NUMERIC,categorical=CATEGORICAL,seed=seed)
    if method == 'predictive_missforest':
        from src.imputation.predictive_missforest.predictive_missforest import PredictiveMissForest, PredictiveMissForestConfig
        params = dict(load_experiment_config(EXPERIMENT/'config.yaml')['effective_methods']['predictive_missforest'])
        if quick:
            params.update(n_estimators=10,n_passes=1)
        return PredictiveMissForest(NUMERIC,CATEGORICAL,config=PredictiveMissForestConfig.from_mapping(params))
    imputer = make_candidate(method,spec,comparison_config)
    if quick:
        if isinstance(imputer,ChainedConditionalImputer):
            imputer.passes=1
            imputer.lgbm_estimators=min(15,imputer.lgbm_estimators)
        if isinstance(imputer,TabPFNExperimentalImputer):
            imputer.max_context_rows=min(200,imputer.max_context_rows)
    return imputer


def _apply(model, records:pd.DataFrame) -> pd.DataFrame:
    result=model.transform(records)
    # The pipeline PMF returns DataFrame (its own fit/transform API).
    if not isinstance(result,pd.DataFrame):
        result=result.imputed
    return result


def _model_id(method: str, noise: float, split_seed: int, quick: bool, seed: int,
              comparison_config: dict) -> str:
    source_fingerprint=sha256((EXPERIMENT/'config.yaml').read_bytes() +
                              (HERE/'comparison_models.py').read_bytes() +
                              (HERE/'robustness_suite.py').read_bytes() +
                              (PIPELINE/'src/imputation/predictive_missforest/predictive_missforest.py').read_bytes()).hexdigest()[:12]
    payload=json.dumps(dict(method=method,noise=noise,split_seed=split_seed,
                            quick=quick,seed=seed,options=comparison_config.get(method,{}),
                            code_fingerprint=source_fingerprint),sort_keys=True,default=str)
    return f'{method}_train{noise:.2f}_split{split_seed}_'+sha256(payload.encode()).hexdigest()[:12]


@dataclass
class FitResult:
    imputer:object
    fit_seconds:float
    loaded:bool
    train_noise:float


class RobustnessSuite:
    def __init__(self, *, output_dir: Path, methods:tuple[str,...]=('median_mode','chained_ridge','lightgbm_chained'),
                 quick:bool=False, split_seed:int=617, seed:int=219, save_models:bool=True,
                 bootstrap_replicates:int=400):
        if any(m not in METHODS for m in methods) or not methods or len(set(methods))!=len(methods):
            raise ValueError(f'Methods must be unique and chosen from {METHODS}')
        self.output_dir=Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True,exist_ok=True)
        self.quick=quick;self.methods=tuple(methods);self.split_seed=split_seed
        self.seed=seed;self.save_models=save_models
        self.bootstrap_replicates=bootstrap_replicates
        import yaml
        self.config=yaml.safe_load((EXPERIMENT/'model_comparison'/'config.yaml').read_text())['methods']
        self.train_cache:dict[tuple[str,float],FitResult]={}
        self.score_files=[]
        self.manifest=dict(created_utc=_utc(),protocol='unseen_records',quick=quick,methods=methods,
             split_seed=split_seed,seed=seed,source='existing synthetic transformer scenario pairs',
             note='Ground truth is never passed to fitted imputers; count-preserving T4 explicitly generates synthetic reexposed values.',
             tabpfn_note='TabPFN extension is an experimental in-context imputer; an impute() call can fit conditional predictors; not equivalent to frozen sklearn replay.',
             software=dict(python=sys.version.split()[0],pandas=pd.__version__))
        (self.output_dir/'models').mkdir(exist_ok=True)
        self._write_manifest()

    def _write_manifest(self):
        (self.output_dir/'run_manifest.json').write_text(json.dumps(self.manifest,indent=2,default=str))

    def _frames(self,noise:float):
        if noise not in (0,.05,.1):
            raise ValueError('Fit noise supported at source scenarios 0, .05 and .1')
        return reference_split(quick=self.quick,split_seed=self.split_seed,noise=float(noise))

    def model(self, method: str, noise: float=.05) -> FitResult:
        key=(method,noise)
        if key in self.train_cache:
            return self.train_cache[key]
        if method not in self.methods and method!='median_mode':
            raise ValueError(f'Method not enabled: {method}')
        train,_,_,_=self._frames(noise)
        model_key=_model_id(method,noise,self.split_seed,self.quick,self.seed,self.config)
        path=self.output_dir/'models'/f'{model_key}.joblib'
        if path.exists():
            imputer=joblib.load(path)
            result=FitResult(imputer,0,True,noise)
        else:
            imputer=_make_imputer(method,quick=self.quick,comparison_config=self.config,seed=self.seed)
            start=perf_counter();imputer.fit(train);elapsed=perf_counter()-start
            result=FitResult(imputer,elapsed,False,noise)
            if self.save_models and method!='tabpfn_unsupervised':
                joblib.dump(imputer,path,compress=3)
        self.train_cache[key]=result
        return result

    def _report(self, *, label: str, method: str, original: pd.DataFrame,
                predictions: pd.DataFrame, truth: pd.DataFrame, split: pd.DataFrame,
                baseline_predictions:pd.DataFrame, fit_seconds:float, transform_seconds:float,
                scored_targets:tuple[str,...] | None=None,
                details:dict | None=None):
        _validate_output(original,predictions,_schema())
        spec=_schema()
        if scored_targets is not None:
            spec=ModelSpec(tuple(c for c in scored_targets if c in NUMERIC),
                           tuple(c for c in scored_targets if c in CATEGORICAL),self.seed)
        metrics,cells=_score(original,predictions,truth,split,spec,method=method,
                            completion=.8,noise=.05,seed=self.split_seed,
                            fit_seconds=fit_seconds,transform_seconds=transform_seconds,
                            baseline_filled=baseline_predictions,export_cells=True)
        details=details or {}
        for df in (metrics,cells):
            df['condition']=label
            for k,v in details.items():df[k]=v
        short=sha256((label+method).encode()).hexdigest()[:8]
        filename=f'{label.replace("/","_")}_{method}_{short}'
        metrics.to_csv(self.output_dir/f'metrics_{filename}.csv',index=False)
        cells.to_csv(self.output_dir/f'cells_{filename}.csv',index=False)
        self.score_files.append((label,method,filename))
        return metrics,cells

    def _reference(self,method:str, *, noise:float=.05):
        train,test,truth,split=self._frames(noise)
        fitted=self.model(method,noise)
        start=perf_counter();pred=_apply(fitted.imputer,test);duration=perf_counter()-start
        if method=='median_mode':base=pred
        else:
            bm=self.model('median_mode',noise);base=_apply(bm.imputer,test)
        return fitted,train,test,truth,split,pred,base,duration

    def t4(self, *, groups:dict=GROUPS, include_count_preserving:bool=False):
        """Frozen predictor-co-missing tests, optionally separate refitted count-controlled tests."""
        results=[];reference={}
        for method in self.methods:
            fitted,_,test,truth,split,reference_pred,base,baseline_seconds=self._reference(method)
            for group,(anchor,sources) in groups.items():
                key=(method,anchor)
                if key not in reference:
                    reference[key]=self._report(label=f't4_{group}_mcar',method=method,
                        original=test,predictions=reference_pred,truth=truth,split=split,
                        baseline_predictions=base,fit_seconds=fitted.fit_seconds,
                        transform_seconds=baseline_seconds,scored_targets=(anchor,),
                        details={'experiment':'T4','group':group,'mask_pattern':'MCAR','protocol_note':'Frozen model, anchor cells unchanged'})
                modified,added=co_mask_source_on_anchor(test,anchor,tuple(sources))
                start=perf_counter();predict=_apply(fitted.imputer,modified);duration=perf_counter()-start
                baseline_after=_apply(self.model('median_mode').imputer,modified)
                # Grouping only adds source NAs, so anchor baseline MUST stay identical.
                assert np.array_equal(base.loc[test[anchor].isna(),anchor].to_numpy(),
                                      baseline_after.loc[test[anchor].isna(),anchor].to_numpy())
                scores,cells=self._report(label=f't4_{group}_co_missing',method=method,
                       original=modified,predictions=predict,truth=truth,split=split,
                       baseline_predictions=baseline_after,fit_seconds=0,transform_seconds=duration,
                       scored_targets=(anchor,),details={'experiment':'T4','group':group,
                       'mask_pattern':'conditional_co_missing','additional_missing_predictor_cells':len(added),
                       'protocol_note':'Added source missingness; overall masking rate NOT held fixed'})
                added.to_csv(self.output_dir/f't4_{group}_additional_mask.csv',index=False)
                results.append(scores)
        if include_count_preserving:
            # A separate experiment; unlike T4 primary it refits on new train masks.
            measured,truth_all=read_pair(EXPERIMENT/'dataset',.8,.05)
            mcar,grouped,reexposed=count_preserving_grouped_pair(measured,truth_all,groups=groups,seed=self.seed)
            grouped.to_csv(self.output_dir/'t4_count_preserving_measured.csv',index=False)
            reexposed.to_csv(self.output_dir/'t4_reexposed_cells.csv',index=False)
            split=make_spatial_split(truth_all,self.split_seed,self.quick)
            train_ix=split.index[split.partition=='train'];test_ix=split.index[split.partition=='test']
            new_train=grouped.loc[train_ix];new_test=grouped.loc[test_ix];new_truth=truth_all.loc[test_ix]
            baseline_model=_make_imputer('median_mode',quick=self.quick,comparison_config=self.config,seed=self.seed).fit(new_train)
            baseline_new=_apply(baseline_model,new_test)
            for method in self.methods:
                candidate=_make_imputer(method,quick=self.quick,comparison_config=self.config,seed=self.seed)
                start=perf_counter();candidate.fit(new_train);fit_seconds=perf_counter()-start
                start=perf_counter();prediction=_apply(candidate,new_test);elapsed=perf_counter()-start
                for group,(anchor,sources) in groups.items():
                    scores,_=self._report(label=f't4_{group}_count_preserving',method=method,
                        original=new_test,predictions=prediction,truth=new_truth,split=split,
                        baseline_predictions=baseline_new,fit_seconds=fit_seconds,
                        transform_seconds=elapsed,scored_targets=(anchor,),
                        details={'experiment':'T4','group':group,'mask_pattern':'count_preserving_refit',
                        'protocol_note':'Per-column counts matched; newly revealed values simulated from truth plus synthetic noise'})
                    results.append(scores)
        return pd.concat(results,ignore_index=True) if results else pd.DataFrame()

    def t5(self, *, predictors=('model_id','install_date'), prevalences=(0.,.05,.10,.25),
           add_training_corruption:bool=False):
        """Controlled test-time predictor swaps, paired on same masked target cells."""
        rows=[]
        for method in self.methods:
            fit,_,test,truth,split,reference_pred,baseline,initial_seconds=self._reference(method)
            clean_metrics,_=self._report(label='t5_clean',method=method,original=test,
                predictions=reference_pred,truth=truth,split=split,
                baseline_predictions=baseline,fit_seconds=fit.fit_seconds,
                transform_seconds=initial_seconds,
                details={'experiment':'T5','predictor':'none','prevalence':0.})
            rows.append(clean_metrics)
            for column in predictors:
                for prevalence in prevalences:
                    if prevalence==0:continue
                    modified,log=_seeded_swap(test,column,float(prevalence),
                                            seed=self.seed+int(prevalence*10000)+list(test.columns).index(column))
                    start=perf_counter();pred=_apply(fit.imputer,modified);elapsed=perf_counter()-start
                    # Always use clean-model baseline. Test observed predictors do not
                    # influence a fixed, training-only median/mode statistic.
                    new_baseline=_apply(self.model('median_mode').imputer,modified)
                    for target in test.columns:
                        missing=test[target].isna()
                        if not np.array_equal(baseline.loc[missing,target],new_baseline.loc[missing,target]):
                            raise AssertionError('Baseline changed under observed-value swap')
                    metrics,cells=self._report(label=f't5_{column}_swap_{prevalence:.2f}',method=method,
                        original=modified,predictions=pred,truth=truth,split=split,
                        baseline_predictions=new_baseline,fit_seconds=0,transform_seconds=elapsed,
                        details={'experiment':'T5','predictor':column,'prevalence':float(prevalence),
                                 'n_corrupted_observed':len(log)})
                    touched=set(log.row_position)
                    cells['row_predictor_corrupted']=cells.row_position.isin(touched)
                    # Add clean-matched per-cell error changes, NOT cross-cohort MAE differences.
                    clean_file=[f for l,m,f in self.score_files if l=='t5_clean' and m==method][-1]
                    clean_cells=pd.read_csv(self.output_dir/f'cells_{clean_file}.csv')
                    pair=cells.merge(clean_cells[['row_position','variable','absolute_error']].rename(
                        columns={'absolute_error':'reference_absolute_error'}),on=['row_position','variable'],
                        how='left',validate='one_to_one')
                    pair['paired_delta_absolute_error']=pair.absolute_error-pair.reference_absolute_error
                    metric_file=self.score_files[-1][2]
                    pair.to_csv(self.output_dir/f'cells_{metric_file}.csv',index=False)
                    log.to_csv(self.output_dir/f'corruption_{column}_{prevalence:.2f}.csv',index=False)
                    rows.append(metrics)
        if add_training_corruption:
            # A separate model-level-contamination experiment; only model_id @25%.
            train,test,truth,split=self._frames(.05)
            train_corrupt,log=_seeded_swap(train,'model_id',.25,self.seed+951)
            log.to_csv(self.output_dir/'t5_training_corruption_log.csv',index=False)
            for method in self.methods:
                model=_make_imputer(method,quick=self.quick,comparison_config=self.config,seed=self.seed)
                start=perf_counter();model.fit(train_corrupt);fit_seconds=perf_counter()-start
                for apply_changed in (False,True):
                    new_test, testlog=(_seeded_swap(test,'model_id',.25,self.seed+952) if apply_changed
                                  else (test,pd.DataFrame()))
                    start=perf_counter();pred=_apply(model,new_test);duration=perf_counter()-start
                    base=_apply(self.model('median_mode').imputer,new_test)
                    metrics,_=self._report(label=f't5_train_corrupt_apply_{int(apply_changed)}',method=method,
                         original=new_test,predictions=pred,truth=truth,split=split,baseline_predictions=base,
                         fit_seconds=fit_seconds,transform_seconds=duration,
                         details={'experiment':'T5','predictor':'model_id','prevalence':.25,
                                  'train_corrupt':True,'apply_corrupt':bool(apply_changed)})
                    rows.append(metrics)
        return pd.concat(rows,ignore_index=True)

    def t6(self, *, train_noises=(0.,.05,.1), apply_noises=(0.,.05,.1,.2,.4),
           methods: tuple[str,...]|None=None):
        """Training × application noise grid with *frozen* fitted model per row."""
        result=[]
        selected=methods or self.methods
        # verify identical partitions/masks for 0 and 5% noise
        _,test0,truth0,split0=self._frames(0)
        _,test05,truth05,split05=self._frames(.05)
        assert test0.isna().equals(test05.isna())
        assert truth0.equals(truth05)
        for method in selected:
            for train_noise in train_noises:
                fitted=self.model(method,float(train_noise))
                baseline=self.model('median_mode',float(train_noise))
                for apply_noise in apply_noises:
                    if apply_noise in (0.,.05,.1):
                        _,test,truth,split=self._frames(float(apply_noise))
                    else:
                        test=scaled_test_noise(test0,test05,float(apply_noise))
                        truth=truth0;split=split0
                    start=perf_counter();pred=_apply(fitted.imputer,test);elapsed=perf_counter()-start
                    baseline_pred=_apply(baseline.imputer,test)
                    scores,_=self._report(label=f't6_train{train_noise:.2f}_apply{apply_noise:.2f}',
                         method=method,original=test,predictions=pred,truth=truth,split=split,
                         baseline_predictions=baseline_pred,fit_seconds=fitted.fit_seconds,
                         transform_seconds=elapsed,details={'experiment':'T6','train_noise':float(train_noise),
                         'apply_noise':float(apply_noise),'frozen_model':method!='tabpfn_unsupervised',
                         'protocol_note':'Same masks and synthetic population; TabPFN conditional predictors may be fitted during impute'})
                    result.append(scores)
        return pd.concat(result,ignore_index=True)

    def consolidate(self):
        all_metrics=[];all_cells=[]
        for label,method,f in self.score_files:
            m=self.output_dir/f'metrics_{f}.csv';c=self.output_dir/f'cells_{f}.csv'
            if m.exists():all_metrics.append(pd.read_csv(m))
            if c.exists():all_cells.append(pd.read_csv(c))
        metrics=pd.concat(all_metrics,ignore_index=True) if all_metrics else pd.DataFrame()
        cells=pd.concat(all_cells,ignore_index=True) if all_cells else pd.DataFrame()
        if not metrics.empty:
            metrics.to_csv(self.output_dir/'all_per_target_metrics.csv',index=False)
            subset=metrics.loc[(metrics.variable_type=='numeric') & (metrics.variable!='random_placebo')]
            summaries=subset.groupby(['experiment','condition','method'],dropna=False)['numeric_skill'].mean().reset_index(name='mean_numeric_skill_ex_placebo')
            summaries.to_csv(self.output_dir/'method_condition_summary.csv',index=False)
        if not cells.empty:
            cells.to_csv(self.output_dir/'all_masked_cells.csv',index=False)
        self.manifest['completed_conditions']=[{'condition':k,'method':m} for k,m,_ in self.score_files]
        self.manifest['finished_utc']=_utc()
        self._write_manifest()
        return metrics,cells


def paired_block_skill_interval(cells: pd.DataFrame, method_a:str,method_b:str,
                                 variable:str, *, n_boot:int=1000,seed:int=617):
    """Paired spatial-block bootstrap for *one condition, one target*.

    Input must be from the SAME condition, same originally masked target cells
    and same baseline; uncertainty is conditional on this fitted run.
    """
    data=cells.loc[cells.variable.eq(variable) & cells.method.isin([method_a,method_b])].copy()
    if data.empty:raise ValueError('No matching scored cells')
    if variable=='model_id':
        raise ValueError('Use categorical metrics for model_id, not numerical skill')
    key=['row_position','variable']
    a=data.loc[data.method==method_a]
    b=data.loc[data.method==method_b]
    pair=a.merge(b[key+['absolute_error','baseline_absolute_error']],on=key,
                 suffixes=('_a','_b'),validate='one_to_one')
    if len(pair)!=len(a) or len(pair)!=len(b):raise AssertionError('Scored cells differ')
    if not np.allclose(pair.baseline_absolute_error_a,pair.baseline_absolute_error_b):
        raise AssertionError('Different baseline errors: cannot compare skills')
    pair['block']=pair.x_tile.astype(str)+'_'+pair.y_tile.astype(str)
    blocks={k:v for k,v in pair.groupby('block',sort=True)}
    keys=list(blocks)
    if len(keys)<2:raise ValueError('Need multiple spatial blocks')
    rng=np.random.default_rng(seed)
    estimates=[]
    for _ in range(n_boot):
        sample=pd.concat([blocks[k] for k in rng.choice(keys,size=len(keys),replace=True)])
        denom=float(sample.baseline_absolute_error_a.mean())
        if denom<=0:continue
        estimates.append(float((sample.absolute_error_b.mean()-sample.absolute_error_a.mean())/denom))
    point=float((pair.absolute_error_b.mean()-pair.absolute_error_a.mean())/pair.baseline_absolute_error_a.mean())
    return {'method_a':method_a,'method_b':method_b,'variable':variable,
            'delta_skill_a_minus_b':point,'ci_low':float(np.quantile(estimates,.025)),
            'ci_high':float(np.quantile(estimates,.975)),
            'n_blocks':len(keys),'n_masked':len(pair),'n_boot':len(estimates)}


def paired_condition_delta_skill(cells: pd.DataFrame, *, method:str, variable:str,
                                  condition_reference:str, condition_intervention:str,
                                  n_boot:int=1000, seed:int=617) -> dict:
    """Paired spatial-block CI for an intervention compared with its reference.

    Returns skill(reference) - skill(intervention), i.e. baseline-scaled
    deterioration. Requires EXACT same target evaluation cells, truth and
    baseline predictions; intended for the primary T4 and T5 matched designs.
    """
    selected=cells.loc[(cells.method==method)&(cells.variable==variable)&
                       cells.condition.isin([condition_reference,condition_intervention])]
    if selected.empty:raise ValueError('No matching intervention observations')
    reference=selected[selected.condition==condition_reference]
    changed=selected[selected.condition==condition_intervention]
    base_cols=['row_position','variable','absolute_error','baseline_absolute_error','truth']
    merged=reference.merge(changed[base_cols],on=['row_position','variable'],
                           how='inner',validate='one_to_one',suffixes=('_ref','_changed'))
    if len(merged)!=len(reference) or len(merged)!=len(changed):
        raise AssertionError('Intervention changed target evaluation cells')
    if not np.allclose(merged.baseline_absolute_error_ref,merged.baseline_absolute_error_changed):
        raise AssertionError('Condition baselines differ: not directly paired')
    if not np.array_equal(merged.truth_ref.astype(str).to_numpy(),merged.truth_changed.astype(str).to_numpy()):
        raise AssertionError('Condition ground truth differs')
    if variable in CATEGORICAL:
        raise ValueError('This helper is for numerical targets only')
    merged['cluster']=merged.x_tile.astype(str)+'_'+merged.y_tile.astype(str)
    groups={key:part for key,part in merged.groupby('cluster')}
    names=list(groups)
    if len(names)<2:raise ValueError('Need at least two held-out spatial blocks')
    rng=np.random.default_rng(seed)
    samples=[]
    for _ in range(n_boot):
        block_sample=pd.concat([groups[k] for k in rng.choice(names,size=len(names),replace=True)])
        denom=block_sample.baseline_absolute_error_ref.mean()
        if denom>0:
            samples.append((block_sample.absolute_error_changed.mean()-block_sample.absolute_error_ref.mean())/denom)
    denom=merged.baseline_absolute_error_ref.mean()
    if not samples or denom<=0:raise ValueError('No positive baseline error')
    return dict(method=method,variable=variable,reference=condition_reference,
                intervention=condition_intervention,
                skill_lost=float((merged.absolute_error_changed.mean()-merged.absolute_error_ref.mean())/denom),
                ci_low=float(np.quantile(samples,.025)),ci_high=float(np.quantile(samples,.975)),
                n_blocks=len(groups),n_masked=len(merged),n_boot=len(samples))
