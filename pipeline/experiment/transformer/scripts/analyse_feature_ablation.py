"""Aggregate transformer T3 ablation results into paired tables, plots and report.

All primary comparisons use the SAME test records/masks, 20% MCAR and ±5% noise.
Numeric relative change is (ablation MAE / full-model MAE - 1) * 100.
Positive means worse. Baseline skill is relative to Median/Mode, same held-out cells.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EXPERIMENT = Path(__file__).resolve().parent.parent
OUT_DEFAULT = EXPERIMENT/'outputs'/'feature_ablation_seed617'
BENCHMARK = EXPERIMENT/'outputs'/'notebook_full_20261111_092108'/'per_target_metrics.csv'
VARIANT_ORDER = ['without_coordinates','without_install_date','without_model_id','without_total_gas','without_altitude','without_placebo']


def load_results(folder: Path, *, require_all: bool=False):
    files=sorted(folder.glob('*_metrics.csv'))
    if not files: raise FileNotFoundError('No ablation metrics found in '+str(folder))
    frames=[pd.read_csv(p) for p in files]
    df=pd.concat(frames,ignore_index=True)
    if df.duplicated(['ablation','variable']).any():raise ValueError('Duplicate rows')
    if require_all and set(VARIANT_ORDER+['all_attributes'])!=set(df.ablation):
        raise ValueError(f'Missing experiments: {set(VARIANT_ORDER+["all_attributes"])-set(df.ablation)}')
    full=df[df.ablation=='all_attributes'].copy()
    if full.empty:raise ValueError('Ablation set requires rerun all_attributes as control')
    m=df.merge(full[['variable','mae','macro_f1','n_masked']].rename(columns={'mae':'mae_full','macro_f1':'macro_f1_full','n_masked':'n_full'}),on='variable',validate='many_to_one')
    if not (m.n_masked==m.n_full).all():raise AssertionError('Target masks differ across ablations!')
    m['mae_change_pct']=np.where((m.variable_type=='numeric')&(m.mae_full>0),100*(m.mae/m.mae_full-1),np.nan)
    m['f1_change']=np.where(m.variable_type=='categorical',m.macro_f1-m.macro_f1_full,np.nan)
    base=pd.read_csv(BENCHMARK)
    base=base[(base.method=='median_mode')&(base.protocol=='unseen_records')&(np.isclose(base.completion,0.8))&(np.isclose(base.noise,0.05))&(base.split_seed==617)]
    m=m.merge(base[['variable','mae','macro_f1','n_masked']].rename(columns={'mae':'median_mode_mae','macro_f1':'median_mode_f1','n_masked':'baseline_n'}),on='variable',validate='many_to_one')
    if not (m.n_masked==m.baseline_n).all():raise AssertionError('Original benchmark masks differ')
    m['numeric_skill']=np.where(m.variable_type=='numeric',1-m.mae/m.median_mode_mae,np.nan)
    return df,m


def make_charts(m:pd.DataFrame,out:Path)->list[Path]:
    figures=[]
    x=m[(m.ablation!='all_attributes')&(m.variable_type=='numeric')&(m.variable!='random_placebo')].copy()
    p=x.pivot(index='variable',columns='ablation',values='mae_change_pct')
    p=p.reindex(index=[v for v in ['install_date','altitude','measured_humid','measured_salt','measured_overgrowth','measured_gas_a','measured_gas_b','measured_gas_c','total_gas','life','leakage','x','y'] if v in p.index])
    p=p.reindex(columns=[c for c in VARIANT_ORDER if c in p.columns]); data=p.to_numpy(dtype=float)
    fig,ax=plt.subplots(figsize=(11,7)); vmax=max(50.0, float(np.ceil(np.nanmax(abs(data[np.isfinite(data)]))/25)*25) if np.isfinite(data).any() else 50.0)
    im=ax.imshow(np.clip(data,-vmax,vmax),cmap='RdBu_r',vmin=-vmax,vmax=vmax,aspect='auto')
    for yi in range(data.shape[0]):
        for xi in range(data.shape[1]):
            value=data[yi,xi]
            if np.isfinite(value) and abs(value)>=10:
                ax.text(xi,yi,f'{value:+.0f}%',ha='center',va='center',fontsize=7,color='white' if abs(value)>vmax*0.6 else 'black')
    ax.set_xticks(range(len(p.columns)),[q.replace('without_','no ') for q in p.columns],rotation=28,ha='right')
    ax.set_yticks(range(len(p.index)),p.index); ax.set_title('T3 · MAE change after removing predictor groups\nrelative to the full 15-attribute model (positive = worse)')
    ax.tick_params(axis='both',labelsize=9)
    fig.colorbar(im,ax=ax,label='Change in numerical MAE (%)',fraction=.03,pad=.04)
    fig.text(.5,.005,'Blank cells: removed attribute was not scored as a target. Colours show signed MAE change; large changes are labelled.',ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.028,1,1]); f=out/'ablation_mae_impact_heatmap.png';fig.savefig(f,dpi=180,bbox_inches='tight');plt.close(fig);figures.append(f)
    # Focused view of source-to-related-target changes.
    focus={
        'without_coordinates':['install_date','altitude','model_id','measured_gas_a','life'],
        'without_install_date':['model_id','measured_gas_a','measured_gas_b','life'],
        'without_model_id':['measured_gas_a','measured_gas_b','measured_gas_c','life'],
        'without_total_gas':['measured_gas_a','measured_gas_b','measured_gas_c','life'],
        'without_altitude':['measured_humid','measured_overgrowth','life'],
    }
    panels=[]
    for ablation, targets in focus.items():
        d=m[(m.ablation==ablation)&(m.variable.isin(targets))];
        if len(d): panels.append((ablation,d))
    if panels:
        fig,axes=plt.subplots(len(panels),1,figsize=(10,max(6,len(panels)*2.2)))
        if len(panels)==1:axes=[axes]
        for ax,(ablation,d) in zip(axes,panels):
            xx=[f'{r.variable}'+(' (F1)' if r.variable_type=='categorical' else '') for r in d.itertuples()]
            # units differ for categorical F1, so chart numeric only here
            dn=d[d.variable_type=='numeric'];ax.barh(dn.variable,dn.mae_change_pct)
            ax.axvline(0,color='black',lw=.6); ax.set_title(ablation.replace('without_','Remove '),loc='left',fontsize=10)
            ax.set_xlabel('Change in MAE vs all-attribute PMF (%) ; positive = worse',fontsize=8)
            ax.tick_params(labelsize=8)
        fig.tight_layout();f=out/'ablation_dependency_profiles.png';fig.savefig(f,dpi=180,bbox_inches='tight');plt.close(fig);figures.append(f)
    # Overall baseline + target skill profile by ablation with common targets only
    selected=['install_date','measured_gas_a','measured_gas_b','measured_gas_c','life','measured_salt','leakage']
    df=m[m.variable.isin(selected)&(m.variable_type=='numeric')].copy()
    piv=df.pivot(index='variable',columns='ablation',values='numeric_skill');piv=piv.reindex(columns=[c for c in ['all_attributes']+VARIANT_ORDER if c in piv.columns]);
    fig,ax=plt.subplots(figsize=(12,5)); im=ax.imshow(piv.to_numpy(),cmap='viridis',vmin=0,vmax=1,aspect='auto')
    ax.set_xticks(range(len(piv.columns)),[c.replace('without_','no ') for c in piv.columns],rotation=30,ha='right')
    ax.set_yticks(range(len(piv.index)),piv.index);fig.colorbar(im,ax=ax,label='Numerical skill vs Median/Mode (same scored cells)',fraction=.035)
    ax.set_title('T3 · Per-attribute numerical skill by predictor set')
    fig.tight_layout();f=out/'ablation_skill_heatmap.png';fig.savefig(f,dpi=180,bbox_inches='tight');plt.close(fig);figures.append(f)
    return figures

def write_report(df,m,out):
    b=m[m.ablation=='all_attributes']; c=m[m.ablation!='all_attributes'].copy()
    c_num=c[c.variable_type=='numeric'].sort_values('mae_change_pct',ascending=False)
    def mkrow(r):return f'| {r.ablation} | `{r.variable}` | {r.mae_full:.4g} | {r.mae:.4g} | {r.mae_change_pct:+.1f}% | {int(r.n_masked)} |'
    worst='\n'.join(map(mkrow,c_num.head(12).itertuples()))
    strongest='\n'.join(map(mkrow,c_num.sort_values('mae_change_pct').head(8).itertuples()))
    ccat=c[c.variable_type=='categorical'].sort_values('f1_change')
    ccat_tbl='\n'.join(f'| {r.ablation} | {r.variable} | {r.macro_f1_full:.4f} | {r.macro_f1:.4f} | {r.f1_change:+.4f} | {int(r.n_masked)} |' for r in ccat.itertuples()) or 'No categorical targets retained in any variants.'
    ablation_table='\n'.join(f'| {variant} | {", ".join(ABLATIONS[variant]) or "None"} |' for variant in ['all_attributes']+VARIANT_ORDER if variant in set(df.ablation))
    main='''# T3 — Transformer feature-relationship ablation: completed analysis\n\n'''
    main+='''**Dataset:** one 10,000-row synthetic transformer population; 20% nominal MCAR missingness and ±5% relative noise in seven measured attributes.\n\n'''
    main+='''**Protocol:** Predictive MissForest fitted on 8,000 spatially separated measured records, applied without retraining to 2,000 unseen records; evaluation restricted to originally missing test cells.\n\n'''
    main+='''**Configuration:** 100 trees, unrestricted depth, minimum leaf size 10, five saved-model fitting passes, `max_features=sqrt`, random seed 219. Exact project implementation reused; no tuning or test-truth use during fitting.\n\n'''
    main+='''**Design:** For each ablation, a source column/group is completely excluded as both a predictor and an imputation target. Comparisons apply only to *remaining targets*, using identical held-out rows and missing masks. This isolates the effect of *predictor availability to the full iterative system*, but cannot by itself establish causal generator mechanisms.\n\n'''
    main+='''## Predictor sets evaluated\n\n| Variant | Excluded attributes |\n|---|---|\n'''+ablation_table+'\n\n'
    main+='''## Largest increases in numerical error (strongest losses)\n\n'''
    main+='''Positive changes mean the ablation made predictions worse. These are per-target changes; unlike a mean across varying target sets, these remain comparable.\n\n'''
    main+='''| Ablation | Target | Full MAE | Ablated MAE | Relative change | Hidden test cells |\n|---|---|---:|---:|---:|---:|\n'''+worst+'\n\n'
    main+='''## Improvements or unexpectedly small losses\n\n'''
    main+='''Negative changes indicate improved MAE after removing the feature. This can occur with redundant or misleading predictors and should not be hidden.\n\n'''
    main+='''| Ablation | Target | Full MAE | Ablated MAE | Relative change | Hidden test cells |\n|---|---|---:|---:|---:|---:|\n'''+strongest+'\n\n'
    main+='''## Categorical result: transformer model ID\n\n'''
    main+='''| Ablation | Target | Full Macro-F1 | Ablated Macro-F1 | Change | Hidden test cells |\n|---|---|---:|---:|---:|---:|\n'''+ccat_tbl+'\n\n'
    main+='''## Interpretation\n\n- **Seed-control:** Retained targets use exactly the same per-target random-forest seed under all ablations, indexed against the full 15-attribute schema. This removes a potential random-number confound caused by dropping columns.\n- **Generator context:** the synthetic source defines `altitude = x + y + 100`, `total_gas = gas_a + gas_b + gas_c`, a date-conditioned model choice, and model-specific gas distributions. `install_date` also correlates strongly with grid coordinates due to generation order. These relationships make predictor removal scientifically relevant.\n- **Evidence type:** observed changes quantify dependence of this fitted imputation system on *available columns*; they do not identify individual tree decisions and are not causal estimates for SAPN asset registers.\n- **Comparison to teammate's work:** our test is intentionally restricted to Predictive MissForest and the 20%/5% anchor; the teammate's separate experiment uses different forest implementations, feature engineering, and evaluation conditions. The scores must not be directly ranked as if protocols matched.\n- **Validity safeguards:** ground truth appears only in scoring; no target re-masking; same spatial split; no replacement of noisy observed cells; dropped columns are omitted at fit and transform rather than populated with ground truth.\n- **Limits:** one generator realization and one split seed; no uncertainty intervals; feature removals alter the number of input dimensions and the order of iterative models; there may be substitution by correlated predictors. These results are a controlled within-dataset diagnostic, not a real-world deployment claim.\n\n'''
    main+='''## Presentation-ready figures\n\n- `ablation_mae_impact_heatmap.png` — target-level change in MAE for each removal.\n- `ablation_dependency_profiles.png` — focused error changes along selected synthetic dependency paths.\n- `ablation_skill_heatmap.png` — relative numerical skill against Median/Mode for selected targets.\n\n'''
    main+='''## Next research requirements\n\n1. Repeat with independent synthetic generator seeds/spatial splits for uncertainty estimates.\n2. Conduct grouped missingness (same marginal rate, different mask correlation) and targeted observed-value corruption as separate experiments.\n3. Investigate the coordinate-to-date generation shortcut through a separately versioned generator, preserving this original benchmark.\n'''
    (out/'T3_report.md').write_text(main,encoding='utf-8')
    return main

def run(out:Path, require_all:bool=False):
    out.mkdir(parents=True,exist_ok=True)
    df,m=load_results(out,require_all=require_all)
    m.to_csv(out/'ablation_comparison.csv',index=False)
    chart=make_charts(m,out)
    report=write_report(df,m,out)
    print(f'{len(df)} target metric rows from {len(df.ablation.unique())} configurations; wrote {len(chart)} figures; {out / "T3_report.md"}')
    return df,m

# imported constant for reporting
from run_feature_ablation import ABLATIONS

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,default=OUT_DEFAULT);p.add_argument('--require-all',action='store_true');args=p.parse_args();run(args.output_dir,args.require_all)
