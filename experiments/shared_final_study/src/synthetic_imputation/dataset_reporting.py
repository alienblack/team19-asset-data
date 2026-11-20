"""Data-derived charts and offline reports for schema-configured CSV runs."""
import base64
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

from .dataset_models import METHODS

COLORS=['#087d78','#b66c4b','#334f72','#8d85a6','#7e9787','#ac8b37']


def build_dataset_report(output, source, predictions, model_data=None):
    output=Path(output)
    summary=json.loads((output/'summary.json').read_text())
    figures=output/'figures';figures.mkdir(exist_ok=True)
    charts=[]
    plt.rcParams.update({'font.family':'sans-serif','font.size':11,'axes.spines.top':False,
        'axes.spines.right':False,'axes.labelcolor':'#4e6174','text.color':'#162b40',
        'xtick.color':'#657182','ytick.color':'#657182','figure.facecolor':'white',
        'axes.facecolor':'white','savefig.facecolor':'white'})

    def save(fig,name,title,caption):
        fig.tight_layout(pad=1.8)
        path=figures/(name+'.png');fig.savefig(path,dpi=180,bbox_inches='tight');plt.close(fig)
        charts.append({'file':'figures/'+path.name,'title':title,'caption':caption,
                       'image':'data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode()})

    ranking=summary['leaderboard']
    numeric_ranks=[r for r in ranking if r['skill'] is not None]
    if numeric_ranks:
        fig,ax=plt.subplots(figsize=(10,4.8))
        rows=numeric_ranks[::-1]
        ax.barh([r['label'] for r in rows],[r['skill'] for r in rows],color=[COLORS[0] if r['method']!='median' else '#b7c2ca' for r in rows])
        ax.axvline(0,color='#8b98a3',lw=.8);ax.set_xlabel('Mean numeric skill relative to the median')
        save(fig,'method_comparison','Numeric recovery by method',
             'Each field contributes one baseline-relative MAE score. Zero matches its training-median reference; negative scores are worse. Controls and fields with zero baseline error are excluded from this mean.')
    fields=summary['field_scores']
    numeric=[c for c in summary['numeric'] if any(r['target']==c for r in fields)][:24]
    methods=[r['method'] for r in ranking]
    if numeric and methods:
        matrix=np.array([[next((r['skill'] for r in fields if r['target']==c and r['method']==m),np.nan) for m in methods] for c in numeric],dtype=float)
        fig,ax=plt.subplots(figsize=(max(9,len(methods)*1.05),max(4,len(numeric)*.45)))
        image=ax.imshow(matrix,cmap='RdYlGn',vmin=-.5,vmax=1,aspect='auto')
        ax.set_xticks(range(len(methods)),[METHODS[m] for m in methods],rotation=35,ha='right')
        ax.set_yticks(range(len(numeric)),numeric)
        if matrix.size<=200:
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    if np.isfinite(matrix[i,j]):ax.text(j,i,f'{matrix[i,j]:.2f}',ha='center',va='center',fontsize=9)
        fig.colorbar(image,ax=ax,shrink=.8,label='Skill')
        save(fig,'attribute_skill','Recovery differs by attribute',
             'The same method can help one field and fail on another. Values below zero indicate more average error than the median. The complete table includes every evaluated field; this plot shows up to 24 numeric fields.')
    categorical=[c for c in summary['categorical'] if any(r['target']==c for r in fields)][:12]
    if categorical and methods:
        if len(categorical)==1:
            fig,ax=plt.subplots(figsize=(10,max(4,len(methods)*.38)))
            values=[next((r['macro_f1'] for r in fields if r['target']==categorical[0] and r['method']==m),np.nan) for m in methods]
            ax.barh([METHODS[m] for m in methods[::-1]],values[::-1],color=COLORS[0])
            ax.set_xlim(0,1.02);ax.set_xlabel('Macro-F1 for '+categorical[0])
        else:
            fig,ax=plt.subplots(figsize=(max(9,len(methods)*.9),max(4,len(categorical)*.5)))
            matrix=np.array([[next((r['macro_f1'] for r in fields if r['target']==c and r['method']==m),np.nan) for m in methods] for c in categorical])
            image=ax.imshow(matrix,cmap='YlGnBu',vmin=0,vmax=1,aspect='auto')
            ax.set_xticks(range(len(methods)),[METHODS[m] for m in methods],rotation=35,ha='right')
            ax.set_yticks(range(len(categorical)),categorical)
            fig.colorbar(image,ax=ax,label='Macro-F1',shrink=.8)
        save(fig,'category_recovery','Categorical recovery',
             'Macro-F1 gives each category equal weight, so a rare class is not hidden by a very common one. Accuracy is reported separately in the results table. These are point classifications, not calibrated probabilities.')
    active=summary['numeric']+summary['categorical']
    missing=source[active].isna().mean().sort_values(ascending=False).head(24)
    fig,ax=plt.subplots(figsize=(10,max(4,len(missing)*.32)))
    ax.barh(missing.index[::-1],100*missing.values[::-1],color=COLORS[0]);ax.set_xlabel('Missing recorded values (%)')
    save(fig,'missing_by_field','Gaps in the supplied dataset',
         'These are the dataset’s original gaps, before the experiment adds any hiding. Entirely unobserved fields cannot be learned from this dataset and must be excluded or supplied with external information.')
    fig,ax=plt.subplots(figsize=(10,4))
    counts=source[active].isna().sum(axis=1).value_counts().sort_index()
    ax.bar(counts.index,counts.values,color='#334f72');ax.set_xlabel('Missing modelled fields in a record');ax.set_ylabel('Records')
    save(fig,'missing_per_record','How many fields are missing together?',
         'Records with several missing fields offer fewer visible clues. This chart describes the supplied data; the configured grouped/scattered stress mask is an additional experimental condition.')
    if len(summary['numeric'])>=2:
        chosen=summary['numeric'][:16]
        correlation=(model_data if model_data is not None else source)[chosen].apply(pd.to_numeric).corr()
        fig,ax=plt.subplots(figsize=(9,7));image=ax.imshow(correlation,cmap='RdBu_r',vmin=-1,vmax=1)
        ax.set_xticks(range(len(chosen)),chosen,rotation=50,ha='right');ax.set_yticks(range(len(chosen)),chosen)
        fig.colorbar(image,ax=ax,shrink=.8,label='Pairwise Pearson correlation')
        save(fig,'numeric_associations','Observed numeric associations',
             'Correlations use available recorded pairs and describe linear association, not causation or guaranteed recoverability. Missingness and recording error can affect them. At most 16 numeric fields are shown.')
    if numeric_ranks:
        best=numeric_ranks[0]['method']
        chosen=sorted(summary['numeric'],key=lambda c:source[c].isna().mean(),reverse=True)[:8]
        for number,c in enumerate(chosen):
            part=predictions[(predictions.method==best)&(predictions.target==c)]
            if part.empty:continue
            actual=pd.to_numeric(part.truth).to_numpy(float)
            guess=pd.to_numeric(part.prediction).to_numpy(float)
            median=pd.to_numeric(part.baseline).to_numpy(float)
            values=np.r_[actual,guess,median]
            if not np.isfinite(values).all():continue
            bins=np.histogram_bin_edges(values,bins=24)
            fig,ax=plt.subplots(figsize=(10,4))
            for value,label,color in [(actual,'Reference values','#8e99aa'),(median,'Median fills','#b24430'),(guess,METHODS[best],COLORS[0])]:
                ax.hist(value,bins=bins,histtype='step',linewidth=1.8,label=label,color=color)
            if c in summary.get('date_fields',[]):
                import matplotlib.dates as dates
                ax.xaxis.set_major_formatter(dates.DateFormatter('%Y-%m-%d'))
                ax.tick_params(axis='x',rotation=30)
            ax.set_xlabel(c);ax.set_ylabel('Scored values');ax.legend(fontsize=9)
            save(fig,f'distribution_{number:02d}',f'{c}: reference and filled distributions',
                 f'The same held-out cells are compared for the median and {METHODS[best]}. Point fills can compress the spread or lose extremes even when average error improves. Repetitions contribute repeated evaluation observations; this is not a population uncertainty estimate.')
    times=[r for r in ranking]
    if times:
        fig,ax=plt.subplots(figsize=(10,4.8));rows=times[::-1]
        ax.barh([r['label'] for r in rows],[r['seconds'] for r in rows],color='#8c9eae')
        ax.set_xlabel('Total validation fit + prediction time (seconds)')
        save(fig,'runtime','Cost of the selected comparison',
             'Runtime includes the selected validation folds and repetitions on this machine. Final full-input completion fits are recorded separately in runtime.csv. These timings are not the historical benchmark’s runtime claims.')
    findings=[]
    if numeric_ranks:
        top=numeric_ranks[0]
        findings.append({'title':'Best mean numeric recovery in this run',
            'text':f"{top['label']} has mean numeric skill {top['skill']:.3f}. This averages field-level reductions in MAE relative to the median; it is not a percentage of correct fills."})
        weak=[r for r in fields if r['method']==top['method'] and r['below_floor']]
        findings.append({'title':'Fields that remain difficult',
            'text':('The highest-ranked numeric method is below the median on '+', '.join(r['target'] for r in weak)+'. Examine those field rows before selecting one method for every column.') if weak else
            'The highest-ranked numeric method is not below the median on any evaluated numeric field. This does not establish that its individual fills meet a practical tolerance.'})
    cat_ranks=[r for r in ranking if r['macro_f1'] is not None]
    if cat_ranks:
        best=max(cat_ranks,key=lambda r:r['macro_f1'])
        floor=next((r for r in cat_ranks if r['method']=='median'),None)
        tied=[r['label'] for r in cat_ranks if abs(r['macro_f1']-best['macro_f1'])<1e-8]
        findings.append({'title':'Categorical recovery is a separate result',
            'text':('No selected method improves categorical macro-F1 over the mode reference in this run. '
                    if floor and best['macro_f1']-floor['macro_f1']<1e-8 else
                    f"The highest mean categorical macro-F1 is {best['macro_f1']:.3f}, attained by {', '.join(tied)}. ")
                    +'Category performance is not mixed into mean numeric skill.'})
    findings.append({'title':'What is and is not validated',
        'text':'Scores use '+summary['score_reference'].lower()+'. Completed exports preserve recorded inputs. Their naturally missing cells are not individually certified, and no calibrated auto-fill gate is fitted in this workflow.'})
    if summary['config']['masking']=='grouped' and 'mlp' in summary['config']['methods']:
        findings.append({'title':'MLP training-mask limitation',
            'text':'The generic masked MLP adds independent training remasking, even in a grouped stress run. Its result can be affected by this training/test mask mismatch; it is not the separate matched-mask MLP experiment from the Subteam A study.'})
    if summary['failures']:
        findings.append({'title':'Some methods did not complete',
            'text':f"{len(summary['failures'])} model-stage failures are listed below. Incomplete validation methods are excluded from the paired ranking rather than scored only on their successful folds."})
    environment=Environment(loader=FileSystemLoader(Path(__file__).parent/'templates'),autoescape=select_autoescape(['html']))
    downloads=[p.name for p in sorted(output.glob('*.csv'))]+['config.json','manifest.json','summary.json','warnings.json','failures.json']
    html=environment.get_template('dataset_report.html').render(s=summary,charts=charts,findings=findings,downloads=downloads)
    (output/'report.html').write_text(html,encoding='utf-8')
    return output/'report.html'
