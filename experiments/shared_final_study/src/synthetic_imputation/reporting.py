"""Build an offline HTML report from archived exports or a freshly completed run."""
import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

REFERENCE = 'completion0.8-noise0.05'
SECTION_NAMES = {'project-guide':'Presentation figures', 'notebook':'All notebook figures',
                 'imputation-A-random-forest':'Random Forest: method and comparison',
                 'imputation-B-gmm':'GMM: method and comparison'}
CAPTIONS = {
 'overall':'Mean numeric skill across models', 'overall_skill':'Performance of every model by field',
 'skill_vs_cost':'Numeric skill and computational cost', 'cost_against_skill':'Numeric skill and computational cost',
 'failure_map':'Field-level performance under each condition',
 'models_by_condition':'Performance under each condition', 'conditions':'Conditions that affect performance',
 'skill_by_visible_links':'Performance as related information disappears',
 'wrong_values':'Propagation of incorrect recorded values', 'fold_design':'Holding out whole spatial areas',
 'skill_by_mechanism':'Changing the missingness mechanism', 'ridge_base_effect':'Contribution of the ridge base',
 'error_split':'How much error is explained by missing information',
 'bias_by_value_band':'Bias at the extremes of the true values',
 'paired_uncertainty':'Paired intervals for differences in model error',
 'selection':'GMM parameter-selection evidence', 'idea':'Illustration of a shared-covariance mixture',
 'fit':'How the GMM is fitted', 'fill':'How the fitted GMM fills a new row',
 'summary_values':'Effect on downstream summary statistics', 'workflow':'Reproducible experimental workflow',
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _link(path, output):
    return Path(os.path.relpath(path, output)).as_posix()


def _read(tables, name):
    paths = [p for p in tables if p.name == name + '.csv']
    if len(paths) != 1:
        raise ValueError(f'Expected exactly one {name}.csv, found {len(paths)}. '
                         'Use the supplied reference_results folder or a completed study run.')
    return pd.read_csv(paths[0])


def build_report(source, output, metadata=None, project_root=None):
    source, output = Path(source).resolve(), Path(output).resolve()
    root = Path(project_root or Path(__file__).resolve().parents[2]).resolve()
    metadata = dict(metadata or {})
    if not source.is_dir():
        raise FileNotFoundError(f'Results directory does not exist: {source}')
    tables = sorted((source / 'tables').rglob('*.csv')) if (source / 'tables').is_dir() else sorted(source.glob('*.csv'))
    if (source / 'completed').is_dir():
        tables += sorted((source / 'completed').glob('*.csv'))
    numeric, category, cost = (_read(tables, n) for n in ('numeric_scores','category_scores','cost'))
    contract = json.loads((source / 'contract.json').read_text()) if (source / 'contract.json').is_file() else {}
    config = metadata.get('config', contract.get('config', {}))
    reference, seed = config.get('representative',REFERENCE), config.get('seed',219)
    for table, name in ((numeric,'numeric'),(category,'category')):
        required = {'scenario','experiment','variant','seed','target','method','n','coverage'}
        if not required <= set(table):
            raise ValueError(f'Malformed {name} scores: missing {sorted(required-set(table))}')
        if table.duplicated(['scenario','experiment','variant','seed','target','method']).any():
            raise ValueError(f'Duplicate {name} score groups')
    if not np.allclose(numeric.skill, 1-numeric.MAE/numeric['median MAE'],equal_nan=True):
        raise ValueError('Numeric skill is inconsistent with model and baseline MAE')
    def reference_rows(table):
        return table[table.scenario.eq(reference) & table.experiment.eq('main')
                     & table.variant.eq('all predictors') & table.seed.eq(seed)]
    num, cat = reference_rows(numeric), reference_rows(category)
    if num.empty or cat.empty:
        raise ValueError(f'No reference results for {reference}, seed {seed}')
    signal = num[num.target.ne('random_placebo')]
    means = signal.groupby('method').skill.mean().sort_values(ascending=False)
    field_count = int(signal.target.nunique())
    timing = cost[cost.experiment.eq('main')].set_index('method')
    accuracy = cat.set_index('method').accuracy.to_dict()
    ranking = []
    for method, score in means.items():
        seconds = float(timing.loc[method,'seconds per target and fold']) if method in timing.index else None
        ranking.append({'method':method,'label':'GMM (Gaussian EM)' if method=='Gaussian EM' else method,
                        'skill':float(score),'percent':100*float(score),'width':max(0,min(100,100*float(score))),
                        'seconds':seconds,'accuracy':accuracy.get('Mode' if method=='Median' else method),
                        'style':'gmm' if method=='Gaussian EM' else 'forest' if method=='Random Forest' else ''})
    leader = ranking[0]
    gmm = next((r for r in ranking if r['method']=='Gaussian EM'),None)
    forest = next((r for r in ranking if r['method']=='Random Forest'),None)
    speedup = forest['seconds']/gmm['seconds'] if forest and gmm and gmm['seconds'] else None
    metric_rows = []
    for r in num.itertuples(index=False):
        metric_rows.append({'method':r.method,'target':r.target,'mae':r.MAE,'rmse':r.RMSE,
                            'skill':r.skill,'accuracy':None,'n':int(r.n)})
    for r in cat.itertuples(index=False):
        metric_rows.append({'method':r.method,'target':r.target,'mae':None,'rmse':None,
                            'skill':None,'accuracy':r.accuracy,'n':int(r.n)})
    csv_downloads = [{'name':p.name,'group':str(p.parent.relative_to(source)),
                      'href':_link(p,output),'bytes':p.stat().st_size} for p in tables]
    json_paths = sorted((source / 'tables').rglob('*.json')) if (source / 'tables').is_dir() else []
    image_paths = sorted((source / 'figures').rglob('*.png'))
    chart_groups = {}
    for p in image_paths:
        content = p.read_bytes()
        if not content.startswith(b'\x89PNG\r\n\x1a\n'):
            raise ValueError(f'Invalid PNG file: {p}')
        parent = p.parent.name
        group = SECTION_NAMES.get(parent,'All run figures')
        caption = CAPTIONS.get(p.stem,p.stem.replace('_',' ').replace('-',' ').capitalize())
        if p.stem.startswith('model_'):
            caption = p.stem[6:].replace('_',' ').title()+' · performance by scenario'
        chart_groups.setdefault(group,[]).append({'name':p.name,'title':caption,
                       'source':str(p.relative_to(source)),
                       'image':'data:image/png;base64,'+base64.b64encode(content).decode('ascii')})
    if not image_paths:
        raise ValueError('No PNG figures found. Complete the notebook figure cells before generating the report.')
    completion_dir = source / 'tables/6_completed_tables' if (source / 'tables').is_dir() else source / 'completed'
    completed = []
    for p in sorted(completion_dir.glob('*.csv')):
        frame = pd.read_csv(p)
        completed.append({'file':p.name,'rows':len(frame),'missing':int(frame.isna().sum().sum()),
                          'href':_link(p,output)})
    population = completed[0]['rows'] if completed else config.get('rows',10000) or 10000
    quick = bool(metadata.get('quick', config.get('rows') is not None))
    kind = 'Quick validation — not research findings' if quick else (
           'Fresh full study' if metadata.get('refitted') else 'Saved-prediction evaluation'
           if metadata.get('mode')=='evaluate' else 'Published full-study results')
    provenance = metadata.get('provenance','Result tables and figures supplied with the project; no models refitted')
    coverage = float(numeric.coverage.min())
    task_counts = cost.groupby('experiment',sort=False).tasks.sum().astype(int).to_dict()
    rows = {k:int(v) for k,v in task_counts.items()}
    run_id = metadata.get('run_id',source.name)
    created = datetime.now(timezone.utc).isoformat()
    source_files = sorted(set(tables+json_paths+image_paths+
                        ([source/'run_metadata.json'] if (source/'run_metadata.json').is_file() else [])))
    model_files = sorted((root / 'src/synthetic_imputation').rglob('*.py'))
    model_files += sorted((root / 'src/synthetic_imputation/templates').glob('*.html'))
    model_files += [p for p in (root/'run_study.py',root/'notebooks/imputation.ipynb',
                    root/'requirements.txt',root/'pyproject.toml') if p.is_file()]
    model_files += sorted((root/'scripts').glob('*.py'))
    manifest = {'schema_version':1,'run_id':run_id,'report_kind':kind,'generated_utc':created,
                'provenance':provenance,'metadata':metadata,'source_files':{
                str(p.relative_to(source)):digest(p) for p in source_files},
                'code_files':{str(p.relative_to(root)):digest(p) for p in model_files},
                'provenance_files':{str(p.relative_to(root)):digest(p) for p in
                    (root/'ORIGINAL_SOURCES.json',source/'contract.json') if p.is_file() and p.is_relative_to(root)}}
    summary = {'run_id':run_id,'report_kind':kind,'quick':quick,'refitted':bool(metadata.get('refitted')),
               'population_rows':population,'reference_scenario':reference,'seed':seed,
               'numeric_fields_in_mean':field_count,'recorded_successful_tasks':int(cost.tasks.sum()),
               'charts':len(image_paths),'csv_exports':len(tables),'leaderboard':ranking,
               'gmm_to_forest_runtime_ratio':speedup,'minimum_prediction_coverage':coverage,
               'tasks_with_warnings':metadata.get('tasks_with_warnings'),
               'completed_tables':completed}
    settings = []
    for p in json_paths:
        if p.name.endswith(('.final.json','.chosen.json')):
            settings.append({'name':str(p.relative_to(source)),
                             'value':json.dumps(json.loads(p.read_text()),sort_keys=True)})
    if config.get('parameters'):
        settings = [{'name':name,'value':json.dumps(value,sort_keys=True)} for name,value in config['parameters'].items()]
    environment = Environment(loader=FileSystemLoader(Path(__file__).parent/'templates'),
                              autoescape=select_autoescape(['html']))
    html = environment.get_template('report.html').render(
        kind=kind,quick=quick,provenance=provenance,run_id=run_id,created=created,
        reference=reference,seed=seed,population=population,field_count=field_count,
        ranking=ranking,leader=leader,gmm=gmm,forest=forest,speedup=speedup,
        charts=chart_groups,chart_count=len(image_paths),downloads=csv_downloads,
        metric_rows=metric_rows,methods=sorted({r['method'] for r in metric_rows}),
        targets=list(num.target.unique())+['model_id'],task_counts=rows,
        task_total=int(cost.tasks.sum()),coverage=coverage,completed=completed,settings=settings,
        source_label=_link(source,root),metadata=metadata)
    output.mkdir(parents=True,exist_ok=True)
    (output/'report.html').write_text(html,encoding='utf-8')
    (output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return output/'report.html'
