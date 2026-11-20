"""One entry point for report-only, quick, full and saved-prediction workflows."""
import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import traceback


def _environment():
    versions = {}
    for name in ('numpy','pandas','scipy','scikit-learn','torch','lightgbm','miceforest',
                 'catboost','matplotlib','joblib','Jinja2'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = 'not installed'
    return {'python': platform.python_version(), 'platform': platform.platform(), 'packages': versions}


def main(argv=None, project_root=None):
    os.environ['MPLBACKEND']='Agg'
    root = Path(project_root or Path(__file__).resolve().parents[2]).resolve()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('report','quick','full','evaluate'), default='report',
                        help='Default: present the supplied full-study results, without refitting')
    parser.add_argument('--source', type=Path, help='Result tables/figures directory for report mode')
    parser.add_argument('--out', type=Path, help='Report output directory')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--run-id', help='Completed saved-prediction folder for evaluate mode')
    parser.add_argument('--resume', help='Interrupted run folder for quick/full mode')
    parser.add_argument('--quick-evaluation', action='store_true', help='Evaluate saved quick-run predictions')
    parser.add_argument('--config', type=Path, help='JSON StudyConfig overrides (recorded in the run contract)')
    parser.add_argument('--keep-inputs', action='store_true', help='Use supplied inputs instead of regenerating them')
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error('--workers must be positive')
    if args.mode == 'evaluate' and not args.run_id:
        parser.error('--mode evaluate requires --run-id')
    if args.run_id and args.mode != 'evaluate':
        parser.error('--run-id is for evaluate mode; use --resume for interrupted fitting')
    if args.resume and args.mode not in ('quick','full'):
        parser.error('--resume is for quick/full mode')
    if args.quick_evaluation and args.mode != 'evaluate':
        parser.error('--quick-evaluation is for evaluate mode')
    if args.source and args.mode != 'report':
        parser.error('--source is for report mode; fitting inputs are selected through --config')
    if args.config and args.mode == 'report':
        parser.error('--config is for quick/full/evaluate; report mode reads existing result exports')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')
    out = (args.out or root / 'reports' / ('published' if args.mode == 'report' else f'{stamp}-{args.mode}')).resolve()
    # A report directory must not replace input data or a saved run directory.
    source = (args.source or root / 'reference_results').resolve() if args.mode == 'report' else None
    if source and (out == source or source.is_relative_to(out)):
        parser.error('--out must be separate from, and not contain, the source directory')
    out.mkdir(parents=True, exist_ok=True)
    metadata = {'mode':args.mode, 'generated_utc':datetime.now(timezone.utc).isoformat(),
                'environment':_environment(), 'refitted':args.mode in ('quick','full')}
    try:
        from .reporting import build_report
        if args.mode == 'report':
            metadata.update(run_id='20261115-121714' if source == root / 'reference_results' else source.name,
                            provenance='Supplied archived full-study tables and figures; no models refitted'
                            if source == root / 'reference_results' else 'Existing result exports; no models refitted')
            if (source/'run_metadata.json').is_file():
                recorded=json.loads((source/'run_metadata.json').read_text())
                for key in ('tasks','failed_tasks','tasks_with_warnings','warning_records'):
                    if key in recorded:
                        metadata[key]=recorded[key]
                metadata['recorded_status_source']=recorded.get('source')
            path = build_report(source, out, metadata, project_root=root)
        else:
            from .execution import generate_inputs, run_notebook
            overrides = json.loads(args.config.read_text(encoding='utf-8')) if args.config else {}
            if not isinstance(overrides, dict):
                raise ValueError('--config must contain a JSON object')
            # Historical/default run locations remain compatible with the original contract.
            run_base = root / overrides.get('output_directory','outputs/imputation-study/runs')
            run_id = args.run_id or args.resume
            if run_id:
                run_dir = (run_base / run_id).resolve()
                if not run_dir.is_relative_to(run_base.resolve()):
                    raise ValueError('Run ID must name a folder inside the run directory')
                if not (run_dir / 'contract.json').is_file():
                    raise FileNotFoundError(f'No saved contract at {run_dir}. Use --mode full or quick, '
                                            'or install package 03 in outputs/imputation-study/runs/.')
                if args.resume:
                    for marker in (run_dir / 'tasks').glob('*.json'):
                        if json.loads(marker.read_text()).get('status') != 'success':
                            marker.unlink()  # incomplete/failed checkpoints are recomputed
            quick = args.mode == 'quick' or args.quick_evaluation
            if quick:
                print('QUICK CHECK: 600-row sample and small models. Results are not research findings.', flush=True)
            options = {'quick':quick, 'workers':args.workers, 'overrides':overrides,
                       'resume':args.resume, 'evaluate':args.run_id, 'auto_report':False}
            log_path = out / 'execution.log'
            console = sys.stdout
            print(f'Executing the study; complete log: {log_path}', flush=True)
            def progress(index, code):
                if 'STUDY.run_model(' in code or 'STUDY.finish(' in code:
                    print(f'Notebook cell {index}: {code.splitlines()[0]}', file=console, flush=True)
            with log_path.open('w', encoding='utf-8', buffering=1) as log:
                with redirect_stdout(log), redirect_stderr(log):
                    if args.mode in ('quick','full') and not args.keep_inputs:
                        generate_inputs(root)
                    study = run_notebook(root, options, progress)
            metadata.update(run_id=Path(study.directory).name, quick=quick,
                            config=json.loads(json.dumps(study.config.__dict__,default=str)),
                            tasks=int(len(study.log)), failed_tasks=int(study.log.status.ne('success').sum()),
                            tasks_with_warnings=int(study.log.warnings.gt(0).sum()),
                            provenance='Fresh model fits and evaluation' if args.mode in ('quick','full')
                            else 'Reevaluation of saved per-cell predictions; run contract checked')
            from .evaluation import warning_table
            metadata['warning_records']=warning_table(study.log).to_dict('records')
            path = build_report(Path(study.directory), out, metadata, project_root=root)
        (out / 'run_status.json').write_text(json.dumps({'status':'success', **metadata,
                      'report':path.name},indent=2),encoding='utf-8')
        print(f'Report ready: {path}', flush=True)
        return 0
    except Exception as error:
        (out / 'run_status.json').write_text(json.dumps({'status':'failed',**metadata,
                   'error':f'{type(error).__name__}: {error}'},indent=2),encoding='utf-8')
        traceback.print_exc()
        print(f'Run failed. See {out / "execution.log"} and run_status.json.', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
