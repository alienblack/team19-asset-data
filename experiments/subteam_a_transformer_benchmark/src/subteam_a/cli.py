"""Dispatch input audits and benchmark runs and record their environment and task contracts."""
import os
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name]='1'
os.environ.setdefault('MPLBACKEND','Agg')

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
from time import perf_counter
import platform

from threadpoolctl import threadpool_limits
from .data import (CORE,MNAR,STRESS,OPTIONAL,OVERHEAD_GRID,ANCHOR,GENERATOR_SEEDS,SPLIT_SEEDS,verify_transformer,sha,SNAPSHOT_YEAR)
from .study import run_task,tuning,DEFAULT,METHODS,write_json
from .evaluate import LIFE_TIGHT_TOL, LIFE_LOOSE_TOL, MIN_CATEGORICAL_FLOOR_ERROR, NUMERIC_TOLERANCES


def main(argv=None):
    """
    Parse optional command-line arguments and run the selected audit or experiment. Run
    artifacts are written under the requested output directory.
    """
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['run','audit','stress','diagnose','failures'])
    p.add_argument('--data',type=Path,default=Path('data'))
    p.add_argument('--out',type=Path,help='Output directory (uses the command-specific default when omitted)')
    p.add_argument('--anchor-run',type=Path,default=Path('outputs/final'),
                   help='Completed full benchmark to use for the stress comparison')
    p.add_argument('--stress-run',type=Path,default=Path('outputs/anchor-stress'),
                   help='Completed grouped and structural-error run for diagnose or failures')
    p.add_argument('--profile',choices=['smoke','pilot','full'],default='full')
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--no-report', action='store_true', help=argparse.SUPPRESS)
    a=p.parse_args(argv)
    if a.workers<1:
        p.error('workers must be positive')
    default_out = Path('outputs') / {
        'stress':'anchor-stress', 'diagnose':'anchor-diagnostics', 'failures':'failure-audit',
    }.get(a.command, 'final')
    data,out=a.data.resolve(),(a.out or default_out).resolve()
    out.mkdir(parents=True,exist_ok=True)
    if a.command == 'failures':
        from .failures import run_failures
        run_failures(data, out, a.anchor_run.resolve(), a.stress_run.resolve(), a.workers)
        return
    if a.command == 'diagnose':
        from .diagnostics import run_diagnostics
        run_diagnostics(data, out, a.anchor_run.resolve(), a.stress_run.resolve(), a.workers)
        return
    if a.command == 'stress':
        from .stress import run_stress
        run_stress(data, out, a.anchor_run.resolve(), a.workers)
        return
    audit=verify_transformer(data/'source')
    audit.to_csv(out/'source_audit.csv',index=False)
    print(f'Validated {len(audit)} supplied transformer pairs (seed 219), cell for cell.',flush=True)
    if a.command=='audit':
        return
    t=perf_counter()
    small=a.profile=='smoke'
    seeds=GENERATOR_SEEDS if a.profile=='full' else GENERATOR_SEEDS[:1]
    splits=SPLIT_SEEDS if a.profile=='full' else SPLIT_SEEDS[:1]
    trials=20 if a.profile=='full' else (4 if a.profile=='pilot' else 0)
    contract=dict(profile=a.profile,trials_per_trainable_method=trials,generator_seeds=seeds,split_seeds=splits,
                  core_scenarios=[s.name for s in CORE],
                  numeric_tolerances={name: asdict(spec) for name, spec in NUMERIC_TOLERANCES.items()},
                  life_tolerance_units={'tight': LIFE_TIGHT_TOL, 'loose': LIFE_LOOSE_TOL},
                  snapshot_year=SNAPSHOT_YEAR, categorical_skill_min_floor_error=MIN_CATEGORICAL_FLOOR_ERROR,
                  interval_coverage_target=.90,minimum_pattern_support=50,
                  source_hashes={str(k.relative_to(data)):sha(k) for k in sorted(data.rglob('*.csv'))},
                  code_hashes={k.name:sha(k) for k in sorted(Path(__file__).parent.glob('*.py'))
                               if k.name != 'report.py' and not k.name.endswith('_report.py')},
                  environment={k:importlib.metadata.version(k) for k in ('numpy','pandas','scipy','scikit-learn','torch','matplotlib','threadpoolctl')},
                  python=platform.python_version())
    path=out/'contract.json'
    contract=json.loads(json.dumps(contract))
    if path.exists() and json.loads(path.read_text())!=contract:
        raise SystemExit(f'{path} describes another input/configuration/code revision. Use a fresh --out.')
    write_json(path,contract)
    with threadpool_limits(limits=1),ProcessPoolExecutor(max_workers=a.workers) as pool:
        tun={pool.submit(tuning,str(data),str(out),gs,ss,trials,small):(gs,ss) for gs in seeds for ss in splits}
        todo=[]
        for fu in as_completed(tun):
            gs,ss=tun[fu]
            cfg=fu.result()['selected']
            print(f'Tuning ready: generator {gs}, split {ss}',flush=True)
            if small:
                scenarios=[next(s for s in CORE if s.name==ANCHOR),MNAR,*STRESS[:1]]
            else:
                scenarios=[*CORE,MNAR]
                if gs==219:
                    scenarios.extend([*STRESS,*OPTIONAL])
            for sc in scenarios:
                exp=(gs,ss)==(219,617)
                todo.append(pool.submit(run_task,str(data),str(out),sc,gs,ss,cfg,('model','rules'),METHODS,True,exp,small))
            if not small:
                sc=next(s for s in CORE if s.name==ANCHOR)
                todo.append(pool.submit(run_task,str(data),str(out),sc,gs,ss,cfg,('model','rules'),METHODS,False,False,small))
        for ss in splits:
            for sc in (OVERHEAD_GRID[:1] if small else OVERHEAD_GRID):
                todo.append(pool.submit(run_task,str(data),str(out),sc,0,ss,DEFAULT,('model',),METHODS,True,ss==617,small))
        for i,fu in enumerate(as_completed(todo),1):
            print(f'[{i}/{len(todo)}] {fu.result()}',flush=True)
    write_json(out/'run_status.json',dict(tasks=len(todo),elapsed_seconds=perf_counter()-t,
                                        finished_utc=datetime.now(timezone.utc).isoformat()))


if __name__=='__main__':
    main()
