"""Run the expanded held-out T4/T5/T6 experiments via the same engine as the master notebook."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
from pathlib import Path
from robustness_suite import RobustnessSuite, METHODS

def parse_floats(raw):
    return tuple(float(x.strip()) for x in raw.split(',') if x.strip())

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment',choices=['t4','t5','t6','all'],default='all')
    p.add_argument('--methods',default='median_mode,chained_ridge,lightgbm_chained')
    p.add_argument('--quick',action='store_true',help='1k train / 1k test, one pass, fewer trees: not report-ready')
    p.add_argument('--split-seed',type=int,default=617)
    p.add_argument('--model-seed',type=int,default=219)
    p.add_argument('--t4-count-preserving',action='store_true',help='Extra separate grouped refit, simulates newly exposed values')
    p.add_argument('--t5-train-corruption',action='store_true')
    p.add_argument('--t5-prevalences',default='0,0.05,0.10,0.25')
    p.add_argument('--t6-train-noises',default='0,0.05,0.10')
    p.add_argument('--t6-apply-noises',default='0,0.05,0.10,0.20,0.40')
    p.add_argument('--no-save-models',action='store_true')
    p.add_argument('--output-dir',type=Path)
    args=p.parse_args()
    methods=tuple(m.strip() for m in args.methods.split(',') if m.strip())
    if not methods or len(set(methods))!=len(methods) or any(m not in METHODS for m in methods):
        p.error(f'Choose unique methods from {METHODS}')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    folder=args.output_dir or (Path(__file__).resolve().parents[1]/'outputs'/f'robustness_{args.experiment}_{"pilot" if args.quick else "full"}_{stamp}')
    suite=RobustnessSuite(output_dir=folder,methods=methods,quick=args.quick,
                          split_seed=args.split_seed,seed=args.model_seed,save_models=not args.no_save_models)
    if args.experiment in ('t4','all'):
        print('T4: predictor co-missingness',flush=True)
        suite.t4(include_count_preserving=args.t4_count_preserving)
    if args.experiment in ('t5','all'):
        print('T5: observed-predictor swap corruption',flush=True)
        suite.t5(prevalences=parse_floats(args.t5_prevalences),add_training_corruption=args.t5_train_corruption)
    if args.experiment in ('t6','all'):
        print('T6: frozen fit across noise severity',flush=True)
        suite.t6(train_noises=parse_floats(args.t6_train_noises),apply_noises=parse_floats(args.t6_apply_noises))
    metric,cells=suite.consolidate()
    print(f'Finished {len(metric)} attribute rows and {len(cells)} masked-cell rows')
    print('Results:',folder.resolve())

if __name__=='__main__':main()
