#!/usr/bin/env python3
"""Run a new three-fold measured-data-only selection pass on the recorded candidates."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from synthetic_imputation.selection import MODELS,refit

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=list(MODELS),required=True)
    p.add_argument('--round',type=int,choices=(1,2,3),default=3)
    p.add_argument('--rows',type=int,help='Diagnostic sample only; omit for full data')
    p.add_argument('--limit',type=int,help='Diagnostic candidate limit only; omit to run every candidate')
    p.add_argument('--seed',type=int,default=219)
    p.add_argument('--out',type=Path)
    a=p.parse_args()
    out=a.out or ROOT/'outputs/selection'/datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')
    print(json.dumps(refit(ROOT,a.model,out,a.round,a.rows,a.seed,a.limit),indent=2))
