#!/usr/bin/env python3
"""Reapply the declared rule to the supplied parameter-selection records."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from synthetic_imputation.selection import MODELS,audit

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=list(MODELS))
    p.add_argument('--round',type=int,choices=(1,2,3),default=3)
    p.add_argument('--out',type=Path,default=ROOT/'reports/selection_audit.json')
    a=p.parse_args()
    result=audit(ROOT,[a.model] if a.model else None,a.round)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    if any(r['matches_recorded'] is False for r in result['models']):
        raise SystemExit('Selection audit differs from the recorded choice')
