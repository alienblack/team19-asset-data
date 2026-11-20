#!/usr/bin/env python3
"""Run from any working directory, without an editable installation."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))

if __name__ == '__main__':
    from synthetic_imputation.cli import main
    raise SystemExit(main(project_root=Path(__file__).resolve().parent))
