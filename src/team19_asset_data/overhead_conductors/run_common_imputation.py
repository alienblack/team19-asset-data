"""Run paired baseline and predictive-forest candidates on prepared pilot views."""

from __future__ import annotations

import argparse
from pathlib import Path

from .common_imputation import run_common_imputation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--estimators", type=int, default=24)
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=219)
    args = parser.parse_args()
    result = run_common_imputation(
        args.pilot_dir, args.output,
        estimators=args.estimators, passes=args.passes, seed=args.seed,
    )
    print(f"Paired candidates written to {result}")
    print("Clean truth was not loaded; reconstruction metrics are a separate step.")


if __name__ == "__main__":
    main()
