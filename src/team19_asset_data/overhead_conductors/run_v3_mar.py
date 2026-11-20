"""Run the predeclared overhead-conductor MAR missing-year comparison."""

from __future__ import annotations

import argparse

from .v3_mar import ARMS, run_mar_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-output", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--arms", nargs="*", choices=[arm.arm_id for arm in ARMS])
    args = parser.parse_args()
    selected = tuple(arm for arm in ARMS if args.arms is None or arm.arm_id in args.arms)
    print(run_mar_evaluation(args.pilot_output, args.output, selected))


if __name__ == "__main__":
    main()
