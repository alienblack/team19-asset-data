"""Run the paired clean/noisy v3 evaluation without retuning on test data."""

from __future__ import annotations

import argparse

from .v3_noise_comparison import ARMS, run_comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-output", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--arms", nargs="*", choices=[arm.arm_id for arm in ARMS])
    args = parser.parse_args()
    selected = tuple(arm for arm in ARMS if args.arms is None or arm.arm_id in args.arms)
    if not selected:
        parser.error("at least one noise arm is required")
    print(run_comparison(args.pilot_output, args.output, selected))


if __name__ == "__main__":
    main()
