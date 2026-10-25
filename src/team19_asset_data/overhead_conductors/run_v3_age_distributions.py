"""Evaluate distribution-based age shares on frozen clean and noisy v3 views."""

from __future__ import annotations

import argparse

from .v3_age_distributions import run_age_distributions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-output", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--noise-output")
    parser.add_argument("--arms", nargs="*")
    args = parser.parse_args()
    print(run_age_distributions(
        args.pilot_output, args.output, args.noise_output,
        tuple(args.arms) if args.arms is not None else None,
    ))


if __name__ == "__main__":
    main()
