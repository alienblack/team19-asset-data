"""Command-line entry point for the revised-methodology Phase 1 pilot."""

from __future__ import annotations

import argparse

from .v3_pilot import PilotSettings, run_pilot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="New or empty output directory")
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 102, 103, 104, 105])
    parser.add_argument("--mask-rate", type=float, default=0.30)
    parser.add_argument("--trees", type=int, default=24)
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--rows", type=int, default=10_000, help="Use 10000 for the documented pilot")
    args = parser.parse_args()
    settings = PilotSettings(tuple(args.seeds), args.mask_rate, args.trees, args.passes)
    print(run_pilot(args.output, settings, args.rows))


if __name__ == "__main__":
    main()
