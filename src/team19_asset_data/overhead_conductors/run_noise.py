"""Command-line runner for controlled overhead-conductor predictor noise."""

from __future__ import annotations

import argparse
from pathlib import Path

from .noise import run_scenarios


ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = ROOT_DIR / "data" / "synthetic" / "overhead_conductors" / "overhead_conductors_complete_reference.csv"
DEFAULT_CONFIG = ROOT_DIR / "config" / "overhead_conductor_noise_rules.yaml"
DEFAULT_OUTPUT = ROOT_DIR / "outputs" / "PGS219-predictor-noise"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--scenarios", nargs="+", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_scenarios(args.input, args.config, args.output_dir, args.scenarios)
    print(f"Noise scenarios completed. Quality report: {report}")


if __name__ == "__main__":
    main()
