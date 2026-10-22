"""Generate or load a synthetic reference and freeze the pilot's input views."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import pandas as pd

from .generate_synthetic_data import (
    Config,
    StrengthConfig,
    calibrate_route_length_marginals,
    generate_dataset,
    validate_generated_dataset,
)
from .input_preparation import load_pilot_config, prepare_pilot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input", type=Path, help="Optional existing complete synthetic reference CSV")
    parser.add_argument("--generation-config", type=Path, default=Path("config/overhead_conductors/generation.yaml"))
    parser.add_argument("--strength-config-dir", type=Path, default=Path("config/overhead_conductors/relationship_strength"))
    args = parser.parse_args()
    protocol = load_pilot_config(args.protocol)
    if args.input:
        reference = pd.read_csv(args.input)
        source = f"complete synthetic CSV: {args.input.resolve()}"
    else:
        generation = Config.from_yaml(args.generation_config)
        generation = replace(
            generation,
            seed=int(protocol["pilot_population"]["generation_seed"]),
            n_rows=int(protocol["pilot_population"]["expected_rows"]),
        )
        strength = int(protocol["pilot_population"]["relationship_strength"])
        profile = StrengthConfig.from_yaml(args.strength_config_dir / f"strength_{strength}.yaml")
        reference = generate_dataset(generation, profile)
        reference = calibrate_route_length_marginals(reference, generation)
        validate_generated_dataset(reference, generation)
        source = f"generator seed={generation.seed}, profile={profile.profile_name}"
    result = prepare_pilot(reference, protocol, args.output, source)
    print(f"Pilot inputs written to {result}")
    print("Model-facing files: observed_inputs/; clean answers: evaluator_only/")


if __name__ == "__main__":
    main()
