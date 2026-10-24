"""Compare 24 and 200 forest trees on the frozen v3 tuning partitions only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .v3_pilot import MATERIALS, PilotSettings, forest_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-output", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    pilot = args.pilot_output.resolve()
    manifest = json.loads((pilot / "manifest.json").read_text(encoding="utf-8"))
    truth = pd.read_csv(pilot / "evaluator_only/clean_reference.csv").set_index("segment_id")
    rows = []
    for seed in manifest["seeds"]:
        run = pilot / f"seed_{seed}"
        parts = pd.read_csv(run / "partitions.csv")
        tuning_ids = set(parts.loc[parts["partition"].eq("tuning"), "segment_id"])
        test_ids = set(parts.loc[parts["partition"].eq("test"), "segment_id"])
        assert not tuning_ids & test_ids
        fitting = pd.read_csv(run / "observed_fitting.csv")
        tuning = pd.read_csv(run / "observed_tuning.csv")
        assert set(tuning["segment_id"]) == tuning_ids
        masks = pd.read_csv(run / "mask_manifest.csv")
        masks = masks.loc[masks["partition"].eq("tuning")]
        settings = json.loads((run / "forest_tuning.json").read_text(encoding="utf-8"))
        for n_trees in (24, 200):
            predictions, _ = forest_predictions(
                fitting, tuning, masks, seed,
                PilotSettings(seeds=(seed,), n_estimators=n_trees), **settings,
            )
            material = predictions.loc[predictions["target"].eq("material")]
            actual = truth.loc[material["segment_id"], "material"].to_numpy()
            probabilities = material[["p_steel", "p_copper", "p_aluminium"]].to_numpy(dtype=float)
            indices = np.array([MATERIALS.index(label) for label in actual])
            true_probabilities = probabilities[np.arange(len(material)), indices]
            year = predictions.loc[predictions["target"].eq("installation_year")]
            actual_year = truth.loc[year["segment_id"], "installation_year"].to_numpy(dtype=float)
            predicted_year = year["point_prediction"].to_numpy(dtype=float)
            rows.append({
                "seed": seed, "partition": "tuning", "n_trees": n_trees,
                "max_depth": settings["max_depth"],
                "min_samples_leaf": settings["min_samples_leaf"],
                "n_passes": settings["n_passes"],
                "material_cells": len(material), "year_cells": len(year),
                "zero_true_class_probabilities": int((true_probabilities == 0).sum()),
                "material_log_loss_clip_1e_minus_12": float(-np.log(np.clip(true_probabilities, 1e-12, 1)).mean()),
                "material_log_loss_clip_1e_minus_6": float(-np.log(np.clip(true_probabilities, 1e-6, 1)).mean()),
                "year_mae": float(np.abs(predicted_year - actual_year).mean()),
            })
    result = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
