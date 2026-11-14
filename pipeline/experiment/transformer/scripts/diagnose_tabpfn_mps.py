"""Isolated Apple Silicon TabPFN-3.5-Fast MPS smoke test.

No SAPN/project data is read. This script is only an MPS compatibility test;
pass/fail is NOT a model benchmark. A native MPS segfault kills only its child.

Usage from pipeline root after copying both scripts to the same scripts/ folder:
    .pipeline/bin/python -u experiment/transformer/scripts/diagnose_tabpfn_mps.py --device mps
    .pipeline/bin/python -u experiment/transformer/scripts/diagnose_tabpfn_mps.py --device cpu
"""
from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
import traceback
from importlib.metadata import PackageNotFoundError, version


def _version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


def child(device: str, context_rows: int, batch_rows: int, n_estimators: int) -> int:
    import numpy as np
    import pandas as pd
    import torch
    from comparison_models import ModelSpec, TabPFNExperimentalImputer

    print("TabPFN diagnostic child started", flush=True)
    print("macOS:", platform.platform(), flush=True)
    print("Python:", sys.version.split()[0], flush=True)
    print("torch:", torch.__version__, "MPS available:", torch.backends.mps.is_available(), flush=True)
    print("tabpfn:", _version("tabpfn"), "extensions:", _version("tabpfn-extensions"), flush=True)
    print("device:", device, "context:", context_rows,
          "batch rows:", batch_rows, "n_estimators:", n_estimators, flush=True)
    print("TABPFN_MPS_MEMORY_FRACTION:", os.getenv("TABPFN_MPS_MEMORY_FRACTION"), flush=True)
    rng = np.random.default_rng(219)
    n_train = max(context_rows + 40, 140)
    n_test = 32
    xx = np.linspace(0, 6, n_train + n_test)
    frame = pd.DataFrame({
        "x": xx,
        "y": xx * 1.5 + rng.normal(size=len(xx)),
        "altitude": xx * 0.75 + 10,
        "model_id": np.tile(np.array(["A", "B", "C", "D"], dtype=object), n_train // 4 + 10)[:len(xx)],
    })
    train = frame.iloc[:n_train].copy()
    test = frame.iloc[n_train:].copy()
    for j, column in enumerate(frame.columns):
        train.loc[train.index[j::6], column] = np.nan
        test.loc[test.index[j::5], column] = np.nan
    columns = ModelSpec(numeric=("x", "y", "altitude"), categorical=("model_id",), seed=219)
    model = TabPFNExperimentalImputer(
        columns, device=device, model_version="v3.5-fast",
        max_context_rows=context_rows, n_permutations=1,
        mps_batch_rows=batch_rows, n_estimators=n_estimators,
        mps_force_float32=True,
    )
    print("BEGIN fit", flush=True)
    model.fit(train)
    print("END fit; BEGIN transform", flush=True)
    output = model.transform(test)
    print("END transform", flush=True)
    if output.isna().any().any():
        raise AssertionError("Imputer left missing cells")
    for c in columns.columns:
        known = test[c].notna()
        if not np.array_equal(test.loc[known,c].to_numpy(), output.loc[known,c].to_numpy()):
            raise AssertionError(f"Modified known cells in {c}")
    if not set(output["model_id"].unique()).issubset({"A", "B", "C", "D"}):
        raise AssertionError("Unexpected categorical outputs")
    print(f"PASS: {len(test)} rows / {len(columns.columns)} attributes / {model.batch_count_} batch(es)", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--context-rows", type=int, default=100)
    parser.add_argument("--batch-rows", type=int, default=16)
    parser.add_argument("--n-estimators", type=int, default=1)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child:
        try:
            return child(args.device, args.context_rows, args.batch_rows, args.n_estimators)
        except Exception:
            traceback.print_exc()
            return 1
    if args.context_rows < 20 or args.batch_rows < 1 or args.n_estimators < 1:
        parser.error("context rows >=20, batch rows >=1 and estimators >=1 are required")
    env = dict(os.environ)
    env.setdefault("TABPFN_MPS_MEMORY_FRACTION", "0.5")
    env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    env.setdefault("PYTHONFAULTHANDLER", "1")
    cmd = [sys.executable, "-u", "-X", "faulthandler", os.path.abspath(__file__),
           "--child", "--device", args.device,
           "--context-rows", str(args.context_rows),
           "--batch-rows", str(args.batch_rows),
           "--n-estimators", str(args.n_estimators)]
    print("Launching isolated worker:", args.device, flush=True)
    result = subprocess.run(cmd, env=env, check=False)
    if result.returncode < 0:
        print(f"Worker terminated by native signal {-result.returncode}. ",
              "This is not catchable inside Python. Try upgrading TabPFN/PyTorch ",
              "or use device=cpu for the benchmark.", flush=True)
        return 1
    if result.returncode:
        print(f"Worker exited with code {result.returncode}; see the traceback above.", flush=True)
        return 1
    print("MPS diagnostic completed successfully." if args.device == "mps"
          else "CPU diagnostic completed successfully.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
