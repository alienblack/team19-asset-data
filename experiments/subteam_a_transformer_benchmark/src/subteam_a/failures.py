"""Rerun the MLP with grouped training masks and evaluate the screening of recorded values."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import json
from os.path import relpath
from time import perf_counter

from threadpoolctl import threadpool_limits

from .data import ANCHOR, ANCHOR_STRESS, FIELD_GROUPS, GROUP_REMASK, sha, verify_transformer
from .screening import FLAG_LEVELS, PRIMARY_LEVEL, screen_pair
from .stress import check_comparable, prepare_anchor
from .study import run_task, write_json


def run_failures(data, out, anchor, stress, workers):
    """
    Run the paired grouped MLP and screening tasks using the same tuning selections and
    partitions as the reference runs.
    """
    start = perf_counter()
    pairs, settings, provenance = prepare_anchor(data, anchor, (ANCHOR,))
    prior = json.loads((stress/'contract.json').read_text())
    if (prior['profile'] != 'anchor-stress' or not (stress/'run_status.json').is_file()
            or prior['source_hashes'] != provenance['source_hashes']
            or prior['reference_hashes'] != provenance['reference_hashes']):
        raise SystemExit('Use the completed stress run based on the same full benchmark.')
    check_comparable(prior['code_hashes'], provenance['code_hashes'])
    names = [f'tasks/{sc.name}-{gs}-{ss}-domain.json' for sc in ANCHOR_STRESS for gs, ss in pairs]
    contract = dict(
        profile='failure-audit', anchor_run=relpath(anchor, out), stress_run=relpath(stress, out),
        stress_reference_hashes={name: sha(stress/name) for name in names},
        grouped_scenario=asdict(GROUP_REMASK), remask_groups=FIELD_GROUPS,
        screening_flag_levels=FLAG_LEVELS, primary_flag_level=PRIMARY_LEVEL, **provenance,
    )
    path = out/'contract.json'
    contract = json.loads(json.dumps(contract))
    if path.exists() and json.loads(path.read_text()) != contract:
        raise SystemExit(f'{path} describes another run. Use a fresh --out.')
    verify_transformer(data/'source').to_csv(out/'source_audit.csv', index=False)
    write_json(path, contract)
    with threadpool_limits(limits=1), ProcessPoolExecutor(max_workers=workers) as pool:
        tasks = []
        for gs, ss in pairs:
            cfg = settings[gs, ss]
            grouped = {'autoencoder': {**cfg['autoencoder'], 'grouped_remask': True}}
            tasks.append(pool.submit(run_task, str(data), str(out), GROUP_REMASK, gs, ss, grouped,
                                     ('model', 'rules'), ('autoencoder',), True, (gs, ss) == pairs[0]))
            tasks.append(pool.submit(screen_pair, data, out, gs, ss, cfg))
        for i, future in enumerate(as_completed(tasks), 1):
            print(f'[{i}/{len(tasks)}] {future.result()}', flush=True)
    write_json(out/'run_status.json', dict(
        tasks=len(pairs), screening_tasks=len(pairs), elapsed_seconds=perf_counter()-start,
        finished_utc=datetime.now(timezone.utc).isoformat(),
    ))
