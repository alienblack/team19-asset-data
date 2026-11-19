"""Run follow-up scenarios against the tuning selections and partitions of the full benchmark."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from itertools import product
import json
from os.path import relpath
from pathlib import Path
from time import perf_counter

from threadpoolctl import threadpool_limits

from .data import (
    ANCHOR, ANCHOR_STRESS, DATE_ERROR_DAYS, DATE_ERROR_RATE, FIELD_GROUPS, GENERATOR_SEEDS,
    MODEL_ERROR_RATE, SPLIT_SEEDS, sha, verify_transformer,
)
from .study import METHODS, run_task, write_json


COMPARABLE_MODULES = ('study.py', 'models.py', 'mixture.py', 'features.py', 'generator.py', 'evaluate.py')
# This revision predates the empty-cohort consistency guard. Its cell scoring is identical.
PRE_SINGLE_EVALUATE = '54583af84a7d2c87bd11aa30fc8978561e3194660b55ea971f69b332440777b1'
# This revision used the same training code with grouped extra re-masking disabled.
PRE_GROUPED_MODELS = 'f313f3fb5c1d248b06443ccdf01b5e55f5325259746395f211420f34cf403c6c'
REFERENCE_METRICS = ('numeric_tolerances', 'life_tolerance_units', 'snapshot_year',
                     'categorical_skill_min_floor_error', 'interval_coverage_target', 'minimum_pattern_support')


def check_comparable(reference, current):
    """
    Check reference hashes against the shared modelling code, accepting the recorded
    compatible revisions.
    """
    for name in COMPARABLE_MODULES:
        if name == 'evaluate.py' and reference.get(name) == PRE_SINGLE_EVALUATE:
            continue
        if name == 'models.py' and reference.get(name) == PRE_GROUPED_MODELS:
            continue
        if reference.get(name) != current[name]:
            raise SystemExit(f'{name} differs from the reference benchmark. Refresh the reference run first.')


def prepare_anchor(data, anchor, baseline_scenarios):
    """
    Validate the reference inputs and collect split pairs, saved settings and fingerprints for
    a follow-up contract.
    """
    source = json.loads((anchor/'contract.json').read_text())
    if (source['profile'] != 'full' or source['generator_seeds'] != list(GENERATOR_SEEDS)
            or source['split_seeds'] != list(SPLIT_SEEDS) or not (anchor / 'run_status.json').is_file()):
        raise SystemExit('The stress run needs a completed full benchmark with matching seeds.')
    pairs = tuple(product(GENERATOR_SEEDS, SPLIT_SEEDS))
    settings = {}
    references = {}
    for gs, ss in pairs:
        tuning = f'tuning/{gs}-{ss}.json'
        settings[gs, ss] = json.loads((anchor/tuning).read_text())['selected']
        names = [tuning, *(f'tasks/{sc}-{gs}-{ss}-domain.json' for sc in baseline_scenarios)]
        references.update({name: sha(anchor/name) for name in names})
    source_hashes = {str(p.relative_to(data)): sha(p) for p in sorted((data / 'source').glob('*.csv'))}
    if source_hashes != {k: v for k, v in source['source_hashes'].items() if k.startswith('source/')}:
        raise SystemExit('Transformer inputs differ from the benchmark used for comparison.')
    code_hashes = {p.name: sha(p) for p in Path(__file__).parent.glob('*.py')
                   if p.name != 'report.py' and not p.name.endswith('_report.py')}
    check_comparable(source['code_hashes'], code_hashes)
    provenance = dict(reference_hashes=references, source_hashes=source_hashes, code_hashes=code_hashes,
                      trials_per_trainable_method=0, generator_seeds=GENERATOR_SEEDS, split_seeds=SPLIT_SEEDS,
                      **{key: source[key] for key in REFERENCE_METRICS})
    return pairs, settings, provenance


def run_scenarios(data, out, contract, scenarios, pairs, settings, arms, methods, workers):
    """
    Run one job per scenario and seed pair, preserving settings and recording completion in
    the output directory.
    """
    start = perf_counter()
    contract_path = out/'contract.json'
    contract = json.loads(json.dumps(contract))
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise SystemExit(f'{contract_path} describes another run. Use a fresh --out.')
    verify_transformer(data/'source').to_csv(out/'source_audit.csv', index=False)
    write_json(contract_path, contract)
    print('Source pairs verified. Reusing the nine anchor tuning selections.', flush=True)
    with threadpool_limits(limits=1), ProcessPoolExecutor(max_workers=workers) as pool:
        tasks = [
            pool.submit(run_task, str(data), str(out), sc, gs, ss, settings[gs, ss], arms,
                        methods, True, (gs, ss) == pairs[0])
            for sc in scenarios for gs, ss in pairs
        ]
        for i, future in enumerate(as_completed(tasks), 1):
            print(f'[{i}/{len(tasks)}] {future.result()}', flush=True)
    write_json(out/'run_status.json', dict(tasks=len(tasks), elapsed_seconds=perf_counter() - start,
                                         finished_utc=datetime.now(timezone.utc).isoformat()))


def run_stress(data, out, anchor, workers):
    """Evaluate grouped gaps and structural errors using the completed anchor run as the reference."""
    pairs, settings, provenance = prepare_anchor(data, anchor, (ANCHOR,))
    contract = dict(
        profile='anchor-stress', anchor_run=relpath(anchor, out), anchor_scenario=ANCHOR,
        scenarios=[asdict(sc) for sc in ANCHOR_STRESS], field_groups=FIELD_GROUPS,
        structural_errors=dict(date_rate=DATE_ERROR_RATE, date_days=DATE_ERROR_DAYS, model_rate=MODEL_ERROR_RATE),
        **provenance,
    )
    run_scenarios(data, out, contract, ANCHOR_STRESS, pairs, settings, ('model', 'rules'), METHODS, workers)
