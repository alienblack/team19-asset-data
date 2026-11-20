"""Run matched single-target and noise-family comparisons using the saved anchor settings."""
from dataclasses import asdict
import json
from os.path import relpath

from .data import ANCHOR, ANCHOR_STRESS, CLEAN_ANCHOR, FAMILY_NOISE, NOISE_FAMILIES, SINGLE_TARGETS, sha
from .stress import check_comparable, prepare_anchor, run_scenarios

METHODS = ('floor', 'forest', 'mixture')
SCENARIOS = (*SINGLE_TARGETS, *FAMILY_NOISE)


def run_diagnostics(data, out, anchor, stress, workers):
    """
    Validate the full and stress reference runs, then score the single-target and one-family
    scenarios.
    """
    pairs, settings, provenance = prepare_anchor(data, anchor, (ANCHOR, CLEAN_ANCHOR))
    ref = json.loads((stress/'contract.json').read_text())
    if (ref['profile'] != 'anchor-stress' or not (stress/'run_status.json').is_file()
            or ref['source_hashes'] != provenance['source_hashes']
            or any(provenance['reference_hashes'].get(name) != digest
                   for name, digest in ref['reference_hashes'].items())):
        raise SystemExit('Use a completed stress run based on the same full benchmark.')
    check_comparable(ref['code_hashes'], provenance['code_hashes'])
    files = [f'tasks/{sc.name}-{gs}-{ss}-domain.json' for sc in ANCHOR_STRESS for gs, ss in pairs]
    contract = dict(
        profile='anchor-diagnostics', anchor_run=relpath(anchor, out), stress_run=relpath(stress, out),
        stress_reference_hashes={name: sha(stress/name) for name in files},
        methods=METHODS, arms=('model',), scenarios=[asdict(sc) for sc in SCENARIOS],
        noise_families=NOISE_FAMILIES, **provenance,
    )
    run_scenarios(data, out, contract, SCENARIOS, pairs, settings, ('model',), METHODS, workers)
