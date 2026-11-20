"""Execution helpers; the original prediction implementations are unchanged."""
from contextlib import contextmanager
from dataclasses import fields, replace
from pathlib import Path
import copy
import json
import os


@contextmanager
def working_directory(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def configured(quick=False, overrides=None):
    from .runner import StudyConfig, quick_config
    config = quick_config() if quick else StudyConfig()
    updates = copy.deepcopy(overrides or {})
    unknown = set(updates) - {f.name for f in fields(config)}
    if unknown:
        raise ValueError(f'Unknown configuration keys: {sorted(unknown)}')
    # A model override changes only the supplied settings, not every other model.
    for key in ('parameters', 'explanation', 'blocks'):
        if key in updates:
            combined = copy.deepcopy(getattr(config, key))
            for name, value in updates[key].items():
                if key != 'blocks':
                    if name not in combined:
                        raise ValueError(f'Unknown model in {key}: {name}')
                    combined[name].update(value)
                else:
                    combined[name] = tuple(value)
            updates[key] = combined
    config = replace(config, **updates)
    if config.folds < 2 or (config.rows is not None and config.rows < config.folds):
        raise ValueError('Use at least two folds, with at least one row per fold')
    return config


def generate_inputs(root):
    """Execute the supplied original generator cell, including its seed and draw order."""
    root = Path(root)
    notebook = root / 'data/datasets/generator/SyntheticDataset.ipynb'
    document = json.loads(notebook.read_text(encoding='utf-8'))
    code = next(''.join(c['source']) for c in document['cells'] if c['cell_type'] == 'code')
    (root / 'data/datasets').mkdir(parents=True, exist_ok=True)
    with working_directory(root):
        exec(compile(code, str(notebook), 'exec'), {'__name__': '__main__'})


def run_notebook(root, options, on_cell=None):
    """Execute the same plain-Python cells used by the interactive notebook, in order.

    No separate benchmark implementation is maintained. Figures and CSVs are written by
    the existing notebook and Study. Console tables go to the CLI log.
    """
    import matplotlib
    matplotlib.use('Agg', force=True)
    import matplotlib.pyplot as plt
    root = Path(root)
    path = root / 'notebooks/imputation.ipynb'
    document = json.loads(path.read_text(encoding='utf-8'))
    namespace = {'__name__': '__main__', '_RUN_OPTIONS': options}
    with working_directory(root):
        for index, cell in enumerate(document['cells']):
            if cell['cell_type'] != 'code':
                continue
            source = ''.join(cell.get('source', []))
            if on_cell:
                on_cell(index, source)
            exec(compile(source, f'{path}:cell-{index}', 'exec'), namespace)
            plt.close('all')  # every notebook figure is saved; headless runs retain no GUI windows
            study = namespace.get('STUDY')
            if study is not None and 'STUDY.run_model(' in source:
                failed = [family for family, log in study.logs.items() if log.status.ne('success').any()]
                if failed:
                    raise RuntimeError(f'Failed model tasks: {failed}. See {study.directory}; resume to retry failed tasks.')
    return namespace['STUDY']
