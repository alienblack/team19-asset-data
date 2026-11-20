"""Define asset fields, recreate benchmark inputs and make disjoint spatial or feeder splits."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from .generator import HEADERS, NOISY, generate

SNAPSHOT_YEAR = 2014

OVERHEAD = {
    'installation_year': None, 'distance_to_sea_km': None, 'length_km': None,
    'voltage_category': None, 'location_group': None, 'corrosion_zone': None,
    'material': None, 'covering': None,
}
LINKS = {
    'x': ('y', 'altitude'), 'y': ('x', 'altitude'),
    'install_date': ('x', 'y'), 'model_id': ('install_date', 'measured_gas_a', 'measured_gas_b', 'measured_gas_c'),
    'altitude': ('x', 'y'), 'measured_humid': ('measured_salt', 'measured_overgrowth'),
    'measured_salt': ('measured_humid', 'measured_overgrowth'), 'measured_overgrowth': ('measured_humid', 'measured_salt'),
    'life': ('install_date', 'model_id', 'measured_humid', 'measured_salt', 'measured_overgrowth', 'total_gas'),
    'random_placebo': (), 'leakage': ('install_date', 'model_id', 'measured_gas_a', 'measured_gas_b', 'measured_gas_c'),
    'total_gas': ('measured_gas_a', 'measured_gas_b', 'measured_gas_c'),
}
for _c in ('a', 'b', 'c'):
    LINKS['measured_gas_' + _c] = ('model_id', 'install_date', 'leakage', 'total_gas')
LINKS.update({
    'installation_year': ('material', 'voltage_category'), 'distance_to_sea_km': ('location_group', 'corrosion_zone'),
    'length_km': ('location_group', 'voltage_category'), 'voltage_category': ('location_group', 'material'),
    'location_group': ('corrosion_zone', 'distance_to_sea_km'), 'corrosion_zone': ('distance_to_sea_km', 'location_group'),
    'material': ('installation_year', 'voltage_category'), 'covering': ('location_group', 'voltage_category'),
})


@dataclass(frozen=True)
class Schema:
    """Describe column names and categorical codes for a transformer or overhead asset matrix."""
    names: tuple
    categories: dict  # column index -> fixed domain labels, in integer-code order
    transformer: bool = True

    @property
    def numeric(self):
        """Return the indexes of numeric columns in schema order."""
        return [j for j in range(len(self.names)) if j not in self.categories]

    def i(self, name):
        """Return the column index for the supplied field name."""
        return self.names.index(name)

    def decode(self, array):
        """Convert an encoded matrix to a named DataFrame with the original category labels."""
        df = pd.DataFrame(array, columns=self.names)
        for j, levels in self.categories.items():
            df[self.names[j]] = [levels[int(x)] if np.isfinite(x) else None for x in array[:, j]]
        return df


TRANSFORMER = Schema(tuple(HEADERS), {HEADERS.index('model_id'): (0, 1, 2, 3)})
OVERHEAD_SCHEMA = Schema(tuple(OVERHEAD), {
    3: ('11/7.6 kV', '19 kV SWER', '33 kV', '66 kV', 'LV'),
    4: ('Regional', 'Rural', 'Urban'),
    5: ('Low', 'Severe', 'Very Severe'),
    6: ('Aluminium', 'Copper', 'Steel'),
    7: ('Bare', 'Covered'),
}, False)


@dataclass(frozen=True)
class Scenario:
    """
    Hold a run label, masking rate, noise amplitude and any target or family needed by the
    simulator.
    """
    name: str
    missing: float
    noise: float
    mechanism: str = 'MCAR'
    structural_errors: bool = False
    target: str | None = None
    noise_family: str | None = None


CORE = tuple(Scenario(f'mcar{m}-n{n:02}', m/100, n/100) for m in (10, 20, 30) for n in (0, 5, 10))
MNAR = Scenario('mnar30-n05', .30, .05, 'MNAR')
STRESS = (Scenario('mcar50-n05', .5, .05), Scenario('mcar70-n05', .7, .05))
OPTIONAL = (Scenario('mar30-n05', .3, .05, 'MAR'), Scenario('mcar30-n20', .3, .2))
OVERHEAD_GRID = tuple(Scenario(f'overhead20-n{n:02}', .2, n/100, 'Overhead') for n in (0, 10, 30, 50))
ANCHOR = 'mcar30-n05'
GROUP_ANCHOR = Scenario('group30-n05', .30, .05, 'GROUP')
GROUP_REMASK = Scenario('group30-n05-mlp-remask', .30, .05, 'GROUP')
OUTLIER_ANCHOR = Scenario('mcar30-n05-outliers', .30, .05, structural_errors=True)
ANCHOR_STRESS = (GROUP_ANCHOR, OUTLIER_ANCHOR)
FIELD_GROUPS = {
    'location': ('x', 'y', 'altitude'),
    'gases': ('measured_gas_a', 'measured_gas_b', 'measured_gas_c', 'total_gas'),
    'environment': ('measured_humid', 'measured_salt', 'measured_overgrowth'),
    'date/model': ('install_date', 'model_id'),
}
CLEAN_ANCHOR = 'mcar30-n00'
NOISE_FAMILIES = {
    'location': FIELD_GROUPS['location'],
    'dates': ('install_date',),
    'gases': FIELD_GROUPS['gases'],
    'environment': FIELD_GROUPS['environment'],
}
SINGLE_TARGETS = tuple(Scenario(f'single30-n05-{c}', .30, .05, 'SINGLE', target=c) for c in HEADERS)
FAMILY_NOISE = tuple(Scenario(f'noise30-n20-{family}', .30, .20, 'NOISE_FAMILY', noise_family=family)
                     for family in NOISE_FAMILIES)
DATE_ERROR_RATE = .05
DATE_ERROR_DAYS = 60
MODEL_ERROR_RATE = .02
GENERATOR_SEEDS = (219, 220, 221)
SPLIT_SEEDS = (617, 631, 653)
PARTS = ('train', 'tune', 'calibration', 'test')


def sha(path):
    """Return the SHA-256 digest of a file for the source or run contract."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_transformer(root):
    """
    Recreate the seed-219 inputs, compare every value and gap with the supplied CSV pairs and
    return their audit table.
    """
    truth, eps, uniform = (np.asarray(v, dtype=float) for v in generate(219))
    rows = []
    for sc in CORE:
        key = f'completion{1-sc.missing:.1f}-noise{sc.noise}'
        pt, pm = (root / f'{key}-{k}.csv' for k in ('truth', 'measured'))
        clean, masked = pd.read_csv(pt, float_precision='round_trip'), pd.read_csv(pm, float_precision='round_trip')
        if tuple(clean) != TRANSFORMER.names or tuple(masked) != TRANSFORMER.names:
            raise ValueError(f'Unexpected source schema: {key}')
        np.testing.assert_array_equal(clean.to_numpy(), truth)
        np.testing.assert_allclose(masked.to_numpy(), measure(truth, eps, uniform, sc), rtol=0, atol=0, equal_nan=True)
        rows.append(dict(scenario=sc.name, rows=len(truth), attributes=len(HEADERS), truth_sha256=sha(pt),
                         measured_sha256=sha(pm), measured_missing=float(masked.isna().to_numpy().mean())))
    return pd.DataFrame(rows)


def share_group_draws(draws, schema, groups):
    """Copy the row-by-field random draws and reuse each group’s first draw across its members."""
    shared = draws.copy()
    for members in groups:
        columns = [schema.i(c) for c in members]
        shared[:, columns] = shared[:, columns[:1]]
    return shared


def measure(truth, eps, uniform, scenario):
    """
    Apply a scenario’s measurement errors and masking to clean values using the saved per-cell
    draws. Return the measured matrix.
    """
    x = truth.copy()
    noisy_fields = NOISE_FAMILIES[scenario.noise_family] if scenario.mechanism == 'NOISE_FAMILY' else NOISY
    noisy = [i for i, name in enumerate(HEADERS) if name in noisy_fields]
    x[:, noisy] = x[:, noisy] + x[:, noisy] * eps[:, noisy] * scenario.noise
    rate = np.full(x.shape, scenario.missing)
    date, xpos = (TRANSFORMER.i(c) for c in ('install_date', 'x'))
    if scenario.mechanism == 'SINGLE':
        rate.fill(0)
        rate[:, TRANSFORMER.i(scenario.target)] = scenario.missing
    if scenario.structural_errors:
        # Date and model have unused, independent noise draws in the source generator.
        date_draw = eps[:, date]
        dates = np.abs(date_draw) > 1 - DATE_ERROR_RATE
        shift = np.where((date_draw < 0) & (x[:, date] >= DATE_ERROR_DAYS), -DATE_ERROR_DAYS, DATE_ERROR_DAYS)
        x[dates, date] += shift[dates]
        model = TRANSFORMER.i('model_id')
        classes = len(TRANSFORMER.categories[model])
        draw = (eps[:, model] + 1) / 2
        swapped = draw < MODEL_ERROR_RATE
        offsets = 1 + (draw[swapped] / MODEL_ERROR_RATE * (classes - 1)).astype(int)
        x[swapped, model] = (x[swapped, model] + offsets) % classes
    if scenario.mechanism == 'GROUP':
        uniform = share_group_draws(uniform, TRANSFORMER, FIELD_GROUPS.values())
    if scenario.mechanism == 'MNAR':
        # Overall risk remains 30%; installation-date masking depends on its own unseen value.
        old = (truth[:, date] < np.median(truth[:, date])).astype(float)
        p = .3 + .50 * (old - old.mean())
        rate[:, date] = p
    if scenario.mechanism == 'MAR':
        # The x value triggering missingness is guaranteed observed for this experiment.
        rate.fill(.3 * len(HEADERS) / (len(HEADERS) - 1))
        rate[:, date] += .5 * ((truth[:, xpos] < 50) - .5)
        rate[:, xpos] = 0
    # Reuse the same draw so raising the missing rate keeps all previously hidden cells hidden.
    x[uniform > 1 - rate] = np.nan
    return x


def transformer_data(root, sc, seed, draw=0):
    """
    Return clean and measured transformer matrices for one seed. Use a separate mask draw for
    calibration when requested.
    """
    truth, eps, uniform = (np.asarray(v, dtype=float) for v in generate(seed))
    if draw:
        uniform = np.random.default_rng(np.random.SeedSequence([seed, 7203, draw])).random(truth.shape)
    if seed == 219 and not draw and sc in CORE:
        key = f'completion{1-sc.missing:.1f}-noise{sc.noise}-measured.csv'
        measured = pd.read_csv(root / key, float_precision='round_trip').to_numpy(dtype=float)
    else:
        measured = measure(truth, eps, uniform, sc)
    return truth, measured


def overhead_data(root, sc):
    """
    Load the supplied overhead pair, encode categories and return clean values, measured
    values and feeder IDs.
    """
    key = f'completion0.8-noise{sc.noise:.1f}'
    frames = [pd.read_csv(root / f'{key}-{suffix}.csv') for suffix in ('truth', 'measured')]
    groups = frames[0]['feeder_id'].to_numpy()
    if not np.array_equal(groups, frames[1]['feeder_id'].to_numpy()):
        raise ValueError('Overhead feeder alignment failed')
    out = []
    for df in frames:
        for j, levels in OVERHEAD_SCHEMA.categories.items():
            c = OVERHEAD_SCHEMA.names[j]
            # Level list is a schema, not a fitted encoder.
            unknown = set(df[c].dropna().unique()) - set(levels)
            if unknown:
                raise ValueError(f'Unrecognised labels for {c}: {unknown}')
            df[c] = df[c].map({level: k for k, level in enumerate(levels)})
        out.append(df[list(OVERHEAD_SCHEMA.names)].to_numpy(dtype=float))
    return out[0], out[1], groups


def split_blocks(truth, split_seed, schema, groups=None):
    """
    Return shuffled row indexes for training, tuning, calibration and test, plus each row’s
    block. Stratify transformer tiles by x band and keep whole overhead feeders together. Use
    clean coordinates only for splitting.
    """
    rng = np.random.default_rng(split_seed)
    if schema.transformer:
        x, y = truth[:, :2].astype(int).T
        # Allocate whole spatial tiles within each x band so all partitions span the installation gradient.
        band = x // 10
        groups = band * 10 + y // 10
    else:
        groups = np.asarray(groups)
        band = np.zeros(len(truth), int)
    part = np.full(len(truth), -1, dtype=int)
    for b in np.unique(band):
        ids = rng.permutation(np.unique(groups[band == b]))
        n = len(ids)
        test, cal, tune = round(.2*n), max(1, round(.1*n)), max(1, round(.1*n))
        for idx, k in zip(np.split(ids, [test, test+cal, test+cal+tune]), [3, 2, 1, 0]):
            part[(band == b) & np.isin(groups, idx)] = k
    if (part < 0).any() or len(np.unique(part)) != 4:
        raise ValueError('Invalid split')
    order = rng.permutation(len(truth))
    return {name: order[part[order] == i] for i, name in enumerate(PARTS)}, groups
