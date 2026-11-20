"""
Prepare deterministic fills and train-only model inputs, then format predictions without
changing observations.
"""
from dataclasses import dataclass
import numpy as np
from scipy.spatial import cKDTree

from .data import SNAPSHOT_YEAR


def exact_fill(x, schema, noise, enabled=True):
    """
    Return a copy with uniquely determined formula values and a mask of cells filled by rule.
    Honour the dataset and noise assumptions.
    """
    a = x.copy()
    if not enabled or not schema.transformer:
        return a, np.zeros(a.shape, bool)
    i = schema.i
    def fill(name, candidate, allowed=True):
        j = i(name)
        use = np.isnan(a[:, j]) & np.isfinite(candidate) & allowed
        a[use, j] = candidate[use]
    def v(name):
        return a[:, i(name)]
    for _ in range(2):
        fill('altitude', v('x') + v('y') + 100)
        for target, other in [('x', 'y'), ('y', 'x')]:
            z = v('altitude') - v(other) - 100
            fill(target, z, (z >= 0) & (z <= 99))
        if noise == 0:
            gases = ['measured_gas_'+x for x in 'abc']
            fill('total_gas', sum(v(c) for c in gases))
            for target in gases:
                z = v('total_gas') - sum(v(c) for c in gases if c != target)
                fill(target, z, z >= 0)
            fill('measured_overgrowth', np.maximum(2*v('measured_humid') - v('measured_salt'), 0))
            positive = v('measured_overgrowth') > 0
            z = (v('measured_overgrowth') + v('measured_salt')) / 2
            fill('measured_humid', z, positive & (z >= 0) & (z <= 1))
            z = 2*v('measured_humid') - v('measured_overgrowth')
            fill('measured_salt', z, positive & (z >= 0) & (z <= 1))
    return a, np.isnan(x) & np.isfinite(a)


@dataclass
class Prediction:
    """Store point estimates, numeric scales, category probabilities and any iteration trace."""
    point: np.ndarray
    scale: np.ndarray
    proba: dict
    trace: tuple = ()

    def subset(self, sl):
        """
        Copy a row slice of the predictions so calibration and test arrays can be handled
        separately.
        """
        return Prediction(self.point[sl].copy(), self.scale[sl].copy(),
                          {k: v[sl].copy() for k, v in self.proba.items()}, self.trace)


class Features:
    """
    Fit shared statistics and spatial lookups on training data and build features from values
    available at prediction time.
    """
    def __init__(self, schema, domain):
        self.schema = schema
        self.domain = domain and schema.transformer

    def fit(self, x):
        """
        Learn numeric summaries, class frequencies and the spatial lookup from the measured
        training matrix. Return this feature builder.
        """
        self.median = np.nanmedian(x, axis=0)
        self.scale = np.maximum(np.nanstd(x, axis=0), 1e-8)
        self.minimum = np.nanmin(x, axis=0)
        self.range = np.maximum(np.nanmax(x, axis=0) - self.minimum, 1e-8)
        self.median = np.nan_to_num(self.median)
        self.scale = np.where(np.isfinite(self.scale), self.scale, 1.)
        self.range = np.where(np.isfinite(self.range), self.range, 1.)
        self.marginal = {}
        for j, levels in self.schema.categories.items():
            counts = np.bincount(x[np.isfinite(x[:, j]), j].astype(int), minlength=len(levels))
            self.marginal[j] = (counts + .5) / (sum(counts) + .5*len(levels))
            self.median[j] = np.argmax(counts)
        if self.schema.transformer:
            j = self.schema.i('install_date')
            keep = np.isfinite(x[:, [0, 1, j]]).all(axis=1)
            self.coords = x[keep, :2]
            self.dates = x[keep, j]
            self.tree = cKDTree(self.coords) if len(self.coords) >= 5 else None
        return self

    def gas_shift(self, x):
        """Return reversible gas corrections based on observed installation day and leakage."""
        i = self.schema.i
        d, l = x[:, i('install_date')], x[:, i('leakage')]
        d, l = np.nan_to_num(d, nan=0), np.nan_to_num(l, nan=0)
        return np.column_stack([d*l, .0005*d*d*l, (d + .00025*d*d)*l])

    def extra(self, raw):
        """
        Return derived features and the sets of original columns each one depends on. The sets
        prevent use of a target through a derived feature.
        """
        if not self.schema.transformer:
            return [], []
        i = self.schema.i
        j = i('install_date')
        values, deps = [], []
        # Missing coordinate rows cannot use the spatial lookup. Exclude the exact site
        # even in training, so a training row cannot read its own install_date.
        v = np.full(len(raw), np.nan)
        have = np.isfinite(raw[:, :2]).all(axis=1)
        if self.tree is not None and have.any():
            dist, idx = self.tree.query(raw[have, :2], k=5)
            w = np.where(dist > 0, 1/np.maximum(dist, .5), 0)
            v[have] = (self.dates[idx]*w).sum(1)/w.sum(1)
        values.append((v-self.median[j])/self.scale[j])
        deps.append({0, 1})  # externally learned dates, no row's date value
        if self.domain:
            l = i('leakage')
            shifts = self.gas_shift(raw)
            for k, name in enumerate(('measured_gas_a','measured_gas_b','measured_gas_c')):
                t = i(name)
                z = raw[:, t] + shifts[:, k]
                z[~np.isfinite(raw[:, [j, l]]).all(1)] = np.nan
                values.append((z-self.median[t])/self.scale[t])
                deps.append({t, j, l})
        return values, deps

    def design(self, raw, state=None):
        """
        Encode observed values or a current fill state, add missing indicators and return
        design columns with their source dependencies.
        """
        state = raw if state is None else state
        vals, deps = [], []
        for j in range(raw.shape[1]):
            a = np.where(np.isfinite(state[:, j]), state[:, j], self.median[j])
            if j in self.schema.categories:
                for k in range(len(self.schema.categories[j])):
                    vals.append((a == k).astype(float))
                    deps.append({j})
            else:
                vals.append((a-self.median[j])/self.scale[j])
                deps.append({j})
            vals.append(~np.isfinite(raw[:, j]))
            deps.append({j})
        extra, owners = self.extra(raw)
        for v, d in zip(extra, owners):
            vals.extend([np.nan_to_num(v, nan=0.), ~np.isfinite(v)])
            deps.extend([d, d])
        return np.column_stack(vals).astype(np.float32), deps

    def finish(self, raw, point, sd, prob, trace=()):
        """
        Apply admissible fill ranges, category decoding and scale floors, then restore all
        observed inputs in the returned prediction.
        """
        point = point.copy()
        i = self.schema.i
        limits = {'x': (0,99), 'y': (0,99), 'altitude': (100,298),
                  'random_placebo': (0,1), 'install_date': (0,np.inf),
                  'measured_humid': (0,np.inf), 'measured_salt': (0,np.inf),
                  'measured_overgrowth': (0,np.inf), 'length_km': (0,np.inf),
                  'distance_to_sea_km': (0, np.inf), 'leakage': (.8, 1.2),
                  'life': (-np.inf, 100_000), 'total_gas': (0, np.inf)}
        if 'installation_year' in self.schema.names:
            year_col = i('installation_year')
            lower_year = self.minimum[year_col]
            if not np.isfinite(lower_year):
                lower_year = -np.inf
            limits['installation_year'] = (lower_year, SNAPSHOT_YEAR)
        for c, (lo, hi) in limits.items():
            if c in self.schema.names:
                point[:, i(c)] = np.clip(point[:, i(c)], lo, hi)
        if 'installation_year' in self.schema.names:
            # Round candidate years before evaluation computes age at the 2014 snapshot.
            point[:, year_col] = np.rint(point[:, year_col])
        for j, p in prob.items():
            p[:] = np.maximum(np.nan_to_num(p, nan=0.), 1e-8)
            p[:] /= p.sum(1, keepdims=True)
            point[:, j] = np.argmax(p, axis=1)
        observed = np.isfinite(raw)
        point[observed] = raw[observed]
        sd = np.maximum(np.nan_to_num(sd, nan=0), self.range * .002)
        if not np.isfinite(point).all() or not np.isfinite(sd).all():
            raise RuntimeError('Non-finite completion')
        return Prediction(point, sd, prob, tuple(trace))
