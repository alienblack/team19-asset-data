"""
Screen observed dates and model codes by predicting them after temporarily hiding each
candidate.
"""
import numpy as np
import pandas as pd

from .data import OUTLIER_ANCHOR, split_blocks
from .evaluate import MIN_PATTERN, pattern_info, qconformal
from .features import Features
from .study import create, load

TARGETS = ('install_date', 'model_id')
METHODS = ('forest', 'mixture')
FLAG_LEVELS = (.01, .05, .10)
PRIMARY_LEVEL = .05


def screening_view(rows, asset_ids, j):
    """
    Select recorded target values and return their IDs, a masked input copy and the held-back
    values.
    """
    observed = np.isfinite(rows[asset_ids, j])
    ids = asset_ids[observed]
    x = rows[ids].copy()
    recorded = x[:, j].copy()
    x[:, j] = np.nan
    return ids, x, recorded


def conformity_cutoffs(cal_scores, cal_keys, scores, keys, alpha=PRIMARY_LEVEL, min_support=MIN_PATTERN):
    """
    Return tail probabilities, cutoffs and support using clean calibration scores in matched
    patterns or the pool.
    """
    p = np.ones(len(scores))
    cutoff = np.full(len(scores), np.inf)
    support = np.zeros(len(scores), dtype=int)
    source = np.full(len(scores), 'pooled', dtype='<U12')
    for key in np.unique(keys):
        group = cal_scores[cal_keys == key]
        level = 'key+burden'
        if len(group) < min_support:
            group = cal_scores
            level = 'pooled'
        selected = keys == key
        if not len(group):
            continue
        group = np.sort(group)
        # Including ties makes this conservative for discrete scores and probabilities.
        p[selected] = (1 + len(group) - np.searchsorted(group, scores[selected], side='left')) / (len(group) + 1)
        cutoff[selected] = qconformal(group, alpha)
        support[selected] = len(group)
        source[selected] = level
    return p, cutoff, support, source


def discrepancy(pred, recorded, j, categorical):
    """Return the scaled numeric residual or one minus the probability of the recorded category."""
    if categorical:
        probability = pred.proba[j][np.arange(len(recorded)), recorded.astype(int)]
        return 1 - probability
    return np.abs(recorded - pred.point[:, j]) / pred.scale[:, j]


def screen_pair(data, out, gs, ss, settings):
    """
    Fit and screen one generator/split pair, then export candidate values, scores and truth
    labels for evaluation.
    """
    dest = out/'screening'/f'{gs}-{ss}.csv.gz'
    if dest.exists():
        return dest.name
    schema, truth, raw, cal_raw, groups = load(str(data), OUTLIER_ANCHOR, gs)
    ids, _ = split_blocks(truth, ss, schema, groups)
    train = raw[ids['train']]
    features = Features(schema, True).fit(train)
    rows = []
    for method in METHODS:
        model = create(method, settings[method], ss).fit(train, features)
        for field in TARGETS:
            j = schema.i(field)
            ci, cx, recorded_cal = screening_view(cal_raw, ids['calibration'], j)
            ti, tx, tv = screening_view(raw, ids['test'], j)
            clean = recorded_cal == truth[ci, j]
            if clean.sum() < MIN_PATTERN:
                raise RuntimeError(f'Insufficient verified clean calibration rows for {field}')
            cp = model.predict(cx[clean])
            tp = model.predict(tx)
            cat = j in schema.categories
            score = discrepancy(tp, tv, j, cat)
            _, cv_keys, cb = pattern_info(cx[clean], schema, j)
            _, tv_keys, tb = pattern_info(tx, schema, j)
            p, cutoff, n, source = conformity_cutoffs(
                discrepancy(cp, recorded_cal[clean], j, cat), cv_keys*10+cb,
                score, tv_keys*10+tb,
            )
            frame = pd.DataFrame(dict(
                generator_seed=gs, split_seed=ss, asset_id=ti, method=method, attribute=field,
                recorded=tv, truth=truth[ti, j], prediction=tp.point[:, j], injected_error=tv != truth[ti, j],
                score=score, p_value=p, cutoff=cutoff, calibration_n=n, calibration_source=source,
                keys_visible=tv_keys, other_gaps=np.isnan(tx).sum(axis=1)-1,
            ))
            if cat:
                frame['recorded_probability'] = 1-score
                frame['probability_threshold'] = np.maximum(0, 1-cutoff)
            else:
                half = cutoff*tp.scale[:, j]
                frame['lower'] = tp.point[:, j]-half
                frame['upper'] = tp.point[:, j]+half
            rows.append(frame)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix('.tmp.gz')
    pd.concat(rows, ignore_index=True).to_csv(tmp, index=False, compression={'method':'gzip', 'mtime':0})
    tmp.replace(dest)
    return dest.name
