"""
Calibrate predictive confidence and score hidden cells, distributions, downstream use and
record consistency.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import f1_score
from .data import LINKS, SNAPSHOT_YEAR

ALPHA = .10
MIN_CATEGORICAL_FLOOR_ERROR = .05
MIN_PATTERN = 50


@dataclass(frozen=True)
class Tolerance:
    """
    Store tight and loose numeric limits, with optional relative scaling and a training-range
    floor.
    """
    tight: float
    loose: float
    relative: bool = False
    range_floor: float = 0.


# Absolute limits for grid cells, days and years avoid tolerances that grow with the calendar offset.
NUMERIC_TOLERANCES = {
    'x': Tolerance(1, 5),
    'y': Tolerance(1, 5),
    'install_date': Tolerance(7, 30),
    'altitude': Tolerance(1, 5),
    'life': Tolerance(100, 500),
    'measured_humid': Tolerance(.10, .15, relative=True),
    'measured_salt': Tolerance(.10, .15, relative=True),
    'measured_overgrowth': Tolerance(.10, .15, relative=True),
    'random_placebo': Tolerance(.10, .15, relative=True),
    'measured_gas_a': Tolerance(.05, .10, relative=True),
    'measured_gas_b': Tolerance(.05, .10, relative=True),
    'measured_gas_c': Tolerance(.05, .10, relative=True),
    'leakage': Tolerance(.05, .10, relative=True),
    'total_gas': Tolerance(.01, .05, relative=True),
    'installation_year': Tolerance(5, 10),
    # The overhead distances retain the same tolerance for both checks.
    'distance_to_sea_km': Tolerance(.05, .05, relative=True, range_floor=.02),
    'length_km': Tolerance(.05, .05, relative=True, range_floor=.02),
}
LIFE_TIGHT_TOL = NUMERIC_TOLERANCES['life'].tight
LIFE_LOOSE_TOL = NUMERIC_TOLERANCES['life'].loose


def tolerance_limit(attribute, reference, train_range, *, tight):
    """
    Return the absolute error bound for a field and reference value using the tight or loose
    policy.
    """
    spec = NUMERIC_TOLERANCES[attribute]
    amount = spec.tight if tight else spec.loose
    if spec.relative:
        return np.maximum(amount * np.abs(reference), spec.range_floor * train_range)
    return amount


def pattern_info(raw, schema, j):
    """Return exact missingness codes, key-input visibility and gap-count buckets for a target."""
    miss = np.isnan(raw)
    exact = miss.astype(np.uint32) @ (1 << np.arange(raw.shape[1],dtype=np.uint32))
    cols = [schema.i(k) for k in LINKS[schema.names[j]]]
    visible = np.isfinite(raw[:,cols]).all(1)
    others = miss.sum(1)-miss[:,j]
    buckets = np.digitize(others, [3, 6, 9, 12])
    return exact, visible, buckets


def qconformal(x, alpha=ALPHA):
    """Return the finite-sample split-conformal residual quantile at the requested error level."""
    n = len(x)
    k = int(np.ceil((n+1)*(1-alpha)))
    return float(np.partition(x,k-1)[k-1]) if 0 < k <= n else np.inf


@dataclass
class Confidence:
    """
    Hold calibrated bands, categorical probabilities and provenance for each confidence
    estimate.
    """
    lower: np.ndarray
    upper: np.ndarray
    pooled_half: np.ndarray
    raw_half: np.ndarray
    proba: dict
    source: np.ndarray
    support: np.ndarray
    temperature: dict


def calibrate(f, cal_raw, cal_truth, cal_rule, cal, raw, rule, pred, min_pattern=MIN_PATTERN):
    """
    Use independent calibration truth to scale test intervals by missing pattern and
    temperature-scale categorical probabilities.
    """
    h = pred.scale.copy()
    pooled, before = h.copy(), h * 1.644853626951
    src, n = np.full(raw.shape, 'observed', dtype='<U16'), np.zeros(raw.shape, int)
    src[rule] = 'rule'
    probs, temperatures = {}, {}
    for j in range(raw.shape[1]):
        keep = np.isnan(cal_raw[:,j]) & ~cal_rule[:,j]
        want = np.isnan(raw[:,j]) & ~rule[:,j]
        if j in f.schema.categories:
            p = cal.proba[j][keep]
            y = cal_truth[keep,j].astype(int)
            labels = np.arange(len(y))
            if len(y):
                def objective(log_t):
                    pp = softmax(np.log(p)/np.exp(log_t), axis=1)
                    return -np.log(np.clip(pp[labels,y],1e-12,1)).mean()
                temp = float(np.exp(minimize_scalar(objective,bounds=(-1.6,2.3),method='bounded').x))
            else:
                temp = 1.
            probs[j] = softmax(np.log(pred.proba[j])/temp,axis=1)
            temperatures[j] = temp
            src[want,j], n[want,j] = 'temperature', len(y)
            continue
        a, v, b = pattern_info(cal_raw,f.schema,j)
        ta, tv, tb = pattern_info(raw,f.schema,j)
        score = np.abs(cal.point[keep,j]-cal_truth[keep,j])/cal.scale[keep,j]
        groups = ((a[keep],ta,'exact'),(v[keep]*10+b[keep],tv*10+tb,'key+burden'),
                  (v[keep],tv,'key'))
        q0 = qconformal(score)
        q, level, support = np.full(len(raw),q0), np.full(len(raw),'pooled',dtype='<U16'), np.full(len(raw),len(score))
        # Start with the pooled cutoff and overwrite it only when a more specific group has support.
        for ck, tk, name in reversed(groups):
            for key in np.unique(tk[want]):
                sub = score[ck==key]
                if len(sub) >= min_pattern:
                    hit = (tk==key) & want
                    q[hit], level[hit], support[hit] = qconformal(sub), name, len(sub)
        h[:,j], pooled[:,j] = h[:,j]*q, pooled[:,j]*q0
        src[want,j], n[want,j] = level[want], support[want]
    for arr in (h, pooled, before):
        arr[rule] = 0.
    return Confidence(pred.point-h,pred.point+h,pooled,before,probs,src,n,temperatures)


def category_metrics(y, p, labels):
    """
    Return accuracy, macro-F1, log loss, Brier score and top-label calibration error from true
    codes and probabilities.
    """
    pred = p.argmax(1)
    max_p = p.max(1)
    correct = (y==pred)
    idx = np.minimum((10*max_p).astype(int), 9)
    ece = sum(abs(max_p[idx==b].mean()-correct[idx==b].mean())*(idx==b).mean()
              for b in np.unique(idx))
    return dict(accuracy=correct.mean(), macro_f1=f1_score(y,pred,labels=np.unique(y),average='macro',zero_division=0),
                log_loss=-np.log(np.maximum(p[np.arange(len(y)),y],1e-12)).mean(),
                brier=np.mean(((p-np.eye(labels)[y])**2).sum(1)), ece=ece)


def interval_rows(y, center, low, high):
    """Return coverage flags, widths and interval scores for the supplied bands and truths."""
    return (y>=low)&(y<=high), high-low, high-low+2/ALPHA*(np.maximum(low-y,0)+np.maximum(y-high,0))


def relative_skill(error, floor_error, categorical):
    """
    Return one minus method error divided by floor error. Suppress undefined ratios and
    categorical comparisons with an almost perfect floor.
    """
    if floor_error <= 0 or (categorical and floor_error < MIN_CATEGORICAL_FLOOR_ERROR):
        return np.nan
    return 1 - error / floor_error


def evaluate(raw, truth, pred, floor, conf, rule, f):
    """
    Score missing cells against known truth and return cell, pattern and distribution
    summaries plus candidate decisions.
    """
    schema = f.schema
    cells, patterns, dist = [], [], []
    miss = np.isnan(raw)
    recommendation = np.full(raw.shape,'observed',dtype='<U8')
    for j,c in enumerate(schema.names):
        use = miss[:,j]
        if not use.any():
            continue
        exact, visible, burden = pattern_info(raw,schema,j)
        y, p, b = (x[:,j] for x in (truth,pred.point,floor.point))
        cat = j in schema.categories
        if cat:
            proba = conf.proba[j]
            raw_proba = pred.proba[j]
            precision = proba.max(1)
            tight_ok = loose_ok = p == y
            e, be = (~tight_ok).astype(float), (b!=y).astype(float)
            auto = precision >= .95
            review = precision >= .65
            q = len(schema.categories[j])
            tv = np.bincount(y.astype(int),minlength=q)/len(y)
            pv = np.bincount(p.astype(int),minlength=q)/len(y)
            ft = np.bincount(y[use].astype(int),minlength=q)/use.sum()
            fp = np.bincount(p[use].astype(int),minlength=q)/use.sum()
            for k,name in enumerate(schema.categories[j]):
                dist.append(dict(attribute=c, level=str(name), truth_share=tv[k], completed_share=pv[k],
                                 hidden_truth_share=ft[k], hidden_fill_share=fp[k],
                                 probability_share=(np.sum(~use & (p==k))+proba[use,k].sum())/len(y),
                                 full_tv=.5*np.abs(tv-pv).sum(), hidden_tv=.5*np.abs(ft-fp).sum()))
        else:
            e, be = np.abs(p-y), np.abs(b-y)
            tight_ok = e <= tolerance_limit(c, y, f.range[j], tight=True) + 1e-10
            loose_ok = e <= tolerance_limit(c, y, f.range[j], tight=False) + 1e-10
            covered, width, score = interval_rows(y,p,conf.lower[:,j],conf.upper[:,j])
            raw_covered = np.abs(p-y) <= conf.raw_half[:,j]+1e-10
            pool_covered = np.abs(p-y) <= conf.pooled_half[:,j]+1e-10
            covered |= rule[:,j] & (e<1e-9)
            # Sufficient bound: every value in the interval is within tolerance,
            # with relative tolerance evaluated at the minimum possible |truth|.
            min_abs = np.maximum(np.abs(p)-width/2,0)
            tight_bound = tolerance_limit(c, min_abs, f.range[j], tight=True)
            loose_bound = tolerance_limit(c, min_abs, f.range[j], tight=False)
            auto = (width/2 <= tight_bound) | rule[:,j]
            loose_eligible = width/2 <= loose_bound
            review = loose_eligible
            dist.append(dict(attribute=c, level='', w1_hidden=wasserstein_distance(y[use],p[use]),
                             w1_completed=wasserstein_distance(y,p),
                             nw1_hidden=wasserstein_distance(y[use],p[use])/f.scale[j],
                             nw1_completed=wasserstein_distance(y,p)/f.scale[j],
                             truth_sd=y[use].std(), fill_sd=p[use].std()))
            for level, keys in (('exact',exact),('key+burden',visible*10+burden)):
                for key in np.unique(keys[use]):
                    sel = use & (keys==key)
                    if sel.sum() >= MIN_PATTERN:
                        patterns.append(dict(attribute=c, pattern_type=level, pattern=int(key), n=int(sel.sum()),
                                             coverage=covered[sel].mean(), coverage_raw=raw_covered[sel].mean(),
                                             coverage_pooled=pool_covered[sel].mean(), interval_width=width[sel].mean(),
                                             cal_support_median=float(np.median(conf.support[sel,j])),
                                             exact_match_share=np.mean(conf.source[sel,j]=='exact')))
        # The tight gate takes priority. The loose gate only routes the gap to review.
        recommendation[use,j] = np.where(auto[use],'auto',np.where(review[use],'review','abstain'))
        for group, sel in [('all',use),('keys_visible',use&visible),('keys_missing',use&~visible)] if LINKS[c] else [('all',use)]:
            if not sel.any():
                continue
            ae, base = e[sel].mean(), be[sel].mean()
            skill = relative_skill(ae, base, cat)
            row = dict(attribute=c, type='categorical' if cat else 'numeric', group=group, n=int(sel.sum()),
                       error=ae, floor_error=base, skill=skill, tolerance=loose_ok[sel].mean(),
                       tolerance_tight=tight_ok[sel].mean(),
                       rule_share=rule[sel,j].mean(), auto_share=auto[sel].mean(),
                       auto_n=int(np.sum(auto&sel)), auto_correct=int(np.sum(auto&sel&tight_ok)),
                       review_share=np.mean(recommendation[sel,j]=='review'),
                       abstain_share=np.mean(recommendation[sel,j]=='abstain'),
                       auto_precision=tight_ok[sel&auto].mean() if (sel&auto).any() else np.nan)
            if cat:
                yy = y[sel].astype(int)
                row.update(category_metrics(yy,proba[sel],q))
                m0 = category_metrics(yy,raw_proba[sel],q)
                row.update({k+'_raw':m0[k] for k in ('log_loss','brier','ece')})
                row['temperature'] = conf.temperature[j]
            else:
                row.update(coverage=covered[sel].mean(), coverage_raw=raw_covered[sel].mean(),
                           coverage_pooled=pool_covered[sel].mean(), interval_width=width[sel].mean(),
                           interval_score=score[sel].mean(), normalized_width=width[sel].mean()/f.scale[j])
                row.update(loose_eligible_n=int(np.sum(sel & loose_eligible)),
                           loose_correct_n=int(np.sum(sel & loose_eligible & loose_ok)))
            cells.append(row)
    return cells, patterns, dist, recommendation



def installation_age_scores(raw, truth, point, floor, age_edges):
    """
    Return hidden-date error and signed bias by true age band, and age-share errors on the
    whole test register.
    """
    names = ('oldest', 'older_middle', 'younger_middle', 'youngest')
    hidden = np.isnan(raw)
    signed_error = point - truth
    floor_error = np.abs(floor - truth)
    truth_band = np.searchsorted(age_edges, truth, side='right')
    filled_band = np.searchsorted(age_edges, point, side='right')
    rows = []
    bands = [('all', np.ones(len(truth), dtype=bool), None, None, None),
             ('older_half', truth_band < 2, filled_band < 2, None, age_edges[1])]
    for index, name in enumerate(names):
        low = age_edges[index - 1] if index > 0 else None
        high = age_edges[index] if index < len(age_edges) else None
        bands.append((name, truth_band == index, filled_band == index, low, high))
    for name, true_members, fill_members, low, high in bands:
        target = hidden & true_members
        truth_n = int(true_members.sum())
        row = dict(age_band=name, lower_install_day=low, upper_install_day=high,
                   hidden_n=int(target.sum()), band_n=truth_n,
                   masked_share=target.sum()/truth_n if truth_n else np.nan,
                   mae=np.abs(signed_error[target]).mean() if target.any() else np.nan,
                   signed_bias_days=signed_error[target].mean() if target.any() else np.nan,
                   floor_mae=floor_error[target].mean() if target.any() else np.nan)
        if fill_members is not None:
            row.update(truth_share=true_members.mean(), completed_share=fill_members.mean(),
                       age_share_error_pp=100*(fill_members.mean()-true_members.mean()))
        rows.append(row)
    return rows


def rule_effects(schema, raw, truth, model_only, rules_first, rule_mask, floor, features):
    """
    Compare paired predictions and split the skill change between exact replacements and all
    remaining gaps.
    """
    rows = []
    for col in schema.numeric:
        if schema.names[col] == 'random_placebo':
            continue
        hidden = np.isnan(raw[:, col])
        if not hidden.any():
            continue
        direct = hidden & rule_mask[:, col]
        remaining = hidden & ~rule_mask[:, col]
        error_model = np.abs(model_only[:, col] - truth[:, col])
        error_rules = np.abs(rules_first[:, col] - truth[:, col])
        error_floor = np.abs(floor[:, col] - truth[:, col])
        improvement = error_model - error_rules
        denom = error_floor[hidden].sum()
        tolerance = tolerance_limit(schema.names[col], truth[:, col], features.range[col], tight=False)
        rows.append(dict(
            attribute=schema.names[col], hidden_n=int(hidden.sum()), direct_n=int(direct.sum()),
            remaining_n=int(remaining.sum()),
            direct_model_within_n=int(np.sum(direct & (error_model <= tolerance + 1e-10))),
            direct_model_mae=error_model[direct].mean() if direct.any() else np.nan,
            direct_floor_mae=error_floor[direct].mean() if direct.any() else np.nan,
            direct_rules_mae=error_rules[direct].mean() if direct.any() else np.nan,
            remaining_model_mae=error_model[remaining].mean() if remaining.any() else np.nan,
            remaining_rules_mae=error_rules[remaining].mean() if remaining.any() else np.nan,
            direct_skill_delta=improvement[direct].sum()/denom if denom else np.nan,
            remaining_skill_delta=improvement[remaining].sum()/denom if denom else np.nan,
        ))
    return rows

def table_checks(f, raw, truth, point, seed):
    """
    Return mixed-table energy distance and correlation error, with an age-share check for
    overhead assets.
    """
    def embed(z):
        arr = [(z[:,f.schema.numeric]-f.median[f.schema.numeric])/f.scale[f.schema.numeric]]
        for j, levels in f.schema.categories.items():
            arr.append(np.eye(len(levels))[z[:,j].astype(int)]/np.sqrt(2))
        return np.column_stack(arr)
    a,b = embed(truth),embed(point)
    idx = np.random.default_rng(seed).choice(len(a), min(350,len(a)), replace=False)
    ea,eb = a[idx],b[idx]
    energy = max(0., 2*cdist(ea,eb).mean()-cdist(ea,ea).mean()-cdist(eb,eb).mean())
    def corr(x):
        x = x-x.mean(0)
        x = x/np.maximum(np.linalg.norm(x,axis=0),1e-12)
        return x.T @ x
    diff = np.abs(corr(a)-corr(b))
    tab = dict(energy_distance=energy, correlation_mae=diff[np.triu_indices(a.shape[1],1)].mean())
    if not f.schema.transformer:
        j = f.schema.i('installation_year')
        cutoff = SNAPSHOT_YEAR - 50
        tab.update(aged_50_truth=np.mean(truth[:, j] <= cutoff),
                   aged_50_completed=np.mean(np.rint(point[:, j]) <= cutoff))
        tab['aged_50_shift_pp']=100*(tab['aged_50_completed']-tab['aged_50_truth'])
    return tab


def consistency(f, raw, point, truth):
    """
    Return agreement rates for the transformer identities in completed and already observed
    records.
    """
    if not f.schema.transformer:
        return []
    formulas = {
        'altitude': (('x','y'),lambda a,b: a+b+100),
        'total_gas': (tuple('measured_gas_'+c for c in 'abc'),lambda a,b,c:a+b+c),
        'measured_overgrowth': (('measured_humid','measured_salt'),lambda a,b:np.maximum(2*a-b,0)),
    }
    rows = []
    for out,(cols,fn) in formulas.items():
        j = f.schema.i(out)
        k = [f.schema.i(c) for c in cols]
        a, b = point[:,j], fn(*point[:,k].T)
        hit = np.abs(a-b) <= tolerance_limit(out, a, f.range[j], tight=False) + 1e-10
        exact = np.isclose(a,b,rtol=1e-9,atol=1e-9)
        use = np.isnan(raw[:,[j,*k]]).any(1)
        rows.append(dict(formula=out,n=int(use.sum()), filled_rule_pass=hit[use].mean() if use.any() else np.nan,
                         filled_exact_pass=exact[use].mean() if use.any() else np.nan,
                         observed_rule_pass=hit[~use].mean() if (~use).any() else np.nan))
    return rows


def downstream(f, train_truth, train_raw, train_fill, test_truth, test_fill):
    """
    Fit the same ridge specification with clean or filled features and score life error and
    ranking against clean test truth.
    """
    i = f.schema.i
    j = i('life')
    keep = np.isfinite(train_raw[:,j])
    if keep.sum() < 30:
        raise ValueError('Downstream training labels unavailable')
    def design(z):
        date = z[:,i('install_date')]/1000
        cat = np.eye(4)[z[:,i('model_id')].astype(int)]
        hazards = z[:,[i(c) for c in ('measured_humid','measured_salt','measured_overgrowth')]]
        gas = z[:,i('total_gas')]/30000
        return np.column_stack([cat,date,gas,hazards,*[cat*(date*h)[:,None] for h in hazards.T]])
    def score(y,p):
        k = max(1,len(y)//10)
        return (np.abs(y-p).mean(), float(spearmanr(y,p).statistic),
                len(np.intersect1d(np.argpartition(y,k-1)[:k],np.argpartition(p,k-1)[:k]))/k)
    y = test_truth[:,j]
    clean = Ridge(alpha=1.).fit(design(train_truth[keep]),train_truth[keep,j])
    filled = Ridge(alpha=1.).fit(design(train_fill[keep]),train_raw[keep,j])
    rows = []
    for name, model, x in (('truth_train_truth_test',clean,test_truth),
                           ('imputed_train_truth_test',filled,test_truth),
                           ('truth_train_imputed_test',clean,test_fill),
                           ('imputed_train_imputed_test',filled,test_fill)):
        mae,rho,overlap = score(y,model.predict(design(x)))
        rows.append(dict(setting=name,mae=mae,spearman=rho,lowest_life_decile_recall=overlap,train_labels=int(keep.sum())))
    return rows


def cell_export(schema, ids, raw, truth, pred, conf, rule, decision, floor):
    """
    Build the per-gap ledger with row key, truth, candidate fill, confidence and decision for
    verification.
    """
    r,j = np.where(np.isnan(raw))
    df = pd.DataFrame(dict(asset_id=ids[r], attribute=np.array(schema.names)[j],
                           truth=truth[r,j], fill=pred.point[r,j], floor=floor.point[r,j],
                           lower=conf.lower[r,j], upper=conf.upper[r,j], scale=pred.scale[r,j],
                           confidence_source=conf.source[r,j], calibration_n=conf.support[r,j], rule=rule[r,j],
                           decision=decision[r,j]))
    for c,p in conf.proba.items():
        sel = j==c
        df.loc[sel,['lower','upper','scale']] = np.nan
        for k in range(p.shape[1]):
            df.loc[sel,f'p_code_{k}'] = p[r[sel],k]
    return df
