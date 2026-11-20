"""Fit frozen chained models, neighbour and median references, and the masked PyTorch imputer."""
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Ridge, LogisticRegression
from .data import FIELD_GROUPS, share_group_draws
from .features import Features, exact_fill


class Floor:
    """Use training medians and modal classes as the comparison baseline."""
    def fit(self, x, f):
        """Store the fitted feature statistics and return this baseline."""
        self.f = f
        return self

    def predict(self, x):
        """
        Return median/mode fills and baseline scales or class probabilities for gaps in the
        supplied rows.
        """
        return self.f.finish(x, np.tile(self.f.median, (len(x), 1)),
                             np.tile(self.f.scale, (len(x), 1)),
                             {j: np.tile(p, (len(x), 1)) for j,p in self.f.marginal.items()})


def class_probs(est, x, size):
    """
    Map an estimator’s class probabilities to all allowed codes, including classes absent in
    training.
    """
    p = np.full((len(x), size), 1e-5)
    p[:, est.classes_.astype(int)] += est.predict_proba(x)
    return p


class Chained:
    """
    Fit one conditional model per target and use a fixed set of models to update the missing
    entries.
    """
    def __init__(self, method, params, seed):
        self.method, self.p, self.seed = method, params, seed

    def fit(self, x, f):
        """
        Fit the ridge/logistic or forest conditionals on observed targets and store the order
        used to fill gaps.
        """
        self.f, self.est = f, {}
        design, deps = f.design(x)
        placebo = set() if not f.schema.transformer else {f.schema.i('random_placebo')}
        observed = np.isfinite(x)
        self.order = np.argsort(np.mean(~observed, axis=0), kind='stable')
        for j in self.order:
            # Exclude the response and placebo even when they enter through a derived column.
            cols = np.array([not ({j} | placebo).intersection(d) for d in deps])
            keep = observed[:, j]
            if keep.sum() < 10:
                continue
            cat = j in f.schema.categories
            if cat and len(np.unique(x[keep, j])) < 2:
                continue
            z, y = design[keep][:, cols], x[keep, j]
            if self.method == 'forest':
                cls = RandomForestClassifier if cat else RandomForestRegressor
                est = cls(n_estimators=self.p['trees'], max_depth=self.p['depth'],
                          min_samples_leaf=self.p['leaf'], max_features=self.p['features'],
                          max_samples=.75, n_jobs=1, random_state=self.seed+int(j))
            elif cat:
                est = LogisticRegression(C=1/self.p['alpha'], max_iter=350)
            else:
                est = Ridge(alpha=self.p['alpha'])
            est.fit(z, y)
            residual = f.scale[j] if cat or self.method == 'forest' else np.std(y-est.predict(z))
            self.est[j] = (cols, est, residual)
        return self

    def predict(self, x):
        """
        Run the frozen conditionals on missing entries and return point predictions and
        uncertainty.
        """
        f = self.f
        out = np.where(np.isfinite(x), x, f.median)
        sd = np.tile(f.scale, (len(x), 1))
        probs = {j: np.tile(v, (len(x), 1)) for j,v in f.marginal.items()}
        miss = ~np.isfinite(x)
        design, deps = f.design(x, out)
        own = {j: [k for k,d in enumerate(deps[:sum(2 if k not in f.schema.categories
                    else len(f.schema.categories[k])+1 for k in range(x.shape[1]))]) if d == {j}]
               for j in self.order}
        trace = []
        for step in range(self.p['passes']):
            old = out.copy()
            for j in self.order:
                rows = miss[:, j]
                if j not in self.est or not rows.any():
                    continue
                cols, est, residual = self.est[j]
                z = design[rows][:, cols]
                if j in probs:
                    p = class_probs(est, z, probs[j].shape[1])
                    probs[j][rows] = p
                    out[rows, j] = p.argmax(1)
                    for k,col in enumerate(own[j][:-1]):
                        design[rows, col] = out[rows, j] == k
                else:
                    if self.method == 'forest':
                        trees = np.stack([tree.predict(z, check_input=False) for tree in est.estimators_])
                        out[rows, j] = trees.mean(0)
                        sd[rows, j] = trees.std(0)
                    else:
                        out[rows, j] = est.predict(z)
                        sd[rows, j] = residual
                    # Stabilize feedback for rare extreme ridge predictions.
                    out[rows,j] = np.clip(out[rows,j], f.median[j]-6*f.scale[j], f.median[j]+6*f.scale[j])
                    design[rows, own[j][0]] = (out[rows,j]-f.median[j])/f.scale[j]
            delta = float(np.mean((np.abs(out-old)/f.scale)[miss])) if miss.any() else 0.
            trace.append(delta)
            if step >= 1 and delta < .003:
                break
        return f.finish(x, out, sd, probs, trace)


class Neighbours:
    """Use observed feature overlap to select comparable training donors for each missing field."""
    def __init__(self, p, seed):
        self.p, self.seed = p, seed

    def representation(self, x):
        """
        Encode rows for neighbour distances, retaining gaps so they do not enter the overlap
        calculation.
        """
        f = self.f
        cols = []
        for j in range(x.shape[1]):
            if f.schema.transformer and f.schema.names[j] == 'random_placebo':
                continue
            a = x[:,j]
            if j in f.schema.categories:
                cols.extend([np.where(np.isfinite(a), (a==k)/np.sqrt(2), np.nan)
                             for k in range(len(f.schema.categories[j]))])
            else:
                cols.append((a-f.median[j])/f.scale[j])
        cols.extend(f.extra(x)[0])
        return np.column_stack(cols).astype(np.float32)

    def fit(self, x, f):
        """Select the training donor sample and cache its feature values and available targets."""
        self.f = f
        idx = np.random.default_rng(self.seed).choice(len(x), min(self.p['donors'], len(x)), replace=False)
        self.x = x[idx]
        z = self.representation(self.x)
        self.obs = np.isfinite(z).astype(np.float32)
        self.z = np.nan_to_num(z)
        self.zz = self.z**2
        self.donors = {j: np.flatnonzero(np.isfinite(self.x[:,j])) for j in range(x.shape[1])}
        return self

    def predict(self, x):
        """
        Return donor-weighted estimates, scales and category probabilities for missing values
        in new rows.
        """
        f = self.f
        out, sd = np.tile(f.median, (len(x),1)), np.tile(f.scale, (len(x),1))
        probs = {j: np.tile(v, (len(x),1)) for j,v in f.marginal.items()}
        for start in range(0, len(x), 256):
            raw = x[start:start+256]
            z = self.representation(raw)
            o = np.isfinite(z).astype(np.float32)
            z = np.nan_to_num(z)
            overlap = o @ self.obs.T
            dist = np.maximum((z**2) @ self.obs.T + o @ self.zz.T - 2*z @ self.z.T, 0)
            dist = dist / np.maximum(overlap, 1) + self.p['penalty'] / np.maximum(overlap,1)
            dist[overlap == 0] = np.inf
            for j, donor in self.donors.items():
                r = np.flatnonzero(np.isnan(raw[:,j]))
                k = min(self.p['k'], len(donor))
                if not len(r) or not k:
                    continue
                dd = dist[np.ix_(r,donor)]
                ii = np.argpartition(dd, k-1, axis=1)[:,:k]
                d = np.take_along_axis(dd, ii, axis=1)
                w = (d+.025)**(-self.p['power'])
                has = w.sum(1) > 0
                if not has.any():
                    continue
                r, w, values = start+r[has], w[has], self.x[donor[ii[has]],j]
                w = w/w.sum(1, keepdims=True)
                if j in probs:
                    probs[j][r] = np.column_stack([(w*(values==c)).sum(1) for c in range(probs[j].shape[1])])
                else:
                    out[r,j] = (w*values).sum(1)
                    sd[r,j] = np.sqrt((w*(values-out[r,j,None])**2).sum(1))
        return f.finish(x,out,sd,probs)


def remask_training_input(original, hide, schema, noise, rules):
    """
    Hide the requested observed entries before applying any exact rules. Return model inputs
    and the hidden targets still requiring prediction.
    """
    masked = np.where(hide, np.nan, original)
    prepared, recovered = exact_fill(masked, schema, noise, rules)
    # Exact recoveries go through the deterministic branch at inference.
    return prepared, hide & ~recovered


class Autoencoder:
    """Learn to reconstruct held-out observations with a masked MLP inspired by ReMasker."""
    def __init__(self, p, seed):
        self.p, self.seed = p, seed

    def fit(self, x, f, *, noise=0., rules=False):
        """
        Train numeric and categorical reconstruction heads on newly hidden training cells.
        Return the fitted model and retain its loss trace.
        """
        import torch
        from torch import nn
        torch.set_num_threads(1)
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.f = f
        self.num = f.schema.numeric
        self.class_heads, width = {}, 2*len(self.num)
        for j, levels in f.schema.categories.items():
            self.class_heads[j] = slice(width, width+len(levels))
            width += len(levels)
        input_dim = f.design(x[:1])[0].shape[1]
        w = self.p['width']
        self.net = nn.Sequential(nn.Linear(input_dim,w), nn.GELU(), nn.Dropout(self.p['dropout']),
                                 nn.Linear(w,w//2), nn.GELU(), nn.Linear(w//2,width))
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.p['lr'], weight_decay=self.p['decay'])
        obs = np.isfinite(x)
        y = torch.as_tensor(np.nan_to_num((x[:,self.num]-f.median[self.num])/f.scale[self.num]), dtype=torch.float32)
        labels = {j: torch.as_tensor(np.nan_to_num(x[:,j]),dtype=torch.long) for j in self.class_heads}
        losses = []
        for _ in range(self.p['epochs']):
            draws = rng.random(x.shape)
            if self.p.get('grouped_remask', False):
                draws = share_group_draws(draws, f.schema, FIELD_GROUPS.values())
            hide = obs & (draws < self.p['remask'])
            raw, hide = remask_training_input(x, hide, f.schema, noise, rules)
            if f.schema.transformer:
                raw[:,f.schema.i('random_placebo')] = np.nan
            z = torch.from_numpy(f.design(raw)[0])
            mask = torch.from_numpy(hide)
            batch_losses = []
            for idx in np.array_split(rng.permutation(len(x)), max(1, int(np.ceil(len(x)/256)))):
                pred = self.net(z[idx])
                loc, log_sd = pred[:,:len(self.num)], pred[:,len(self.num):2*len(self.num)].clamp(-3.,2.)
                err = loc-y[idx]
                # The small MSE term stabilizes point learning when the variance is large.
                cell = log_sd + .5*err.square()*torch.exp(-2*log_sd) + .15*err.square()
                target = mask[idx][:,self.num]
                if not target.any():
                    continue
                loss = cell[target].mean()
                for j, sl in self.class_heads.items():
                    keep = mask[idx,j]
                    if keep.any():
                        loss = loss + .5*nn.functional.cross_entropy(pred[keep,sl], labels[j][idx][keep])
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 4.)
                opt.step()
                batch_losses.append(float(loss.detach()))
            losses.append(float(np.mean(batch_losses)))
        self.losses = tuple(losses)
        return self

    def predict(self, x):
        """
        Return numeric point and scale estimates and categorical probabilities for gaps in new
        rows.
        """
        import torch
        from scipy.special import softmax
        f = self.f
        raw = x.copy()
        if f.schema.transformer:
            raw[:,f.schema.i('random_placebo')] = np.nan
        self.net.eval()
        with torch.no_grad():
            a = self.net(torch.from_numpy(f.design(raw)[0])).numpy()
        out, sd = np.tile(f.median,(len(x),1)), np.tile(f.scale,(len(x),1))
        n = len(self.num)
        out[:,self.num] = a[:,:n]*f.scale[self.num]+f.median[self.num]
        sd[:,self.num] = np.exp(np.clip(a[:,n:2*n],-3,2))*f.scale[self.num]
        return f.finish(x,out,sd,{j:softmax(a[:,sl],axis=1) for j,sl in self.class_heads.items()},self.losses)
