"""Fit a Gaussian mixture to incomplete assets and predict from conditional distributions."""
import numpy as np
from scipy.special import softmax, logsumexp
from sklearn.cluster import KMeans


class MaskedMixture:
    """
    Use shared-covariance Gaussian components and observed-data EM to estimate missing numeric
    and categorical fields.
    """
    def __init__(self, params, seed):
        self.p, self.seed = params, seed

    def encode(self, x):
        """
        Convert asset rows to the mixture representation, including categorical encoding and
        optional gas correction.
        """
        f = self.f
        z = (x[:,f.schema.numeric]-self.center)/self.scale
        if f.domain:
            z[:,self.gas_idx] += f.gas_shift(x)/self.scale[self.gas_idx]
        cols = [z]
        for j, levels in f.schema.categories.items():
            cols.append(np.where(np.isfinite(x[:,j,None]),
                                 self.p['class_weight']*(x[:,j,None] == np.arange(len(levels))), np.nan))
        return np.column_stack(cols)

    @staticmethod
    def preparation(z):
        """
        Group encoded rows by observed pattern so repeated conditional calculations can be
        shared.
        """
        mask = np.isfinite(z)
        pattern, row = np.unique(mask, axis=0, return_inverse=True)
        return np.nan_to_num(z), pattern.astype(float), row

    @staticmethod
    def expect(prep, mu, cov, prior, moments=True):
        """
        Return component responsibilities and conditional moments for the supplied incomplete
        patterns and mixture parameters.
        """
        z, mask, row = prep
        n, d = z.shape
        s = cov[None,:,:] * mask[:,:,None] * mask[:,None,:]
        s[:,np.arange(d),np.arange(d)] += 1-mask
        inv = np.linalg.inv(s)
        _, ldet = np.linalg.slogdet(s)
        cross = cov[None,:,:]*mask[:,:,None]
        gain = inv @ cross
        obs_mu = mu[None,:,:]*mask[row,None,:]
        diff = z[:,None,:]-obs_mu
        proj = np.einsum('nkd,ndm->nkm',diff, inv[row])
        logp = -.5*((proj*diff).sum(2)+ldet[row,None]+mask[row].sum(1)[:,None]*np.log(2*np.pi))
        logp += np.log(prior)[None,:]
        resp = softmax(logp, axis=1)
        if not moments:
            return resp
        fill = mu[None,:,:]+np.einsum('nkd,ndm->nkm',diff,gain[row])
        cond = cov[None,:,:]-np.swapaxes(cross,1,2) @ gain
        return resp, fill, cond, float(logsumexp(logp,axis=1).mean())

    def fit(self, x, f):
        """Fit the mixture to measured training rows, record its trace and return the fitted model."""
        self.f = f
        self.center, self.scale = f.median[f.schema.numeric].copy(), f.scale[f.schema.numeric].copy()
        if f.domain:
            self.gas_idx = [f.schema.numeric.index(f.schema.i('measured_gas_'+c)) for c in 'abc']
            # Recenter shifted gas so missing initialization is on the correct scale.
            self.center[self.gas_idx] = np.nanmedian(x[:,[f.schema.i('measured_gas_'+c) for c in 'abc']]
                                                     + f.gas_shift(x), axis=0)
        z = self.encode(x)
        prep = self.preparation(z)
        base, mask, row = prep
        k = self.p['clusters']
        init = KMeans(n_clusters=k,n_init=1,max_iter=20,random_state=self.seed).fit(base)
        mu = init.cluster_centers_
        cov = np.cov(base,rowvar=False)+np.eye(base.shape[1])*.2
        prior = np.maximum(np.bincount(init.labels_, minlength=k),1)/len(base)
        hist = []
        for _ in range(self.p['steps']):
            r, q, cond, ll = self.expect(prep,mu,cov,prior)
            hist.append(ll)
            nk = r.sum(0)+1e-8
            mu = np.einsum('nk,nkd->kd',r,q)/nk[:,None]
            err = q-mu[None,:,:]
            e = (err*np.sqrt(r[:,:,None])).reshape(-1,base.shape[1])
            cov = (e.T @ e + np.einsum('n,ndm->dm',np.bincount(row),cond))/len(base)
            cov = .5*(cov+cov.T)+np.eye(cov.shape[0])*self.p['reg']
            prior = nk/nk.sum()
            if len(hist) >= 5 and abs(hist[-1]-hist[-2]) < .002:
                break
        self.mu, self.cov, self.prior, self.trace = mu,cov,prior,tuple(hist)
        r = self.expect(prep,mu,cov,prior,moments=False)
        self.class_by_cluster = {}
        for j,levels in f.schema.categories.items():
            p = np.column_stack([r[x[:,j]==c].sum(0) for c in range(len(levels))])+.5
            self.class_by_cluster[j] = p/p.sum(1,keepdims=True)
        return self

    def predict(self,x):
        """
        Condition the fitted mixture on observed test entries and return fills, scales and
        category probabilities.
        """
        f = self.f
        prep = self.preparation(self.encode(x))
        r,q,cond,_ = self.expect(prep,self.mu,self.cov,self.prior)
        out,sd = np.tile(f.median,(len(x),1)),np.tile(f.scale,(len(x),1))
        n = len(f.schema.numeric)
        m = np.einsum('nk,nkd->nd',r,q[:,:,:n])
        v = np.einsum('nk,nkd->nd',r,(q[:,:,:n]-m[:,None,:])**2)
        v += np.diagonal(cond,axis1=1,axis2=2)[prep[2],:n]
        out[:,f.schema.numeric] = m*self.scale+self.center
        sd[:,f.schema.numeric] = np.sqrt(np.maximum(v,0))*self.scale
        if f.domain:
            out[:,[f.schema.numeric[k] for k in self.gas_idx]] -= f.gas_shift(x)
        probs = {}
        for j,p in self.class_by_cluster.items():
            probs[j] = r @ p
        return f.finish(x,out,sd,probs,self.trace)
