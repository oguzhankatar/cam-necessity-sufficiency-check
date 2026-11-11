"""cross-check: a from-first-principles re-implementation of the necessity /
sufficiency estimator.  Nothing is imported from the project.

Definitions, taken from the project's own prose (nslib docstring):
    N(t) = z(A) - z(A with cell t replaced by B's cell t)
    S(t) = z(B with cell t replaced by A's cell t) - z(B)
Everything is a LITERAL loop: build the perturbed (T,C) cell matrix, run the
head on it.  No rank-1 algebra, no batching tricks, float64 everywhere the head
allows it.
"""
import numpy as np
import torch

try:
    from scipy.stats import spearmanr, kendalltau
except Exception:                                                # pragma: no cover
    spearmanr = kendalltau = None


@torch.no_grad()
def NS_literal(head, XA, XB, c):
    """head: (B,T,C)->(B,K) callable.  XA,XB: (T,C).  Returns N,S as float64."""
    T = XA.shape[0]
    zA = float(head(XA[None])[0, c])
    zB = float(head(XB[None])[0, c])
    N = np.empty(T, dtype=np.float64)
    S = np.empty(T, dtype=np.float64)
    for t in range(T):
        Xn = XA.clone(); Xn[t] = XB[t]
        Xs = XB.clone(); Xs[t] = XA[t]
        N[t] = zA - float(head(Xn[None])[0, c])
        S[t] = float(head(Xs[None])[0, c]) - zB
    return N, S, zA, zB


@torch.no_grad()
def subset_delta(head, XA, XB, c, masks):
    """z(A) - z(A with the cells in each mask replaced by B).  Literal loop."""
    zA = float(head(XA[None])[0, c])
    out = np.empty(len(masks), dtype=np.float64)
    for i, mk in enumerate(masks):
        X = XA.clone(); X[mk] = XB[mk]
        out[i] = zA - float(head(X[None])[0, c])
    return out


def stats(N, S):
    d = {}
    d['corr_pearson'] = float(np.corrcoef(N, S)[0, 1]) if (N.std() > 0 and S.std() > 0) else np.nan
    if spearmanr is not None:
        d['corr_spearman'] = float(spearmanr(N, S).statistic)
        d['corr_kendall'] = float(kendalltau(N, S).statistic)
    # "the same number" -- the claim the paper actually makes in words
    sc = max(np.abs(N).max(), np.abs(S).max()) + 1e-300
    d['max_abs_diff_rel'] = float(np.abs(N - S).max() / sc)
    d['mean_abs_diff_rel'] = float(np.abs(N - S).mean() / sc)
    # slope/intercept of S on N -- corr==1 only pins these up to affine
    if N.std() > 0:
        d['slope'] = float(np.polyfit(N, S, 1)[0])
        d['intercept_rel'] = float(np.polyfit(N, S, 1)[1] / sc)
    return d


def additivity(head, XA, XB, c, N, rs, nsub=64, npair=256):
    """Rhalf and I2, literal loops (the project's own definitions)."""
    T = XA.shape[0]
    masks = []
    for _ in range(nsub):
        m = np.zeros(T, dtype=bool); m[rs.choice(T, T // 2, replace=False)] = True
        masks.append(m)
    D = subset_delta(head, XA, XB, c, masks)
    add = np.array([N[m].sum() for m in masks])
    Rh = float(np.abs(D - add).mean() / (np.abs(D).mean() + 1e-300))
    pm = []
    p = rs.randint(0, T, (npair, 2)); p = p[p[:, 0] != p[:, 1]]
    for a, b in p:
        m = np.zeros(T, dtype=bool); m[a] = True; m[b] = True; pm.append(m)
    D2 = subset_delta(head, XA, XB, c, pm)
    add2 = np.array([N[m].sum() for m in pm])
    I2 = float(np.abs(D2 - add2).mean() / (np.abs(D2).mean() + 1e-300))
    return Rh, I2


def report(tag, head, XA, XB, c, rs, do_add=True):
    N, S, zA, zB = NS_literal(head, XA, XB, c)
    d = stats(N, S)
    if do_add:
        d['Rhalf'], d['I2'] = additivity(head, XA, XB, c, N, rs)
    d['sumN_over_zA_minus_zB'] = float(N.sum() / (zA - zB)) if abs(zA - zB) > 1e-12 else np.nan
    d['sumAbsN_over_absZA'] = float(np.abs(N).sum() / (abs(zA) + 1e-300))
    d['tag'] = tag
    return d


def line(d):
    return (f"{d['tag']:<46s} r={d['corr_pearson']:+.8f} "
            f"rho={d.get('corr_spearman', float('nan')):+.6f} "
            f"tau={d.get('corr_kendall', float('nan')):+.6f} "
            f"maxdiff={d['max_abs_diff_rel']:.2e} "
            f"Rhalf={d.get('Rhalf', float('nan')):.2e} "
            f"I2={d.get('I2', float('nan')):.2e} "
            f"slope={d['slope']:+.4f}")
