"""C6 measurement library: the paper's float64 additivity certificate (Sec. 2.2)
applied to an arbitrary `forward_head`, plus the frozen check, read-only.

Nothing here edits scripts/heldout/frozen_rule.py.  The per-cell loop, the
half-subset residual and the pair residual are imported verbatim from
scripts/crosscheck/rtlib.py (the functions the held-out certificate in
scripts/heldout/hlib.py uses):

  eps1 = max|N - S| / max(|N|,|S|)              (rtlib.stats  'max_abs_diff_rel')
  eps2 = Rhalf, 64 random half-subsets          (rtlib.additivity)
  eps3 = I2,    256 random cell pairs           (rtlib.additivity)

each the worst over three probe pairs, certified additive iff all < 1e-9.
Label (hlib.gt_label): ADDITIVE / VACUOUS (additive, and spatial mass < 1e-9 or
one active cell) / NORM-ONLY (blocked as built, additive after the post-
aggregator normalisations are deleted) / HARD-BLOCKED.

Cells.  A tensor feature map is read in the model's own layout (BCHW, BHWC or
BTC, as hlib.CellView).  A LIST feature (CoaT) is read as the concatenation of
all its token sequences; each sequence's token 0 is a class token (prefix).
"""
import os, sys
import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'crosscheck'))
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'heldout'))
import rtlib                      # noqa: E402
import frozen_rule as FR          # noqa: E402  (read-only import)

TOL = 1e-9


class Cells:
    """feature <-> (B, T, C) cell matrix; `prefix` = indices of non-spatial tokens."""

    def __init__(s, f, fmt=None, C=None, prefix=()):
        s.islist = isinstance(f, (list, tuple))
        if s.islist:
            s.lens = [t.shape[1] for t in f]
            s.C = f[0].shape[2]
            assert all(t.shape[2] == s.C for t in f), 'list features with unequal C'
            s.fmt = 'LIST'
            starts = np.cumsum([0] + s.lens[:-1])
            s.prefix = [int(i) for i in starts]          # token 0 of every sequence
            s.T = int(sum(s.lens))
            return
        s.shape = tuple(f.shape)
        if fmt is None:
            if f.dim() == 3:
                fmt = 'BTC'
            elif C is not None and f.shape[3] == C and f.shape[1] != C:
                fmt = 'BHWC'
            else:
                fmt = 'BCHW'
        s.fmt = fmt
        if fmt == 'BTC':
            s.T, s.C = f.shape[1], f.shape[2]
        elif fmt == 'BCHW':
            s.C, s.T = f.shape[1], f.shape[2] * f.shape[3]
        else:
            s.C, s.T = f.shape[3], f.shape[1] * f.shape[2]
        s.prefix = list(prefix)

    def cells(s, f):
        if s.islist:
            return torch.cat(list(f), dim=1)
        if s.fmt == 'BTC':
            return f
        if s.fmt == 'BCHW':
            return f.flatten(2).transpose(1, 2)
        return f.reshape(f.shape[0], -1, s.C)

    def feat(s, X):
        if s.islist:
            return list(torch.split(X, s.lens, dim=1))
        if s.fmt == 'BTC':
            return X
        B = X.shape[0]
        if s.fmt == 'BCHW':
            return X.transpose(1, 2).reshape(B, s.C, s.shape[2], s.shape[3])
        return X.reshape(B, s.shape[1], s.shape[2], s.C)


def cells_for_timm(m, f):
    if isinstance(f, (list, tuple)):
        return Cells(f)
    C = int(getattr(m, 'num_features', 0) or 0) or None
    npfx = int(getattr(m, 'num_prefix_tokens', 0) or 0) if f.dim() == 3 else 0
    return Cells(f, C=C, prefix=range(npfx))


def certify(head, X, cv, seed=0, nsub=64, npair=256):
    """X: (P, T, C) cell matrices, pairs (0,1),(2,3),...  head: (B,T,C)->(B,K)."""
    rs = np.random.RandomState(seed)
    out = []
    with torch.no_grad():
        z = head(X)
    for i in range(0, X.shape[0] - 1, 2):
        XA, XB = X[i], X[i + 1]
        c = int(z[i].argmax())
        N, S, zA, zB = rtlib.NS_literal(head, XA, XB, c)
        d = rtlib.stats(N, S)
        if XA.shape[0] >= 2:
            d['Rhalf'], d['I2'] = rtlib.additivity(head, XA, XB, c, N, rs, nsub=nsub, npair=npair)
        else:
            d['Rhalf'] = d['I2'] = 0.0
        tot = float(np.abs(N).sum())
        sp_idx = np.setdiff1d(np.arange(len(N)), np.array(cv.prefix, dtype=int))
        d['spatial_mass'] = float(np.abs(N[sp_idx]).sum() / tot) if tot > 0 else 0.0
        mx = float(np.abs(N).max())
        d['n_active'] = int((np.abs(N) > 1e-12 * mx).sum()) if mx > 0 else 0
        d['absN_max'] = mx
        d['dz'] = float(zA - zB)
        out.append(d)
    r = dict(eps1=max(d['max_abs_diff_rel'] for d in out),
             eps2=max(d['Rhalf'] for d in out),
             eps3=max(d['I2'] for d in out),
             corr_min=min((d['corr_pearson'] for d in out if d['corr_pearson'] == d['corr_pearson']),
                          default=float('nan')),
             spatial_mass=max(d['spatial_mass'] for d in out),
             n_active=max(d['n_active'] for d in out),
             absN_max=min(d['absN_max'] for d in out),
             T=int(X.shape[1]))
    r['additive'] = bool(r['eps1'] < TOL and r['eps2'] < TOL and r['eps3'] < TOL)
    r['vacuous'] = bool(r['additive'] and (r['spatial_mass'] < 1e-9 or r['n_active'] <= 1))
    return r


def label(ship, nonorm):
    if ship['additive']:
        return 'VACUOUS' if ship['vacuous'] else 'ADDITIVE'
    if nonorm is not None and nonorm['additive']:
        return 'NORM-ONLY'
    return 'HARD-BLOCKED'


# frozen code -> paper term, and the measured label each code predicts
PAPER = {'IN-SCOPE': 'degenerate', 'DEGENERATE': 'class-token',
         'NORM-BLOCKED': 'normalisation-blocked', 'NONLIN-BLOCKED': 'nonlinearity-blocked',
         'AGG-BLOCKED': 'aggregator-blocked'}
GT_OF = {'IN-SCOPE': 'ADDITIVE', 'DEGENERATE': 'VACUOUS', 'NORM-BLOCKED': 'NORM-ONLY',
         'NONLIN-BLOCKED': 'HARD-BLOCKED', 'AGG-BLOCKED': 'HARD-BLOCKED'}


def is_norm_name(mod):
    """independent of frozen_rule tables: any class whose name contains 'Norm',
    except batch normalisation (affine at inference)."""
    n = type(mod).__name__
    return ('Norm' in n or 'norm' in n) and not n.startswith('BatchNorm')


def set_mod(m, name, new):
    parent = m
    parts = name.split('.')
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new)


def measure(m, f, cv, norm_names=()):
    """certificate as built, and with the named modules replaced by Identity."""
    head = lambda X: m.forward_head(cv.feat(X))
    X = cv.cells(f)
    ship = certify(head, X, cv)
    nonorm = None
    if norm_names:
        mods = dict(m.named_modules())
        saved = {n: mods[n] for n in norm_names}
        for n in norm_names:
            set_mod(m, n, nn.Identity())
        try:
            nonorm = certify(head, X, cv)
        finally:
            for n, mod in saved.items():
                set_mod(m, n, mod)
    return ship, nonorm, label(ship, nonorm)
