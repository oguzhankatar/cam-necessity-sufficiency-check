"""Ground-truth machinery for the held-out validation (stage 6).

The frozen stage-5 rule makes a STRUCTURAL prediction.  This module produces the
NUMERICAL ground truth it is scored against.  Two independent tests are run:

  (A) `seg_dev`  -- stage 5's own numeric criterion, copied from scripts/scope/slib.py:
      is `forward_head` affine along the segment between two real feature maps?
      This is NECESSARY for additivity over cells but not sufficient: the cell-swap
      lattice spans up to min(T,C) directions out of the pooled vector, the segment
      spans exactly one.

  (B) the LITERAL float64 per-cell loop from scripts/crosscheck/rtlib.py --
      N(t) = z(A) - z(A with cell t <- B), S(t) = z(B with cell t <- A) - z(B),
      plus Rhalf over random half-subsets and I2 over random pairs.  A model is
      certified ADDITIVE only when maxNS, Rhalf and I2 are all < 1e-9, over
      EVERY probe pair.  `corrNS` is recorded and never used as a criterion.

(B) is the certificate.  (A) is kept so that the held-out run also answers
"would stage 5's own weaker test have got the same answer?".

The cell matrix is the model's own feature map at the aggregation point, read in
the model's own layout: the channel axis is identified as the axis whose length
is `m.num_features`, so a channels-last body (MambaOut, Swin, Hiera) is not
silently transposed.  Nothing is re-implemented: the readout under test is
literally `m.forward_head(<feature map rebuilt from the cell matrix>)`.
"""
import os, sys, json, warnings
import numpy as np
import torch
import torch.nn as nn

warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
VAL = os.path.join(ROOT, 'data', 'in1k', 'val')
OUT = os.path.join(ROOT, 'outputs', 'heldout')
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'crosscheck'))
sys.path.insert(0, HERE)
import rtlib                                                  # noqa: E402
import frozen_rule as FR                                      # noqa: E402

TOL = 1e-9          # the binding-rule certificate threshold
TOL_SEG = 1e-11     # stage 5's own seg_dev threshold


def save(name, obj):
    p = os.path.join(OUT, name)
    json.dump(obj, open(p, 'w'), indent=1, default=float)
    print('  -> %s' % p, flush=True)
    return p


# --------------------------------------------------------------- data
_DS = {}


def tfm(cfg):
    from torchvision import transforms
    isz = cfg['input_size'][-1]
    ip = {'bicubic': transforms.InterpolationMode.BICUBIC,
          'bilinear': transforms.InterpolationMode.BILINEAR,
          'nearest': transforms.InterpolationMode.NEAREST}[cfg.get('interpolation', 'bicubic')]
    return transforms.Compose([
        transforms.Resize(int(isz / cfg.get('crop_pct', 0.9)), interpolation=ip),
        transforms.CenterCrop(isz), transforms.ToTensor(),
        transforms.Normalize(cfg['mean'], cfg['std'])])


def val_images(cfg, idx):
    from torchvision import datasets
    key = (cfg['input_size'][-1], cfg.get('crop_pct', 0.9),
           tuple(cfg['mean']), tuple(cfg['std']), cfg.get('interpolation', 'bicubic'))
    if key not in _DS:
        _DS[key] = datasets.ImageFolder(VAL, tfm(cfg))
    ds = _DS[key]
    return torch.stack([ds[i][0] for i in idx])


# --------------------------------------------------- stage-5's seg_dev (copied)
@torch.no_grad()
def seg_dev(fwd, fa, fb, LAM=(0.0, .07, .13, .25, .5, .61, .75, .9, 1.0)):
    z = torch.stack([fwd(fa + l * (fb - fa))[0] for l in LAM])
    lam = torch.tensor(LAM, device=z.device, dtype=z.dtype)[:, None]
    lerp = z[0][None] + lam * (z[-1] - z[0])[None]
    sc = (z[-1] - z[0]).abs().max()
    return float((z - lerp).abs().max() / (sc + 1e-300)), float(sc)


# ----------------------------------------------------------- cell-matrix view
class CellView:
    """Feature map <-> (T, C) cell matrix, in the model's own layout."""

    def __init__(s, m, f):
        if not torch.is_tensor(f):
            raise ValueError('forward_features returned %s' % type(f).__name__)
        s.shape = tuple(f.shape)
        C = int(getattr(m, 'num_features'))
        s.fallback = None
        if f.dim() == 3:
            if f.shape[2] != C:
                s.fallback = 'num_features=%d != C axis of %s' % (C, s.shape)
                C = f.shape[2]
            s.fmt, s.T = 'BTC', f.shape[1]
        elif f.dim() == 4:
            if f.shape[1] == C and f.shape[3] != C:
                s.fmt, s.T = 'BCHW', f.shape[2] * f.shape[3]
            elif f.shape[3] == C and f.shape[1] != C:
                s.fmt, s.T = 'BHWC', f.shape[1] * f.shape[2]
            else:
                # ambiguous (C on both axes) or matching neither: assume NCHW,
                # and let the permutation diagnostic in h1_run flag it.
                s.fallback = 'num_features=%d vs shape %s -> assumed BCHW' % (C, s.shape)
                s.fmt, s.T, C = 'BCHW', f.shape[2] * f.shape[3], f.shape[1]
        else:
            raise ValueError('feature dim %d' % f.dim())
        s.C = C
        s.H = s.shape[2] if s.fmt == 'BCHW' else (s.shape[1] if s.fmt == 'BHWC' else None)
        s.W = s.shape[3] if s.fmt == 'BCHW' else (s.shape[2] if s.fmt == 'BHWC' else None)
        s.npfx = int(getattr(m, 'num_prefix_tokens', 0) or 0) if s.fmt == 'BTC' else 0

    def cells(s, f):
        """(B,...) feature map -> (B, T, C)"""
        if s.fmt == 'BTC':
            return f
        if s.fmt == 'BCHW':
            return f.flatten(2).transpose(1, 2)
        return f.reshape(f.shape[0], -1, s.C)

    def feat(s, X):
        """(B, T, C) -> feature map"""
        if s.fmt == 'BTC':
            return X
        if s.fmt == 'BCHW':
            return X.transpose(1, 2).reshape(X.shape[0], s.C, s.H, s.W)
        return X.reshape(X.shape[0], s.H, s.W, s.C)


def make_head(m, cv):
    @torch.no_grad()
    def head(X):
        return m.forward_head(cv.feat(X))
    return head


# ------------------------------------------------------------ module surgery
def set_mod(m, n, new):
    parent = m
    parts = n.split('.')
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new)


def kill_norms(m, post_names):
    mods = dict(m.named_modules())
    killed = [n for n in post_names if n in mods and FR.is_norm(mods[n])]
    for n in killed:
        set_mod(m, n, nn.Identity())
    return killed, mods


def restore(m, names, mods):
    for n in names:
        set_mod(m, n, mods[n])


# ------------------------------------------------------------ the certificate
def certify(head, cells, cls_idx, npfx, seed=0, nsub=64, npair=256):
    """cells: (P, T, C) stack of real cell matrices; probe pairs (0,1),(2,3),...

    Returns the worst maxNS / Rhalf / I2 over the pairs, plus the spatial mass.
    """
    rs = np.random.RandomState(seed)
    out = []
    P = cells.shape[0]
    for i in range(0, P - 1, 2):
        XA, XB = cells[i], cells[i + 1]
        c = cls_idx[i]
        N, S, zA, zB = rtlib.NS_literal(head, XA, XB, c)
        d = rtlib.stats(N, S)
        if XA.shape[0] < 2:
            # exactly one cell at this site (PiT's forward_features returns only
            # the class token).  Rhalf needs a half-subset and I2 needs a pair of
            # DISTINCT cells, so both statistics are undefined rather than
            # non-zero; a one-cell readout is additive by definition.
            d['Rhalf'], d['I2'], d['degenerate_T'] = 0.0, 0.0, True
        else:
            d['Rhalf'], d['I2'] = rtlib.additivity(head, XA, XB, c, N, rs,
                                                   nsub=nsub, npair=npair)
            d['degenerate_T'] = False
        tot = float(np.abs(N).sum())
        sp = float(np.abs(N[npfx:]).sum())
        d['spatial_mass'] = (sp / tot) if tot > 0 else 0.0
        # attribute-free vacuity: how many cells carry any necessity at all.
        # A class-token readout puts everything on exactly one cell.
        mx = float(np.abs(N).max())
        d['n_active'] = int((np.abs(N) > 1e-12 * mx).sum()) if mx > 0 else 0
        d['T'] = int(len(N))
        d['absN_total_over_absZA'] = float(tot / (abs(zA) + 1e-300))
        d['sumN_over_dz'] = (float(N.sum() / (zA - zB)) if abs(zA - zB) > 1e-12
                             else float('nan'))
        d['cls'] = int(c)
        d['pair'] = [i, i + 1]
        out.append(d)
    agg = dict(
        maxNS=max(d['max_abs_diff_rel'] for d in out),
        Rhalf=max(d['Rhalf'] for d in out),
        I2=max(d['I2'] for d in out),
        corrNS_min=min(d['corr_pearson'] for d in out),
        spatial_mass=max(d['spatial_mass'] for d in out),
        sumN_over_dz=float(np.mean([d['sumN_over_dz'] for d in out])),
        n_active=max(d['n_active'] for d in out),
        degenerate_T=bool(out[0]['degenerate_T']),
        T=out[0]['T'], pairs=out)
    agg['additive'] = bool(agg['maxNS'] < TOL and agg['Rhalf'] < TOL and agg['I2'] < TOL)
    return agg


GT_OF = {'IN-SCOPE': 'ADDITIVE', 'DEGENERATE': 'VACUOUS',
         'NORM-BLOCKED': 'NORM-ONLY', 'NONLIN-BLOCKED': 'HARD-BLOCKED',
         'AGG-BLOCKED': 'HARD-BLOCKED'}
# stage 5's own coarser numeric label (seg_dev based)
R5_OF = {'IN-SCOPE': 'AFFINE-SHIPPED', 'DEGENERATE': 'AFFINE-SHIPPED',
         'NORM-BLOCKED': 'NORM-ONLY', 'NONLIN-BLOCKED': 'HARD-BLOCKED',
         'AGG-BLOCKED': 'HARD-BLOCKED'}


def gt_label(ship, nonorm):
    if ship['additive']:
        # two vacuity criteria, reported side by side in h9:
        #   spatial_mass  -- needs the model's declared num_prefix_tokens
        #   n_active <= 1 -- attribute-free: all of the logit on one cell
        vac = ship['spatial_mass'] < 1e-9 or ship.get('n_active', 99) <= 1
        return 'VACUOUS' if vac else 'ADDITIVE'
    if nonorm is not None and nonorm['additive']:
        return 'NORM-ONLY'
    return 'HARD-BLOCKED'
