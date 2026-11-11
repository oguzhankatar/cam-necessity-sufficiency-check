"""Shared helpers for the C3 additional analyses (natural-image localisation).

Reads the archived per-record arrays in outputs/wsol READ-ONLY and re-runs the
tests of scripts/wsol/w3_stats.py exactly (same functions from wlib, same
seeds, same resample counts).  Nothing under scripts/ or outputs/ is written.
"""
import os
import sys
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
C3 = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(C3, '..', '..'))
WS = os.path.join(ROOT, 'scripts', 'wsol')
WO = os.path.join(ROOT, 'outputs', 'wsol')
OUT = os.path.join(C3, 'out')
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, WS)
import wlib as W  # noqa: E402  (pure functions; importing writes nothing new)

DATASETS = ['cub', 'voc12', 'voc12seg', 'coco17']
BACKBONES = ['resnet50', 'vgg16gap', 'vitb16']
METRICS = {'cub': ['MaxBoxAccV2', 'best_iou', 'pointing_strict', 'pointing_tol15'],
           'voc12': ['pointing_strict', 'pointing_tol15'],
           'voc12seg': ['pointing_strict', 'pointing_tol15', 'peak_mask_iou'],
           'coco17': ['pointing_strict', 'pointing_tol15', 'peak_mask_iou']}
EXTENT = {'MaxBoxAccV2', 'best_iou', 'peak_mask_iou'}
MAXLOC = {'pointing_strict', 'pointing_tol15'}
MNAME = {'MaxBoxAccV2': 'MaxBoxAccV2', 'best_iou': 'best IoU',
         'pointing_strict': 'pointing, strict', 'pointing_tol15': 'pointing, 15 px',
         'peak_mask_iou': 'peak-mask IoU'}
ALPHA = 0.05
DELTAS = (0.3, 0.5, 0.7)


def arr(ds, bk, kind, mp, suf=''):
    p = os.path.join(WO, '%s_%s_%s%s_%s.npy' % (kind, ds, bk, suf, mp))
    return np.load(p) if os.path.exists(p) else None


def recidx(ds, bk, suf=''):
    return np.load(os.path.join(WO, 'recidx_%s_%s%s.npy' % (ds, bk, suf)))


def per_record(ds, bk, metric, mp, suf=''):
    """the per-record (per-image for CUB) vector a test is run on, or the
    curve matrix for MaxBoxAccV2."""
    if metric == 'MaxBoxAccV2':
        return arr(ds, bk, 'curve', mp, suf)
    if metric == 'best_iou':
        c = arr(ds, bk, 'curve', mp, suf)
        return None if c is None else c.max(1).astype(np.float64)
    if metric.startswith('pointing'):
        p = arr(ds, bk, 'pt', mp, suf)
        return None if p is None else p[0 if metric == 'pointing_strict' else 1].astype(np.float64)
    return arr(ds, bk, 'iou', mp, suf)


def test(ds, bk, metric, a, b, suf=''):
    """exactly the test w3_stats.py runs for this (metric, a, b)."""
    xa, xb = per_record(ds, bk, metric, a, suf), per_record(ds, bk, metric, b, suf)
    if xa is None or xb is None:
        return None
    if metric == 'MaxBoxAccV2':
        d, lo, hi, p = W.boot_mbav2_diff(xa, xb, n=W.PROTOCOL['n_boot_mbav2'],
                                         seed=W.PROTOCOL['seed'], deltas=DELTAS)
        return dict(va=W.maxboxacc_v2(xa, DELTAS)[0], vb=W.maxboxacc_v2(xb, DELTAS)[0],
                    diff=d, ci=[lo, hi], p=p, test='paired bootstrap', n=len(xa))
    d, lo, hi = W.boot_diff(xa, xb, n=W.PROTOCOL['n_boot'], seed=W.PROTOCOL['seed'])
    if metric.startswith('pointing'):
        mc = W.mcnemar(xa.astype(bool), xb.astype(bool))
        p, t = mc['p'], 'mcnemar'
    else:
        p, t = W.wilcoxon(xa, xb)['p'], 'wilcoxon'
    return dict(va=float(xa.mean()), vb=float(xb.mean()), diff=d, ci=[lo, hi], p=p,
                test=t, n=len(xa))


def holm(p):
    return W.holm(p)


def verdict(diff, p_holm):
    if p_holm < ALPHA:
        return 'win' if diff > 0 else 'loss'
    return 'tie'


def arm_of(a):
    return 'relocation' if a.endswith('_ln') else 'aggregator'


def half_of(a):
    return 'necessity' if a.startswith('N_') else 'sufficiency'


def separated_maps(bk):
    return ['N_lse', 'S_lse'] + (['N_ln', 'S_ln'] if bk == 'vitb16' else [])


# ---------------------------------------------------- image-clustered tests
def signflip_p(xa, xb, g, B=20000, seed=0, chunk=2000):
    """two-sided paired sign-flip permutation test with the cluster (image) as
    the exchangeable unit.  Statistic: sum over records of (xa - xb), i.e. the
    record-level mean difference times n.  Under H0 the two maps' labels are
    exchangeable within each image, so each image's summed difference flips
    sign as a block.  p = (1 + #|T*| >= |T|) / (B + 1)."""
    d = np.asarray(xa, np.float64) - np.asarray(xb, np.float64)
    k = int(g.max()) + 1
    D = np.bincount(g, weights=d, minlength=k)
    D = D[D != 0]
    if D.size == 0:
        return 1.0
    T0 = abs(D.sum())
    rs = np.random.RandomState(seed)
    ge = 0
    done = 0
    tol = 1e-12 * max(1.0, np.abs(D).sum())
    while done < B:
        m = min(chunk, B - done)
        S = rs.randint(0, 2, size=(m, D.size)).astype(np.float64) * 2 - 1
        ge += int((np.abs(S @ D) >= T0 - tol).sum())
        done += m
    return (1.0 + ge) / (B + 1.0)


def clusterboot_p(xa, xb, g, B=2000, seed=0):
    """clustered (image) bootstrap of the record-level mean difference; the
    p value is 2*min(P(b<=0), P(b>=0)) floored at 1/B, the same construction
    w3 uses for MaxBoxAccV2.  Returns (p, lo, hi)."""
    xa = np.asarray(xa, np.float64); xb = np.asarray(xb, np.float64)
    k = int(g.max()) + 1
    cnt = np.bincount(g, minlength=k).astype(np.float64)
    sx = np.bincount(g, weights=xa, minlength=k)
    sy = np.bincount(g, weights=xb, minlength=k)
    rs = np.random.RandomState(seed)
    pick = rs.randint(0, k, (B, k))
    cc = cnt[pick].sum(1)
    b = sx[pick].sum(1) / cc - sy[pick].sum(1) / cc
    if np.all(xa == xb):
        return 1.0, 0.0, 0.0
    p = 2.0 * min((b <= 0).mean(), (b >= 0).mean())
    return float(min(1.0, max(p, 1.0 / B))), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def save(name, obj):
    p = os.path.join(OUT, name)
    json.dump(obj, open(p, 'w', encoding='utf-8'), indent=1, default=W._jd)
    print('  ->', p)
    return p
