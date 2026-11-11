"""analyses/medical_extra (medical analyses) -- shared helpers.

Read only: scripts/medical (mlib, m4_stats, m5_isic, m1_model), outputs/medical,
analyses/medical_baselines (medical_baselines_lib, t5_nih, t6_score, maps_*.npz, t6_score.json),
analyses/closed_form_maps (closed_form_lib: closed form of Eq. 4), analyses/extended_tables.
Writes only under analyses/medical_extra/{out,cache}.

The feature caches of the original runs (F_*.npy, checkpoints) lived in the
system temp directory and no longer exist; every feature used here is rebuilt
from the local images and the distributed weights, into analyses/medical_extra/cache.
"""
import os
import sys
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
M = os.path.abspath(os.path.join(HERE, '..'))
ROOT = os.path.abspath(os.path.join(M, *['..'] * 2))
OUT = os.path.join(M, 'out')
CACHE = os.path.join(M, 'cache')
os.makedirs(OUT, exist_ok=True)
os.makedirs(CACHE, exist_ok=True)
os.environ.setdefault('C2CACHE', CACHE)
os.environ.setdefault('MEDCACHE', CACHE)

C2S = os.path.join(ROOT, 'analyses', 'medical_baselines', 'scripts')
C2O = os.path.join(ROOT, 'analyses', 'medical_baselines', 'out')
C5S = os.path.join(ROOT, 'analyses', 'closed_form_maps', 'scripts')
SO = os.path.join(ROOT, 'analyses', 'extended_tables', 'out')
MOUT = os.path.join(ROOT, 'outputs', 'medical')
for p in (C5S, C2S, os.path.join(ROOT, 'scripts', 'medical')):
    if p not in sys.path:
        sys.path.insert(0, p)

import medical_baselines_lib as L          # noqa: E402  (imports mlib)
import closed_form_lib as C          # noqa: E402
from medical_baselines_lib import mlib     # noqa: E402
import m4_stats as m4      # noqa: E402

G = 7
OFF = ['lse8.0', 'lse-8.0']
QDICE = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60]   # m5_isic.QGRID / t6_score


def save(name, obj):
    p = os.path.join(OUT, name)
    json.dump(obj, open(p, 'w'), indent=1, default=float)
    print('wrote', p, flush=True)
    return p


def jload(p):
    return json.load(open(p))


def units(kind):
    return [[t] for t in range(G * G)] if kind == '1x1' else [list(map(int, x)) for x in mlib.blocks_of(G, 3)]


def cf_maps(sa, SB, T, kind):
    """closed-form (Eq. 4, float64, log1p/expm1) N and S of an LSE_T head, one unit
    substituted at a time, averaged over the reference bank SB (list of score vectors)."""
    us = units(kind)
    NS = [C.ns_exact_units(sa, sb, T, us) for sb in SB]
    return np.mean([x[0] for x in NS], 0), np.mean([x[1] for x in NS], 0)


def cf_collapsed(sa, SB, kind):
    """closed form at T = 0 (GAP): sum over the unit of (s_a - s_b)/n, averaged."""
    us = units(kind)
    return np.mean([C.ns_exact_units(sa, sb, 0.0, us)[0] for sb in SB], 0)


# ------------------------------------------------------------------ scoring
def score(maps, gts, metrics, names=None):
    """the archive's scoring (t6_score.score_arm / m4_stats / m5_isic):
    hit = mlib.pointing_hit, ap = mlib.pixel_ap, iou / dice at a top-q threshold
    with q cross-tuned over two folds (RandomState(0).randint(0,2,n_kept)).
    maps: dict name -> (n, 49) array; gts: list of 224x224 masks.
    Returns dict name -> metric -> per-kept-item vector, and the kept index."""
    keep = [i for i, g in enumerate(gts) if 0 < g.sum() < g.size]
    names = sorted(maps) if names is None else names
    third = [m for m in metrics if m in ('iou', 'dice')]
    Q = m4.QGRID if 'iou' in third else QDICE
    S = {mn: {mt: np.full(len(keep), np.nan) for mt in metrics} for mn in names}
    sweep = {mn: {q: np.full(len(keep), np.nan) for q in Q} for mn in names} if third else None
    for k, i in enumerate(keep):
        g = gts[i]
        for mn in names:
            mp = maps[mn][i]
            S[mn]['hit'][k] = mlib.pointing_hit(mp, G, g, 224)
            if 'ap' in metrics:
                S[mn]['ap'][k] = mlib.pixel_ap(mp, G, g, 224)
            if third:
                f = mlib.iou_at if third[0] == 'iou' else mlib.dice_at
                for q in Q:
                    sweep[mn][q][k] = f(mp, G, g, 224, q)
    chosen = {}
    if third:
        mt = third[0]
        fold = np.random.RandomState(0).randint(0, 2, len(keep))
        for mn in names:
            v = np.full(len(keep), np.nan)
            ch = []
            for f in (0, 1):
                tr, te = fold != f, fold == f
                best, bq = -1, Q[0]
                for q in Q:
                    a = sweep[mn][q][tr]
                    if np.nanmean(a) > best:
                        best, bq = np.nanmean(a), q
                v[te] = sweep[mn][bq][te]
                ch.append(bq)
            S[mn][mt] = v
            chosen[mn] = ch
    return S, np.array(keep), chosen


def cmp(a, b, binary):
    """the archive's paired comparison: mean difference with a 10,000-resample
    item bootstrap (mlib.paired, seed 0), Wilcoxon; exact McNemar for hit."""
    d = L.cmp(a, b, binary)
    d['P'] = d['p_mcnemar'] if binary else d['p_wilcoxon']
    return d


def grid(S, metrics):
    out = {}
    for unit in ('1x1', '3x3'):
        Cn = 'collapsed_%s' % unit
        for mt in metrics:
            for temp in OFF:
                Nn, Sn = 'N_%s_%s' % (temp, unit), 'S_%s_%s' % (temp, unit)
                out['H1_%s_%s_%s' % (temp, unit, mt)] = cmp(S[Nn][mt], S[Cn][mt], mt == 'hit')
                out['H2_%s_%s_%s' % (temp, unit, mt)] = cmp(S[Nn][mt], S[Sn][mt], mt == 'hit')
                out['Sminus_%s_%s_%s' % (temp, unit, mt)] = cmp(S[Sn][mt], S[Cn][mt], mt == 'hit')
    return out


def brief(c, keys=('diff', 'lo', 'hi', 'P', 'mean_a', 'mean_b', 'n')):
    return {k: c[k] for k in keys if k in c}


def fmt(c):
    return '%+.4f [%+.4f, %+.4f] P=%.3g' % (c['diff'], c['lo'], c['hi'], c['P'])


def mcnemar_from_wilcoxon(c):
    """Exact McNemar P of a paired hit cell for which only the Wilcoxon P was stored
    (the first ISIC run).  For a binary outcome, b - c = diff * n and the Wilcoxon P,
    a monotone function of (b - c) / sqrt(b + c), fix the discordant counts b and c;
    the counts are checked by re-running scipy's Wilcoxon on the rebuilt differences.
    Same computation as analyses/extended_tables/scripts/s2_medical_grid.py."""
    from scipy.stats import norm, wilcoxon, binomtest
    n, p = c['n'], c['p_wilcoxon']
    net = int(round(c['diff'] * n))
    if net == 0:
        return 1.0
    tot = int(round((net / norm.isf(p / 2)) ** 2))
    b, cc = (tot + net) // 2, (tot - net) // 2
    assert b - cc == net and 0 <= cc and tot <= n
    d = np.r_[np.ones(b), -np.ones(cc), np.zeros(n - tot)]
    assert abs(np.log(float(wilcoxon(d, zero_method='wilcox').pvalue)) - np.log(p)) < 1e-6
    return float(binomtest(min(b, cc), tot, 0.5).pvalue)


def paper_p(c, key):
    """P as the paper reports it: exact McNemar for hit, Wilcoxon signed-rank otherwise."""
    if 'p_mcnemar' in c:
        return c['p_mcnemar']
    if key.endswith('_hit'):
        return mcnemar_from_wilcoxon(c)
    return c['p_wilcoxon']


def orig_cell(arch, key):
    """a cell of an archived 'prespecified' block, with P chosen as the paper does."""
    c = dict(arch['prespecified'][key])
    c['P'] = paper_p(c, key)
    return c
