"""W3 -- the pre-specified paired comparisons, effect sizes and CIs.

    python w3_stats.py            (every w2_*.json present)

Pre-specified contrasts, per (dataset, backbone) and per metric:

    C1  N_lse    vs collapsed          "does necessity beat the collapsed map?"
    C2  S_lse    vs collapsed          "does sufficiency beat it?"
    C3  N_lse    vs center             "does necessity beat the centre prior?"
    C4  S_lse    vs center
    C5  N_ln     vs collapsed          (ViT only: LayerNorm relocation)
    C6  S_ln     vs collapsed
    C7  N_ln     vs center
    C8  S_ln     vs center

`center` is the designated primary trivial control -- it is the one that beat
every published CAM in stage 3, so it is the control a reader will ask about.
`actnorm` and `random` are reported as secondary controls.  Holm-Bonferroni is
applied within each (dataset, backbone, metric) family.

Tests
  MaxBoxAccV2       paired bootstrap over images of the difference (the metric
                    is a max over thresholds, so it is recomputed inside every
                    resample).
  per-image best IoU / peak mask IoU   Wilcoxon signed-rank + rank-biserial
                    effect size + paired bootstrap of the mean difference.
  pointing game     McNemar exact + paired bootstrap of the difference of
                    proportions.
"""
import os, sys, glob, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import wlib as W

PAIRS = [('N_lse', 'collapsed'), ('S_lse', 'collapsed'),
         ('N_lse', 'center'), ('S_lse', 'center'),
         ('N_ln', 'collapsed'), ('S_ln', 'collapsed'),
         ('N_ln', 'center'), ('S_ln', 'center'),
         # secondary: the 3x3-block perturbation scale
         ('N_lse_b3', 'collapsed_b3'), ('S_lse_b3', 'collapsed_b3'),
         ('N_lse_b3', 'center'), ('S_lse_b3', 'center'),
         # context rows a reader will want
         ('collapsed', 'center'), ('gradcam', 'center'), ('cam', 'center'),
         ('collapsed', 'random'), ('collapsed', 'actnorm'),
         # the most direct reading of the question, added AFTER the VOC runs and
         # therefore reported as secondary (raw p): does the SPLIT itself separate
         # the two halves, as opposed to the model edit moving both of them?
         ('N_lse', 'S_lse'), ('N_ln', 'S_ln'), ('N_lse_b3', 'S_lse_b3'),
         ('N_ln_b3', 'S_ln_b3'), ('N_ln', 'collapsed_b3'), ('S_ln', 'collapsed_b3'),
         ('N_ln_b3', 'collapsed_b3'), ('S_ln_b3', 'collapsed_b3')]
PRIMARY = 8                      # the first 8 pairs form the Holm family


def arr(ds, bk, kind, mp):
    p = os.path.join(W.OUT, '%s_%s_%s_%s.npy' % (kind, ds, bk, mp))
    return np.load(p) if os.path.exists(p) else None


out = {}
for f in sorted(glob.glob(os.path.join(W.OUT, 'w2_*.json'))):
    j = json.load(open(f))
    ds, bk = j['dataset'], j['backbone']
    if os.path.basename(f) != 'w2_%s_%s.json' % (ds, bk):
        continue          # smoke tests, alternative basis, tau sweep: not primary
    key = '%s|%s' % (ds, bk)
    fam = {}
    for metric in ('MaxBoxAccV2', 'best_iou', 'pointing_strict', 'pointing_tol15',
                   'peak_mask_iou'):
        rows, pv = [], []
        for a, b in PAIRS:
            if a not in j['maps'] or b not in j['maps']:
                continue
            r = dict(a=a, b=b, metric=metric)
            if metric in ('MaxBoxAccV2', 'best_iou'):
                ca, cb = arr(ds, bk, 'curve', a), arr(ds, bk, 'curve', b)
                if ca is None:
                    continue
                if metric == 'MaxBoxAccV2':
                    d, lo, hi, pb = W.boot_mbav2_diff(
                        ca, cb, n=W.PROTOCOL['n_boot_mbav2'],
                        seed=W.PROTOCOL['seed'], deltas=tuple(j['deltas']))
                    r.update(va=W.maxboxacc_v2(ca, tuple(j['deltas']))[0],
                             vb=W.maxboxacc_v2(cb, tuple(j['deltas']))[0],
                             diff=d, ci=[lo, hi], test='paired bootstrap', p=pb)
                else:
                    xa, xb = ca.max(1).astype(np.float64), cb.max(1).astype(np.float64)
                    d, lo, hi = W.boot_diff(xa, xb, n=W.PROTOCOL['n_boot'],
                                            seed=W.PROTOCOL['seed'])
                    w = W.wilcoxon(xa, xb)
                    r.update(va=float(xa.mean()), vb=float(xb.mean()), diff=d,
                             ci=[lo, hi], test='wilcoxon', p=w['p'],
                             effect_rank_biserial=w['rank_biserial'], n=len(xa))
            elif metric.startswith('pointing'):
                pa, pb = arr(ds, bk, 'pt', a), arr(ds, bk, 'pt', b)
                if pa is None:
                    continue
                k = 0 if metric == 'pointing_strict' else 1
                xa, xb = pa[k].astype(bool), pb[k].astype(bool)
                d, lo, hi = W.boot_diff(xa.astype(float), xb.astype(float),
                                        n=W.PROTOCOL['n_boot'], seed=W.PROTOCOL['seed'])
                mc = W.mcnemar(xa, xb)
                r.update(va=float(xa.mean()), vb=float(xb.mean()), diff=d,
                         ci=[lo, hi], test='mcnemar', p=mc['p'],
                         n10=mc['n10'], n01=mc['n01'], n=len(xa))
            else:
                ia, ib = arr(ds, bk, 'iou', a), arr(ds, bk, 'iou', b)
                if ia is None:
                    continue
                d, lo, hi = W.boot_diff(ia, ib, n=W.PROTOCOL['n_boot'],
                                        seed=W.PROTOCOL['seed'])
                w = W.wilcoxon(ia, ib)
                r.update(va=float(ia.mean()), vb=float(ib.mean()), diff=d,
                         ci=[lo, hi], test='wilcoxon', p=w['p'],
                         effect_rank_biserial=w['rank_biserial'], n=len(ia))
            rows.append(r); pv.append(r['p'])
        if not rows:
            continue
        prim = [i for i, r in enumerate(rows)
                if (r['a'], r['b']) in PAIRS[:PRIMARY] and np.isfinite(r['p'])]
        if prim:
            adj = W.holm([pv[i] for i in prim])
            for i, a_ in zip(prim, adj):
                rows[i]['p_holm'] = a_
        fam[metric] = rows
        print('%-26s %-16s %d contrasts' % (key, metric, len(rows)), flush=True)
    out[key] = dict(dataset=ds, backbone=bk, n_records=j['n_records'],
                    n_eval=j['n_eval'], contrasts=fam,
                    classification=j['classification'],
                    certificate={k: {kk: v[kk] for kk in
                                     ('ADDITIVE', 'maxNS', 'Rhalf', 'I2',
                                      'corrNS_recorded_never_used')}
                                 for k, v in j['certificate'].items()})
W.save('w3_stats.json', out)
