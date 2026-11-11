"""m7 -- diagnostics on the NIH result (run AFTER the pre-specified comparison).

  D1  Grad-CAM's invariance to the aggregator.  For a readout whose per-cell
      contribution is linear (z = pool_t(W a_t + b)), the Grad-CAM channel
      weights are mean_t (d pool / d z_t) W = W / T for ANY pooling that is a
      convex combination, so Grad-CAM cannot see the AND-OR knob at all.
      Measured here, not asserted.
  D2  Why the pre-specified T = +8 necessity map fails: concentration.
  D3  The stage-3 deflation analysis, transplanted to human ground truth:
      per-image Spearman of a map against the radiologist mask, raw and
      partialled on the two trivial controls.
"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlib
import m4_stats as m4

SIZE = 224


def partial_spearman(x, y, Z):
    """Spearman(x, y) after linearly removing the columns of Z from both."""
    from scipy.stats import rankdata, spearmanr
    rx, ry = rankdata(x), rankdata(y)
    RZ = np.column_stack([rankdata(Z[:, j]) for j in range(Z.shape[1])]
                         + [np.ones(len(x))])
    bx = np.linalg.lstsq(RZ, rx, rcond=None)[0]
    by = np.linalg.lstsq(RZ, ry, rcond=None)[0]
    ex, ey = rx - RZ @ bx, ry - RZ @ by
    if ex.std() < 1e-12 or ey.std() < 1e-12:
        return np.nan
    return float(np.corrcoef(ex, ey)[0, 1])


def r2_on_controls(x, Z):
    from scipy.stats import rankdata
    rx = rankdata(x)
    RZ = np.column_stack([rankdata(Z[:, j]) for j in range(Z.shape[1])]
                         + [np.ones(len(x))])
    bt = np.linalg.lstsq(RZ, rx, rcond=None)[0]
    res = rx - RZ @ bt
    return float(1 - res.var() / rx.var()) if rx.var() > 0 else np.nan


def main():
    R = json.load(open(os.path.join(mlib.OUT, 'm3_maps.json')))
    G = R['G']
    T = R['T']
    boxes = mlib.nih_boxes()
    items = R['items']
    res = {'n_items': len(items)}

    # ---- D1: Grad-CAM invariance -----------------------------------------
    worst = 0.
    for it in items:
        a = np.asarray(it['maps']['gradcam_gap'])
        s = max(abs(a).max(), 1e-300)
        for k in ('gradcam_lse8.0', 'gradcam_lse-8.0'):
            worst = max(worst, float(np.abs(a - np.asarray(it['maps'][k])).max() / s))
    res['gradcam_invariance_worst_rel'] = worst

    # ---- D2: concentration of each map -----------------------------------
    conc = {}
    for mn in sorted(items[0]['maps']):
        top1, eff = [], []
        for it in items:
            v = np.abs(np.asarray(it['maps'][mn]))
            s = v.sum()
            if s <= 0:
                continue
            p = v / s
            top1.append(float(p.max()))
            eff.append(float(np.exp(-(p[p > 0] * np.log(p[p > 0])).sum())))
        conc[mn] = dict(top1_share=float(np.mean(top1)),
                        effective_cells=float(np.mean(eff)), T=T)
    res['concentration'] = conc

    # ---- D3: deflation against the human mask ----------------------------
    keys = ['collapsed_1x1', 'N_lse8.0_1x1', 'S_lse8.0_1x1', 'N_lse-8.0_1x1',
            'S_lse-8.0_1x1', 'gradcam_gap', 'actnorm', 'center', 'random']
    raw = {k: [] for k in keys}
    par = {k: [] for k in keys}
    r2c = {k: [] for k in keys}
    import torch
    import torch.nn.functional as Fn
    for it in items:
        g = m4.gt_mask(boxes[it['image']], it['finding'])
        if g.sum() == 0:
            continue
        gt = Fn.avg_pool2d(torch.from_numpy(g.astype(np.float32))[None, None],
                           SIZE // G)[0, 0].numpy().reshape(-1)
        if gt.std() < 1e-9:
            continue
        Z = np.column_stack([np.asarray(it['maps']['center']),
                             np.asarray(it['maps']['actnorm'])])
        from scipy.stats import spearmanr
        for k in keys:
            v = np.asarray(it['maps'][k])
            if v.std() < 1e-12:
                continue
            raw[k].append(float(spearmanr(v, gt).statistic))
            if k not in ('center', 'actnorm'):
                par[k].append(partial_spearman(v, gt, Z))
                r2c[k].append(r2_on_controls(v, Z))
    res['deflation'] = {k: dict(raw=mlib.boot_ci(raw[k]),
                                partial=mlib.boot_ci(par[k]) if par[k] else None,
                                r2_on_trivial=float(np.nanmean(r2c[k])) if r2c[k] else None)
                        for k in keys}
    mlib.save('m7_diag.json', res)
    print('gradcam invariance worst rel diff: %.3e' % worst)
    for k in keys:
        d = res['deflation'][k]
        print('%-16s raw rho %+.4f [%+.4f,%+.4f]  partial %s  R2|trivial %s'
              % (k, d['raw']['mean'], d['raw']['lo'], d['raw']['hi'],
                 ('%+.4f' % d['partial']['mean']) if d['partial'] else '   -   ',
                 ('%.3f' % d['r2_on_trivial']) if d['r2_on_trivial'] is not None else ' - '))


if __name__ == '__main__':
    main()
