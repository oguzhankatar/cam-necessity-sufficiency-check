"""C2 task 7 -- ISIC Dice threshold tuning, made explicit.

m5_isic scores Dice at a top-q threshold (q = fraction of the 224x224 image kept),
q chosen from QGRID by 2-fold cross-tuning (fold = RandomState(0).randint(0,2,n)).
For every map of the ISIC replicate (out/maps_isic.npz) we report
  * mean Dice at every q of the grid,
  * the q chosen in each tuning fold and the resulting cross-tuned Dice,
  * the in-sample oracle (best single q) and Dice at q = 0.20 (the mean lesion
    fraction, 0.200 in m5_isic.json), to show how much the tuning step moves it.
Output: out/t7_isic_dice.json
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import medical_baselines_lib as L
from medical_baselines_lib import mlib
import m5_isic as m5

G = 7
Q = m5.QGRID
MAPS = ['collapsed_1x1', 'collapsed_3x3', 'N_lse8.0_1x1', 'N_lse-8.0_1x1', 'S_lse8.0_1x1',
        'S_lse-8.0_1x1', 'N_lse8.0_3x3', 'N_lse-8.0_3x3', 'S_lse8.0_3x3', 'S_lse-8.0_3x3',
        'cam', 'gradcam_gap', 'actnorm', 'center', 'random']


def dice_sweep(mp, g):
    u = mlib.upsample(mp, G, 224).reshape(-1)
    gg = g.reshape(-1) > 0
    srt = np.sort(u)
    out = []
    for q in Q:
        k = max(1, int(round(q * len(u))))
        p = u >= srt[-k]
        out.append(2 * float((p & gg).sum()) / float(p.sum() + gg.sum()))
    return out


def main():
    M, meta = L.load_maps('maps_isic.npz')
    gts = [m5.gt_of(n) for n, c in meta['items']]
    n = len(gts)
    fold = np.random.RandomState(0).randint(0, 2, n)
    res = dict(n=n, qgrid=Q, gt_fraction_mean=float(np.mean([g.mean() for g in gts])), maps={})
    for mn in MAPS:
        D = np.array([dice_sweep(M[mn][i], gts[i]) for i in range(n)])     # (n, |Q|)
        means = D.mean(0)
        v = np.full(n, np.nan)
        chosen = {}
        for f in (0, 1):
            tr, te = fold != f, fold == f
            j = int(np.argmax(D[tr].mean(0)))
            chosen['fold%d' % f] = Q[j]
            v[te] = D[te, j]
        res['maps'][mn] = dict(dice_by_q={str(q): float(m) for q, m in zip(Q, means)},
                               chosen_q=chosen, dice_crosstuned=float(v.mean()),
                               dice_oracle=float(means.max()), q_oracle=Q[int(np.argmax(means))],
                               dice_q020=float(means[Q.index(0.20)]))
        print('%-16s cross-tuned %.3f (q %s) oracle %.3f q=.20 %.3f' % (
            mn, v.mean(), chosen, means.max(), means[Q.index(0.20)]), flush=True)
    L.save('t7_isic_dice.json', res)


if __name__ == '__main__':
    main()
