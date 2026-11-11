"""m13 -- the paired test the class-independent-baseline claim was missing.

The article states that a map derived only from activation magnitude, which
carries no class information, "outperformed all 15 model-derived maps we
tested". That ordering was read off point estimates: m4 reports a bootstrap
interval for each map separately and no paired test between them, and the
intervals overlap heavily (actnorm 0.240 [0.214, 0.266] against 0.227 for the
collapsed map). A reader asked for the test, correctly.

This runs it, on the same items, the same ground truth and the same estimator
m4 uses: exact McNemar on the paired binary hit, activation magnitude against
every other map, with Holm correction across the 17 comparisons. It also runs
the same comparison with cardiomegaly excluded, because that finding is large
and central and dominates a pointing-game hit rate.

Nothing here re-derives a map; it only scores what m3 already produced.
"""
import os
import sys
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mlib
from m4_stats import gt_mask, mcnemar, SIZE

BASELINE = 'actnorm'
NOT_MODEL_DERIVED = {'actnorm', 'center', 'random'}


def score(excl=()):
    R = json.load(open(os.path.join(mlib.OUT, 'm3_maps.json')))
    G = R['G']
    boxes = mlib.nih_boxes()
    names = sorted(R['items'][0]['maps'].keys())
    hit = {n: [] for n in names}
    n_items = 0
    for it in R['items']:
        if it['finding'] in excl:
            continue
        g = gt_mask(boxes[it['image']], it['finding'])
        if g.sum() == 0 or g.sum() == g.size:
            continue
        n_items += 1
        for n in names:
            hit[n].append(mlib.pointing_hit(np.asarray(it['maps'][n]), G, g, SIZE))
    return {n: np.asarray(v) for n, v in hit.items()}, n_items


def holm(ps):
    order = np.argsort(ps)
    out = np.empty(len(ps))
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(ps) - rank) * ps[i])
        out[i] = min(1.0, run)
    return out


def run(tag, excl):
    hit, n = score(excl)
    base = hit[BASELINE]
    others = [n_ for n_ in sorted(hit) if n_ != BASELINE]
    rows = []
    for o in others:
        m = mcnemar(base, hit[o])
        rows.append(dict(other=o, model_derived=o not in NOT_MODEL_DERIVED,
                         actnorm=float(base.mean()), other_rate=float(hit[o].mean()),
                         diff=float(base.mean() - hit[o].mean()), **m))
    p = holm(np.array([r['p_mcnemar'] for r in rows]))
    for r, ph in zip(rows, p):
        r['p_holm'] = float(ph)
    rows.sort(key=lambda r: -r['other_rate'])

    md = [r for r in rows if r['model_derived']]
    beaten = [r for r in md if r['diff'] > 0]
    sig = [r for r in beaten if r['p_holm'] < 0.05]
    print(f'\n[{tag}]  n = {n} items,  activation magnitude = {base.mean():.3f}')
    print(f'  model-derived maps: {len(md)}; point estimate below activation '
          f'magnitude in {len(beaten)}; significantly below after Holm: {len(sig)}')
    print(f"  {'map':18s}{'rate':>7s}{'diff':>8s}{'P (McNemar)':>13s}{'P Holm':>9s}")
    for r in rows[:6]:
        print(f"  {r['other']:18s}{r['other_rate']:7.3f}{r['diff']:+8.3f}"
              f"{r['p_mcnemar']:13.3g}{r['p_holm']:9.3g}")
    return dict(tag=tag, n_items=n, excluded=list(excl),
                actnorm_hit=float(base.mean()),
                n_model_derived=len(md), n_below_point_estimate=len(beaten),
                n_below_significant=len(sig), rows=rows)


if __name__ == '__main__':
    out = dict(
        note='exact McNemar, activation magnitude against every other map, '
             'Holm-corrected across the comparisons; the same items, ground '
             'truth and estimator as m4',
        all_findings=run('all findings', ()),
        no_cardiomegaly=run('cardiomegaly excluded', ('Cardiomegaly',)))
    mlib.save('m13_actnorm_test.json', out)
