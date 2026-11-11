"""W9 -- the per-cell effect sizes behind the 192-contrast tally.

A win/lose/tie count over 96 significance tests is not adequate reporting for a
primary outcome: it hides how large the wins are, how large the losses are, and
whether the two are comparable. `w3_stats.json` carries `diff`, its bootstrap
interval and the Holm-corrected P for every contrast.

This collects them into one record, per contrast and summarised by arm,
comparator and backbone. That record gives
Tables S1 and S2; the main text keeps the counts as a summary line and points
there.
"""
import os
import sys
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import wlib as W
from primary_set import select

ROOT = os.path.dirname(os.path.dirname(HERE))
ALPHA = 0.05

# the arm a contrast belongs to is carried by the name of the dissociated map:
# `_ln` is the relocated normalisation, `_lse` the replaced aggregator
ARM = {'ln': 'relocated normalisation', 'lse': 'replaced aggregator'}
PAIR = {('N', 'collapsed'): 'necessity vs collapsed',
        ('S', 'collapsed'): 'sufficiency vs collapsed',
        ('N', 'center'): 'necessity vs centre prior',
        ('S', 'center'): 'sufficiency vs centre prior'}
METRIC = {'MaxBoxAccV2': 'MaxBoxAccV2', 'best_iou': 'best IoU',
          'pointing_strict': 'pointing, strict',
          'pointing_tol15': 'pointing, 15 px',
          'peak_mask_iou': 'peak-mask IoU'}


def parts(c):
    half, arm = c['a'].split('_')
    return half, arm, c['b']


def fmt(x, nd=4):
    return ('%+.*f' % (nd, x)) if x is not None else ''


def iqr(v):
    v = np.asarray(v, float)
    if not v.size:
        return '--'
    if v.size == 1:
        return '%+.3f' % v[0]
    q1, q3 = np.percentile(v, [25, 75])
    return '%+.3f [%+.3f, %+.3f]' % (np.median(v), q1, q3)


def main():
    prim = select(W.load('w3_stats.json') if hasattr(W, 'load') else
                  json.load(open(os.path.join(W.OUT, 'w3_stats.json'))))
    assert len(prim) == 192, 'expected 192 primary contrasts, got %d' % len(prim)

    rows = []
    for c in prim:
        half, arm, comp = parts(c)
        rows.append(dict(
            dataset=c['dataset'], backbone=c['backbone'],
            metric=METRIC.get(c['metric'], c['metric']),
            arm=ARM[arm], pair=PAIR[(half, comp)],
            va=c['va'], vb=c['vb'], diff=c['diff'],
            lo=c['ci'][0], hi=c['ci'][1], p_holm=c['p_holm'],
            # The Methods say the test is named alongside each value. It was
            # not: S2 printed neither the test nor the sample size, so a
            # reader could not tell a McNemar row from a Wilcoxon one.
            test=c.get('test', ''), n=c.get('n', 0),
            sig='win' if (c['p_holm'] < ALPHA and c['diff'] > 0)
                else ('loss' if c['p_holm'] < ALPHA else 'tie')))

    summary = []
    for arm in ('relocated normalisation', 'replaced aggregator'):
        for comp in ('collapsed', 'centre prior'):
            sub0 = [r for r in rows if r['arm'] == arm
                    and r['pair'].endswith(comp)]
            if not sub0:
                continue
            bks = sorted({r['backbone'] for r in sub0})
            # 'all' first, then a row per backbone wherever the arm spans more
            # than one -- the aggregator arm reverses between backbones, and a
            # pooled tally hides that
            for bk in ['all'] + (bks if len(bks) > 1 else []):
                sub = sub0 if bk == 'all' else [r for r in sub0
                                                if r['backbone'] == bk]
                w = [r['diff'] for r in sub if r['sig'] == 'win']
                l_ = [r['diff'] for r in sub if r['sig'] == 'loss']
                a = [r['diff'] for r in sub]
                summary.append(dict(arm=arm, comparator=comp, backbone=bk,
                                    n=len(sub), win=len(w), loss=len(l_),
                                    tie=len(sub) - len(w) - len(l_),
                                    median_win=float(np.median(w)) if w else None,
                                    median_loss=float(np.median(l_)) if l_ else None,
                                    median_all=float(np.median(a))))

    W.save('w9_percell.json', dict(n_contrasts=len(rows), alpha=ALPHA,
                                   summary=summary, rows=rows))

    print('  192 contrasts -> outputs/wsol/w9_percell.json')
    for s in summary:
        if s['backbone'] != 'all':
            continue
        print('  %-24s vs %-12s n=%3d  win %2d  lose %2d  tie %2d   '
              'median win %s  median loss %s'
              % (s['arm'], s['comparator'], s['n'], s['win'], s['loss'],
                 s['tie'],
                 '%+.3f' % s['median_win'] if s['median_win'] is not None else '  --  ',
                 '%+.3f' % s['median_loss'] if s['median_loss'] is not None else '  --  '))
    for s in summary:
        if s['backbone'] == 'all' or s['comparator'] != 'collapsed':
            continue
        print('    %-24s %-10s n=%3d  win %2d  lose %2d  tie %2d'
              % (s['arm'], s['backbone'], s['n'], s['win'], s['loss'], s['tie']))


if __name__ == '__main__':
    main()
