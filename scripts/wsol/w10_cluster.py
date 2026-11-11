# -*- coding: utf-8 -*-
"""E1, vectorised: does clustering the localisation bootstrap by image move a verdict?

A record-level mean over a resample of images is the ratio of the selected
images' summed values to their summed counts, so the whole bootstrap reduces to
two matrix products over per-image aggregates. No per-draw concatenation.
"""
import glob
import os
import re

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
W = os.path.join(ROOT, 'outputs', 'wsol')
OUT = open(os.path.join(W, 'w10_cluster.txt'), 'w', encoding='utf-8')


def say(*a):
    print(*a)
    OUT.write(' '.join(str(x) for x in a) + '\n')


def agg(v, g, k):
    s = np.zeros(k)
    np.add.at(s, g, v)
    return s


def both(x, y, g, n=2000, seed=0):
    k = g.max() + 1
    cnt = np.bincount(g, minlength=k).astype(np.float64)
    sx, sy = agg(x, g, k), agg(y, g, k)
    d = float(x.mean() - y.mean())
    rs = np.random.RandomState(seed)
    pick = rs.randint(0, k, (n, k))
    cc = cnt[pick].sum(1)
    bc = sx[pick].sum(1) / cc - sy[pick].sum(1) / cc
    m = rs.randint(0, len(x), (n, len(x)))
    bi = x[m].mean(1) - y[m].mean(1)
    return (d, float(np.percentile(bi, 2.5)), float(np.percentile(bi, 97.5)),
            float(np.percentile(bc, 2.5)), float(np.percentile(bc, 97.5)))


rows = []
for f in sorted(glob.glob(os.path.join(W, '*.npy'))):
    b = os.path.basename(f)
    if b.startswith(('recidx', 'curve')):
        continue
    m = re.match(r'(pt|iou)_([a-z0-9]+)_([a-z0-9]+?)_(.+)\.npy$', b)
    if not m:
        continue
    kind, ds, bk, arm = m.groups()
    if arm.startswith('collapsed'):
        continue
    ridx = os.path.join(W, 'recidx_%s_%s.npy' % (ds, bk))
    base = os.path.join(W, '%s_%s_%s_collapsed.npy' % (kind, ds, bk))
    if not (os.path.exists(ridx) and os.path.exists(base)):
        continue
    g = np.load(ridx)
    XA, XB = np.load(f), np.load(base)
    if XA.ndim != XB.ndim or XA.shape[0] != XB.shape[0]:
        continue
    # pt_* arrays are (2, N): row 0 the strict pointing game, row 1 the
    # tolerant one. Scoring only row 0 dropped half of every such array.
    nr = XA.shape[0] if XA.ndim == 2 else 1
    for r in range(nr):
        xa = np.asarray(XA[r] if XA.ndim == 2 else XA, np.float64)
        xb = np.asarray(XB[r] if XB.ndim == 2 else XB, np.float64)
        if not (len(xa) == len(xb) == len(g)):
            continue
        # An arm bit-identical to the baseline is not a contrast: it scores
        # the collapsed map against itself, so both intervals are [0, 0] and
        # the width ratio is 0/0. Twenty-six such rows dragged the median
        # below 1 and inflated the count the Methods quote.
        if np.array_equal(xa, xb):
            continue
        d, l1, h1, l2, h2 = both(xa, xb, g)
        tag = arm if nr == 1 else '%s|%s' % (arm, ('strict', 'tol15')[r] if kind == 'pt' else 'r%d' % r)
        rows.append((ds, bk, tag, d, l1, h1, l2, h2, (l1 > 0 or h1 < 0), (l2 > 0 or h2 < 0)))

if not rows:
    raise SystemExit(
        'no per-record arrays found under %s.\n'
        'This script needs the per-record .npy arrays, which ship in the FULL '
        'replication tier and not in the core one. Rebuild with\n'
        '    python scripts/make_replication_package.py --full\n'
        'or obtain the full archive, then re-run.' % W)

say('contrasts recomputed: %d' % len(rows))
say('%-8s %-9s %-27s %9s %20s %20s' % ('dataset', 'backbone', 'arm', 'diff',
                                       'shipped 95% CI', 'clustered 95% CI'))
moved = 0
for ds, bk, arm, d, l1, h1, l2, h2, s1, s2 in rows:
    flag = ''
    if s1 != s2:
        moved += 1
        flag = '   VERDICT MOVES'
    say('%-8s %-9s %-27s %+9.4f [%+.4f,%+.4f] [%+.4f,%+.4f]%s'
        % (ds, bk, arm[:27], d, l1, h1, l2, h2, flag))
say('')
say('median width ratio, clustered over shipped: %.2f'
    % np.median([(r[7] - r[6]) / max(r[5] - r[4], 1e-12) for r in rows]))
say('verdicts that move: %d of %d' % (moved, len(rows)))
OUT.close()

# the record the number checker reads, emitted by the script that computes it
import json
import statistics
json.dump(dict(
    note='E1: paired bootstrap resampling images rather than object records',
    n_contrasts=len(rows),
    n_verdicts_moved=moved,
    median_width_ratio=statistics.median((r[7] - r[6]) / max(r[5] - r[4], 1e-12) for r in rows),
    max_width_ratio=max((r[7] - r[6]) / max(r[5] - r[4], 1e-12) for r in rows),
    rows=[dict(dataset=r[0], backbone=r[1], arm=r[2], diff=r[3],
               ci_iid=[r[4], r[5]], ci_cluster=[r[6], r[7]],
               ratio=(r[7] - r[6]) / max(r[5] - r[4], 1e-12),
               sig_iid=r[8], sig_cluster=r[9]) for r in rows]),
    open(os.path.join(W, 'w10_cluster.json'), 'w', encoding='utf-8'), indent=1)
print('wrote', os.path.join(W, 'w10_cluster.json'))
