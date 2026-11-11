"""The EXACT definition of the stage-7A primary contrast set.

`w9_report.py` builds its headline counts by collecting every row of
`outputs/wsol/w3_stats.json` that carries a `p_holm` field.  That field is
written by `w3_stats.py` for, and only for, the first `PRIMARY = 8` entries of
its `PAIRS` list.  This module re-derives the same set from the map names alone,
so a figure script can reproduce the counts without depending on that
implementation detail, and asserts that the two agree.

THE SET
-------
unit          1x1 cells only.  No `_b3` row is primary, and the comparator is
              `collapsed`, never `collapsed_b3`.
pairs         (N_lse, collapsed) (S_lse, collapsed) (N_lse, center) (S_lse, center)
              (N_ln,  collapsed) (S_ln,  collapsed) (N_ln,  center) (S_ln,  center)
              The four `_ln` pairs exist only where the backbone has a
              LayerNorm-relocation head, i.e. the four `vitb16` cells.
cells         all 12 (dataset x backbone); none excluded.
metrics       whichever the dataset defines:
                cub                  MaxBoxAccV2, best_iou, pointing_strict,
                                     pointing_tol15
                voc12                pointing_strict, pointing_tol15
                voc12seg, coco17     pointing_strict, pointing_tol15, peak_mask_iou
significance  `p_holm` < 0.05.  Holm-Bonferroni is applied inside each
              (dataset, backbone, metric) family over exactly these rows, so the
              family size is 4 for a CNN cell and 8 for a ViT cell.  Secondary
              rows (the `_b3` scale, the context rows, and the post-hoc N-vs-S
              rows) carry a raw `p` only and are NOT counted.
direction     `diff` > 0 means the dissociated map (a) scored higher than its
              comparator (b).

Expected size: cub 3 cells x 4 metrics, voc12 3 x 2, voc12seg 3 x 3, coco17 3 x 3,
times 4 pairs (8 on the ViT cell of each dataset)  =  192.
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import wlib as W

PRIMARY_PAIRS = [('N_lse', 'collapsed'), ('S_lse', 'collapsed'),
                 ('N_lse', 'center'), ('S_lse', 'center'),
                 ('N_ln', 'collapsed'), ('S_ln', 'collapsed'),
                 ('N_ln', 'center'), ('S_ln', 'center')]
ALPHA = 0.05


def select(w3):
    """w3: the parsed outputs/wsol/w3_stats.json -> list of primary contrasts."""
    out = []
    for cell, v in w3.items():
        for metric, rows in v['contrasts'].items():
            for r in rows:
                if (r['a'], r['b']) not in PRIMARY_PAIRS:
                    continue
                if 'p_holm' not in r:
                    raise AssertionError(
                        'primary pair without p_holm: %s %s %s vs %s'
                        % (cell, metric, r['a'], r['b']))
                out.append(dict(cell=cell, dataset=v['dataset'],
                                backbone=v['backbone'], metric=metric,
                                a=r['a'], b=r['b'], va=r['va'], vb=r['vb'],
                                diff=r['diff'], ci=r['ci'], test=r['test'],
                                # Supplementary Table S2 prints the sample
                                # size beside the test, so it has to survive
                                # this projection. The threshold-swept rows
                                # carry none of their own and take the cell's.
                                n=r.get('n', v.get('n_records')),
                                p=r['p'], p_holm=r['p_holm']))
    # the set must equal "every row that carries p_holm"
    n_flag = sum(1 for v in w3.values() for rows in v['contrasts'].values()
                 for r in rows if 'p_holm' in r)
    assert n_flag == len(out), (n_flag, len(out))
    return out


def counts(sel, comparator=None):
    s = [r for r in sel if comparator is None or r['b'] == comparator]
    better = sum(1 for r in s if r['diff'] > 0 and r['p_holm'] < ALPHA)
    worse = sum(1 for r in s if r['diff'] < 0 and r['p_holm'] < ALPHA)
    return dict(n=len(s), better=better, worse=worse,
                tied=len(s) - better - worse)


if __name__ == '__main__':
    w3 = W.load('w3_stats.json')
    sel = select(w3)
    print('cells in w3_stats.json : %d  (%s)'
          % (len(w3), ', '.join(sorted(w3))))
    print('primary contrasts      : %d' % len(sel))
    for name, comp in (('overall', None), ('vs collapsed', 'collapsed'),
                       ('vs center', 'center')):
        c = counts(sel, comp)
        print('%-14s n=%-4d better=%-4d worse=%-4d tied=%-4d'
              % (name, c['n'], c['better'], c['worse'], c['tied']))
    # breakdown, so a figure can check its own slicing
    print('\nby dataset x backbone:')
    for cell in sorted(set(r['cell'] for r in sel)):
        s = [r for r in sel if r['cell'] == cell]
        c = counts(s)
        print('  %-22s n=%-4d better=%-3d worse=%-3d tied=%-3d  metrics=%s'
              % (cell, c['n'], c['better'], c['worse'], c['tied'],
                 ','.join(sorted(set(r['metric'] for r in s)))))
    print('\nby map:')
    for m in ('N_lse', 'S_lse', 'N_ln', 'S_ln'):
        s = [r for r in sel if r['a'] == m]
        c = counts(s)
        print('  %-8s n=%-4d better=%-3d worse=%-3d tied=%-3d'
              % (m, c['n'], c['better'], c['worse'], c['tied']))
    W.save('primary_set.json',
           dict(alpha=ALPHA, primary_pairs=[list(p) for p in PRIMARY_PAIRS],
                n=len(sel), overall=counts(sel),
                vs_collapsed=counts(sel, 'collapsed'),
                vs_center=counts(sel, 'center'), contrasts=sel))
