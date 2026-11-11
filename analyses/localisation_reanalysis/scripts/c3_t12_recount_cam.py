"""C3 tasks 1 and 2.

1. Re-derive the 192 primary contrasts from the per-record arrays with the
   exact w3 tests; assert they reproduce w3_stats.json / w9_percell.json; then
   break the 39 wins (and 26 losses) against the collapsed map down by metric,
   arm, half and backbone (extent vs maximum-location).
2. Re-run the 96 'vs collapsed' contrasts with plain CAM (`cam`, = X w_c, no
   reference) as the comparator.  Holm family: the w3 family with `collapsed`
   replaced by `cam`, i.e. {N,S} x {cam, center} per (dataset, backbone,
   metric) -- size 4 on CNN cells, 8 on ViT cells, exactly as in w9.  A
   sensitivity variant uses the family of the cam rows alone (size 2 / 4).
   Also: collapsed minus cam per cell and metric, same tests.
"""
import json
from collections import Counter, defaultdict
import numpy as np
from localisation_reanalysis_lib import (W, WO, DATASETS, BACKBONES, METRICS, EXTENT, MAXLOC, MNAME,
                   test, holm, verdict, arm_of, half_of, separated_maps, save)


def run_family(ds, bk, metric, comparator):
    rows = []
    for a in separated_maps(bk):
        for b in (comparator, 'center'):
            r = test(ds, bk, metric, a, b)
            r.update(dataset=ds, backbone=bk, metric=metric, a=a, b=b)
            rows.append(r)
    adj = holm([r['p'] for r in rows])
    for r, q in zip(rows, adj):
        r['p_holm'] = q
        r['sig'] = verdict(r['diff'], q)
    # sensitivity: Holm over the comparator rows only
    sub = [r for r in rows if r['b'] == comparator]
    for r, q in zip(sub, holm([r['p'] for r in sub])):
        r['p_holm_own'] = q
        r['sig_own'] = verdict(r['diff'], q)
    return rows


def main():
    w3 = json.load(open(WO + '/w3_stats.json'))
    allrows = {'collapsed': [], 'cam': []}
    for ds in DATASETS:
        for bk in BACKBONES:
            for m in METRICS[ds]:
                for comp in ('collapsed', 'cam'):
                    allrows[comp] += run_family(ds, bk, m, comp)

    # ---------------- reproduction check against w3_stats.json
    ref = {}
    for cell, v in w3.items():
        for m, rows in v['contrasts'].items():
            for r in rows:
                if 'p_holm' in r:
                    ref[(v['dataset'], v['backbone'], m, r['a'], r['b'])] = r
    mx_d = mx_p = 0.0
    nmatch = 0
    for r in allrows['collapsed']:
        q = ref[(r['dataset'], r['backbone'], r['metric'], r['a'], r['b'])]
        mx_d = max(mx_d, abs(q['diff'] - r['diff']))
        mx_p = max(mx_p, abs(q['p_holm'] - r['p_holm']))
        nmatch += 1
    assert nmatch == 192 == len(ref), (nmatch, len(ref))
    print('reproduction of the 192 primary contrasts: max |diff| %.2e, max |p_holm| %.2e'
          % (mx_d, mx_p))

    # ---------------- task 1: the 96 vs collapsed
    vc = [r for r in allrows['collapsed'] if r['b'] == 'collapsed']
    vz = [r for r in allrows['collapsed'] if r['b'] == 'center']
    tally = lambda rs: dict(win=sum(r['sig'] == 'win' for r in rs),
                            loss=sum(r['sig'] == 'loss' for r in rs),
                            tie=sum(r['sig'] == 'tie' for r in rs), n=len(rs))
    print('vs collapsed', tally(vc), ' vs centre', tally(vz))
    t1 = {}
    for sig in ('win', 'loss', 'tie'):
        tab = defaultdict(Counter)
        for r in vc:
            if r['sig'] == sig:
                tab[arm_of(r['a'])][r['metric']] += 1
        t1[sig] = {arm: dict(tab[arm]) for arm in tab}
    ext = defaultdict(Counter)
    for r in vc:
        ext[(arm_of(r['a']), r['sig'])]['extent' if r['metric'] in EXTENT else 'maxloc'] += 1
    by_bk = defaultdict(Counter)
    for r in vc:
        by_bk[(arm_of(r['a']), r['backbone'], 'extent' if r['metric'] in EXTENT else 'maxloc')][r['sig']] += 1
    by_half = defaultdict(Counter)
    for r in vc:
        by_half[(arm_of(r['a']), half_of(r['a']), 'extent' if r['metric'] in EXTENT else 'maxloc')][r['sig']] += 1
    ext_wins = sorted([r for r in vc if r['sig'] == 'win' and r['metric'] in EXTENT],
                      key=lambda r: -r['diff'])
    print('\nTask 1 -- wins vs collapsed by arm x metric')
    for arm in ('relocation', 'aggregator'):
        print('  %-10s' % arm, {MNAME[k]: v for k, v in t1['win'].get(arm, {}).items()})
    print('  losses:', {a: {MNAME[k]: v for k, v in d.items()} for a, d in t1['loss'].items()})
    for k, v in sorted(ext.items()):
        print('  ', k, dict(v))
    for k, v in sorted(by_bk.items()):
        print('  bk', k, dict(v))
    for k, v in sorted(by_half.items()):
        print('  half', k, dict(v))
    print('  extent wins:')
    for r in ext_wins:
        print('    %-8s %-8s %-14s %-6s %+.4f [%+.4f, %+.4f] p_holm=%.2g'
              % (r['dataset'], r['backbone'], MNAME[r['metric']], r['a'], r['diff'],
                 r['ci'][0], r['ci'][1], r['p_holm']))
    big = max(vc, key=lambda r: r['diff'] if r['sig'] == 'win' else -9)
    print('  largest win:', big['dataset'], big['backbone'], big['metric'], big['a'], '%+.4f' % big['diff'])

    # ---------------- task 2: plain CAM as comparator
    vcam = [r for r in allrows['cam'] if r['b'] == 'cam']
    print('\nTask 2 -- vs plain CAM (family {N,S}x{cam,centre}):', tally(vcam))
    own = dict(win=sum(r['sig_own'] == 'win' for r in vcam),
               loss=sum(r['sig_own'] == 'loss' for r in vcam),
               tie=sum(r['sig_own'] == 'tie' for r in vcam))
    print('   sensitivity, family of cam rows only:', own)
    t2 = []
    for arm in ('relocation', 'aggregator'):
        for bk in ['all'] + BACKBONES:
            for comp, rs in (('collapsed', vc), ('cam', vcam)):
                sub = [r for r in rs if arm_of(r['a']) == arm and (bk == 'all' or r['backbone'] == bk)]
                if not sub:
                    continue
                t = tally(sub)
                t.update(arm=arm, backbone=bk, comparator=comp,
                         win_extent=sum(r['sig'] == 'win' and r['metric'] in EXTENT for r in sub),
                         loss_extent=sum(r['sig'] == 'loss' and r['metric'] in EXTENT for r in sub),
                         median_diff=float(np.median([r['diff'] for r in sub])))
                t2.append(t)
                print('  %-10s %-8s vs %-9s n=%2d win %2d loss %2d tie %2d  (extent: win %d loss %d)  median diff %+.4f'
                      % (arm, bk, comp, t['n'], t['win'], t['loss'], t['tie'],
                         t['win_extent'], t['loss_extent'], t['median_diff']))
    # transitions collapsed -> cam verdict for the same separated map
    trans = Counter()
    for r1, r2 in zip(vc, vcam):
        assert (r1['dataset'], r1['backbone'], r1['metric'], r1['a']) == (r2['dataset'], r2['backbone'], r2['metric'], r2['a'])
        trans[(arm_of(r1['a']), r1['sig'], r2['sig'])] += 1
    print('  verdict transitions (arm, vs collapsed -> vs cam):')
    for k, v in sorted(trans.items()):
        print('    ', k, v)

    # collapsed minus cam, per cell and metric (raw p and Holm over the cell's metrics)
    cc = []
    for ds in DATASETS:
        for bk in BACKBONES:
            rows = []
            for m in METRICS[ds]:
                r = test(ds, bk, m, 'collapsed', 'cam')
                r.update(dataset=ds, backbone=bk, metric=m)
                rows.append(r)
            for r, q in zip(rows, holm([r['p'] for r in rows])):
                r['p_holm_cell'] = q
                r['sig'] = verdict(r['diff'], q)
            cc += rows
    print('\n  collapsed - cam per cell (Holm over the metrics of the cell):')
    for r in cc:
        print('    %-8s %-8s %-15s coll %.4f cam %.4f diff %+.4f [%+.4f,%+.4f] p=%.2g %s'
              % (r['dataset'], r['backbone'], MNAME[r['metric']], r['va'], r['vb'], r['diff'],
                 r['ci'][0], r['ci'][1], r['p'], r['sig']))
    print('  collapsed vs cam tally:', Counter(r['sig'] for r in cc))
    print('  by backbone:', {bk: dict(Counter(r['sig'] for r in cc if r['backbone'] == bk)) for bk in BACKBONES})

    save('t1_t2_contrasts.json', dict(
        reproduction=dict(n=192, max_abs_diff=mx_d, max_abs_p_holm=mx_p),
        task1=dict(tally_vs_collapsed=tally(vc), tally_vs_centre=tally(vz),
                   by_arm_metric=t1,
                   extent_vs_maxloc={'%s|%s' % k: dict(v) for k, v in ext.items()},
                   by_arm_backbone_family={'%s|%s|%s' % k: dict(v) for k, v in by_bk.items()},
                   by_arm_half_family={'%s|%s|%s' % k: dict(v) for k, v in by_half.items()},
                   extent_wins=ext_wins),
        task2=dict(family='Holm within (dataset, backbone, metric) over {N,S}(x{lse,ln}) x {cam, center}',
                   tally_vs_cam=tally(vcam), tally_vs_cam_ownfamily=own, summary=t2,
                   transitions={'%s|%s->%s' % k: v for k, v in trans.items()},
                   rows_vs_cam=vcam, collapsed_minus_cam=cc),
        rows_vs_collapsed=vc, rows_vs_centre=vz))


if __name__ == '__main__':
    main()
