"""C3 task 3: Holm verdicts of the 192 primary contrasts recomputed
from image-clustered tests.

For every primary contrast the record-level test of w3 (McNemar / Wilcoxon)
is replaced by
  (A) an image-level paired sign-flip permutation test (20,000 flips,
      seed 0): the exchangeable unit is the image, so all records of an image
      flip together;                                     <- primary variant
  (B) a clustered-bootstrap p (2,000 image resamples, seed 0,
      p = 2 min(P(b<=0), P(b>=0)), floored at 1/2000), the construction w3
      already uses for MaxBoxAccV2;
  (D) a cluster-level Wilcoxon signed-rank test on the per-image mean of the
      record differences (the w3 test family, applied to image units);
  (C) the same sign-flip test with every record its own cluster, to separate
      the effect of clustering (C -> A) from the effect of changing the test
      (original -> C).
MaxBoxAccV2 rows keep w3's paired bootstrap over images (already clustered;
CUB has one record per image).  Holm is then re-applied inside each
(dataset, backbone, metric) family over all of the family's primary rows.
"""
import json
from collections import Counter, defaultdict
import numpy as np
from localisation_reanalysis_lib import (W, WO, DATASETS, BACKBONES, METRICS, MNAME, per_record, recidx,
                   holm, verdict, separated_maps, signflip_p, clusterboot_p, save)


def main():
    w3 = json.load(open(WO + '/w3_stats.json'))
    ref = {}
    for v in w3.values():
        for m, rows in v['contrasts'].items():
            for r in rows:
                if 'p_holm' in r:
                    ref[(v['dataset'], v['backbone'], m, r['a'], r['b'])] = r
    out = []
    for ds in DATASETS:
        for bk in BACKBONES:
            g_img = recidx(ds, bk)
            g_rec = np.arange(len(g_img))
            for m in METRICS[ds]:
                fam = []
                for a in separated_maps(bk):
                    for b in ('collapsed', 'center'):
                        q = ref[(ds, bk, m, a, b)]
                        r = dict(dataset=ds, backbone=bk, metric=m, a=a, b=b,
                                 diff=q['diff'], p_orig=q['p'], p_holm_orig=q['p_holm'],
                                 test_orig=q['test'])
                        if m == 'MaxBoxAccV2':
                            r.update(p_perm_img=q['p'], p_perm_rec=q['p'], p_cboot=q['p'], p_wil_img=q['p'])
                        else:
                            xa = per_record(ds, bk, m, a); xb = per_record(ds, bk, m, b)
                            r['p_perm_img'] = signflip_p(xa, xb, g_img)
                            r['p_perm_rec'] = (r['p_perm_img'] if ds == 'cub'
                                               else signflip_p(xa, xb, g_rec))
                            p, lo, hi = clusterboot_p(xa, xb, g_img)
                            r.update(p_cboot=p, ci_cboot=[lo, hi])
                            k = int(g_img.max()) + 1
                            cnt = np.bincount(g_img, minlength=k)
                            ma = np.bincount(g_img, weights=xa, minlength=k) / cnt
                            mb = np.bincount(g_img, weights=xb, minlength=k) / cnt
                            r['p_wil_img'] = W.wilcoxon(ma, mb)['p']
                        fam.append(r)
                for key in ('orig', 'perm_img', 'perm_rec', 'cboot', 'wil_img'):
                    adj = holm([r['p_' + key] for r in fam])
                    for r, qv in zip(fam, adj):
                        r['p_holm_' + key] = qv
                        r['sig_' + key] = verdict(r['diff'], qv)
                for r in fam:
                    assert abs(r['p_holm_orig'] - ref[(ds, bk, m, r['a'], r['b'])]['p_holm']) < 1e-12
                out += fam
                print(ds, bk, m, 'done', flush=True)

    summ = {}
    print('\nverdict changes vs the shipped record-level verdicts (192 primary contrasts)')
    for key, lab in (('perm_img', 'image-level sign-flip permutation'),
                     ('cboot', 'image-clustered bootstrap p'),
                     ('wil_img', 'Wilcoxon on image-mean differences'),
                     ('perm_rec', 'record-level sign-flip permutation (test change only)')):
        per_ds = {}
        for ds in DATASETS + ['all']:
            sub = [r for r in out if ds == 'all' or r['dataset'] == ds]
            ch = [r for r in sub if r['sig_' + key] != r['sig_orig']]
            per_ds[ds] = dict(n=len(sub), changed=len(ch),
                              transitions=dict(Counter('%s->%s' % (r['sig_orig'], r['sig_' + key]) for r in ch)),
                              changed_vs_collapsed=sum(r['b'] == 'collapsed' for r in ch))
        # pure clustering effect: record-level permutation -> image-level permutation
        summ[key] = dict(label=lab, per_dataset=per_ds)
        print(' ', lab)
        for ds, v in per_ds.items():
            print('    %-9s n=%3d changed=%2d (vs collapsed %d) %s'
                  % (ds, v['n'], v['changed'], v['changed_vs_collapsed'], v['transitions']))
    pure = {}
    for ds in DATASETS + ['all']:
        sub = [r for r in out if ds == 'all' or r['dataset'] == ds]
        ch = [r for r in sub if r['sig_perm_img'] != r['sig_perm_rec']]
        pure[ds] = dict(n=len(sub), changed=len(ch),
                        transitions=dict(Counter('%s->%s' % (r['sig_perm_rec'], r['sig_perm_img']) for r in ch)))
    summ['clustering_only'] = pure
    print('  clustering alone (record-level permutation -> image-level permutation):')
    for ds, v in pure.items():
        print('    %-9s n=%3d changed=%2d %s' % (ds, v['n'], v['changed'], v['transitions']))
    # tallies under the clustered verdicts
    for key in ('orig', 'perm_img', 'cboot'):
        for comp in ('collapsed', 'center'):
            sub = [r for r in out if r['b'] == comp]
            c = Counter(r['sig_' + key] for r in sub)
            summ.setdefault('tallies', {})['%s|%s' % (key, comp)] = dict(c)
            print('  tally %-9s vs %-9s win %2d loss %2d tie %2d' % (key, comp, c['win'], c['loss'], c['tie']))
    # which rows change under the primary clustered variant
    print('  rows whose verdict changes under the image-level permutation:')
    for r in out:
        if r['sig_perm_img'] != r['sig_orig']:
            print('    %-8s %-8s %-15s %-6s vs %-9s diff %+.4f  p_holm orig %.3g -> img %.3g  %s->%s'
                  % (r['dataset'], r['backbone'], MNAME[r['metric']], r['a'], r['b'], r['diff'],
                     r['p_holm_orig'], r['p_holm_perm_img'], r['sig_orig'], r['sig_perm_img']))
    save('t3_clustered.json', dict(design=__doc__, summary=summ, rows=out))


if __name__ == '__main__':
    main()
