"""POST HOC paired contrast within H3: CAM of H3 with its LayerNorm ignored minus the H3
sufficiency map (MaxBoxAccV2), on stored values and on within-image ranks.

    python k2_h3_pair.py

Reads (never writes) analyses/end_to_end_cub/out/per_image/H3_s<seed>.npz (curve_S, pt_strict_S, best_iou_S),
analyses/end_to_end_cub/out/per_image/H0_s<seed>.npz (control), analyses/end_to_end_cub/post/out/per_image/H3_s<seed>_natural.npz
(curve_cam, pt_strict_cam, best_iou_cam), analyses/end_to_end_cub/post/out/per_image/k2_rank_curves.npz (rank
curves) and the stored k2_natural.json / k2_rank.json (controls). Writes
analyses/end_to_end_cub/post/out/k2_h3_pair.json.

Bootstrap exactly as analyses/end_to_end_cub/scripts/k2_stats.py: one image resample shared by every seed and map,
RandomState(wlib.PROTOCOL['seed']).randint(0, n, (1000, n)); difference = mean over seeds of the
seed-matched difference; 95% percentile interval; P = 2 min(P*(b <= 0), P*(b >= 0)) clipped to
[1/1000, 1]. A single contrast, so no Holm adjustment. Controls: H3 CAM minus H0 collapsed must
reproduce the stored k2_natural.json value (raw) and k2_rank.json value (ranks).
"""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
POST = os.path.dirname(HERE)
K2 = os.path.dirname(POST)
ROOT = os.path.abspath(os.path.join(K2, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'scripts', 'wsol'))
import wlib as W   # noqa: E402  (numpy only at import)

NB = 1000
DELTAS = tuple(W.PROTOCOL['deltas'])
SEEDS = [0, 1, 2]
OUT = os.path.join(K2, 'out')
POUT = os.path.join(POST, 'out')


def main():
    C = {}      # (version, head, map, seed) -> curves (n, 100)
    PI = {}     # (head, map, seed) -> dict(pt_strict, best_iou)
    for s in SEEDS:
        h3 = np.load(os.path.join(OUT, 'per_image', 'H3_s%d.npz' % s))
        h0 = np.load(os.path.join(OUT, 'per_image', 'H0_s%d.npz' % s))
        nt = np.load(os.path.join(POUT, 'per_image', 'H3_s%d_natural.npz' % s))
        C['raw', 'H3', 'S', s] = h3['curve_S']
        C['raw', 'H0', 'collapsed', s] = h0['curve_collapsed']
        C['raw', 'H3', 'cam', s] = nt['curve_cam']
        PI['H3', 'S', s] = dict(pointing_strict=h3['pt_strict_S'], best_iou=h3['best_iou_S'])
        PI['H3', 'cam', s] = dict(pointing_strict=nt['pt_strict_cam'], best_iou=nt['best_iou_cam'])
    rk = np.load(os.path.join(POUT, 'per_image', 'k2_rank_curves.npz'))
    cols = [str(c) for c in rk['cols']]
    vers = [str(v) for v in rk['versions']]
    for vi, vn in enumerate(vers):
        for s in SEEDS:
            for h, m in (('H3', 'S'), ('H3', 'cam'), ('H0', 'collapsed')):
                C[vn + '_npz', h, m, s] = rk['curves'][:, vi, cols.index('%s_%s_s%d' % (h, m, s))]
    # the raw curves of k2_rank_curves.npz must equal the stored per-image curves
    raw_vs_npz = max(float(np.abs(C['raw', h, m, s].astype(np.float64) -
                                  C['raw_npz', h, m, s].astype(np.float64)).max())
                     for s in SEEDS for h, m in (('H3', 'S'), ('H3', 'cam'), ('H0', 'collapsed')))
    n = len(C['raw', 'H3', 'S', 0])

    IND = {k: np.concatenate([(v >= d) for d in DELTAS], 1).astype(np.float64) for k, v in C.items()}

    def mba_w(vn, h, m, s, w):
        x = (w @ IND[vn, h, m, s]) / w.sum()
        return float(x.reshape(len(DELTAS), -1).max(1).mean())

    rs = np.random.RandomState(W.PROTOCOL['seed'])
    IDX = rs.randint(0, n, (NB, n))
    WTS = np.stack([np.bincount(k, minlength=n) for k in IDX]).astype(np.float64)
    one = np.ones(n)
    chk = 0.0
    for b in range(3):
        chk = max(chk, abs(mba_w('raw', 'H3', 'S', 0, WTS[b]) -
                           W.maxboxacc_v2(C['raw', 'H3', 'S', 0], DELTAS, IDX[b])[0]))
    assert chk < 1e-12, chk

    def summarise(bs, d0, per_seed, **kw):
        p = 2.0 * min((bs <= 0).mean(), (bs >= 0).mean())
        return dict(kw, diff=float(d0), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                    p=float(min(1.0, max(p, 1.0 / NB))), diff_per_seed=[float(x) for x in per_seed])

    def contrast(vn, A, B):
        f = lambda w: np.mean([mba_w(vn, *A, s, w) - mba_w(vn, *B, s, w) for s in SEEDS])  # noqa: E731
        bs = np.array([f(WTS[b]) for b in range(NB)])
        return summarise(bs, f(one), [mba_w(vn, *A, s, one) - mba_w(vn, *B, s, one) for s in SEEDS],
                         a='%s_%s' % A, b='%s_%s' % B, metric='MaxBoxAccV2', version=vn)

    def contrast_pi(A, B, metric):
        dimg = np.mean([PI[A + (s,)][metric].astype(np.float64) - PI[B + (s,)][metric].astype(np.float64)
                        for s in SEEDS], 0)
        bs = (WTS @ dimg) / n
        per_seed = [float((PI[A + (s,)][metric].astype(np.float64) -
                           PI[B + (s,)][metric].astype(np.float64)).mean()) for s in SEEDS]
        return summarise(bs, dimg.mean(), per_seed, a='%s_%s' % A, b='%s_%s' % B, metric=metric,
                         version='raw')

    A, B, CO = ('H3', 'cam'), ('H3', 'S'), ('H0', 'collapsed')
    res = dict(post_hoc=True, n_test=n, n_boot=NB, seeds=SEEDS, holm='none (single contrast)',
               description='H3 CAM (LayerNorm ignored) minus H3 sufficiency, seed-matched, mean over seeds',
               bootstrap_vs_wlib_max_abs=chk, raw_curves_vs_rank_npz_max_abs=raw_vs_npz)
    res['raw_MaxBoxAccV2'] = contrast('raw', A, B)
    res['rank_MaxBoxAccV2'] = contrast('rank_npz', A, B)
    res['raw_best_iou'] = contrast_pi(A, B, 'best_iou')
    res['raw_pointing_strict'] = contrast_pi(A, B, 'pointing_strict')
    # controls against the stored contrasts with H0 collapsed
    kn = json.load(open(os.path.join(POUT, 'k2_natural.json')))
    kr = json.load(open(os.path.join(POUT, 'k2_rank.json')))
    st_raw = [q for q in kn['natural_vs_collapsed_MaxBoxAccV2'] if q['a'] == 'H3_cam'][0]
    st_rank = [q for q in kr['contrasts']['rank'] if q['a'] == 'H3_cam'][0]
    ctl = {}
    for vn, st in (('raw', st_raw), ('rank_npz', st_rank)):
        r = contrast(vn, A, CO)
        ctl[vn] = dict(recomputed=r, stored_diff=st['diff'], stored_ci=st['ci'],
                       max_abs_err=max(abs(r['diff'] - st['diff']), abs(r['ci'][0] - st['ci'][0]),
                                       abs(r['ci'][1] - st['ci'][1])))
    res['control_H3cam_vs_H0collapsed'] = ctl
    assert all(v['max_abs_err'] < 1e-12 for v in ctl.values()), ctl
    with open(os.path.join(POUT, 'k2_h3_pair.json'), 'w', encoding='utf-8') as fh:
        json.dump(res, fh, indent=1)
    for k in ('raw_MaxBoxAccV2', 'rank_MaxBoxAccV2', 'raw_best_iou', 'raw_pointing_strict'):
        r = res[k]
        print('%-22s %+.4f [%+.4f, %+.4f] P %.3g  per seed %s'
              % (k, r['diff'], r['ci'][0], r['ci'][1], r['p'],
                 ', '.join('%+.4f' % x for x in r['diff_per_seed'])))
    print('controls max abs err', {k: v['max_abs_err'] for k, v in ctl.items()},
          'raw vs npz curves', raw_vs_npz, 'wlib check', chk)


if __name__ == '__main__':
    main()
