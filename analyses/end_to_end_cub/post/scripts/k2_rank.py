"""POST HOC (written after K2 and the natural-map analysis were finished): rank-transform
sensitivity of MaxBoxAccV2.

    python k2_rank.py

Every stored 14 x 14 map (analyses/end_to_end_cub/out/per_image/<H>_s<seed>.npz 'map_*', and
analyses/end_to_end_cub/post/out/per_image/<H>_s<seed>_natural.npz 'map_*') is scored twice with exactly the K2
localisation pipeline (wlib.upsample 224 bilinear + min-max, wlib.box_curve 100 thresholds,
wlib.maxboxacc_v2 at 0.3/0.5/0.7; boxes rescaled to the 224 x 224 frame as in k2_eval):
  raw   the stored map itself (must reproduce the stored per-image curves and MaxBoxAccV2),
  rank  the map replaced by its within-image ranks, scipy.stats.rankdata(method='average') / 196.
Maps: H0 collapsed, H0 CAM, H1/H2/H3 necessity and sufficiency (K2), and every natural map of
analyses/end_to_end_cub/post (H0 Grad-CAM, H1 CAM, H1 Grad-CAM, H2 attention, H2 attention x evidence, H2 Grad-CAM,
H3 Grad-CAM, H3 CAM with LayerNorm ignored).
Contrasts: mean over seeds of [MaxBoxAccV2(map_s) - MaxBoxAccV2(H0 collapsed_s)], computed
separately for raw (vs raw collapsed) and rank (vs rank collapsed); bootstrap of k2_stats.py
(RandomState(wlib.PROTOCOL['seed']).randint(0, n, (1000, n)), one resample shared by all seeds and
maps; 95% percentile interval; P = 2 min tail share clipped to [1/1000, 1]); Holm within the
families of the raw analyses (FAMILIES below). Nothing existing is modified; writes
analyses/end_to_end_cub/post/out/k2_rank.json, analyses/end_to_end_cub/post/out/per_image/k2_rank_curves.npz and analyses/end_to_end_cub/post/RESULTS_rank.md.
"""
import os, sys, json, time
import numpy as np
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__))
POST = os.path.dirname(HERE)
K2 = os.path.dirname(POST)
sys.path.insert(0, os.path.join(K2, 'scripts'))
import k2lib as L   # noqa: E402
W = L.W

NB = 1000
EVAL_RES = W.PROTOCOL['eval_res']
NTHR = W.PROTOCOL['n_thresholds']
THR = np.arange(NTHR) / NTHR                       # as k2_eval.THR
DELTAS = tuple(W.PROTOCOL['deltas'])
ALPHA = 0.05
POUT = os.path.join(POST, 'out')
# (head, map, source): source 'k2' = analyses/end_to_end_cub/out/per_image, 'nat' = analyses/end_to_end_cub/post/out/per_image natural
MAPS = [('H0', 'collapsed', 'k2'), ('H0', 'cam', 'k2'), ('H0', 'gradcam', 'nat'),
        ('H1', 'N', 'k2'), ('H1', 'S', 'k2'), ('H2', 'N', 'k2'), ('H2', 'S', 'k2'),
        ('H3', 'N', 'k2'), ('H3', 'S', 'k2'),
        ('H1', 'cam', 'nat'), ('H1', 'gradcam', 'nat'), ('H2', 'attn', 'nat'),
        ('H2', 'attn_ev', 'nat'), ('H2', 'gradcam', 'nat'), ('H3', 'gradcam', 'nat'),
        ('H3', 'cam', 'nat')]
LBL = {('H0', 'collapsed'): 'H0 collapsed', ('H0', 'cam'): 'H0 CAM',
       ('H0', 'gradcam'): 'H0 Grad-CAM',
       ('H1', 'N'): 'H1 necessity', ('H1', 'S'): 'H1 sufficiency',
       ('H2', 'N'): 'H2 necessity', ('H2', 'S'): 'H2 sufficiency',
       ('H3', 'N'): 'H3 necessity', ('H3', 'S'): 'H3 sufficiency',
       ('H1', 'cam'): 'H1 CAM (per-cell class score)', ('H1', 'gradcam'): 'H1 Grad-CAM',
       ('H2', 'attn'): 'H2 attention weights', ('H2', 'attn_ev'): 'H2 attention x class evidence',
       ('H2', 'gradcam'): 'H2 Grad-CAM', ('H3', 'gradcam'): 'H3 Grad-CAM',
       ('H3', 'cam'): 'H3 CAM (LayerNorm ignored)'}
# Holm families exactly as in the raw analyses:
#   primary (k2_stats.py): the 6 N/S maps of H1-H3;
#   natural (k2_natural_stats.py FAM): the 7 natural maps of H1-H3;
#   H0 references (k2_natural_stats.py H0REF): H0 CAM, H0 Grad-CAM.
FAMILIES = [('primary', 'primary (K2 protocol): 6 N/S maps vs H0 collapsed',
             [('H1', 'N'), ('H1', 'S'), ('H2', 'N'), ('H2', 'S'), ('H3', 'N'), ('H3', 'S')]),
            ('natural', 'post hoc natural family: 7 natural maps of H1-H3 vs H0 collapsed',
             [('H1', 'cam'), ('H1', 'gradcam'), ('H2', 'attn'), ('H2', 'attn_ev'), ('H2', 'gradcam'),
              ('H3', 'gradcam'), ('H3', 'cam')]),
            ('h0_refs', 'post hoc H0 references vs H0 collapsed', [('H0', 'cam'), ('H0', 'gradcam')])]


def rank196(m):
    from scipy.stats import rankdata
    return rankdata(m, method='average') / m.shape[0]


def work(task):
    """task = (M (nimg, nmap, 196) float64, gbs list) -> curves (nimg, 2, nmap, 100) float32."""
    import cv2
    cv2.setNumThreads(1)
    M, gbs = task
    out = np.zeros((M.shape[0], 2, M.shape[1], NTHR), np.float32)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            for v, mp in enumerate((M[i, j], rank196(M[i, j]))):
                out[i, v, j] = W.box_curve(W.upsample(mp, 14, 14, EVAL_RES), gbs[i], THR)
    return out


def main():
    t0 = time.time()
    from PIL import Image
    man = W.cub_manifest()
    recs = man['ev']
    n = len(recs)
    gbs = []
    for r in recs:                                   # box rescaling exactly as k2_eval / k2_natural
        w0, h0 = Image.open(r['path']).size
        sx, sy = EVAL_RES / w0, EVAL_RES / h0
        gbs.append([[b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy] for b in r['boxes']])
    cols, maps, stored_curve, stored_mba = [], [], [], []
    for s in L.SEEDS:
        for h, m, src in MAPS:
            if src == 'k2':
                z = np.load(os.path.join(L.OUT, 'per_image', '%s.npz' % L.tag(h, s)))
                js = json.load(open(os.path.join(L.OUT, 'eval_%s.json' % L.tag(h, s))))
            else:
                z = np.load(os.path.join(POUT, 'per_image', '%s_natural.npz' % L.tag(h, s)))
                js = json.load(open(os.path.join(POUT, 'natural_%s.json' % L.tag(h, s))))
            cols.append((h, m, s)); maps.append(z['map_' + m]); stored_curve.append(z['curve_' + m])
            stored_mba.append(js['per_map'][m]['MaxBoxAccV2'])
    M = np.stack(maps, 1)                            # (n, 30, 196)
    assert M.shape == (n, len(cols), 196)
    print('loaded %d images x %d maps  [%.0fs]' % (n, len(cols), time.time() - t0), flush=True)
    CH = 16
    tasks = [(M[a:a + CH], gbs[a:a + CH]) for a in range(0, n, CH)]
    with Pool(30) as P:
        res = P.map(work, tasks, chunksize=1)
    CV = np.concatenate(res, 0)                      # (n, 2, n_maps x 3 seeds, 100)
    print('curves done [%.0fs]' % (time.time() - t0), flush=True)
    np.savez_compressed(os.path.join(POUT, 'per_image', 'k2_rank_curves.npz'), curves=CV,
                        cols=np.array(['%s_%s_s%d' % c for c in cols]), versions=np.array(['raw', 'rank']))

    # ---- verification on raw maps
    ver = []
    for j, (h, m, s) in enumerate(cols):
        v, _ = W.maxboxacc_v2(CV[:, 0, j], DELTAS)
        ver.append(dict(map='%s_%s' % (h, m), seed=s, MaxBoxAccV2_recomputed=v,
                        MaxBoxAccV2_stored=stored_mba[j], abs_diff=abs(v - stored_mba[j]),
                        curve_max_abs_diff=float(np.abs(CV[:, 0, j] - stored_curve[j]).max())))
    ver_max = max(x['abs_diff'] for x in ver)
    ver_curve = max(x['curve_max_abs_diff'] for x in ver)
    print('verification: max |MBA raw - stored| %.3e, max curve diff %.3e' % (ver_max, ver_curve))
    assert ver_max < 1e-9, ver_max

    # ---- MaxBoxAccV2 per seed and bootstrap (k2_stats convention)
    ci_ = {c: j for j, c in enumerate(cols)}
    VER = ['raw', 'rank']
    IND = {}
    for vi, vn in enumerate(VER):
        for j, c in enumerate(cols):
            IND[vn, c] = np.concatenate([(CV[:, vi, j] >= d) for d in DELTAS], 1).astype(np.float64)

    def mba_w(vn, h, m, s, w):
        x = (w @ IND[vn, (h, m, s)]) / w.sum()
        return float(x.reshape(len(DELTAS), -1).max(1).mean())

    rs = np.random.RandomState(W.PROTOCOL['seed'])
    IDX = rs.randint(0, n, (NB, n))
    WTS = np.stack([np.bincount(k, minlength=n) for k in IDX]).astype(np.float64)
    one = np.ones(n)
    chk = 0.0
    for b in range(3):
        for (h, m, _) in MAPS[:3]:
            for vi, vn in enumerate(VER):
                j = ci_[(h, m, 0)]
                chk = max(chk, abs(mba_w(vn, h, m, 0, WTS[b]) -
                                   W.maxboxacc_v2(CV[:, vi, j], DELTAS, IDX[b])[0]))
    assert chk < 1e-12, chk

    def contrast(vn, A, B=('H0', 'collapsed')):
        f = lambda w: np.mean([mba_w(vn, *A, s, w) - mba_w(vn, *B, s, w) for s in L.SEEDS])  # noqa: E731
        d0 = f(one)
        bs = np.array([f(WTS[b]) for b in range(NB)])
        p = 2.0 * min((bs <= 0).mean(), (bs >= 0).mean())
        return dict(a='%s_%s' % A, b='%s_%s' % B, version=vn, diff=float(d0),
                    ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                    p=float(min(1.0, max(p, 1.0 / NB))),
                    diff_per_seed=[mba_w(vn, *A, s, one) - mba_w(vn, *B, s, one) for s in L.SEEDS])

    # reproduce the stored primary contrasts (raw) of k2_contrasts.json
    kc = json.load(open(os.path.join(L.OUT, 'k2_contrasts.json')))
    prim_err = 0.0
    out = dict(post_hoc=True, n_test=n, n_boot=NB, seeds=L.SEEDS,
               verification=dict(max_abs_MaxBoxAccV2_diff=ver_max, max_abs_curve_diff=ver_curve,
                                 per_map=ver, bootstrap_vs_wlib_max_abs=chk),
               mba={}, contrasts={})
    for vn in VER:
        out['mba'][vn] = {}
        for h, m, _ in MAPS:
            ps = [mba_w(vn, h, m, s, one) for s in L.SEEDS]
            out['mba'][vn]['%s_%s' % (h, m)] = dict(per_seed=ps, mean=float(np.mean(ps)))
        out['contrasts'][vn] = []
        for fk, fname, mem in FAMILIES:
            rows = [contrast(vn, A) for A in mem]
            for r, q in zip(rows, W.holm([r['p'] for r in rows])):
                r['p_holm'] = q
                r['family'] = fk
                r['verdict'] = 'no difference' if q >= ALPHA else ('better' if r['diff'] > 0 else 'worse')
            out['contrasts'][vn] += rows
    # reproduce every stored raw contrast vs H0 collapsed (k2_contrasts primary; k2_natural
    # natural family and H0 references family): difference, interval, Holm P
    kn = json.load(open(os.path.join(POUT, 'k2_natural.json')))
    stored = {('primary', q['a']): q for q in kc['primary']}
    stored.update({('natural', q['a']): q for q in kn['natural_vs_collapsed_MaxBoxAccV2']})
    stored.update({('h0_refs', q['a']): q for q in kn['secondary']['H0_refs_vs_collapsed_MaxBoxAccV2']})
    n_rep, p_same = 0, True
    for r in out['contrasts']['raw']:
        q = stored[(r['family'], r['a'])]
        assert q['b'] == 'H0_collapsed'
        prim_err = max(prim_err, abs(q['diff'] - r['diff']),
                       abs(q['ci'][0] - r['ci'][0]), abs(q['ci'][1] - r['ci'][1]),
                       abs(q['p_holm'] - r['p_holm']))
        p_same &= q['p_holm'] == r['p_holm']
        n_rep += 1
    out['verification']['stored_contrast_reproduction_max_abs'] = prim_err
    out['verification']['stored_contrasts_reproduced'] = n_rep
    out['verification']['stored_holm_p_identical'] = bool(p_same)
    print('stored raw contrasts reproduced (%d): max abs err %.3e, Holm P identical %s'
          % (n_rep, prim_err, p_same))
    assert prim_err < 1e-9, prim_err
    rr = {r['a']: r for r in out['contrasts']['raw']}
    for r in out['contrasts']['rank']:
        q = rr[r['a']]
        r['sign_change_vs_raw'] = bool(np.sign(r['diff']) != np.sign(q['diff']))
        r['verdict_change_vs_raw'] = r['verdict'] != q['verdict']
    L.save_json(os.path.join(POUT, 'k2_rank.json'), out)
    write_md(out)
    print('done [%.0fs]' % (time.time() - t0))


def write_md(o):
    V = o['verification']
    T = ['# K2 post hoc: rank-transform sensitivity of MaxBoxAccV2', '',
         '**Post hoc sensitivity analysis, run after K2 and the natural-map analysis were complete; '
         'not part of the K2 protocol. analyses/end_to_end_cub/RESULTS.md and analyses/end_to_end_cub/post/RESULTS_natural.md are unchanged.**', '',
         'Each stored 14 x 14 map (ground-truth class, %d CUB-200-2011 test images, 3 seeds) is scored '
         'twice with the unchanged K2 pipeline (wlib.upsample to 224 bilinear + min-max, wlib.box_curve '
         'at 100 thresholds, wlib.maxboxacc_v2 at 0.3/0.5/0.7, boxes rescaled as in k2_eval.py): **raw** = '
         'the stored map; **rank** = the map replaced by its within-image ranks (average ranks for ties) '
         'divided by 196. Differences are A minus H0 collapsed, seed-matched, mean of 3 seeds, raw against '
         'raw and rank against rank; 95%% percentile bootstrap over test images with the 1000 image '
         'resamples of k2_stats.py; P = 2 x smaller tail share, floored at 1/1000; Holm within the families '
         'of the raw analyses: primary (6 N/S maps of H1-H3, analyses/end_to_end_cub/RESULTS.md), post hoc natural family '
         '(7 natural maps of H1-H3, RESULTS_natural.md), H0 references (H0 CAM, H0 Grad-CAM; '
         'RESULTS_natural.md). Grad-CAM maps with no positive cell are constant after ranking and, as in '
         'the raw analysis, become a zero map after min-max.' % o['n_test'], '',
         '## Verification', '',
         'Raw maps through this script reproduce the stored MaxBoxAccV2 of all %d map x seed cells with '
         'largest absolute difference %.1e and the stored per-image curves with largest difference %.1e. '
         'Weighted-indicator bootstrap vs wlib.maxboxacc_v2 on resampled indices: %.1e. The %d stored raw '
         'contrasts against H0 collapsed (analyses/end_to_end_cub/out/k2_contrasts.json primary; analyses/end_to_end_cub/post/out/k2_natural.json '
         'natural and H0-reference families; difference, interval and Holm P) are reproduced to %.1e '
         '(Holm P identical: %s).'
         % (len(V['per_map']), V['max_abs_MaxBoxAccV2_diff'], V['max_abs_curve_diff'],
            V['bootstrap_vs_wlib_max_abs'], V['stored_contrasts_reproduced'],
            V['stored_contrast_reproduction_max_abs'], V['stored_holm_p_identical']), '',
         '## MaxBoxAccV2, raw and rank', '',
         '| map | raw, mean | raw, per seed | rank, mean | rank, per seed | rank minus raw |',
         '|---|---|---|---|---|---|']
    for h, m, _ in MAPS:
        k = '%s_%s' % (h, m); a, b = o['mba']['raw'][k], o['mba']['rank'][k]
        T.append('| %s | %.4f | %s | %.4f | %s | %+.4f |'
                 % (LBL[h, m], a['mean'], ', '.join('%.4f' % x for x in a['per_seed']),
                    b['mean'], ', '.join('%.4f' % x for x in b['per_seed']), b['mean'] - a['mean']))
    rr = {r['a']: r for r in o['contrasts']['raw']}
    FN = {fk: fname for fk, fname, _ in FAMILIES}
    T += ['', '## Difference from H0 collapsed: raw vs rank', '',
          'Raw and rank columns are each against H0 collapsed scored the same way (raw %.4f, rank %.4f).'
          % (o['mba']['raw']['H0_collapsed']['mean'], o['mba']['rank']['H0_collapsed']['mean']), '',
          '| A | raw difference [95% interval] | raw P (Holm) | rank difference [95% interval] | '
          'rank, per seed | rank P (Holm) | sign change | verdict change |',
          '|---|---|---|---|---|---|---|---|']
    cur = None
    for r in o['contrasts']['rank']:
        if r['family'] != cur:
            cur = r['family']
            T.append('| *%s* | | | | | | | |' % FN[cur])
        q = rr[r['a']]
        h, m = r['a'].split('_', 1)
        T.append('| %s | %+.4f [%+.4f, %+.4f] | %.3g (%s) | %+.4f [%+.4f, %+.4f] | %s | %.3g (%s) | %s | %s |'
                 % (LBL[h, m], q['diff'], q['ci'][0], q['ci'][1], q['p_holm'], q['verdict'],
                    r['diff'], r['ci'][0], r['ci'][1],
                    ', '.join('%+.4f' % x for x in r['diff_per_seed']), r['p_holm'], r['verdict'],
                    'yes' if r['sign_change_vs_raw'] else 'no',
                    'yes' if r['verdict_change_vs_raw'] else 'no'))
    for vn, title in (('raw', 'raw maps'), ('rank', 'rank-transformed maps')):
        T += ['', '## Difference from H0 collapsed, %s (full rows)' % title, '',
              '| family | A | difference | 95% interval | per seed | P (Holm) | verdict |',
              '|---|---|---|---|---|---|---|']
        for r in o['contrasts'][vn]:
            h, m = r['a'].split('_', 1)
            T.append('| %s | %s | %+.4f | [%+.4f, %+.4f] | %s | %.3g | %s |'
                     % (r['family'], LBL[h, m], r['diff'], r['ci'][0], r['ci'][1],
                        ', '.join('%+.4f' % x for x in r['diff_per_seed']), r['p_holm'], r['verdict']))
    open(os.path.join(POST, 'RESULTS_rank.md'), 'w', encoding='utf-8').write('\n'.join(T) + '\n')


if __name__ == '__main__':
    main()
