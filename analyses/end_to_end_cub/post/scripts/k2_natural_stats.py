"""POST HOC contrasts for the natural maps, with the bootstrap of analyses/end_to_end_cub/scripts/k2_stats.py.

    python k2_natural_stats.py

Reads (never writes) analyses/end_to_end_cub/out/per_image/H0_s*.npz, analyses/end_to_end_cub/out/eval_*.json and analyses/end_to_end_cub/out/k2_contrasts.json;
reads analyses/end_to_end_cub/post/out/per_image/*_natural.npz; writes analyses/end_to_end_cub/post/out/k2_natural.json and
analyses/end_to_end_cub/post/RESULTS_natural.md.

Contrast = mean over seeds s of [metric(map_{H,s}) - metric(H0 collapsed_s)], seed-matched;
95% percentile bootstrap over test images, 1,000 resamples drawn exactly as in k2_stats
(RandomState(wlib.PROTOCOL['seed']).randint(0, n, (1000, n)), one resample shared by every seed and
map); P = 2 min(P*(b <= 0), P*(b >= 0)) clipped to [1/1000, 1]; Holm (wlib.holm) at 0.05.
Post hoc family (MaxBoxAccV2): the 7 natural maps of the separable heads,
  H1 CAM, H1 Grad-CAM, H2 attention, H2 attention x evidence, H2 Grad-CAM, H3 Grad-CAM, H3 CAM.
Same 7 on strict pointing and on best IoU (separate Holm families), and against H0 CAM.
H0 references (recomputed H0 CAM, H0 Grad-CAM) vs H0 collapsed: own family of 2 per metric.
The code is a copy of the k2_stats contrast (its functions are local to its main()); it is checked
by reproducing a stored primary contrast of k2_contrasts.json.
"""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
POST = os.path.dirname(HERE)
K2 = os.path.dirname(POST)
sys.path.insert(0, os.path.join(K2, 'scripts'))
import k2lib as L   # noqa: E402
W = L.W

NB = 1000
DELTAS = tuple(W.PROTOCOL['deltas'])
ALPHA = 0.05
POUT = os.path.join(POST, 'out')
NAT = {'H0': ['cam', 'gradcam'], 'H1': ['cam', 'gradcam'],
       'H2': ['attn', 'attn_ev', 'gradcam'], 'H3': ['gradcam', 'cam']}
FAM = [('H1', 'cam'), ('H1', 'gradcam'), ('H2', 'attn'), ('H2', 'attn_ev'), ('H2', 'gradcam'),
       ('H3', 'gradcam'), ('H3', 'cam')]
H0REF = [('H0', 'cam'), ('H0', 'gradcam')]
LBL = {('H0', 'collapsed'): 'H0 collapsed (stored)', ('H0', 'cam_stored'): 'H0 CAM (stored)',
       ('H0', 'cam'): 'H0 CAM (recomputed)', ('H0', 'gradcam'): 'H0 Grad-CAM',
       ('H1', 'cam'): 'H1 CAM (per-cell class score)', ('H1', 'gradcam'): 'H1 Grad-CAM',
       ('H2', 'attn'): 'H2 attention weights', ('H2', 'attn_ev'): 'H2 attention x class evidence',
       ('H2', 'gradcam'): 'H2 Grad-CAM', ('H3', 'gradcam'): 'H3 Grad-CAM',
       ('H3', 'cam'): 'H3 CAM (LayerNorm ignored)',
       ('H1', 'N'): 'H1 necessity', ('H1', 'S'): 'H1 sufficiency', ('H2', 'N'): 'H2 necessity',
       ('H2', 'S'): 'H2 sufficiency', ('H3', 'N'): 'H3 necessity', ('H3', 'S'): 'H3 sufficiency'}
key = lambda h, m: '%s_%s' % (h, m)   # noqa: E731


def main():
    D = {}; E = {}; NJ = {}
    for s in L.SEEDS:
        st = dict(np.load(os.path.join(L.OUT, 'per_image', 'H0_s%d.npz' % s)))
        D['H0', 'collapsed', s] = {f: st[f + '_collapsed'] for f in ('curve', 'pt_strict', 'best_iou')}
        D['H0', 'cam_stored', s] = {f: st[f + '_cam'] for f in ('curve', 'pt_strict', 'best_iou')}
        for h in ('H1', 'H2', 'H3'):
            st = dict(np.load(os.path.join(L.OUT, 'per_image', '%s_s%d.npz' % (h, s))))
            for m in ('N', 'S'):
                D[h, m, s] = {f: st[f + '_' + m] for f in ('curve', 'pt_strict', 'best_iou')}
        for h in L.HEADS:
            E[h, s] = json.load(open(os.path.join(L.OUT, 'eval_%s.json' % L.tag(h, s))))
            NJ[h, s] = json.load(open(os.path.join(POUT, 'natural_%s.json' % L.tag(h, s))))
            nt = dict(np.load(os.path.join(POUT, 'per_image', '%s_natural.npz' % L.tag(h, s))))
            for m in NAT[h]:
                D[h, m, s] = {f: nt[f + '_' + m] for f in ('curve', 'pt_strict', 'best_iou', 'map')}
    n = len(D['H0', 'collapsed', 0]['curve'])
    assert all(len(v['curve']) == n for v in D.values())

    # ---- sanity: recomputed H0 CAM vs the stored evaluation
    san = []
    for s in L.SEEDS:
        a, b = D['H0', 'cam', s], D['H0', 'cam_stored', s]
        st = np.load(os.path.join(L.OUT, 'per_image', 'H0_s%d.npz' % s))
        ej = E['H0', s]['per_map']['cam']; nj = NJ['H0', s]['per_map']['cam']
        san.append(dict(seed=s,
                        MaxBoxAccV2_recomputed=nj['MaxBoxAccV2'], MaxBoxAccV2_stored=ej['MaxBoxAccV2'],
                        abs_diff={k: abs(nj[k] - ej[k]) for k in
                                  ('MaxBoxAccV2', 'pointing_strict', 'pointing_tol15', 'best_iou')},
                        per_delta_abs_diff=max(abs(nj['per_delta'][d] - ej['per_delta'][d])
                                               for d in ej['per_delta']),
                        curve_max_abs_diff=float(np.abs(a['curve'] - b['curve']).max()),
                        map_max_abs_diff=float(np.abs(a['map'] - st['map_cam']).max()),
                        map_max_abs=float(np.abs(st['map_cam']).max()),
                        pointing_identical=bool((a['pt_strict'] == b['pt_strict']).all())))

    # ---- MaxBoxAccV2 through indicator matrices (k2_stats)
    IND = {}
    for (h, m, s), v in D.items():
        IND[h, m, s] = np.concatenate([(v['curve'] >= d) for d in DELTAS], 1).astype(np.float64)

    def mba_w(h, m, s, w):
        v = (w @ IND[h, m, s]) / w.sum()
        return float(v.reshape(len(DELTAS), -1).max(1).mean())

    rs = np.random.RandomState(W.PROTOCOL['seed'])
    IDX = rs.randint(0, n, (NB, n))
    WTS = np.stack([np.bincount(k, minlength=n) for k in IDX]).astype(np.float64)
    one = np.ones(n)
    chk = 0.0
    for b in range(3):
        for (h, m) in [('H0', 'collapsed')] + FAM[:3]:
            chk = max(chk, abs(mba_w(h, m, 0, WTS[b]) - W.maxboxacc_v2(D[h, m, 0]['curve'], DELTAS, IDX[b])[0]))
            chk = max(chk, abs(mba_w(h, m, 0, one) - W.maxboxacc_v2(D[h, m, 0]['curve'], DELTAS)[0]))
    assert chk < 1e-12, chk

    def per_img(h, m, s, metric):
        f = 'pt_strict' if metric == 'pointing_strict' else 'best_iou'
        return D[h, m, s][f].astype(np.float64)

    def contrast(A, B, metric):
        if metric == 'MaxBoxAccV2':
            f = lambda w: np.mean([mba_w(*A, s, w) - mba_w(*B, s, w) for s in L.SEEDS])  # noqa: E731
            d0 = f(one)
            bs = np.array([f(WTS[b]) for b in range(NB)])
            per_seed = [mba_w(*A, s, one) - mba_w(*B, s, one) for s in L.SEEDS]
        else:
            dimg = np.mean([per_img(*A, s, metric) - per_img(*B, s, metric) for s in L.SEEDS], 0)
            d0 = float(dimg.mean())
            bs = (WTS @ dimg) / n
            per_seed = [float((per_img(*A, s, metric) - per_img(*B, s, metric)).mean()) for s in L.SEEDS]
        p = 2.0 * min((bs <= 0).mean(), (bs >= 0).mean())
        return dict(a=key(*A), b=key(*B), a_label=LBL[A], b_label=LBL[B], metric=metric, diff=float(d0),
                    ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                    p=float(min(1.0, max(p, 1.0 / NB))), diff_per_seed=[float(x) for x in per_seed])

    def family(name, rows):
        adj = W.holm([r['p'] for r in rows])
        for r, q in zip(rows, adj):
            r['p_holm'] = q
            r['verdict'] = ('no difference' if q >= ALPHA else ('better' if r['diff'] > 0 else 'worse'))
            r['family'] = name
            r['ci_excludes_0'] = bool(r['ci'][0] > 0 or r['ci'][1] < 0)
        return rows

    # reproduce one stored primary contrast (H3 sufficiency vs H0 collapsed)
    stored = [r for r in json.load(open(os.path.join(L.OUT, 'k2_contrasts.json')))['primary']
              if r['a'] == 'H3_S'][0]
    rep = contrast(('H3', 'S'), ('H0', 'collapsed'), 'MaxBoxAccV2')
    repro = dict(contrast='H3_S - H0_collapsed, MaxBoxAccV2',
                 diff_abs_err=abs(rep['diff'] - stored['diff']),
                 ci_abs_err=max(abs(x - y) for x, y in zip(rep['ci'], stored['ci'])),
                 p_equal=rep['p'] == stored['p'])

    CO = ('H0', 'collapsed')
    out = dict(post_hoc=True, n_test=n, n_boot=NB, seeds=L.SEEDS, alpha=ALPHA,
               family_definition=[key(*x) for x in FAM], mba_wlib_check_max_abs=chk,
               sanity_H0_cam_recomputed_vs_stored=san, bootstrap_reproduces_k2_stats=repro)
    out['natural_vs_collapsed_MaxBoxAccV2'] = family(
        'post hoc: MaxBoxAccV2 vs H0 collapsed (7 natural maps)', [contrast(x, CO, 'MaxBoxAccV2') for x in FAM])
    sec = {}
    for met in ('pointing_strict', 'best_iou'):
        sec['vs_collapsed_' + met] = family('post hoc: %s vs H0 collapsed (7 natural maps)' % met,
                                            [contrast(x, CO, met) for x in FAM])
    for met in ('MaxBoxAccV2', 'pointing_strict', 'best_iou'):
        sec['vs_cam_' + met] = family('post hoc: %s vs H0 CAM (stored) (7 natural maps)' % met,
                                      [contrast(x, ('H0', 'cam_stored'), met) for x in FAM])
        sec['H0_refs_vs_collapsed_' + met] = family('post hoc: %s H0 references vs H0 collapsed' % met,
                                                    [contrast(x, CO, met) for x in H0REF])
    out['secondary'] = sec

    desc = {}
    for h in L.HEADS:
        for m in NAT[h]:
            v = {met: [NJ[h, s]['per_map'][m][met] for s in L.SEEDS]
                 for met in ('MaxBoxAccV2', 'pointing_strict', 'pointing_tol15', 'best_iou')}
            for met in list(v):
                v[met + '_mean'] = float(np.mean(v[met]))
            v['all_zero_maps'] = [NJ[h, s]['all_zero_maps'][m] for s in L.SEEDS]
            desc[key(h, m)] = v
    out['maps'] = desc
    out['checks'] = {h: [NJ[h, s]['checks'] for s in L.SEEDS] for h in L.HEADS}
    out['stored_reference'] = {k: E['H0', 0]['per_map'][k] for k in ('collapsed', 'cam')}
    out['stored_H0'] = {m: {met: float(np.mean([E['H0', s]['per_map'][m][met] for s in L.SEEDS]))
                            for met in ('MaxBoxAccV2', 'pointing_strict', 'pointing_tol15', 'best_iou')}
                        for m in ('collapsed', 'cam')}
    L.save_json(os.path.join(POUT, 'k2_natural.json'), out)
    write_md(out)
    print(json.dumps(dict(sanity=san, repro=repro), indent=1, default=W._jd))
    for r in out['natural_vs_collapsed_MaxBoxAccV2']:
        print('%-35s %+.4f [%+.4f, %+.4f] P_holm %.3g' % (r['a_label'], r['diff'], *r['ci'], r['p_holm']))


def fmt(r):
    return ('| %s | %s | %+.4f | [%+.4f, %+.4f] | %s | %.3g | %s |'
            % (r['a_label'], r['b_label'], r['diff'], r['ci'][0], r['ci'][1],
               ', '.join('%+.4f' % x for x in r['diff_per_seed']), r['p_holm'], r['verdict']))


def write_md(o):
    M = o['maps']
    L_ = ['# K2 post hoc: the natural map of each head against the H0 collapsed map', '',
          '**Post hoc analysis, run after K2 was complete; not part of the K2 protocol '
          '(protocols/K2_end_to_end_protocol.md). The pre-specified K2 results in analyses/end_to_end_cub/RESULTS.md are '
          'unchanged.**', '',
          'Same 12 checkpoints, same %d CUB-200-2011 test images, same preprocessing (whole image resized '
          'to 448 x 448, no crop), ground-truth class, box rescaling, map up-sampling and min-max scaling, '
          'MaxBoxAccV2 (0.3/0.5/0.7, 100 thresholds), strict pointing and best IoU as analyses/end_to_end_cub/scripts/k2_eval.py '
          '(imported, not re-implemented). Backbone in float32 (TF32 off); every map, including Grad-CAM '
          '(autograd through a float64 copy of the trained head), in float64. Differences are A minus B, '
          'seed-matched, mean over the 3 seeds; 95%% percentile bootstrap over test images (%d resamples, '
          'the same image resamples as k2_stats.py); P = 2 x smaller tail share, floored at 1/1000; Holm '
          'within each family at alpha = 0.05.' % (o['n_test'], o['n_boot']), '',
          '## Natural maps', '',
          '| head | map | definition (class c = ground truth, x_t = cell t of the 14 x 14 layer4 map) |',
          '|---|---|---|',
          '| H0 | CAM | <w_c, x_t> (recomputed; must equal the stored H0 CAM) |',
          '| H0 | Grad-CAM | ReLU(sum_k alpha_k x_tk), alpha_k = mean_t dz_c/dx_tk; for GAP + linear = ReLU(<w_c, x_t>)/196 |',
          '| H1 | CAM | s_t = <w_c, x_t> + b_c, the per-cell class score that log-sum-exp pools |',
          '| H1 | Grad-CAM | standard; alpha = mean_t softmax_t(s) w_c = w_c/196, so = ReLU(<w_c, x_t>)/196 |',
          '| H2 | attention | a_t = softmax_t(e_t), the gated-attention weights (class independent) |',
          '| H2 | attention x evidence | a_t <w_c, x_t>: the head pools then applies the linear layer, '
          'z_c = <w_c, sum_t a_t x_t> + b_c = sum_t a_t <w_c, x_t> + b_c; bias excluded |',
          '| H2 | Grad-CAM | standard (gradient through softmax attention and pooling) |',
          '| H3 | Grad-CAM | standard; = ReLU(<J^T w_c, x_t>)/196, J = LayerNorm Jacobian at the pooled vector |',
          '| H3 | CAM | <w_c, x_t>, LayerNorm ignored (reference) |', '',
          'Grad-CAM closed forms (H0, H1, H3) were checked against autograd on every image: largest relative '
          'difference %.1e. Largest |float64 head logit - float32 network logit| %.1e.'
          % (max(c['gradcam_vs_closed_form_rel'] for v in o['checks'].values() for c in v),
             max(c['head64_logit_vs_float32_abs'] for v in o['checks'].values() for c in v)), '',
          '## Sanity check: H0 CAM recomputed with this pipeline vs the stored K2 evaluation', '',
          '| seed | MaxBoxAccV2 recomputed | stored | abs. difference | largest per-image curve difference | largest map difference | strict pointing identical |',
          '|---|---|---|---|---|---|---|']
    for v in o['sanity_H0_cam_recomputed_vs_stored']:
        L_.append('| %d | %.10f | %.10f | %.1e | %.1e | %.1e (max |map| %.0f) | %s |'
                  % (v['seed'], v['MaxBoxAccV2_recomputed'], v['MaxBoxAccV2_stored'], v['abs_diff']['MaxBoxAccV2'],
                     v['curve_max_abs_diff'], v['map_max_abs_diff'], v['map_max_abs'], v['pointing_identical']))
    r = o['bootstrap_reproduces_k2_stats']
    L_ += ['', 'Bootstrap check: re-running the stored primary contrast %s with this code gives an absolute '
           'error of %.1e in the difference and %.1e in the interval; P identical: %s.'
           % (r['contrast'], r['diff_abs_err'], r['ci_abs_err'], r['p_equal']), '',
           '## Localisation of every natural map (mean of 3 seeds)', '',
           '| map | MaxBoxAccV2 | per seed | strict pointing | best IoU | pointing, 15 px | all-zero maps (per seed) |',
           '|---|---|---|---|---|---|---|']
    sh = o['stored_H0']
    for m, lb in (('collapsed', 'H0 collapsed (stored, K2)'), ('cam', 'H0 CAM (stored, K2)')):
        L_.append('| %s | %.4f | | %.4f | %.4f | %.4f | |' % (lb, sh[m]['MaxBoxAccV2'], sh[m]['pointing_strict'],
                                                         sh[m]['best_iou'], sh[m]['pointing_tol15']))
    for k, v in M.items():
        h, m = k.split('_', 1)
        L_.append('| %s | %.4f | %s | %.4f | %.4f | %.4f | %s |'
                  % (LBL[h, m], v['MaxBoxAccV2_mean'], ', '.join('%.4f' % x for x in v['MaxBoxAccV2']),
                     v['pointing_strict_mean'], v['best_iou_mean'], v['pointing_tol15_mean'],
                     ', '.join(str(x) for x in v['all_zero_maps'])))
    L_ += ['', 'All-zero maps (Grad-CAM with no positive cell) become a zero map after min-max; as in '
           'wlib.upsample they then cover the whole image at every threshold.']
    hdr = ['| A | B | difference | 95% interval | per seed | P (Holm) | verdict |', '|---|---|---|---|---|---|---|']
    L_ += ['', '## Post hoc family: MaxBoxAccV2 against H0 collapsed (7 natural maps of H1-H3)', ''] + hdr
    L_ += [fmt(x) for x in o['natural_vs_collapsed_MaxBoxAccV2']]
    S = o['secondary']
    for k, t in [('vs_collapsed_pointing_strict', 'Strict pointing against H0 collapsed'),
                 ('vs_collapsed_best_iou', 'Best IoU against H0 collapsed'),
                 ('vs_cam_MaxBoxAccV2', 'MaxBoxAccV2 against H0 CAM'),
                 ('vs_cam_pointing_strict', 'Strict pointing against H0 CAM'),
                 ('vs_cam_best_iou', 'Best IoU against H0 CAM'),
                 ('H0_refs_vs_collapsed_MaxBoxAccV2', 'H0 references against H0 collapsed, MaxBoxAccV2'),
                 ('H0_refs_vs_collapsed_pointing_strict', 'H0 references against H0 collapsed, strict pointing'),
                 ('H0_refs_vs_collapsed_best_iou', 'H0 references against H0 collapsed, best IoU')]:
        L_ += ['', '## %s' % t, ''] + hdr + [fmt(x) for x in S[k]]
    L_ += ['', 'Families: the 7 natural maps of H1-H3 form one Holm family per metric and per reference '
           '(H0 collapsed, H0 CAM); the two H0 references form their own family of 2 per metric. '
           'Every contrast is reported whatever its sign.', '',
           'Scripts: analyses/end_to_end_cub/post/scripts/k2_natural.py (maps and metrics, per model), k2_natural_driver.py, '
           'k2_natural_stats.py (this file and analyses/end_to_end_cub/post/out/k2_natural.json). Per-image arrays: '
           'analyses/end_to_end_cub/post/out/per_image/<H>_s<seed>_natural.npz.']
    open(os.path.join(POST, 'RESULTS_natural.md'), 'w', encoding='utf-8').write('\n'.join(L_) + '\n')


if __name__ == '__main__':
    main()
