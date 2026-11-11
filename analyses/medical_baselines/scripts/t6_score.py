"""C2 tasks 2-5, 8 -- score every arm's per-item maps and build the tables.

Inputs: out/maps_*.npz (t2_rsna.py, t5_nih.py) and the archived JSONs (read only).
Output: out/t6_score.json and printed tables.
Scoring is the archive's own: mlib.pointing_hit / pixel_ap / iou_at / dice_at,
m4's 2-fold cross-tuned IoU (QGRID, RandomState(0)), m5's cross-tuned Dice.
"""
import os
import sys
import csv
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import medical_baselines_lib as L
from medical_baselines_lib import mlib
import m4_stats as m4

G = 7
ARCH = {'nih_m4': 'm4_stats.json', 'nih_m10': 'm10_condB.json', 'nih_m6': 'm6_own.json',
        'rsna_xrv': 'm11_rsna.json', 'isic': 'm5_isic.json', 'rsna_disjoint': None}
NPZ = {'nih_m4': 'maps_nih_m4.npz', 'nih_m10': 'maps_nih_m10.npz', 'nih_m6': 'maps_nih_m6.npz',
       'rsna_xrv': 'maps_rsna_xrv.npz', 'rsna_disjoint': 'maps_rsna_disjoint.npz',
       'isic': 'maps_isic.npz'}
REPLICATE = {'nih_m10', 'nih_m6', 'isic'}          # GPU re-training, not bit-exact


# ------------------------------------------------------------- ground truth
def rsna_boxes():
    rows = list(csv.DictReader(open(os.path.join(L.ROOT, 'data', 'rsna', 'stage_2_train_labels.csv'))))
    b = {}
    for r in rows:
        if r['Target'] == '1' and r['x']:
            b.setdefault(r['patientId'], []).append(
                (float(r['x']), float(r['y']), float(r['width']), float(r['height'])))
    return b


def gts(arm, meta):
    if arm.startswith('nih'):
        bx = mlib.nih_boxes()
        return [m4.gt_mask(bx[n], f) for n, f in meta['items']], [f for n, f in meta['items']]
    if arm.startswith('rsna'):
        bx = rsna_boxes()
        return [L.box_mask_224(bx[n]) for n in meta['items']], ['Pneumonia'] * len(meta['items'])
    import m5_isic as m5
    return [m5.gt_of(n) for n, c in meta['items']], [str(c) for n, c in meta['items']]


# ------------------------------------------------------------- average box
def plateau_point(M):
    """argmax of a full-resolution map; on a plateau of equal maxima take the
    plateau pixel nearest the plateau's centroid (np.argmax would take its
    top-left corner)."""
    mx = M.max()
    ys, xs = np.where(M >= mx - 1e-12)
    cy, cx = ys.mean(), xs.mean()
    k = np.argmin((ys - cy) ** 2 + (xs - cx) ** 2)
    return ys[k], xs[k]


def avgbox_maps(arm, meta, gt, fnd):
    """returns (full-res hit vector, 7x7 map array) for the average-box baseline.
    NIH: leave-one-patient-out within finding over the 880 boxed test images (all
    NIH boxes are in the test split), plus a 5-fold patient-level cross-fit.
    RSNA: mean box mask of the stage_2_train positives that are NOT among the
    1,385 evaluation studies."""
    n = len(gt)
    hit_full = np.zeros(n)
    hit_cf = np.full(n, np.nan)
    maps7 = np.zeros((n, G * G))
    pool = lambda M: M.reshape(G, 32, G, 32).mean((1, 3)).reshape(-1)
    if arm.startswith('nih'):
        imgs = [it[0] for it in meta['items']]
        fnd = np.array(fnd)
        GT = np.stack(gt).astype(np.float32)
        pid = np.array([i.split('_')[0] for i in imgs])
        uim = sorted(set(pid))
        fo = dict(zip(uim, np.random.RandomState(0).randint(0, 5, len(uim))))
        fold = np.array([fo[i] for i in pid])
        for i in range(n):
            same = (fnd == fnd[i]) & (pid != pid[i])
            M = GT[same].mean(0)
            y, x = plateau_point(M)
            hit_full[i] = float(gt[i][y, x] > 0)
            maps7[i] = pool(M)
            cf = (fnd == fnd[i]) & (fold != fold[i])
            M2 = GT[cf].mean(0)
            y, x = plateau_point(M2)
            hit_cf[i] = float(gt[i][y, x] > 0)
        extra = dict(avgbox_cf5_hit=hit_cf)
    else:
        bx = rsna_boxes()
        ev = set(meta['items'])
        other = [p for p in bx if p not in ev]
        M = np.mean([L.box_mask_224(bx[p]).astype(np.float64) for p in other], 0)
        y, x = plateau_point(M)
        for i in range(n):
            hit_full[i] = float(gt[i][y, x] > 0)
            maps7[i] = pool(M)
        extra = dict(n_reference_patients=len(other), point=[int(y), int(x)])
    return hit_full, maps7, extra


# ------------------------------------------------------------- scoring
def score_arm(arm):
    M, meta = L.load_maps(NPZ[arm])
    gt, fnd = gts(arm, meta)
    keep = [i for i, g in enumerate(gt) if 0 < g.sum() < g.size]
    hb, m7, ab_extra = (avgbox_maps(arm, meta, gt, fnd) if not arm.startswith('isic')
                        else (None, None, {}))
    if m7 is not None:
        M['avgbox7'] = m7
    names = sorted(M)
    metrics = ['hit', 'ap'] + (['iou'] if arm == 'nih_m4' else []) + (['dice'] if arm == 'isic' else [])
    S = {mn: {mt: np.full(len(gt), np.nan) for mt in metrics} for mn in names}
    Q = m4.QGRID if arm == 'nih_m4' else [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60]
    sweep = {mn: {q: np.full(len(gt), np.nan) for q in Q} for mn in names} if len(metrics) > 2 else None
    for i in keep:
        g = gt[i]
        for mn in names:
            mp = M[mn][i]
            S[mn]['hit'][i] = mlib.pointing_hit(mp, G, g, 224)
            S[mn]['ap'][i] = mlib.pixel_ap(mp, G, g, 224)
            if sweep is not None:
                f = mlib.iou_at if arm == 'nih_m4' else mlib.dice_at
                for q in Q:
                    sweep[mn][q][i] = f(mp, G, g, 224, q)
    kept = np.array(keep)
    for mn in names:
        for mt in metrics[:2]:
            S[mn][mt] = S[mn][mt][kept]
    if sweep is not None:
        mt = metrics[2]
        nk = len(kept)
        fold = np.random.RandomState(0).randint(0, 2, nk)
        for mn in names:
            v = np.full(nk, np.nan)
            for f in (0, 1):
                tr, te = fold != f, fold == f
                best, bq = -1, Q[0]
                for q in Q:
                    a = sweep[mn][q][kept][tr]
                    if np.nanmean(a) > best:
                        best, bq = np.nanmean(a), q
                v[te] = sweep[mn][bq][kept][te]
            S[mn][mt] = v
    if hb is not None:
        S['avgbox_full'] = {'hit': hb[kept]}
        if 'avgbox_cf5_hit' in ab_extra:
            S['avgbox_cf5_full'] = {'hit': ab_extra.pop('avgbox_cf5_hit')[kept]}
    items = [meta['items'][i] for i in keep]
    return S, metrics, items, [fnd[i] for i in keep], meta, ab_extra, [gt[i] for i in keep]


def summary(S):
    return {mn: {mt: float(np.nanmean(v)) for mt, v in d.items()} for mn, d in S.items()}


# ------------------------------------------------------------- main
def main():
    res = {}
    arms = [a for a in os.environ.get('ARMS', ','.join(NPZ)).split(',') if a]
    for arm in arms:
        if not os.path.exists(os.path.join(L.OUT, NPZ[arm])):
            print('missing', arm)
            continue
        S, metrics, items, fnd, meta, abx, gtk = score_arm(arm)
        A = L.load_json(ARCH[arm]) if ARCH[arm] else None
        r = dict(n=len(items), metrics=metrics, summary=summary(S), meta_check={
            k: v for k, v in meta.items() if k not in ('items', 'fold_of_item', 'banks')})
        # 0. agreement with the archive
        if A is not None:
            dev_ = {}
            for mn, d in A['summary'].items():
                if mn in S:
                    for mt in d:
                        if mt in S[mn]:
                            dev_['%s/%s' % (mn, mt)] = r['summary'][mn][mt] - d[mt]['mean']
            r['max_abs_dev_from_archive'] = max(abs(v) for v in dev_.values())
            r['dev_from_archive'] = dev_
        # 1. pre-specified grid H1 (N vs collapsed), H2 (N vs S), and the
        #    matched halves (N at T=-8 and S at T=+8, each minus collapsed)
        grid = {}
        for unit in ('1x1', '3x3'):
            C = 'collapsed_%s' % unit
            for mt in metrics:
                for temp in L.OFF:
                    grid['H1_%s_%s_%s' % (temp, unit, mt)] = dict(
                        A='N_%s_%s' % (temp, unit), B=C,
                        **L.cmp(S['N_%s_%s' % (temp, unit)][mt], S[C][mt], mt == 'hit'))
                    grid['H2_%s_%s_%s' % (temp, unit, mt)] = dict(
                        A='N_%s_%s' % (temp, unit), B='S_%s_%s' % (temp, unit),
                        **L.cmp(S['N_%s_%s' % (temp, unit)][mt], S['S_%s_%s' % (temp, unit)][mt], mt == 'hit'))
                    grid['Sminus_%s_%s_%s' % (temp, unit, mt)] = dict(
                        A='S_%s_%s' % (temp, unit), B=C,
                        **L.cmp(S['S_%s_%s' % (temp, unit)][mt], S[C][mt], mt == 'hit'))
        r['grid'] = grid
        # 2. comparators (hit) + avg box + McNemar against the strongest model map
        model_maps = [mn for mn in S if mn.startswith(('N_', 'S_', 'collapsed', 'gradcam', 'cam'))]
        best = max(model_maps, key=lambda mn: np.nanmean(S[mn]['hit']))
        comp = {}
        for mn in ['collapsed_1x1', 'cam', 'gradcam_gap', 'actnorm', 'center', 'random',
                   'avgbox_full', 'avgbox_cf5_full', 'avgbox7', best]:
            if mn in S:
                comp[mn] = mlib.boot_ci(S[mn]['hit'])
        r['comparators_hit'] = comp
        r['strongest_model_map'] = best
        tests = {}
        for ab in ('avgbox_full', 'avgbox_cf5_full', 'avgbox7'):
            if ab in S:
                for other in (best, 'collapsed_1x1', 'cam', 'gradcam_gap', 'actnorm'):
                    tests['%s_vs_%s' % (ab, other)] = L.cmp(S[ab]['hit'], S[other]['hit'], True)
        for a_, b_ in (('cam', 'collapsed_1x1'), ('cam', 'gradcam_gap'), ('collapsed_1x1', 'gradcam_gap'),
                       ('cam', 'actnorm'), ('collapsed_1x1', 'actnorm')):
            tests['%s_vs_%s' % (a_, b_)] = L.cmp(S[a_]['hit'], S[b_]['hit'], True)
        r['comparator_tests'] = tests
        # Grad-CAM degenerate (all-zero) maps: argmax falls on pixel (0,0)
        M, _ = L.load_maps(NPZ[arm])
        gc = M['gradcam_gap']
        r['gradcam_allzero_fraction'] = float(np.mean(np.all(gc <= 0, 1)))
        # per finding (NIH): hit of comparators
        if arm.startswith('nih'):
            fnd = np.array(fnd)
            r['per_finding_hit'] = {f: dict(n=int((fnd == f).sum()), **{
                mn: float(np.nanmean(np.asarray(S[mn]['hit'])[fnd == f]))
                for mn in ['collapsed_1x1', 'cam', 'gradcam_gap', 'actnorm', 'center', 'avgbox_full',
                           'N_lse-8.0_1x1', 'S_lse8.0_1x1', 'N_lse8.0_1x1']})
                for f in sorted(set(fnd))}
        r['avgbox_extra'] = abx
        # 3. subset explained by a model that did not train on it (xrv split)
        if arm in ('rsna_xrv', 'nih_m4'):
            fn = 'rsna_local_xrv_heldout.json' if arm == 'rsna_xrv' else 'nih_boxed_xrv_heldout.json'
            held = set(json.load(open(os.path.join(L.OUT, fn))))
            ids = [it if isinstance(it, str) else it[0] for it in items]
            hm = np.array([i in held for i in ids])
            sub = dict(n=int(hm.sum()))
            for key in ('H1_lse8.0_1x1_hit', 'H1_lse-8.0_1x1_hit', 'Sminus_lse8.0_1x1_hit'):
                g_ = grid[key]
                sub[key] = L.cmp(np.asarray(S[g_['A']]['hit'])[hm], np.asarray(S[g_['B']]['hit'])[hm], True)
            sub['hit'] = {mn: float(np.nanmean(np.asarray(S[mn]['hit'])[hm]))
                          for mn in ['collapsed_1x1', 'cam', 'gradcam_gap', 'actnorm', 'center',
                                     'N_lse8.0_1x1', 'N_lse-8.0_1x1', 'S_lse8.0_1x1', 'avgbox_full']}
            sub['hit_trainset'] = {mn: float(np.nanmean(np.asarray(S[mn]['hit'])[~hm]))
                                   for mn in sub['hit']}
            sub['collapsed_held_vs_train_fisher_p'] = float(__import__('scipy.stats', fromlist=['x']).fisher_exact(
                [[np.sum(np.asarray(S['collapsed_1x1']['hit'])[hm]), np.sum(1 - np.asarray(S['collapsed_1x1']['hit'])[hm])],
                 [np.sum(np.asarray(S['collapsed_1x1']['hit'])[~hm]), np.sum(1 - np.asarray(S['collapsed_1x1']['hit'])[~hm])]])[1])
            r['xrv_heldout_subset'] = sub
        # 4. resolution ceiling (any 7x7 map, this pointing protocol)
        reach = reachable_pixels()
        ceil_ = np.array([float(any(g[y, x] for y, x in reach)) for g in gtk])
        r['ceiling_any7x7_hit'] = float(ceil_.mean())
        if arm.startswith('nih'):
            r['ceiling_by_finding'] = {f: float(ceil_[np.array(fnd) == f].mean()) for f in sorted(set(fnd))}
        res[arm] = r
        print('== %s n=%d best=%s ceiling=%.3f' % (arm, len(items), best, r['ceiling_any7x7_hit']), flush=True)
        if 'max_abs_dev_from_archive' in r:
            print('   max |dev| from archive %.4g' % r['max_abs_dev_from_archive'], flush=True)
    L.save(os.environ.get('OUTNAME', 't6_score.json'), res)


_REACH = None


def reachable_pixels():
    """every pixel at which mlib.pointing_hit's argmax can land for a 7x7 map."""
    global _REACH
    if _REACH is None:
        rs = np.random.RandomState(0)
        pts = set()
        for k in range(49):
            e = np.zeros(49)
            e[k] = 1
            for base in (e, e + 1e-3 * rs.rand(49)):
                u = mlib.upsample(base, G, 224)
                pts.add(np.unravel_index(int(np.argmax(u)), u.shape))
        for _ in range(20000):
            u = mlib.upsample(rs.randn(49), G, 224)
            pts.add(np.unravel_index(int(np.argmax(u)), u.shape))
        _REACH = sorted(pts)
    return _REACH


if __name__ == '__main__':
    main()
