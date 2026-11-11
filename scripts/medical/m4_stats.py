"""m4 -- score every map from m3 against the radiologist ground truth and run
the PRE-SPECIFIED comparison.

Metrics
  hit      pointing game: argmax of the bilinearly upsampled map inside the GT
  ap       threshold-free pixel average precision
  iou      IoU of the top-q fraction of the map, q cross-tuned 2-fold by image

Pre-specified decisive comparison (primary metric = hit, unit = 1x1, T = +8):
  H1  N-map  >  collapsed map
  H2  N-map  >  S-map
Paired: exact McNemar for the binary hit rate, Wilcoxon signed-rank + bootstrap
CI for the continuous metrics.
"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlib

SIZE = 224
ORIG = 1024.
QGRID = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]

PRIMARY = dict(metric='hit', unit='1x1', temp='lse8.0')


def gt_mask(boxes, finding):
    g = np.zeros((SIZE, SIZE), dtype=np.uint8)
    for (lab, x, y, w, h) in boxes:
        if lab != finding:
            continue
        s = SIZE / ORIG
        x0, y0 = int(round(x * s)), int(round(y * s))
        x1, y1 = int(round((x + w) * s)), int(round((y + h) * s))
        g[max(0, y0):min(SIZE, y1), max(0, x0):min(SIZE, x1)] = 1
    return g


def mcnemar(a, b):
    """exact McNemar on paired binary outcomes a, b."""
    from scipy.stats import binomtest
    a, b = np.asarray(a) > 0, np.asarray(b) > 0
    n01 = int((~a & b).sum())
    n10 = int((a & ~b).sum())
    n = n01 + n10
    p = float(binomtest(n10, n, 0.5).pvalue) if n else 1.0
    return dict(a_only=n10, b_only=n01, p_mcnemar=p)


def main():
    R = json.load(open(os.path.join(mlib.OUT, os.environ.get('INNAME', 'm3_maps.json'))))
    G = R['G']
    boxes = mlib.nih_boxes()
    items = R['items']
    mapnames = sorted(items[0]['maps'].keys())

    sc = {mn: dict(hit=[], ap=[], iou={q: [] for q in QGRID}) for mn in mapnames}
    excl = set(x for x in os.environ.get('EXCL', '').split(',') if x)
    keep = []
    for it in items:
        if it['finding'] in excl:
            continue
        g = gt_mask(boxes[it['image']], it['finding'])
        if g.sum() == 0 or g.sum() == g.size:
            continue
        keep.append(it)
        for mn in mapnames:
            mp = it['maps'][mn]
            sc[mn]['hit'].append(mlib.pointing_hit(mp, G, g, SIZE))
            sc[mn]['ap'].append(mlib.pixel_ap(mp, G, g, SIZE))
            for q in QGRID:
                sc[mn]['iou'][q].append(mlib.iou_at(mp, G, g, SIZE, q))
    n = len(keep)
    print('scored %d item-findings' % n, flush=True)
    findings = np.array([it['finding'] for it in keep])
    gtfrac = np.array([gt_mask(boxes[it['image']], it['finding']).mean()
                       for it in keep])

    # ---- 2-fold cross-tuned IoU -----------------------------------------
    rs = np.random.RandomState(0)
    fold = rs.randint(0, 2, n)
    iou_ct = {}
    for mn in mapnames:
        v = np.full(n, np.nan)
        for f in (0, 1):
            tr, te = fold != f, fold == f
            best, bq = -1, QGRID[0]
            for q in QGRID:
                a = np.asarray(sc[mn]['iou'][q], float)[tr]
                mval = np.nanmean(a)
                if mval > best:
                    best, bq = mval, q
            v[te] = np.asarray(sc[mn]['iou'][bq], float)[te]
        iou_ct[mn] = v

    summ = {}
    for mn in mapnames:
        summ[mn] = dict(
            hit=mlib.boot_ci(sc[mn]['hit']),
            ap=mlib.boot_ci(sc[mn]['ap']),
            iou=mlib.boot_ci(iou_ct[mn]),
            iou_by_q={str(q): float(np.nanmean(sc[mn]['iou'][q])) for q in QGRID})

    def cmp(A, B, metric):
        if metric == 'hit':
            a, b = np.asarray(sc[A]['hit']), np.asarray(sc[B]['hit'])
            d = mlib.paired(a, b)
            d.update(mcnemar(a, b))
            d['mean_a'], d['mean_b'] = float(np.nanmean(a)), float(np.nanmean(b))
        elif metric == 'ap':
            a, b = np.asarray(sc[A]['ap']), np.asarray(sc[B]['ap'])
            d = mlib.paired(a, b)
            d['mean_a'], d['mean_b'] = float(np.nanmean(a)), float(np.nanmean(b))
        else:
            a, b = iou_ct[A], iou_ct[B]
            d = mlib.paired(a, b)
            d['mean_a'], d['mean_b'] = float(np.nanmean(a)), float(np.nanmean(b))
        d['A'], d['B'], d['metric'] = A, B, metric
        return d

    pre = {}
    for unit in ('1x1', '3x3'):
        for temp in ('lse8.0', 'lse-8.0'):
            Nm = 'N_%s_%s' % (temp, unit)
            Sm = 'S_%s_%s' % (temp, unit)
            Cm = 'collapsed_%s' % unit
            for metric in ('hit', 'ap', 'iou'):
                pre['H1_%s_%s_%s' % (temp, unit, metric)] = cmp(Nm, Cm, metric)
                pre['H2_%s_%s_%s' % (temp, unit, metric)] = cmp(Nm, Sm, metric)

    # controls the paper needs anyway
    extra = {}
    for metric in ('hit', 'ap', 'iou'):
        extra['collapsed_vs_center_%s' % metric] = cmp('collapsed_1x1', 'center', metric)
        extra['gradcam_gap_vs_center_%s' % metric] = cmp('gradcam_gap', 'center', metric)
        extra['S_lse8_vs_center_%s' % metric] = cmp('S_lse8.0_1x1', 'center', metric)
        extra['N_lse8_vs_center_%s' % metric] = cmp('N_lse8.0_1x1', 'center', metric)
        extra['N_lse8_vs_random_%s' % metric] = cmp('N_lse8.0_1x1', 'random', metric)
        extra['collapsed_vs_gradcam_gap_%s' % metric] = cmp('collapsed_1x1', 'gradcam_gap', metric)

    per_find = {}
    for f in sorted(set(findings)):
        m = findings == f
        per_find[f] = dict(n=int(m.sum()), gt_frac=float(gtfrac[m].mean()),
                           **{mn: float(np.nanmean(np.asarray(sc[mn]['hit'])[m]))
                              for mn in mapnames})

    # in-line control: at the degenerate point the two maps are one vector
    dmax = 0.
    for it in keep:
        for u in ('1x1', '3x3'):
            a = np.asarray(it['maps']['collapsed_%s' % u])
            b_ = np.asarray(it['maps']['collapsedS_%s' % u])
            sc_ = max(abs(a).max(), abs(b_).max()) + 1e-300
            dmax = max(dmax, float(np.abs(a - b_).max() / sc_))
    control = dict(worst_rel_N_minus_S_at_GAP=dmax, n_items=len(keep))

    # confident subgroup: items where the model's own logit is above median
    lg = np.array([it['logit_gap'] for it in keep])
    conf = lg >= np.median(lg)
    sub = {}
    for temp in ('lse8.0', 'lse-8.0'):
        for metric in ('hit', 'ap'):
            a = np.asarray(sc['N_%s_1x1' % temp][metric], float)
            b_ = np.asarray(sc['collapsed_1x1'][metric], float)
            s_ = np.asarray(sc['S_%s_1x1' % temp][metric], float)
            d1 = mlib.paired(a[conf], b_[conf])
            d2 = mlib.paired(a[conf], s_[conf])
            d1.update(mean_a=float(np.nanmean(a[conf])), mean_b=float(np.nanmean(b_[conf])))
            d2.update(mean_a=float(np.nanmean(a[conf])), mean_b=float(np.nanmean(s_[conf])))
            sub['H1_%s_%s' % (temp, metric)] = d1
            sub['H2_%s_%s' % (temp, metric)] = d2

    res = dict(n=n, primary=PRIMARY, maps=mapnames, control=control,
               confident_subgroup=sub,
               gt_fraction_mean=float(gtfrac.mean()),
               summary=summ, prespecified=pre, extra=extra,
               per_finding_hit=per_find,
               n_by_finding={f: int((findings == f).sum()) for f in sorted(set(findings))})
    res['excluded_findings'] = sorted(excl)
    mlib.save(os.environ.get('OUTNAME', 'm4_stats.json'), res)

    P = pre['H1_%s_%s_%s' % (PRIMARY['temp'], PRIMARY['unit'], PRIMARY['metric'])]
    Q = pre['H2_%s_%s_%s' % (PRIMARY['temp'], PRIMARY['unit'], PRIMARY['metric'])]
    print('\nPRIMARY (hit rate, 1x1, T=+8), n=%d' % n)
    print('  N  %.4f   collapsed %.4f   diff %+.4f [%+.4f,%+.4f]  p=%.4g'
          % (P['mean_a'], P['mean_b'], P['diff'], P['lo'], P['hi'], P['p_mcnemar']))
    print('  N  %.4f   S         %.4f   diff %+.4f [%+.4f,%+.4f]  p=%.4g'
          % (Q['mean_a'], Q['mean_b'], Q['diff'], Q['lo'], Q['hi'], Q['p_mcnemar']))


if __name__ == '__main__':
    main()
