"""W8 -- the paired test for the pre-normalisation control.

w7 compares point estimates and counts signs; a sign count over twelve cells is
not a test. w6 saves per-record arrays under the same names and the same record
order the main run uses, so each control map can be paired image by image
against the maps it is compared with.

Paired binary outcomes (the two pointing games) take an exact McNemar test;
continuous ones (peak-mask IoU, per-image best IoU) take the two-sided Wilcoxon
signed-rank test; MaxBoxAccV2 takes a paired bootstrap over images, because it
maximises over thresholds and so must be recomputed inside every resample.
Every metric a dataset defines is tested, which was not true of the first
version of this script. The Holm step-down correction is applied across the whole
family of comparisons for one control map, which is the family that carries the
claim.

The claim under test is the paper's: that a map which cannot encode the
necessity/sufficiency distinction -- because its readout is additive, so the two
are provably the same array -- is not worse than the separated pair.
"""
import os
import sys
import glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import wlib as W

CONTROLS = ['collapsed_pre', 'cam_pre']
SEPARATED = ['N_ln', 'S_ln']
ALPHA = 0.05


def load(ds, name, kind):
    p = os.path.join(W.OUT, f'{kind}_{ds}_vitb16_{name}.npy')
    return np.load(p) if os.path.exists(p) else None


def mcnemar(a, b):
    from scipy.stats import binomtest
    a, b = np.asarray(a) > 0, np.asarray(b) > 0
    n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
    n = n10 + n01
    return (float(binomtest(n10, n, 0.5).pvalue) if n else 1.0), n10, n01


def wilcoxon(a, b):
    from scipy.stats import wilcoxon as wx
    d = np.asarray(a, float) - np.asarray(b, float)
    if not np.any(d):
        return 1.0
    return float(wx(a, b, zero_method='wilcox', alternative='two-sided').pvalue)


def holm(ps):
    order = np.argsort(ps)
    out, run = np.empty(len(ps)), 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(ps) - rank) * ps[i])
        out[i] = min(1.0, run)
    return out


def main():
    rows = []
    for f in sorted(glob.glob(os.path.join(W.OUT, 'w6_prenorm_*.json'))):
        ds = os.path.basename(f)[len('w6_prenorm_'):-len('.json')]
        for ctrl in CONTROLS:
            pt_c = load(ds, f'prenorm_{ctrl}', 'pt')
            if pt_c is None:
                continue
            for sep in SEPARATED:
                pt_s = load(ds, sep, 'pt')
                if pt_s is None or pt_s.shape != pt_c.shape:
                    continue
                for j, metric in enumerate(('pointing_strict', 'pointing_tol15')):
                    p, n10, n01 = mcnemar(pt_c[j], pt_s[j])
                    rows.append(dict(dataset=ds, control=ctrl, separated=sep,
                                     metric=metric, test='mcnemar',
                                     n=int(pt_c.shape[1]),
                                     control_rate=float((pt_c[j] > 0).mean()),
                                     separated_rate=float((pt_s[j] > 0).mean()),
                                     diff=float((pt_c[j] > 0).mean() - (pt_s[j] > 0).mean()),
                                     control_only=n10, separated_only=n01, p=p))
                io_c, io_s = load(ds, f'prenorm_{ctrl}', 'iou'), load(ds, sep, 'iou')
                if io_c is not None and io_s is not None and io_c.shape == io_s.shape:
                    rows.append(dict(dataset=ds, control=ctrl, separated=sep,
                                     metric='peak_mask_iou', test='wilcoxon',
                                     n=int(io_c.size),
                                     control_rate=float(io_c.mean()),
                                     separated_rate=float(io_s.mean()),
                                     diff=float(io_c.mean() - io_s.mean()),
                                     p=wilcoxon(io_c, io_s)))

                # The first version of this script tested only the metrics that
                # come as per-record vectors, which silently dropped CUB's two
                # curve-derived metrics -- including MaxBoxAccV2, the metric
                # this paper elsewhere calls the one built to defeat degenerate
                # nulls. A reader caught it. Both are testable from the curves
                # w6 persists, by exactly the tests the main run uses: a paired
                # bootstrap for MaxBoxAccV2, because it is a max over
                # thresholds and has to be recomputed inside every resample,
                # and Wilcoxon for the per-image best IoU.
                cv_c = load(ds, f'prenorm_{ctrl}', 'curve')
                cv_s = load(ds, sep, 'curve')
                if cv_c is not None and cv_s is not None and cv_c.shape == cv_s.shape:
                    dl = tuple(W.PROTOCOL['deltas'])
                    d, lo, hi, pb = W.boot_mbav2_diff(
                        cv_c, cv_s, n=W.PROTOCOL['n_boot_mbav2'],
                        seed=W.PROTOCOL['seed'], deltas=dl)
                    rows.append(dict(dataset=ds, control=ctrl, separated=sep,
                                     metric='MaxBoxAccV2',
                                     test='paired bootstrap',
                                     n=int(cv_c.shape[0]),
                                     control_rate=float(W.maxboxacc_v2(cv_c, dl)[0]),
                                     separated_rate=float(W.maxboxacc_v2(cv_s, dl)[0]),
                                     diff=float(d), ci=[float(lo), float(hi)],
                                     p=float(pb)))
                    xa = cv_c.max(1).astype(np.float64)
                    xb = cv_s.max(1).astype(np.float64)
                    rows.append(dict(dataset=ds, control=ctrl, separated=sep,
                                     metric='best_iou', test='wilcoxon',
                                     n=int(xa.size),
                                     control_rate=float(xa.mean()),
                                     separated_rate=float(xb.mean()),
                                     diff=float(xa.mean() - xb.mean()),
                                     p=wilcoxon(xa, xb)))

    if not rows:
        sys.exit('no paired arrays found; re-run w6_prenorm.py')

    for ctrl in CONTROLS:
        fam = [r for r in rows if r['control'] == ctrl]
        if not fam:
            continue
        for r, ph in zip(fam, holm(np.array([r['p'] for r in fam]))):
            r['p_holm'] = float(ph)

    print(f"{'control':16s}{'dataset':10s}{'vs':6s}{'metric':18s}"
          f"{'ctrl':>8s}{'sep':>8s}{'diff':>9s}{'test':>10s}{'P Holm':>10s}")
    print('-' * 95)
    summary = {}
    for ctrl in CONTROLS:
        fam = [r for r in rows if r['control'] == ctrl]
        worse = [r for r in fam if r['diff'] < 0 and r['p_holm'] < ALPHA]
        better = [r for r in fam if r['diff'] > 0 and r['p_holm'] < ALPHA]
        for r in fam:
            print(f"{r['control']:16s}{r['dataset']:10s}{r['separated']:6s}"
                  f"{r['metric']:18s}{r['control_rate']:8.4f}"
                  f"{r['separated_rate']:8.4f}{r['diff']:+9.4f}"
                  f"{r['test']:>10s}{r['p_holm']:10.3g}")
        summary[ctrl] = dict(n=len(fam), better=len(better), worse=len(worse),
                             no_difference=len(fam) - len(better) - len(worse))
        print(f"  -> {ctrl}: of {len(fam)} paired comparisons against the two "
              f"separated maps, the control is significantly better in "
              f"{len(better)}, significantly worse in {len(worse)}, and "
              f"indistinguishable in {len(fam) - len(better) - len(worse)}")
        print('-' * 95)

    W.save('w8_control_test.json',
           dict(alpha=ALPHA, note='paired tests, Holm-corrected within each '
                                  'control map; McNemar for the binary '
                                  'pointing outcomes, Wilcoxon signed-rank for '
                                  'the continuous ones',
                summary=summary, rows=rows))


if __name__ == '__main__':
    main()
