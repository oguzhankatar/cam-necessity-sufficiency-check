"""S5 (P3) -- re-verify the nine in-scope models with the cross-check's estimator.

The head adapter is reused from `scripts/probe25_lnplace.py` (the `Site` class),
so the object under test is identical.  Everything measured about it is
recomputed:

  * N and S with `scripts/crosscheck/rtlib.py` -- a literal per-cell loop that
    rebuilds the whole perturbed cell matrix and re-runs the head, in float64,
    importing nothing from the project's own estimator.
  * `max_t |N-S| / scale`, `Rhalf`, `I2` -- the direct additivity certificate.
    corrNS is reported next to them but is NOT the criterion.
  * top-1 for all three orderings on the FULL 50 000-image val set, recomputed
    from the images, not read from the old JSON.

Every number is then differenced against `outputs/probe25_lnplace_*.json` and
any disagreement is printed and stored.
"""
import os, sys, json, time, argparse, warnings, numpy as np, torch, torch.nn as nn
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import slib
sys.path.insert(0, os.path.join(slib.ROOT, 'scripts'))
sys.path.insert(0, os.path.join(slib.ROOT, 'scripts', 'crosscheck'))
import rtlib
from torch.utils.data import DataLoader, Subset

MODELS = {
    'beitv2_base_patch16_224': 'beitv2_base_patch16_224.in1k_ft_in1k',
    'convnext_tiny': 'convnext_tiny.fb_in1k',
    'eva02_tiny_patch14_336': 'eva02_tiny_patch14_336.mim_in22k_ft_in1k',
    'naflexvit_base_patch16_gap': 'naflexvit_base_patch16_gap.e300_s576_in1k',
    'swin_tiny_patch4_window7_224': 'swin_tiny_patch4_window7_224.ms_in1k',
    'vit_base_patch16_rope_reg1_gap_256': 'vit_base_patch16_rope_reg1_gap_256.sbb_in1k',
    'vit_betwixt_patch16_reg4_gap_256': 'vit_betwixt_patch16_reg4_gap_256.sbb_in1k',
    'vit_little_patch16_reg4_gap_256': 'vit_little_patch16_reg4_gap_256.sbb_in1k',
    'vit_medium_patch16_reg4_gap_256': 'vit_medium_patch16_reg4_gap_256.sbb_in1k',
}
MODES = ('post', 'pre', 'none')

ap = argparse.ArgumentParser()
ap.add_argument('models', nargs='*', default=sorted(MODELS))
ap.add_argument('--nprobe', type=int, default=32)
ap.add_argument('--nbank', type=int, default=16)
ap.add_argument('--neval', type=int, default=50000)
ap.add_argument('--bs', type=int, default=64)
ap.add_argument('--workers', type=int, default=8)
ap.add_argument('--tag', default='s5_reverify')
A = ap.parse_args()


def load_site(key):
    os.environ['MODEL'] = MODELS[key]
    import importlib
    import probe25_lnplace as P
    importlib.reload(P)
    return P.Site(MODELS[key]), P


def run(key):
    t0 = time.time()
    st, P = load_site(key)
    ds = P.datasets_imagefolder() if hasattr(P, 'datasets_imagefolder') else None
    from torchvision import datasets
    ds = datasets.ImageFolder(slib.VAL, P.transform(st.cfg))
    rs = np.random.RandomState(0)
    perm = rs.permutation(len(ds))
    print('[%s] family=%s shipped=%s C=%d input=%d'
          % (key, st.family, st.shipped, st.C, st.cfg['input_size'][-1]), flush=True)

    # ---- sanity: the composed head reproduces the shipped model
    x0 = torch.stack([ds[i][0] for i in perm[:8]]).cuda()
    with torch.no_grad():
        ref = st.m(x0)
        ours = st.head(st.cells(x0), st.shipped)
    cerr = float((ref - ours).abs().max() / (ref.abs().max() + 1e-12))
    print('  compose rel err %.2e %s' % (cerr, 'PASS' if cerr < 2e-4 else 'FAIL'),
          flush=True)
    assert cerr < 2e-4

    # -------------------------------------------------- accuracy, full 50 000
    nev = min(A.neval, len(ds))
    ev = np.arange(len(ds)) if nev >= len(ds) else np.sort(perm[:nev])
    ld = DataLoader(Subset(ds, ev.tolist()), batch_size=A.bs,
                    num_workers=A.workers, pin_memory=True)
    hit = {k: 0 for k in MODES}
    n = 0
    with torch.no_grad():
        for xb, yb in ld:
            X = st.cells(xb.cuda(non_blocking=True))
            for k in MODES:
                hit[k] += (st.head(X, k).argmax(1).cpu() == yb).sum().item()
            n += len(yb)
            if n % 12800 == 0:
                print('   acc %6d/%d  ' % (n, nev)
                      + ' '.join('%s=%.4f' % (k, hit[k] / n) for k in MODES), flush=True)
    acc = {k: hit[k] / n for k in MODES}
    print('  top-1 (%d): ' % n + '  '.join('%s=%.4f' % (k, v) for k, v in acc.items()),
          flush=True)

    # ------------------------------------------- rtlib, float64, literal loop
    st.m = st.m.double()
    st.w = st.w.double()
    if st.b is not None:
        st.b = st.b.double()
    sel = perm[:A.nprobe + A.nbank]
    xp = torch.stack([ds[i][0] for i in sel[:A.nprobe]]).cuda().double()
    xbk = torch.stack([ds[i][0] for i in sel[A.nprobe:]]).cuda().double()
    with torch.no_grad():
        XP = torch.cat([st.cells(xp[i:i + 8]) for i in range(0, A.nprobe, 8)])
        XBK = st.cells(xbk)
        pred = st.head(XP, st.shipped).argmax(1).cpu().numpy()
    out_modes = {}
    for mode in MODES:
        rr = []
        rs2 = np.random.RandomState(1)
        for q in range(A.nprobe):
            XA = XP[q]
            XB = XBK[rs2.randint(A.nbank)]
            rr.append(rtlib.report('%s/%s/%d' % (key, mode, q),
                                   lambda Z: st.head(Z, mode), XA, XB,
                                   int(pred[q]), rs2))
        d = dict(corrNS_mean=float(np.nanmean([r['corr_pearson'] for r in rr])),
                 corrNS_min=float(np.nanmin([r['corr_pearson'] for r in rr])),
                 corrNS_max=float(np.nanmax([r['corr_pearson'] for r in rr])),
                 maxNS=float(np.nanmax([r['max_abs_diff_rel'] for r in rr])),
                 Rhalf=float(np.nanmax([r['Rhalf'] for r in rr])),
                 I2=float(np.nanmax([r['I2'] for r in rr])),
                 sumN_over_dz=float(np.nanmean([r['sumN_over_zA_minus_zB'] for r in rr])),
                 n=len(rr), acc=acc[mode])
        d['additive'] = bool(d['maxNS'] < 1e-9 and d['Rhalf'] < 1e-9 and d['I2'] < 1e-9)
        out_modes[mode] = d
        print('  %-5s corrNS=%+.8f [%.6f,%.6f]  maxNS=%.2e Rhalf=%.2e I2=%.2e '
              'sumN/dz=%.6f  ADDITIVE=%s'
              % (mode, d['corrNS_mean'], d['corrNS_min'], d['corrNS_max'],
                 d['maxNS'], d['Rhalf'], d['I2'], d['sumN_over_dz'], d['additive']),
              flush=True)

    # --------------------------------------------------- diff vs probe25 JSON
    jp = os.path.join(slib.ROOT, 'outputs', 'probe25_lnplace_%s.json' % key)
    diff = {}
    if os.path.exists(jp):
        old = json.load(open(jp))
        for k in MODES:
            diff['acc_' + k] = dict(new=acc[k], old=old[k]['acc'],
                                    delta_pts=100 * (acc[k] - old[k]['acc']))
            diff['corrNS_' + k] = dict(new=out_modes[k]['corrNS_mean'],
                                       old=old[k]['corrNS'],
                                       delta=out_modes[k]['corrNS_mean'] - old[k]['corrNS'],
                                       old_ci=old[k]['corrNS_ci'])
        flags = []
        for k in MODES:
            if abs(diff['acc_' + k]['delta_pts']) > 0.02:
                flags.append('acc_%s %+.3f pts' % (k, diff['acc_' + k]['delta_pts']))
            lo, hi = old[k]['corrNS_ci']
            if not (lo - 1e-6 <= out_modes[k]['corrNS_mean'] <= hi + 1e-6):
                flags.append('corrNS_%s %.6f outside published CI [%.6f,%.6f]'
                             % (k, out_modes[k]['corrNS_mean'], lo, hi))
        diff['flags'] = flags
        print('  DIFF vs probe25 JSON: %s'
              % ('none' if not flags else ' | '.join(flags)), flush=True)
    return dict(model=key, weights=MODELS[key], family=st.family,
                shipped=st.shipped, C=st.C, T=int(XP.shape[1]),
                input=st.cfg['input_size'][-1], compose_rel_err=cerr,
                n_eval=n, modes=out_modes, diff_vs_probe25=diff,
                secs=time.time() - t0)


if __name__ == '__main__':
    allr = []
    for k in A.models:
        try:
            allr.append(run(k))
        except Exception as e:
            import traceback
            traceback.print_exc()
            allr.append(dict(model=k, error='%s: %s' % (type(e).__name__, e)))
        slib.save(A.tag + '.json', allr)
