"""S6 (P4) -- HOW MUCH OF THE `post` COLUMN IS THE BASELINE?

The project measures necessity/sufficiency against a baseline B whose cells are
taken from ANOTHER validation image (a disjoint 16-image bank).  probe25's own
docstring gives the reason: a per-image channel mean has the SAME pooled vector
as the image, so every cell difference would be identically zero.  That rules
out the degenerate choice but does not pin the baseline down, and the cross-check
showed the shipped `post` number moves a lot with it.

This measures, for all nine in-scope models and both the exact orderings:

  BANK        a cell matrix from another val image                (the paper's)
  ZEROS       all-zero cells
  DSMEAN      one constant cell vector = mean over bank cells and positions
  GAUSS       Gaussian cells matched to the bank's per-channel mean/std
  SHUFFLE     a random spatial permutation of A's own cells (m_B == m_A)
  SAMECLASS   another image predicted to the same class as A
  BLUR        A's own cells after a 3x3 spatial box blur
  s = 1 .. 0.01  B = A + s (B_bank - A):  the same baseline, brought closer

Estimator: `scripts/crosscheck/rtlib.py`, literal per-cell loop, float64.
The project's PASS rule (|corrNS - 1| < 1e-5 AND Rhalf < 1e-3) is applied to
each cell, so we can say exactly which baselines would flip the verdict.
"""
import os, sys, json, time, argparse, warnings, numpy as np, torch, torch.nn as nn
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import slib
sys.path.insert(0, os.path.join(slib.ROOT, 'scripts'))
sys.path.insert(0, os.path.join(slib.ROOT, 'scripts', 'crosscheck'))
import rtlib
from s5_reverify import MODELS

MODES = ('post', 'pre', 'none')
ap = argparse.ArgumentParser()
ap.add_argument('models', nargs='*', default=sorted(MODELS))
ap.add_argument('--nimg', type=int, default=12)
ap.add_argument('--tag', default='s6_baseline')
A = ap.parse_args()


def load_site(key):
    os.environ['MODEL'] = MODELS[key]
    import importlib
    import probe25_lnplace as P
    importlib.reload(P)
    st = P.Site(MODELS[key])
    st.m = st.m.double()
    st.w = st.w.double()
    if st.b is not None:
        st.b = st.b.double()
    return st, P


def run(key):
    t0 = time.time()
    st, P = load_site(key)
    from torchvision import datasets
    ds = datasets.ImageFolder(slib.VAL, P.transform(st.cfg))
    rs = np.random.RandomState(0)
    perm = rs.permutation(len(ds))
    NB = 16
    sel = perm[:A.nimg + NB]
    xp = torch.stack([ds[i][0] for i in sel[:A.nimg]]).cuda().double()
    xb = torch.stack([ds[i][0] for i in sel[A.nimg:]]).cuda().double()
    with torch.no_grad():
        XP = torch.cat([st.cells(xp[i:i + 4]) for i in range(0, A.nimg, 4)])
        XB = torch.cat([st.cells(xb[i:i + 4]) for i in range(0, NB, 4)])
        pred = st.head(XP, st.shipped).argmax(1).cpu().numpy()
    T, C = XP.shape[1], XP.shape[2]
    print('[%s] shipped=%s T=%d C=%d' % (key, st.shipped, T, C), flush=True)

    # a same-class partner for each probe image, found among the next 600 val images
    same = {}
    pool_idx = perm[A.nimg + NB: A.nimg + NB + 600]
    with torch.no_grad():
        for i in range(0, len(pool_idx), 16):
            xx = torch.stack([ds[j][0] for j in pool_idx[i:i + 16]]).cuda().double()
            XX = st.cells(xx)
            pp = st.head(XX, st.shipped).argmax(1).cpu().numpy()
            for q in range(A.nimg):
                if q not in same:
                    hits = np.where(pp == pred[q])[0]
                    if len(hits):
                        same[q] = XX[hits[0]].clone()
        del XX

    mu = XB.reshape(-1, C).mean(0)
    sd = XB.reshape(-1, C).std(0)
    g = torch.Generator(device='cuda').manual_seed(3)

    def blur(X):
        h = int(round(T ** .5))
        if h * h != T:
            return None
        Y = X.T.reshape(1, C, h, h)
        k = torch.ones(C, 1, 3, 3, device=X.device, dtype=X.dtype) / 9.
        Y = torch.nn.functional.conv2d(torch.nn.functional.pad(Y, (1, 1, 1, 1),
                                                               mode='replicate'),
                                       k, groups=C)
        return Y.reshape(C, T).T.contiguous()

    def baselines(q):
        XA = XP[q]
        B0 = XB[q % NB]
        out = {'bank': B0,
               'zeros': torch.zeros_like(XA),
               'dsmean': mu[None].expand(T, C).contiguous(),
               'gauss': mu + sd * torch.randn(T, C, device=XA.device,
                                              dtype=XA.dtype, generator=g),
               'shuffle': XA[torch.randperm(T, device=XA.device, generator=g)]}
        if q in same:
            out['sameclass'] = same[q]
        b = blur(XA)
        if b is not None:
            out['blur'] = b
        for s in (0.5, 0.25, 0.1, 0.05, 0.01):
            out['bank_s%g' % s] = XA + s * (B0 - XA)
        return out

    res = {}
    for mode in MODES:
        acc = {}
        for q in range(A.nimg):
            XA = XP[q]
            for bn, XBq in baselines(q).items():
                d = rtlib.report('%s/%s/%s/%d' % (key, mode, bn, q),
                                 lambda Z: st.head(Z, mode), XA, XBq,
                                 int(pred[q]), np.random.RandomState(1))
                acc.setdefault(bn, []).append(d)
        res[mode] = {}
        for bn, rr in acc.items():
            corr = np.array([r['corr_pearson'] for r in rr], dtype=np.float64)
            corr = corr[np.isfinite(corr)]
            e = dict(corrNS=float(corr.mean()) if len(corr) else float('nan'),
                     corrNS_sd=float(corr.std()) if len(corr) else float('nan'),
                     maxNS=float(np.nanmax([r['max_abs_diff_rel'] for r in rr])),
                     Rhalf=float(np.nanmax([r['Rhalf'] for r in rr])),
                     n=len(corr))
            e['project_PASS'] = bool(abs(e['corrNS'] - 1) < 1e-5 and e['Rhalf'] < 1e-3)
            res[mode][bn] = e
        order = ['bank', 'zeros', 'dsmean', 'gauss', 'sameclass', 'blur', 'shuffle',
                 'bank_s0.5', 'bank_s0.25', 'bank_s0.1', 'bank_s0.05', 'bank_s0.01']
        print('  %-5s ' % mode + '  '.join(
            '%s=%s%.6f%s' % (b, '+' if res[mode][b]['corrNS'] >= 0 else '',
                             res[mode][b]['corrNS'],
                             '*' if res[mode][b]['project_PASS'] else '')
            for b in order if b in res[mode]), flush=True)
    std = ['bank', 'zeros', 'dsmean', 'gauss', 'sameclass', 'blur']
    vals = [res['post'][b]['corrNS'] for b in std if b in res['post']]
    out = dict(model=key, shipped=st.shipped, T=T, C=C, n_img=A.nimg,
               baselines=res,
               post_std_min=float(min(vals)), post_std_max=float(max(vals)),
               post_std_range=float(max(vals) - min(vals)),
               post_shuffle=res['post']['shuffle']['corrNS'],
               post_any_pass=[b for b in res['post'] if res['post'][b]['project_PASS']],
               exact_all_pass={k: all(res[k][b]['project_PASS'] for b in res[k])
                               for k in ('pre', 'none')},
               secs=time.time() - t0)
    print('  post over the 6 standard baselines: %.4f .. %.4f (range %.4f); '
          'shuffle %.4f; pre all-pass=%s none all-pass=%s'
          % (out['post_std_min'], out['post_std_max'], out['post_std_range'],
             out['post_shuffle'], out['exact_all_pass']['pre'],
             out['exact_all_pass']['none']), flush=True)
    del st
    torch.cuda.empty_cache()
    return out


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
