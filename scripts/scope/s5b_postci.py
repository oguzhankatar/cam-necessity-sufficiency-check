"""S5b -- the `post` column with MY OWN confidence interval, so that a
"disagreement" with the published value means something.

S5 re-measured `post` on 32 images; probe25 published it on 256.  A point
estimate outside a published CI is not evidence of an error when the two draws
have very different n and the per-image spread is large (on eva02 the per-image
corrNS runs from -0.51 to +0.93).  This recomputes the `post` column with
n = 96 images, float64, `rtlib`'s literal loop, keeps the PER-IMAGE values, and
bootstraps a 95 % CI over images.  A disagreement is declared only when the two
intervals are disjoint.
"""
import os, sys, json, time, argparse, warnings, numpy as np, torch
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import slib
sys.path.insert(0, os.path.join(slib.ROOT, 'scripts'))
sys.path.insert(0, os.path.join(slib.ROOT, 'scripts', 'crosscheck'))
import rtlib
from s5_reverify import MODELS

ap = argparse.ArgumentParser()
ap.add_argument('models', nargs='*', default=sorted(MODELS))
ap.add_argument('--nimg', type=int, default=96)
ap.add_argument('--nbank', type=int, default=16)
ap.add_argument('--tag', default='s5b_postci')
A = ap.parse_args()


def boot(v, B=4000, seed=0):
    v = np.asarray(v, dtype=np.float64)
    v = v[np.isfinite(v)]
    r = np.random.RandomState(seed)
    mu = v.mean()
    bs = v[r.randint(0, len(v), (B, len(v)))].mean(1)
    return float(mu), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def run(key):
    t0 = time.time()
    os.environ['MODEL'] = MODELS[key]
    import importlib
    import probe25_lnplace as P
    importlib.reload(P)
    st = P.Site(MODELS[key])
    st.m = st.m.double()
    st.w = st.w.double()
    if st.b is not None:
        st.b = st.b.double()
    from torchvision import datasets
    ds = datasets.ImageFolder(slib.VAL, P.transform(st.cfg))
    rs = np.random.RandomState(0)
    perm = rs.permutation(len(ds))
    sel = perm[:A.nimg + A.nbank]
    with torch.no_grad():
        XP = torch.cat([st.cells(torch.stack([ds[i][0] for i in sel[k:k + 4]])
                                 .cuda().double()) for k in range(0, A.nimg, 4)])
        XB = torch.cat([st.cells(torch.stack([ds[i][0] for i in sel[A.nimg + k:A.nimg + k + 4]])
                                 .cuda().double()) for k in range(0, A.nbank, 4)])
        pred = st.head(XP, st.shipped).argmax(1).cpu().numpy()
    out = dict(model=key, shipped=st.shipped, T=int(XP.shape[1]), C=st.C,
               n_img=A.nimg)
    rs2 = np.random.RandomState(1)
    per = []
    for q in range(A.nimg):
        d = rtlib.report('%s/post/%d' % (key, q), lambda Z: st.head(Z, 'post'),
                         XP[q], XB[rs2.randint(A.nbank)], int(pred[q]), rs2,
                         do_add=False)
        per.append(d['corr_pearson'])
    mu, lo, hi = boot(per)
    out['post'] = dict(corrNS=mu, ci=[lo, hi], per_image_min=float(np.nanmin(per)),
                       per_image_max=float(np.nanmax(per)),
                       per_image_sd=float(np.nanstd(per)), n=len(per))
    jp = os.path.join(slib.ROOT, 'outputs', 'probe25_lnplace_%s.json' % key)
    old = json.load(open(jp))
    olo, ohi = old['post']['corrNS_ci']
    out['published'] = dict(corrNS=old['post']['corrNS'], ci=[olo, ohi],
                            n=old['meta']['n_img'])
    out['ci_disjoint'] = bool(hi < olo or lo > ohi)
    out['delta'] = mu - old['post']['corrNS']
    print('%-36s mine=%+.6f [%+.6f,%+.6f] (n=%d, per-image %+.3f..%+.3f)  '
          'published=%+.6f [%+.6f,%+.6f] (n=%d)  delta=%+.4f  DISJOINT=%s'
          % (key, mu, lo, hi, len(per), out['post']['per_image_min'],
             out['post']['per_image_max'], old['post']['corrNS'], olo, ohi,
             old['meta']['n_img'], out['delta'], out['ci_disjoint']), flush=True)
    out['secs'] = time.time() - t0
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
    ok = [r for r in allr if 'ci_disjoint' in r]
    print('\nCI-disjoint with the published value: %d/%d -> %s'
          % (sum(r['ci_disjoint'] for r in ok), len(ok),
             [r['model'] for r in ok if r['ci_disjoint']]))
