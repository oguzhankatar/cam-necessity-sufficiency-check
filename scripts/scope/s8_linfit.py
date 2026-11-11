"""S8 -- what is the BEST an affine readout can do on this architecture?

The edits in S3 are particular affine repairs.  This measures the ceiling: the
least-squares linear map from the model's own POOLED vector to the model's own
SHIPPED LOGITS,

        z_hat = W m + c,      W, c = argmin  E ||W m + c - z_shipped||^2

fitted by ridge-free normal equations in float64 on a held-out slice of the val
set and using NO LABELS (the target is the network's own output).  Because the
pooled vector is an exact linear function of the cells, this readout is additive
over the cells by construction, and no affine post-pool map reading the same
pooled vector approximates the shipped logits better IN L2.

CAREFUL: that is an L2 statement, not an accuracy statement.  Its top-1 is NOT
an upper bound on what an affine head can achieve -- a different affine map can
classify better while fitting the logits worse -- so it is reported as a
reference point ("how much of the head's function is affine?"), and the
argmax agreement with the shipped model is reported next to it.

Reported on the 40 000 val images NOT used for the fit; the shipped head is
scored on exactly the same 40 000 so the difference is apples to apples.
"""
import os, sys, json, time, argparse, warnings, numpy as np, torch, torch.nn as nn
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import slib, shead
import timm
from torch.utils.data import DataLoader, Subset

ap = argparse.ArgumentParser()
ap.add_argument('models', nargs='+')
ap.add_argument('--nfit', type=int, default=10000)
ap.add_argument('--bs', type=int, default=64)
ap.add_argument('--workers', type=int, default=6)
ap.add_argument('--tag', default='s8_linfit')
A = ap.parse_args()


def run(name):
    t0 = time.time()
    m = timm.create_model(name, pretrained=True).eval().cuda()
    ds = slib.val_ds(m.pretrained_cfg)
    rs = np.random.RandomState(0)
    perm = rs.permutation(len(ds))
    x2 = torch.stack([ds[i][0] for i in perm[:2]]).cuda()
    fam = type(m).__name__
    ag = shead.AggMamba(m, x2) if fam == 'MambaOut' else shead.Agg(m, x2)
    X2 = ag.cells(x2)
    T, C = X2.shape[1], X2.shape[2]

    def POST(v):
        if isinstance(ag, shead.AggMamba):
            return ag.post(v)
        return ag.post(v, torch.zeros(v.shape[0], C, 1, 1, device=v.device,
                                      dtype=v.dtype))
    with torch.no_grad():
        cerr = float((m(x2) - POST(X2.mean(1))).abs().max() / (m(x2).abs().max() + 1e-12))
    assert cerr < 2e-4, 'adapter mismatch %.2e' % cerr

    fit_idx = np.sort(perm[:A.nfit])
    hold_idx = np.sort(perm[A.nfit:])
    # ---- normal equations in float64
    G = torch.zeros(C + 1, C + 1, device='cuda', dtype=torch.float64)
    B = None
    ld = DataLoader(Subset(ds, fit_idx.tolist()), batch_size=A.bs,
                    num_workers=A.workers, pin_memory=True)
    with torch.no_grad():
        for xb, _ in ld:
            mm = ag.cells(xb.cuda(non_blocking=True)).mean(1).double()
            z = POST(mm.float()).double()
            u = torch.cat([mm, torch.ones(len(mm), 1, device='cuda',
                                          dtype=torch.float64)], 1)
            G += u.T @ u
            B = (u.T @ z) if B is None else B + u.T @ z
    Wc = torch.linalg.solve(G + 1e-8 * torch.eye(C + 1, device='cuda',
                                                 dtype=torch.float64), B)
    W, c = Wc[:C].float(), Wc[C].float()
    print('[%s] linfit solved on %d images (C=%d)' % (name, len(fit_idx), C), flush=True)

    # ---- accuracy: shipped vs linfit, on the held-out 40 k and on all 50 k
    res = {}
    for tag, idx in (('holdout', hold_idx),):
        ld = DataLoader(Subset(ds, idx.tolist()), batch_size=A.bs,
                        num_workers=A.workers, pin_memory=True)
        hs = hl = n = 0
        agree = 0
        with torch.no_grad():
            for xb, yb in ld:
                mm = ag.cells(xb.cuda(non_blocking=True)).mean(1)
                zs = POST(mm)
                zl = mm @ W + c
                hs += (zs.argmax(1).cpu() == yb).sum().item()
                hl += (zl.argmax(1).cpu() == yb).sum().item()
                agree += (zs.argmax(1) == zl.argmax(1)).sum().item()
                n += len(yb)
        res[tag] = dict(n=n, acc_shipped=hs / n, acc_linfit=hl / n,
                        dacc_pts=100 * (hl - hs) / n, argmax_agree=agree / n)
        print('  %-8s n=%d shipped=%.4f linfit=%.4f (%+.2f pts) argmax-agree=%.4f'
              % (tag, n, res[tag]['acc_shipped'], res[tag]['acc_linfit'],
                 res[tag]['dacc_pts'], res[tag]['argmax_agree']), flush=True)
    ag.close()
    del m
    torch.cuda.empty_cache()
    return dict(model=name, family=fam, T=T, C=C, n_fit=len(fit_idx),
                compose_rel_err=cerr, res=res, secs=time.time() - t0)


if __name__ == '__main__':
    allr = []
    for nm in A.models:
        try:
            allr.append(run(nm))
        except Exception as e:
            import traceback
            traceback.print_exc()
            allr.append(dict(model=nm, error='%s: %s' % (type(e).__name__, e)))
        slib.save(A.tag + '.json', allr)
