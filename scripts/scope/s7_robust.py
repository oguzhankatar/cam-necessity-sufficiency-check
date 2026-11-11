"""S7 -- is the reached degeneracy robust to the baseline and the perturbation unit?

The cross-check's standing objections to any single exactness number are (i) it is a
statistic of a (model, image, BASELINE) triple and (ii) the deficit depends on the
PERTURBATION UNIT (1x1 cell vs 3x3 / 5x5 blocks).  Both objections apply to a
number that is claimed to be near 1; they cannot rescue a number that is far from
it.  So this script checks the reached configurations of the blocked families at
TWO perturbation units and FOUR baselines, and the shipped configuration at the
same grid for contrast.

Everything is a literal loop in float64: the necessity/sufficiency pair is built
by rebuilding the whole perturbed cell matrix and re-running the head, and the
block version does exactly the same with a set of cells instead of one cell.  No
algebraic fast path anywhere; the 1x1 case is cross-checked against
`rtlib.NS_literal` and must agree to 1e-15.
"""
import os, sys, json, time, warnings, numpy as np, torch, torch.nn as nn
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import slib, shead
sys.path.insert(0, os.path.join(slib.ROOT, 'scripts', 'crosscheck'))
import rtlib
import timm

NIMG = int(os.environ.get('NIMG', 6))
TARGETS = {                        # model -> the reaching edit to re-test
    'mobilenetv3_large_100': 'poollast',
    'ghostnet_100': 'delact',
    'maxvit_tiny_tf_224': 'poollast',
    'caformer_s18': 'poollast',
}
ACT_NM = {}


def blocks(T, unit, rs):
    """partition the sqrt(T) x sqrt(T) grid into `unit` x `unit` blocks"""
    h = int(round(T ** .5))
    if h * h != T or unit == 1:
        return [[t] for t in range(T)]
    out = []
    for i in range(0, h, unit):
        for j in range(0, h, unit):
            out.append([r * h + c for r in range(i, min(i + unit, h))
                        for c in range(j, min(j + unit, h))])
    return out


@torch.no_grad()
def NS_blocks(head, XA, XB, c, bl):
    zA = float(head(XA[None])[0, c])
    zB = float(head(XB[None])[0, c])
    N = np.empty(len(bl)); S = np.empty(len(bl))
    for k, idx in enumerate(bl):
        Xn = XA.clone(); Xn[idx] = XB[idx]
        Xs = XB.clone(); Xs[idx] = XA[idx]
        N[k] = zA - float(head(Xn[None])[0, c])
        S[k] = float(head(Xs[None])[0, c]) - zB
    return N, S, zA


@torch.no_grad()
def Radd(head, XA, XB, c, bl, N, zA, rs, nsub=48):
    """E|Delta(H) - sum_{k in H} N_k| / E|Delta(H)| over random halves of BLOCKS"""
    D, add = [], []
    for _ in range(nsub):
        sel = rs.choice(len(bl), max(1, len(bl) // 2), replace=False)
        idx = [t for k in sel for t in bl[k]]
        X = XA.clone(); X[idx] = XB[idx]
        D.append(zA - float(head(X[None])[0, c]))
        add.append(N[sel].sum())
    D, add = np.array(D), np.array(add)
    return float(np.abs(D - add).mean() / (np.abs(D).mean() + 1e-300))


def main():
    rows = []
    for name, edit in TARGETS.items():
        t0 = time.time()
        m = timm.create_model(name, pretrained=True).eval().cuda()
        ds = slib.val_ds(m.pretrained_cfg)
        rs = np.random.RandomState(0)
        perm = rs.permutation(len(ds))
        x2 = torch.stack([ds[i][0] for i in perm[:2]]).cuda()
        ag = shead.Agg(m, x2)
        X2 = ag.cells(x2)
        T, C = X2.shape[1], X2.shape[2]
        m = m.double()
        mods = dict(m.named_modules())
        # post-pool leaves
        ev, hs = [], []

        def mk(nm):
            def f(mod, inp, out):
                if len(inp) and torch.is_tensor(inp[0]) and torch.is_tensor(out):
                    ev.append(nm)
            return f
        for nm, mod in m.named_modules():
            if len(list(mod.children())) == 0:
                hs.append(mod.register_forward_hook(mk(nm)))
        with torch.no_grad():
            m(x2.double())
        for h in hs:
            h.remove()
        pn = ev[ev.index(ag.pool_name) + 1:]
        acts = [n for n in pn if n in mods and slib.is_act(mods[n])]
        norms = [n for n in pn if n in mods and slib.is_norm(mods[n])]

        def POST(v):
            return ag.post(v, torch.zeros(v.shape[0], C, 1, 1, device=v.device,
                                          dtype=v.dtype))

        def set_mod(n, new):
            p = m
            for q in n.split('.')[:-1]:
                p = getattr(p, q)
            setattr(p, n.split('.')[-1], new)

        def cfg(c):
            for n in acts + norms:
                set_mod(n, mods[n])
            if c == 'delact':
                for n in acts:
                    set_mod(n, nn.Identity())
            if c == 'delboth':
                for n in acts + norms:
                    set_mod(n, nn.Identity())

        def head_of(c):
            if c == 'poollast':
                return lambda Z: POST(Z.reshape(-1, C)).reshape(Z.shape[0], T, -1).mean(1)
            return lambda Z: POST(Z.mean(1))

        xp = torch.stack([ds[i][0] for i in perm[:NIMG + 8]]).cuda().double()
        with torch.no_grad():
            XP = ag.cells(xp)
        bank = XP[NIMG:]
        with torch.no_grad():
            pred = POST(XP[:NIMG].mean(1)).argmax(1).cpu().numpy()
        mu = bank.reshape(-1, C).mean(0)
        g = torch.Generator(device='cuda').manual_seed(5)

        def bmake(kind, q):
            XA = XP[q]
            if kind == 'bank':
                return bank[q % bank.shape[0]]
            if kind == 'zeros':
                return torch.zeros_like(XA)
            if kind == 'dsmean':
                return mu[None].expand(T, C).contiguous()
            return XA[torch.randperm(T, device=XA.device, generator=g)]

        rec = dict(model=name, edit=edit, T=T, C=C, acts=acts, norms=norms, cells={})
        for c in ('shipped', edit):
            cfg(c)
            h = head_of(c)
            for unit in (1, 3):
                bl = blocks(T, unit, rs)
                for bk in ('bank', 'zeros', 'dsmean', 'shuffle'):
                    mm, rr, cc = [], [], []
                    rs2 = np.random.RandomState(2)
                    for q in range(NIMG):
                        XA, XB = XP[q], bmake(bk, q)
                        N, S, zA = NS_blocks(h, XA, XB, int(pred[q]), bl)
                        sc = max(abs(N).max(), abs(S).max()) + 1e-300
                        mm.append(np.abs(N - S).max() / sc)
                        rr.append(Radd(h, XA, XB, int(pred[q]), bl, N, zA, rs2))
                        if N.std() > 0 and S.std() > 0:
                            cc.append(float(np.corrcoef(N, S)[0, 1]))
                        if unit == 1 and bk == 'bank' and q == 0:
                            Nr, Sr, _, _ = rtlib.NS_literal(h, XA, XB, int(pred[q]))
                            e = float(max(np.abs(Nr - N).max(), np.abs(Sr - S).max())
                                      / (np.abs(Nr).max() + 1e-300))
                            rec['crosscheck_vs_rtlib_%s' % c] = e
                            assert e < 1e-12, 'block loop disagrees with rtlib'
                    k = '%s|unit%d|%s' % (c, unit, bk)
                    rec['cells'][k] = dict(maxNS=float(np.max(mm)),
                                           Radd=float(np.max(rr)),
                                           corrNS=float(np.mean(cc)) if cc else float('nan'),
                                           nblocks=len(bl))
                    print('  %-34s maxNS=%.2e Radd=%.2e corrNS=%+.6f (%d blocks)'
                          % (k, rec['cells'][k]['maxNS'], rec['cells'][k]['Radd'],
                             rec['cells'][k]['corrNS'], len(bl)), flush=True)
        cfg('shipped')
        rec['secs'] = time.time() - t0
        rows.append(rec)
        print('[%s] done in %.0fs' % (name, rec['secs']), flush=True)
        ag.close()
        del m
        torch.cuda.empty_cache()
        slib.save('s7_robust.json', rows)


if __name__ == '__main__':
    main()
