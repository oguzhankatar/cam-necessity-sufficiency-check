"""S3 -- ONE-EDIT REACHABILITY for the architectures whose shipped readout is
non-additive over cells.

For each model we try a small set of SINGLE edits to the head and ask
  (a) does the readout become ADDITIVE over the cells, certified DIRECTLY, and
  (b) what does the edit cost on the full 50 000-image ILSVRC-2012 val set.

EDITS (each is one line where the head is written):
  shipped   z = g(mean_t x_t)                        reference
  poollast  z = mean_t g(x_t)                        move the WHOLE post-pool map
            inside the pool.  For a norm+linear head this IS the project's `pre`
            ordering; for the mobile head pool->1x1conv->act->fc it is exactly
            "apply the activation before the pool" (the 1x1 conv commutes with
            the mean, so nothing else changes).
  delact    every post-pool activation -> Identity.  For the mobile head this is
            literally "fold the 1x1 conv into the classifier".
  delnorm   every post-pool normalisation -> Identity (stage 4's one-line fix).
  delboth   both.
  linact    every post-pool activation -> the per-channel affine map fitted by
            least squares to its own input/output over 2048 val images, i.e.
            "replace the activation by its affine part".

TWO CRITERIA, and the difference between them matters.

  dev_seg : max_lam |g(m_A + lam(m_B - m_A)) - chord| / |g(m_B) - g(m_A)|.
            This is the cross-check's architecture-agnostic test.  It is the right
            criterion for a head of the form z = g(mean), because the cell-swap
            lattice lives in the affine hull of that segment -- but it is the
            WRONG criterion for `poollast`, whose head is z = mean_t g(x_t):
            that head is additive over cells by construction while being a
            genuinely nonlinear function of lam.  Reported, flagged n/a there.

  ADDITIVITY, certified directly with the cross-check's own independent float64
            literal-loop estimator (scripts/crosscheck/rtlib.py): N(t), S(t) built
            by rebuilding the whole perturbed cell matrix and re-running the
            head, then
               maxNS  = max_t |N(t)-S(t)| / scale      (the identity itself)
               Rhalf  = E_H |Delta(H) - sum_{t in H} N(t)| / E|Delta(H)|
               I2     = the same over random cell PAIRS.
            corrNS is recorded but is NOT used as the certificate (the cross-check
            built non-additive heads that return corrNS = +1.00000000).
"""
import os, sys, json, time, argparse, warnings, numpy as np, torch, torch.nn as nn
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import slib, shead
sys.path.insert(0, os.path.join(slib.ROOT, 'scripts', 'crosscheck'))
import rtlib
import timm
from torch.utils.data import DataLoader, Subset

TOL = 1e-11
ap = argparse.ArgumentParser()
ap.add_argument('models', nargs='+')
ap.add_argument('--neval', type=int, default=50000)
ap.add_argument('--nprobe', type=int, default=8)
ap.add_argument('--bs', type=int, default=64)
ap.add_argument('--workers', type=int, default=8)
ap.add_argument('--tag', default='s3_edits')
A = ap.parse_args()


def leaf_order(m, x):
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
        m(x)
    for h in hs:
        h.remove()
    return ev


class LinAct(nn.Module):
    """y = a*x + b, per channel: the least-squares affine part of the activation."""

    def __init__(s, a, b):
        super().__init__()
        s.register_buffer('a', a)
        s.register_buffer('b', b)

    def forward(s, x):
        if x.dim() == 4:
            return x * s.a.view(1, -1, 1, 1) + s.b.view(1, -1, 1, 1)
        return x * s.a + s.b


def run(name):
    t0 = time.time()
    m = timm.create_model(name, pretrained=True).eval().cuda()
    cfg = m.pretrained_cfg
    ds = slib.val_ds(cfg)
    rs = np.random.RandomState(0)
    perm = rs.permutation(len(ds))
    x2 = torch.stack([ds[i][0] for i in perm[:2]]).cuda()
    fam = type(m).__name__

    if fam == 'MambaOut':
        ag = shead.AggMamba(m, x2)
        pool_nm = ag.pool_name
        pnames = ['head.' + n for n, mod in m.head.named_modules()
                  if n and len(list(mod.children())) == 0]
    else:
        ag = shead.Agg(m, x2)
        pool_nm = ag.pool_name
        ev = leaf_order(m, x2)
        pnames = ev[ev.index(pool_nm) + 1:]
    print('[%s] family=%s pool=%s' % (name, fam, pool_nm), flush=True)

    X2 = ag.cells(x2)
    T, C = X2.shape[1], X2.shape[2]

    def POST(v):
        if isinstance(ag, shead.AggMamba):
            return ag.post(v)
        return ag.post(v, torch.zeros(v.shape[0], C, 1, 1, device=v.device,
                                      dtype=v.dtype))

    with torch.no_grad():
        ref = m(x2)
        comp = POST(X2.mean(1))
    cerr = float((ref - comp).abs().max() / (ref.abs().max() + 1e-12))
    print('  compose check: rel err %.2e %s  T=%d C=%d'
          % (cerr, 'PASS' if cerr < 2e-4 else 'FAIL', T, C), flush=True)
    assert cerr < 2e-4, 'hook adapter does not reproduce the shipped model'

    mods = dict(m.named_modules())
    act_nm = [n for n in pnames if n in mods and slib.is_act(mods[n])]
    norm_nm = [n for n in pnames if n in mods and slib.is_norm(mods[n])]
    other = [n for n in pnames if n in mods
             and not slib.is_act(mods[n]) and not slib.is_norm(mods[n])]
    print('  post-pool leaves=%d  acts=%s  norms=%s' % (len(pnames), act_nm, norm_nm),
          flush=True)

    def set_mod(n, new):
        parent = m
        parts = n.split('.')
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], new)

    # ---------------------------------------------------- linact calibration
    lin_par = {}
    r2 = {}
    if act_nm:
        # NOTE: many timm activations are inplace=True, so a plain forward hook
        # sees inp[0] IS out and the fitted map comes out as the identity.
        # Capture the input in a PRE-hook, cloned.
        buf = {n: [[], []] for n in act_nm}
        hs = []

        def flat(t):
            t = t.detach().float()
            ch = t.shape[1] if t.dim() == 4 else t.shape[-1]
            return (t.permute(0, 2, 3, 1).reshape(-1, ch) if t.dim() == 4
                    else t.reshape(-1, ch)).cpu()

        def mk_pre(n):
            def f(mod, inp):
                buf[n][0].append(flat(inp[0].clone()))
            return f

        def mk_post(n):
            def f(mod, inp, out):
                buf[n][1].append(flat(out.clone()))
            return f
        for n in act_nm:
            hs.append(mods[n].register_forward_pre_hook(mk_pre(n)))
            hs.append(mods[n].register_forward_hook(mk_post(n)))
        ld = DataLoader(Subset(ds, perm[:2048].tolist()), batch_size=A.bs,
                        num_workers=A.workers)
        with torch.no_grad():
            for xb, _ in ld:
                m(xb.cuda(non_blocking=True))
        for h in hs:
            h.remove()
        for n in act_nm:
            u = torch.cat(buf[n][0]).double()
            w = torch.cat(buf[n][1]).double()
            mu, mw = u.mean(0), w.mean(0)
            a = ((u - mu) * (w - mw)).mean(0) / (((u - mu) ** 2).mean(0) + 1e-12)
            b = mw - a * mu
            lin_par[n] = (a.cuda(), b.cuda())
            resid = ((w - (a * u + b)) ** 2).mean(0)
            r2[n] = float((1 - resid / (w.var(0) + 1e-30)).mean())
        print('  linact per-channel R2 = %s' % {k: round(v, 4) for k, v in r2.items()},
              flush=True)

    CFGS = ['shipped', 'poollast', 'delact', 'delnorm', 'delboth', 'linact']
    if not act_nm:
        CFGS = [c for c in CFGS if c not in ('delact', 'linact', 'delboth')]
    if not norm_nm:
        CFGS = [c for c in CFGS if c not in ('delnorm', 'delboth')]

    def snap():
        mm = dict(m.named_modules())
        return {n: mm[n] for n in set(act_nm + norm_nm)}

    orig = snap()

    def restore():
        for n, o in orig.items():
            set_mod(n, o)

    def apply_cfg(c, dbl=False):
        restore()
        if c in ('delact', 'delboth'):
            for n in act_nm:
                set_mod(n, nn.Identity())
        if c in ('delnorm', 'delboth'):
            for n in norm_nm:
                set_mod(n, nn.Identity())
        if c == 'linact':
            for n in act_nm:
                a, b = lin_par[n]
                set_mod(n, LinAct(a.double() if dbl else a.float(),
                                  b.double() if dbl else b.float()))

    def head_c(c):
        if c == 'poollast':
            def h(X):
                B = X.shape[0]
                return POST(X.reshape(B * T, C)).reshape(B, T, -1).mean(1)
        else:
            def h(X):
                return POST(X.mean(1))
        return h

    # ------------------------------------------------ fp64: segment + rtlib
    res = {}
    m = m.double()
    orig = snap()
    xp = torch.stack([ds[i][0] for i in perm[:A.nprobe + 4]]).cuda().double()
    with torch.no_grad():
        XP = ag.cells(xp)
    XB_bank = XP[A.nprobe:]
    with torch.no_grad():
        pred = POST(XP[:A.nprobe].mean(1)).argmax(1).cpu().numpy()

    for c in CFGS:
        apply_cfg(c, dbl=True)
        h = head_c(c)
        if c == 'poollast':
            dseg = None
        else:
            dseg, _ = slib.seg_dev(lambda Xq: POST(Xq.mean(1)),
                                   XP[0:1], XP[1:2])
        rr = []
        rr_rs = np.random.RandomState(1)
        for q in range(A.nprobe):
            XA = XP[q]
            XB = XB_bank[q % XB_bank.shape[0]]
            d = rtlib.report('%s/%s/%d' % (name, c, q), h, XA, XB,
                             int(pred[q]), rr_rs)
            rr.append(d)
        agg = {k: float(np.nanmax([abs(r[k]) for r in rr]))
               for k in ('max_abs_diff_rel', 'Rhalf', 'I2')}
        agg['corrNS_min'] = float(np.nanmin([r['corr_pearson'] for r in rr]))
        agg['corrNS_mean'] = float(np.nanmean([r['corr_pearson'] for r in rr]))
        add = agg['max_abs_diff_rel'] < 1e-9 and agg['Rhalf'] < 1e-9 and agg['I2'] < 1e-9
        res[c] = dict(dev_seg_fp64=dseg, additive=bool(add), **agg)
        print('  [%-9s] dev_seg=%s  maxNS=%.2e Rhalf=%.2e I2=%.2e corrNS>=%.6f'
              '  ADDITIVE=%s'
              % (c, 'n/a    ' if dseg is None else '%.2e' % dseg,
                 agg['max_abs_diff_rel'], agg['Rhalf'], agg['I2'],
                 agg['corrNS_min'], add), flush=True)
    restore()
    m = m.float()
    orig = snap()
    for n in list(lin_par):
        a, b = lin_par[n]
        lin_par[n] = (a.float(), b.float())

    # --------------------------------------------------- accuracy, full 50 k
    nev = min(A.neval, len(ds))
    ev = np.arange(len(ds)) if nev >= len(ds) else np.sort(perm[:nev])
    ld = DataLoader(Subset(ds, ev.tolist()), batch_size=A.bs,
                    num_workers=A.workers, pin_memory=True)
    hit = {c: 0 for c in CFGS}
    n = 0
    with torch.no_grad():
        for xb, yb in ld:
            xb = xb.cuda(non_blocking=True)
            Xc = ag.cells(xb)
            for c in CFGS:
                apply_cfg(c)
                hit[c] += (head_c(c)(Xc).argmax(1).cpu() == yb).sum().item()
            n += len(yb)
            if n % 12800 == 0:
                print('   acc %6d/%d  ' % (n, nev) +
                      ' '.join('%s=%.4f' % (c, hit[c] / n) for c in CFGS), flush=True)
    restore()
    acc = {c: hit[c] / n for c in CFGS}
    for c in CFGS:
        res[c]['acc'] = acc[c]
        res[c]['dacc_pts'] = 100 * (acc[c] - acc['shipped'])
    reach = [c for c in CFGS if c != 'shipped' and res[c]['additive']]
    best = max(reach, key=lambda c: acc[c]) if reach else None
    out = dict(model=name, family=fam, pool=pool_nm, T=T, C=C,
               compose_rel_err=cerr, n_eval=n, post_leaves=pnames,
               acts=act_nm, norms=norm_nm, other=other, linact_r2=r2,
               cfgs=res, reachable=bool(reach), reaching_edits=reach,
               best_edit=best,
               best_cost_pts=None if best is None else res[best]['dacc_pts'],
               secs=time.time() - t0)
    print('  ' + '  '.join('%s:%.4f(%+.2f)' % (c, acc[c], 100 * (acc[c] - acc['shipped']))
                           for c in CFGS), flush=True)
    print('  REACHABLE=%s  best=%s  cost=%s pts'
          % (bool(reach), best, out['best_cost_pts']), flush=True)
    ag.close()
    del m
    torch.cuda.empty_cache()
    return out


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
