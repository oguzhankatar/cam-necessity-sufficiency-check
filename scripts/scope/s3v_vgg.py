"""S3v -- VGG16 needs its own script, and the reason is the finding.

timm's `vgg16` has NO mean pool in its readout.  `forward_head` is

    fc6 = Conv2d(512, 4096, kernel_size=7)   <-- consumes the whole 7x7 grid
    ReLU -> fc7 = Conv2d(4096, 4096, 1) -> ReLU -> avgpool(1 cell) -> fc8

so the aggregation over the 49 cells is a DENSE, position-weighted linear map,
not a mean.  A dense linear map is itself additive over the cells
(z = sum_t W_t x_t + b), so the blocker is not the aggregator but the ReLU that
follows it.  There is no "move the pool" edit to make here because there is no
pool; the only single edit available is to delete (or linearise) the two
classifier ReLUs, after which fc6-fc7-fc8 collapse to one dense linear map over
the cells and the readout is exactly additive.

Same estimator as everywhere else: rtlib, literal loop, float64.  Accuracy on
the full 50 000-image val set.
"""
import os, sys, time, warnings, numpy as np, torch, torch.nn as nn
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import slib
sys.path.insert(0, os.path.join(slib.ROOT, 'scripts', 'crosscheck'))
import rtlib
import timm
from torch.utils.data import DataLoader, Subset

NEVAL = int(os.environ.get('NEVAL', 50000))
NPROBE = int(os.environ.get('NPROBE', 8))
ACTS = ['pre_logits.act1', 'pre_logits.act2']


class LinAct(nn.Module):
    def __init__(s, a, b):
        super().__init__()
        s.register_buffer('a', a)
        s.register_buffer('b', b)

    def forward(s, x):
        return x * s.a.view(1, -1, 1, 1) + s.b.view(1, -1, 1, 1)


def set_mod(m, n, new):
    p = m
    for q in n.split('.')[:-1]:
        p = getattr(p, q)
    setattr(p, n.split('.')[-1], new)


def main():
    m = timm.create_model('vgg16', pretrained=True).eval().cuda()
    ds = slib.val_ds(m.pretrained_cfg)
    rs = np.random.RandomState(0)
    perm = rs.permutation(len(ds))
    mods = dict(m.named_modules())
    orig = {n: mods[n] for n in ACTS}
    print('[vgg16] aggregator = %s' % m.pre_logits.fc1, flush=True)

    # ---- affine surrogate for each ReLU (inplace-safe: pre-hook for the input)
    buf = {n: [[], []] for n in ACTS}
    hs = []

    def flat(t):
        t = t.detach().float()
        return t.permute(0, 2, 3, 1).reshape(-1, t.shape[1]).cpu()

    def mk_pre(n):
        def f(mod, inp):
            buf[n][0].append(flat(inp[0].clone()))
        return f

    def mk_post(n):
        def f(mod, inp, out):
            buf[n][1].append(flat(out.clone()))
        return f
    for n in ACTS:
        hs.append(mods[n].register_forward_pre_hook(mk_pre(n)))
        hs.append(mods[n].register_forward_hook(mk_post(n)))
    with torch.no_grad():
        for xb, _ in DataLoader(Subset(ds, perm[:1024].tolist()), batch_size=32,
                                num_workers=8):
            m(xb.cuda())
    for h in hs:
        h.remove()
    lin_par, r2 = {}, {}
    for n in ACTS:
        u = torch.cat(buf[n][0]).double()
        w = torch.cat(buf[n][1]).double()
        mu, mw = u.mean(0), w.mean(0)
        a = ((u - mu) * (w - mw)).mean(0) / (((u - mu) ** 2).mean(0) + 1e-12)
        b = mw - a * mu
        lin_par[n] = (a.cuda(), b.cuda())
        r2[n] = float((1 - ((w - (a * u + b)) ** 2).mean(0)
                       / (w.var(0) + 1e-30)).mean())
    print('  linact per-channel R2 = %s' % {k: round(v, 4) for k, v in r2.items()},
          flush=True)
    del buf

    CFGS = ['shipped', 'delact', 'linact']

    def apply_cfg(c, dbl=False):
        for n in ACTS:
            set_mod(m, n, orig[n])
        if c == 'delact':
            for n in ACTS:
                set_mod(m, n, nn.Identity())
        if c == 'linact':
            for n in ACTS:
                a, b = lin_par[n]
                set_mod(m, n, LinAct(a.double() if dbl else a.float(),
                                     b.double() if dbl else b.float()))

    # ------------------------------------------------ rtlib additivity, fp64
    res = {}
    m = m.double()
    orig = {n: dict(m.named_modules())[n] for n in ACTS}
    xp = torch.stack([ds[i][0] for i in perm[:NPROBE + 4]]).cuda().double()
    with torch.no_grad():
        F = m.forward_features(xp)                    # (B,512,7,7)
    B, C, H, W = F.shape
    T = H * W
    X = F.reshape(B, C, T).transpose(1, 2).contiguous()   # (B,T,C)

    def head(Z):                                      # (b,T,C) -> logits
        return m.forward_head(Z.transpose(1, 2).reshape(-1, C, H, W))
    with torch.no_grad():
        pred = head(X[:NPROBE]).argmax(1).cpu().numpy()
    for c in CFGS:
        apply_cfg(c, dbl=True)
        rr, rs2 = [], np.random.RandomState(1)
        for q in range(NPROBE):
            rr.append(rtlib.report('vgg16/%s/%d' % (c, q), head, X[q],
                                   X[NPROBE + q % 4], int(pred[q]), rs2))
        e = dict(maxNS=float(np.nanmax([r['max_abs_diff_rel'] for r in rr])),
                 Rhalf=float(np.nanmax([r['Rhalf'] for r in rr])),
                 I2=float(np.nanmax([r['I2'] for r in rr])),
                 corrNS_min=float(np.nanmin([r['corr_pearson'] for r in rr])),
                 corrNS_mean=float(np.nanmean([r['corr_pearson'] for r in rr])))
        e['additive'] = bool(e['maxNS'] < 1e-9 and e['Rhalf'] < 1e-9 and e['I2'] < 1e-9)
        res[c] = e
        print('  [%-7s] maxNS=%.2e Rhalf=%.2e I2=%.2e corrNS>=%.6f ADDITIVE=%s'
              % (c, e['maxNS'], e['Rhalf'], e['I2'], e['corrNS_min'], e['additive']),
              flush=True)
    apply_cfg('shipped', dbl=True)        # restore BEFORE re-snapshotting `orig`,
    m = m.float()                         # otherwise the last fp64 cfg becomes
    orig = {n: dict(m.named_modules())[n] for n in ACTS}   # the "shipped" model
    for n in ACTS:
        a, b = lin_par[n]
        lin_par[n] = (a.float(), b.float())

    # ------------------------------------------------------- accuracy, 50 000
    nev = min(NEVAL, len(ds))
    ev = np.arange(len(ds)) if nev >= len(ds) else np.sort(perm[:nev])
    ld = DataLoader(Subset(ds, ev.tolist()), batch_size=64, num_workers=8,
                    pin_memory=True)
    hit = {c: 0 for c in CFGS}
    n = 0
    with torch.no_grad():
        for xb, yb in ld:
            f = m.forward_features(xb.cuda(non_blocking=True))
            for c in CFGS:
                apply_cfg(c)
                hit[c] += (m.forward_head(f).argmax(1).cpu() == yb).sum().item()
            n += len(yb)
            if n % 12800 == 0:
                print('   acc %6d/%d  ' % (n, nev)
                      + ' '.join('%s=%.4f' % (c, hit[c] / n) for c in CFGS), flush=True)
    acc = {c: hit[c] / n for c in CFGS}
    for c in CFGS:
        res[c]['acc'] = acc[c]
        res[c]['dacc_pts'] = 100 * (acc[c] - acc['shipped'])
    print('  ' + '  '.join('%s:%.4f(%+.2f)' % (c, acc[c], 100 * (acc[c] - acc['shipped']))
                           for c in CFGS), flush=True)
    slib.save('s3v_vgg16.json',
              [dict(model='vgg16', aggregator='pre_logits.fc1 Conv2d(512,4096,k=7) DENSE',
                    T=T, C=C, n_eval=n, acts=ACTS, linact_r2=r2, cfgs=res,
                    reaching_edits=[c for c in CFGS if c != 'shipped' and res[c]['additive']])])


if __name__ == '__main__':
    main()
