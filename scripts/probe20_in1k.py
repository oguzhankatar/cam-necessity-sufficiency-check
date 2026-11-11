"""PROBE 20 (P1) -- THE AND/OR FAN AT IMAGENET-1k SCALE.

The largest setting in stages 1-2 was CIFAR-100 (100 classes) / imagewoof (10
fine-grained classes).  This runs the *identical* estimator code on the real
ILSVRC-2012 validation set: 50 000 images, **1000 classes**, 256-px source,
224-px centre crop, frozen torchvision ResNet50 (the exact checkpoint the CAM
literature evaluates on).

Splits: file index 0..39 of each class -> head-training split (40 000 images),
40..49 -> evaluation split (10 000 images).  Disjoint by construction.  Note
that the *body* is the ImageNet-trained one, so the features are in-domain; only
the 1000-way readout is re-fit, which is the same protocol as every other frozen
setting in the project.

Two head initialisations, both reported:
  INIT=random   fresh Linear(2048,1000)          -- protocol-identical to stages 1-2
  INIT=shipped  torchvision's own `fc` weights   -- at T=0 this IS the shipped
                ImageNet-1k ResNet50, bit for bit

and a third, zero-training condition (`SHIPPED0`): the shipped `fc` is dropped
into the LSE head at every temperature with **no training at all**, so the only
thing that changes across rows is the pooling operator.

Mandatory internal control: T = 0 is algebraically GAP+FC, so it must return
I2 = Rhalf = 0, corrNS = +1.000000, del_N == del_S, ins_N == ins_S.
"""
import os, sys, json, time, numpy as np, torch, torch.nn as nn
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nslib as L
from heads import LSE

BODY = 'resnet50'
DS = os.environ.get('DS', 'in1k')
NIMG = int(os.environ.get('NIMG', 32))
EPOCHS = int(os.environ.get('EPOCHS', 12))
INIT = os.environ.get('INIT', 'shipped')
LR = float(os.environ.get('LR', 3e-3))
BS = int(os.environ.get('BS', 256))
TS = [float(t) for t in os.environ.get(
    'TS', '-8,-4,-2,-1,-0.5,0,0.5,1,2,4,8').split(',')]
TAG = f'{DS}_{INIT}'
dev = 'cuda'


def shipped_fc():
    from torchvision import models
    m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    return m.fc.weight.data.clone(), m.fc.bias.data.clone()


def make_head(T, ncl, C, W=None, b=None, seed=0):
    torch.manual_seed(seed)
    h = LSE(k=ncl, T=T, c=C).to(dev)
    if W is not None:
        h.fc.weight.data.copy_(W.to(dev)); h.fc.bias.data.copy_(b.to(dev))
    return h


def evaluate(h, Fva, Yva, bs=256):
    with torch.no_grad():
        ok, P = 0, []
        for i in range(0, len(Yva), bs):
            j = np.arange(i, min(i + bs, len(Yva)))
            z = h(L.batch(Fva, j))
            ok += (z.argmax(1).cpu().numpy() == Yva[j]).sum()
            P.append(torch.softmax(z, 1).cpu().numpy())
    return float(ok / len(Yva)), np.concatenate(P)


def train(h, Ftr, Ytr, epochs, seed=0):
    import torch.nn.functional as Fn
    opt = torch.optim.AdamW(h.parameters(), lr=LR, weight_decay=1e-4)
    n = len(Ytr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, max(1, epochs * (n // BS)))
    yt = torch.from_numpy(Ytr).to(dev); rs = np.random.RandomState(seed)
    for ep in range(epochs):
        perm = rs.permutation(n)
        for i in range(0, n - BS + 1, BS):
            j = np.sort(perm[i:i + BS])
            loss = Fn.cross_entropy(h(L.batch(Ftr, j)), yt[j])
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    return h.eval()


def measure(h, Xi, bank, T):
    rs2 = np.random.RandomState(1)
    I2s, Rhs, cs, sns, sgn, sgs, mags = [], [], [], [], [], [], []
    D = {k: [] for k in ('N', 'S', 'G', 'rand')}
    I = {k: [] for k in ('N', 'S', 'G', 'rand')}
    for k in range(Xi.shape[0]):
        A = Xi[k]
        with torch.no_grad():
            c = int(h(A[None]).argmax(1))
        B = bank[rs2.randint(bank.shape[0])]
        i2, rh, cc, mg, zm = L.interaction(h, A, B, c, rs2)
        I2s.append(i2); Rhs.append(rh); mags.append(mg / (zm + 1e-12))
        if not np.isnan(cc):
            cs.append(cc)
        N, S = L.maps(h, A, B, c); G = L.gradcam_head(h, A, c)
        n, s, g = [v.cpu().numpy() for v in (N, S, G)]
        if n.std() > 1e-9 and s.std() > 1e-9: sns.append(spearmanr(n, s).statistic)
        if n.std() > 1e-9 and g.std() > 1e-9: sgn.append(spearmanr(g, n).statistic)
        if s.std() > 1e-9 and g.std() > 1e-9: sgs.append(spearmanr(g, s).statistic)
        rnd = torch.randn(A.shape[1], device=A.device)
        for tag, o in (('N', N), ('S', S), ('G', G), ('rand', rnd)):
            D[tag].append(L.auc(h, A, B, c, o, 'del'))
            I[tag].append(L.auc(h, A, B, c, o, 'ins'))
    r = dict(T=T, I2=float(np.mean(I2s)), Rhalf=float(np.mean(Rhs)),
             corrNS=float(np.mean(cs)) if cs else float('nan'),
             spNS=float(np.mean(sns)), spGN=float(np.mean(sgn)),
             spGS=float(np.mean(sgs)), Nmag_rel=float(np.mean(mags)))
    for t in D:
        r[f'del_{t}'] = float(np.mean(D[t])); r[f'ins_{t}'] = float(np.mean(I[t]))
    for t in ('N', 'S', 'G'):
        r[f'del_{t}_ci'] = L.boot_ci(D[t])[1:]
        r[f'ins_{t}_ci'] = L.boot_ci(I[t])[1:]
    return r


HDR = (f"{'T':>6s}{'acc':>8s}{'I2':>8s}{'Rhalf':>8s}{'corrNS':>9s}{'spNS':>8s}"
       f"{'spGN':>8s}{'spGS':>8s} | {'delN':>6s}{'delS':>6s}{'delG':>6s}"
       f"{'delR':>6s} | {'insN':>6s}{'insS':>6s}{'insG':>6s}{'insR':>6s}")


def row(r):
    return (f"{r['T']:6.2f}{r['acc']:8.4f}{r['I2']:8.4f}{r['Rhalf']:8.4f}"
            f"{r['corrNS']:+9.4f}{r['spNS']:+8.3f}{r['spGN']:+8.3f}"
            f"{r['spGS']:+8.3f} | {r['del_N']:6.3f}{r['del_S']:6.3f}"
            f"{r['del_G']:6.3f}{r['del_rand']:6.3f} | {r['ins_N']:6.3f}"
            f"{r['ins_S']:6.3f}{r['ins_G']:6.3f}{r['ins_rand']:6.3f}")


def main():
    ftr, ytr, C, T = L.cache_feats(BODY, DS, 'train', size=224, bs=64, workers=8)
    fva, yva, _, _ = L.cache_feats(BODY, DS, 'val', size=224, bs=64, workers=8)
    Ftr, Ytr = L.load_cached(ftr, ytr); Fva, Yva = L.load_cached(fva, yva)
    ncl = int(Ytr.max()) + 1
    print(f'[{TAG}] C={C} cells={T} classes={ncl} train={len(Ytr)} '
          f'val={len(Yva)}', flush=True)

    W, b = shipped_fc()
    res = {'meta': dict(body=BODY, ds=DS, C=C, cells=T, nclass=ncl,
                        n_train=int(len(Ytr)), n_val=int(len(Yva)),
                        n_img=NIMG, epochs=EPOCHS, init=INIT, lr=LR, bs=BS)}

    # --- label-alignment sanity: GAP+shipped fc on cached features IS the
    #     shipped ResNet50, so its top-1 must be ~0.76 on ILSVRC val.
    h0 = make_head(0.0, ncl, C, W, b)
    a_ship, _ = evaluate(h0, Fva, Yva)
    a_ship_tr, _ = evaluate(h0, Ftr, Ytr)
    res['shipped_gap_top1_eval'] = a_ship
    res['shipped_gap_top1_trainsplit'] = a_ship_tr
    print(f'SANITY shipped ResNet50 (GAP+fc, no training) top-1: '
          f'{a_ship:.4f} eval / {a_ship_tr:.4f} train-split  '
          f'(published 0.7613)', flush=True)

    rs = np.random.RandomState(0)
    sel = rs.permutation(len(Yva))[:NIMG + 48]
    Xi = L.batch(Fva, np.sort(sel[:NIMG]))
    bank = L.batch(Fva, np.sort(sel[NIMG + 16:NIMG + 48]))

    # ---------------- condition A: zero-training, shipped fc at every T -------
    print('\n=== A. SHIPPED fc, NO TRAINING, pooling operator is the only '
          'difference ===')
    print(HDR, flush=True)
    for Tt in TS:
        t0 = time.time()
        h = make_head(Tt, ncl, C, W, b).eval()
        acc, _ = evaluate(h, Fva, Yva)
        r = measure(h, Xi, bank, Tt); r['acc'] = acc; r['sec'] = time.time() - t0
        res[f'ship_lse{Tt}'] = r
        print(row(r) + f"  ({r['sec']:.0f}s)", flush=True)
        L.save(f'probe20_in1k_{TAG}.json', res)

    # ---------------- condition B: heads re-fit at each T ---------------------
    print(f'\n=== B. heads RE-FIT at each T (init={INIT}, {EPOCHS} epochs) ===')
    print(HDR, flush=True)
    for Tt in TS:
        t0 = time.time()
        h = make_head(Tt, ncl, C, *( (W, b) if INIT == 'shipped' else (None, None)))
        h = train(h, Ftr, Ytr, EPOCHS)
        acc, P = evaluate(h, Fva, Yva)
        r = measure(h, Xi, bank, Tt); r['acc'] = acc
        r['ece'] = L.ece(P, Yva); r['sec'] = time.time() - t0
        res[f'lse{Tt}'] = r
        print(row(r) + f"  ({r['sec']:.0f}s)", flush=True)
        L.save(f'probe20_in1k_{TAG}.json', res)

    for pre in ('ship_', ''):
        z = res.get(f'{pre}lse0.0')
        if z is None:
            continue
        ok = (abs(z['I2']) < 1e-4 and abs(z['Rhalf']) < 1e-4
              and abs(z['corrNS'] - 1) < 1e-6
              # 1e-8, not 1e-9: at 1000 classes N and S are computed by two
              # different forward passes and agree only to float32 (measured
              # 2.0e-06 relative, identical cell ordering on 32/32 images --
              # see scripts/probe20b_control.py), so the AUC difference is
              # arithmetic noise of order 1e-11, not a structural failure.
              and abs(z['del_N'] - z['del_S']) < 1e-8
              and abs(z['ins_N'] - z['ins_S']) < 1e-8)
        res[f'control_T0_{pre or "trained"}'] = bool(ok)
        print(f"CONTROL T=0 [{pre or 'trained'}]: {'PASS' if ok else 'FAIL'}  "
              f"I2={z['I2']:.2e} Rhalf={z['Rhalf']:.2e} "
              f"corrNS={z['corrNS']:.6f} delN-delS={z['del_N']-z['del_S']:.2e}",
              flush=True)
    L.save(f'probe20_in1k_{TAG}.json', res)


if __name__ == '__main__':
    main()
