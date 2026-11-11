"""PROBE 29 (P4) -- THE IMAGENET-1k ZERO-TRAINING FAN, RE-MEASURED AT n >= 200.

Stage 3's Probe 20 measured the AND/OR fan on the real 1000-class ILSVRC-2012
validation set with the shipped torchvision ResNet50 `fc` dropped into the LSE
head at every temperature -- so the ONLY difference between rows is the pooling
operator and no training happens at all.  It used 32 images.  This re-runs the
identical estimator code on 224 images with bootstrap 95 % CIs over images for
every column, so the proposal's §3 table can be quoted with error bars.

Both conditions:
  A  shipped `fc`, zero training           (the headline condition)
  B  heads re-fit at every T from the shipped `fc` init

Mandatory internal control: T = 0 is algebraically GAP + FC, so it must return
corrNS = +1.000000, I2 = Rhalf = 0 and del_N == del_S, ins_N == ins_S.
"""
import os, sys, time, numpy as np, torch
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nslib as L
import probe20_in1k as P20

NIMG = int(os.environ.get('NIMG', 224))
TS = [float(t) for t in os.environ.get('TS', '-8,-1,0,1,8').split(',')]
EPOCHS = int(os.environ.get('EPOCHS', 12))
COND = os.environ.get('COND', 'AB')
dev = 'cuda'
KEYS = ('I2', 'Rhalf', 'corrNS', 'spNS', 'spGN', 'spGS', 'Nmag_rel')


def measure_ci(h, Xi, bank, T):
    """Probe 20's `measure`, but keeping the per-image values for bootstrap."""
    rs2 = np.random.RandomState(1)
    acc = {k: [] for k in KEYS}
    D = {k: [] for k in ('N', 'S', 'G', 'rand')}
    I = {k: [] for k in ('N', 'S', 'G', 'rand')}
    for k in range(Xi.shape[0]):
        A = Xi[k]
        with torch.no_grad():
            c = int(h(A[None]).argmax(1))
        B = bank[rs2.randint(bank.shape[0])]
        i2, rh, cc, mg, zm = L.interaction(h, A, B, c, rs2)
        acc['I2'].append(i2); acc['Rhalf'].append(rh)
        acc['Nmag_rel'].append(mg / (zm + 1e-12))
        acc['corrNS'].append(cc)
        N, S = L.maps(h, A, B, c); G = L.gradcam_head(h, A, c)
        n, s, g = [v.cpu().numpy() for v in (N, S, G)]
        acc['spNS'].append(spearmanr(n, s).statistic
                           if n.std() > 1e-9 and s.std() > 1e-9 else np.nan)
        acc['spGN'].append(spearmanr(g, n).statistic
                           if n.std() > 1e-9 and g.std() > 1e-9 else np.nan)
        acc['spGS'].append(spearmanr(g, s).statistic
                           if s.std() > 1e-9 and g.std() > 1e-9 else np.nan)
        rnd = torch.randn(A.shape[1], device=A.device)
        for tag, o in (('N', N), ('S', S), ('G', G), ('rand', rnd)):
            D[tag].append(L.auc(h, A, B, c, o, 'del'))
            I[tag].append(L.auc(h, A, B, c, o, 'ins'))
    r = {'T': T}
    for k, v in acc.items():
        v = [z for z in v if np.isfinite(z)]
        mu, lo, hi = L.boot_ci(v)
        r[k] = mu; r[k + '_ci'] = [lo, hi]; r[k + '_n'] = len(v)
    for t in D:
        for tag, arr in (('del', D[t]), ('ins', I[t])):
            mu, lo, hi = L.boot_ci(arr)
            r[f'{tag}_{t}'] = mu; r[f'{tag}_{t}_ci'] = [lo, hi]
    return r


HDR = (f"{'T':>6s}{'acc':>8s}{'corrNS':>10s}{'[95% CI]':>20s}{'Rhalf':>9s}"
       f"{'[95% CI]':>18s}{'I2':>8s}{'spGN':>8s}{'spGS':>8s} | "
       f"{'delN':>6s}{'delS':>6s}{'delG':>6s}{'delR':>6s}")


def row(r):
    return (f"{r['T']:6.2f}{r['acc']:8.4f}{r['corrNS']:+10.6f}"
            f"  [{r['corrNS_ci'][0]:+.4f},{r['corrNS_ci'][1]:+.4f}]"
            f"{r['Rhalf']:9.3f}  [{r['Rhalf_ci'][0]:.3f},{r['Rhalf_ci'][1]:.3f}]"
            f"{r['I2']:8.4f}{r['spGN']:+8.3f}{r['spGS']:+8.3f} | "
            f"{r['del_N']:6.3f}{r['del_S']:6.3f}{r['del_G']:6.3f}"
            f"{r['del_rand']:6.3f}")


def main():
    ftr, ytr, C, T = L.cache_feats('resnet50', 'in1k', 'train', size=224,
                                   bs=64, workers=8)
    fva, yva, _, _ = L.cache_feats('resnet50', 'in1k', 'val', size=224,
                                   bs=64, workers=8)
    Ftr, Ytr = L.load_cached(ftr, ytr); Fva, Yva = L.load_cached(fva, yva)
    ncl = int(Ytr.max()) + 1
    W, b = P20.shipped_fc()
    res = {'meta': dict(body='resnet50', ds='in1k', C=C, cells=T, nclass=ncl,
                        n_img=NIMG, n_train=int(len(Ytr)), n_val=int(len(Yva)),
                        epochs=EPOCHS, init='shipped')}
    h0 = P20.make_head(0.0, ncl, C, W, b)
    a0, _ = P20.evaluate(h0, Fva, Yva)
    res['shipped_gap_top1_eval'] = a0
    print(f'[in1k] C={C} cells={T} classes={ncl} n_img={NIMG}; SANITY shipped '
          f'ResNet50 GAP+fc top-1 = {a0:.4f} (published .7613, this mirror '
          f'is short-side-256 JPEG q92)', flush=True)
    rs = np.random.RandomState(0)
    sel = rs.permutation(len(Yva))[:NIMG + 48]
    Xi = L.batch(Fva, np.sort(sel[:NIMG]))
    bank = L.batch(Fva, np.sort(sel[NIMG + 16:NIMG + 48]))
    if 'A' in COND:
        print('\n=== A. SHIPPED fc, ZERO TRAINING, only the pooling operator '
              'differs ===')
        print(HDR, flush=True)
        for Tt in TS:
            t0 = time.time()
            h = P20.make_head(Tt, ncl, C, W, b).eval()
            acc, _ = P20.evaluate(h, Fva, Yva)
            r = measure_ci(h, Xi, bank, Tt); r['acc'] = acc
            r['sec'] = time.time() - t0
            res[f'ship_lse{Tt}'] = r
            print(row(r) + f"  ({r['sec']:.0f}s)", flush=True)
            L.save('probe29_fan200.json', res)
    if 'B' in COND:
        print(f'\n=== B. heads RE-FIT at each T ({EPOCHS} epochs, shipped '
              f'init) ===')
        print(HDR, flush=True)
        for Tt in TS:
            t0 = time.time()
            h = P20.train(P20.make_head(Tt, ncl, C, W, b), Ftr, Ytr, EPOCHS)
            acc, P = P20.evaluate(h, Fva, Yva)
            r = measure_ci(h, Xi, bank, Tt); r['acc'] = acc
            r['ece'] = L.ece(P, Yva); r['sec'] = time.time() - t0
            res[f'lse{Tt}'] = r
            print(row(r) + f"  ({r['sec']:.0f}s)", flush=True)
            L.save('probe29_fan200.json', res)
    for pre, lab in (('ship_', 'A zero-training'), ('', 'B re-fit')):
        z = res.get(f'{pre}lse0.0')
        if z is None:
            continue
        ok = (abs(z['I2']) < 1e-4 and abs(z['Rhalf']) < 1e-4
              and abs(z['corrNS'] - 1) < 1e-6
              and abs(z['del_N'] - z['del_S']) < 1e-6)
        res[f'control_T0_{pre or "refit"}'] = bool(ok)
        print(f"CONTROL T=0 ({lab}): {'PASS' if ok else 'FAIL'} "
              f"corrNS={z['corrNS']:.8f} I2={z['I2']:.2e} Rhalf={z['Rhalf']:.2e} "
              f"del_N-del_S={z['del_N']-z['del_S']:.2e}", flush=True)
    L.save('probe29_fan200.json', res)


if __name__ == '__main__':
    main()
