"""PROBE 10 -- REPLICATE the AND/OR fan on a different BODY and/or DATASET.

Stage 1 established the fan on exactly one setting: ResNet50 layer4 features,
imagenette, frozen body, LSE-T head.  That single setting is the weakest link
in the whole project, so this script re-runs the *identical* estimator code
with the body and the dataset as free variables.

Usage:
  BODY=convnext_t DS=imagenette python scripts/probe10_fan.py
  BODY=vgg16      DS=imagenette python scripts/probe10_fan.py
  BODY=resnet50   DS=imagewoof  python scripts/probe10_fan.py
  BODY=resnet50   DS=cifar100 MAXPC=250 SIZE=160 python scripts/probe10_fan.py

The mandatory internal control is the T = 0 row: LSE at T = 0 is algebraically
GAP + FC, so it must return I2 = 0.0000, Rhalf = 0.0000, corrNS = +1.0000,
spNS = +1.000, del_N == del_S and ins_N == ins_S.  If it does not, the
measurement is broken and every other row is void.
"""
import os, sys, json, time, numpy as np, torch
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nslib as L

BODY = os.environ.get('BODY', 'convnext_t')
DS = os.environ.get('DS', 'imagenette')
NIMG = int(os.environ.get('NIMG', 32))
EPOCHS = int(os.environ.get('EPOCHS', 8))
MAXPC = os.environ.get('MAXPC')
SIZE = os.environ.get('SIZE')
TS = [float(t) for t in os.environ.get(
    'TS', '-8,-4,-2,-1,-0.5,0,0.5,1,2,4,8').split(',')]
TAG = f'{BODY}_{DS}'


def main():
    mp = int(MAXPC) if MAXPC else None
    sz = int(SIZE) if SIZE else None
    ftr, ytr, C, T = L.cache_feats(BODY, DS, 'train', size=sz, max_per_class=mp)
    fva, yva, _, _ = L.cache_feats(BODY, DS, 'val', size=sz)
    Ftr, Ytr = L.load_cached(ftr, ytr); Fva, Yva = L.load_cached(fva, yva)
    nclass = int(Ytr.max()) + 1
    print(f'[{TAG}] C={C} cells={T} classes={nclass} '
          f'train={len(Ytr)} val={len(Yva)}', flush=True)

    rs = np.random.RandomState(0)
    sel = rs.permutation(len(Yva))[:NIMG + 48]
    Xi = L.batch(Fva, np.sort(sel[:NIMG]))
    bank = L.batch(Fva, np.sort(sel[NIMG + 16:NIMG + 48]))

    res = {'meta': dict(body=BODY, ds=DS, C=C, cells=T, nclass=nclass,
                        n_train=int(len(Ytr)), n_val=int(len(Yva)),
                        n_img=NIMG, epochs=EPOCHS)}
    hdr = (f"{'T':>6s}{'acc':>8s}{'I2':>8s}{'Rhalf':>8s}{'corrNS':>9s}{'spNS':>8s}"
           f"{'spGN':>8s}{'spGS':>8s} | {'delN':>6s}{'delS':>6s}{'delG':>6s}"
           f"{'delR':>6s} | {'insN':>6s}{'insS':>6s}{'insG':>6s}{'insR':>6s}")
    print(hdr, flush=True)
    for Tt in TS:
        t0 = time.time()
        name = f'lse{Tt}'
        h, acc, P, _ = L.train_head(name, Ftr, Ytr, Fva, Yva, nclass, C,
                                    epochs=EPOCHS)
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
        r = dict(T=Tt, acc=acc, I2=float(np.mean(I2s)), Rhalf=float(np.mean(Rhs)),
                 corrNS=float(np.mean(cs)) if cs else float('nan'),
                 spNS=float(np.mean(sns)), spGN=float(np.mean(sgn)),
                 spGS=float(np.mean(sgs)), Nmag_rel=float(np.mean(mags)),
                 ece=L.ece(P, Yva), sec=time.time() - t0)
        for t in D:
            r[f'del_{t}'] = float(np.mean(D[t])); r[f'ins_{t}'] = float(np.mean(I[t]))
        for t in ('N', 'S', 'G'):
            r[f'del_{t}_ci'] = L.boot_ci(D[t])[1:]
            r[f'ins_{t}_ci'] = L.boot_ci(I[t])[1:]
        res[name] = r
        print(f"{Tt:6.2f}{acc:8.4f}{r['I2']:8.4f}{r['Rhalf']:8.4f}{r['corrNS']:+9.4f}"
              f"{r['spNS']:+8.3f}{r['spGN']:+8.3f}{r['spGS']:+8.3f} | "
              f"{r['del_N']:6.3f}{r['del_S']:6.3f}{r['del_G']:6.3f}{r['del_rand']:6.3f}"
              f" | {r['ins_N']:6.3f}{r['ins_S']:6.3f}{r['ins_G']:6.3f}"
              f"{r['ins_rand']:6.3f}  ({r['sec']:.0f}s)", flush=True)
        L.save(f'probe10_fan_{TAG}.json', res)

    # ---- internal control -------------------------------------------------
    z = res.get('lse0.0')
    if z is not None:
        # tolerance 1e-4: the estimator is exact, but the features are cached
        # fp16 -> fp32 so the residual is float round-off, not model structure.
        ok = (abs(z['I2']) < 1e-4 and abs(z['Rhalf']) < 1e-4
              and abs(z['corrNS'] - 1) < 1e-6
              and abs(z['del_N'] - z['del_S']) < 1e-9
              and abs(z['ins_N'] - z['ins_S']) < 1e-9)
        res['control_T0_exact'] = bool(ok)
        print(f"\nCONTROL T=0 exact-GAP degeneracy: {'PASS' if ok else 'FAIL'}"
              f"  (I2={z['I2']:.2e} Rhalf={z['Rhalf']:.2e} "
              f"corrNS={z['corrNS']:.6f} delN-delS={z['del_N']-z['del_S']:.2e})")
        L.save(f'probe10_fan_{TAG}.json', res)


if __name__ == '__main__':
    main()
