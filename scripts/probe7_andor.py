"""PROBE 7 -- the OR/AND axis: which causal quantity the aggregator lets you SEE.

Probe 4 found an asymmetry nobody predicted.  As the LSE temperature T grows
(pooling -> max, an OR over positions):
    sufficiency stays sharp and keeps winning deletion AND insertion,
    necessity DEGRADES to worse than useless (del 0.316 -> 0.505).
Reading: under an OR-like pool no single cell is necessary (delete the argmax
and the runner-up takes over), but many single cells are sufficient.

PREDICTION: the mirror image must hold.  LSE with T < 0 is a soft-MIN, an AND
over positions.  There, every cell should be necessary and none sufficient, so
NECESSITY should become the informative map and SUFFICIENCY should collapse.

If the mirror holds, the pooling operator does not merely "restore interaction"
-- it SELECTS which causal quantity is measurable at all.  T = 0 (GAP) is the
exact point where the two coincide, which is why they are identical for every
standard CNN.
"""
import os, sys, json, numpy as np, torch
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(__file__))
from heads import build, LSE
import probe1_aggregator as P1
from probe1_aggregator import load, batch, train, interaction, OUT
from probe4_utility import maps, gradcam_head, auc

dev = 'cuda'
TS = [-8.0, -4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0, 8.0]


def main():
    Ftr, Ytr = load('train'); Fva, Yva = load('val')
    rs = np.random.RandomState(0)
    sel = rs.permutation(len(Yva))[:96]
    Xi = batch(Fva, np.sort(sel[:32])); bank = batch(Fva, np.sort(sel[48:80]))
    res = {}
    print(f"{'T':>6s}{'acc':>8s}{'I2':>8s}{'Rhalf':>8s}{'corrNS':>9s}{'spNS':>8s}"
          f"{'spGN':>8s}{'spGS':>8s} | {'delN':>6s}{'delS':>6s}{'delG':>6s}"
          f"{'delR':>6s} | {'insN':>6s}{'insS':>6s}{'insG':>6s}")
    for T in TS:
        name = f'lse{T}'
        P1.GROUP.setdefault(name, 'nonlin-agg')
        h, acc = train(name, Ftr, Ytr, Fva, Yva)
        rs2 = np.random.RandomState(1)
        I2s, Rhs, cs, sns, sgn, sgs = [], [], [], [], [], []
        D = {k: [] for k in ('N', 'S', 'G', 'rand')}
        I = {k: [] for k in ('N', 'S', 'G')}
        for k in range(Xi.shape[0]):
            A = Xi[k]
            with torch.no_grad():
                c = int(h(A[None]).argmax(1))
            B = bank[rs2.randint(bank.shape[0])]
            with torch.no_grad():
                i2, rh, cc, _ = interaction(h, A, B, c, rs=rs2)
            I2s.append(i2); Rhs.append(rh)
            if not np.isnan(cc): cs.append(cc)
            N, S = maps(h, A, B, c); G = gradcam_head(h, A, c)
            n, s, g = [v.cpu().numpy() for v in (N, S, G)]
            if n.std() > 1e-9 and s.std() > 1e-9: sns.append(spearmanr(n, s).statistic)
            if n.std() > 1e-9 and g.std() > 1e-9: sgn.append(spearmanr(g, n).statistic)
            if s.std() > 1e-9 and g.std() > 1e-9: sgs.append(spearmanr(g, s).statistic)
            rnd = torch.randn(A.shape[1], device=A.device)
            for tag, o in (('N', N), ('S', S), ('G', G), ('rand', rnd)):
                D[tag].append(auc(h, A, B, c, o, 'del'))
                if tag != 'rand': I[tag].append(auc(h, A, B, c, o, 'ins'))
        r = dict(T=T, acc=acc, I2=float(np.mean(I2s)), Rhalf=float(np.mean(Rhs)),
                 corrNS=float(np.mean(cs)), spNS=float(np.mean(sns)),
                 spGN=float(np.mean(sgn)), spGS=float(np.mean(sgs)),
                 **{f'del_{t}': float(np.mean(D[t])) for t in D},
                 **{f'ins_{t}': float(np.mean(I[t])) for t in I})
        res[name] = r
        print(f"{T:6.2f}{acc:8.4f}{r['I2']:8.4f}{r['Rhalf']:8.4f}{r['corrNS']:+9.4f}"
              f"{r['spNS']:+8.3f}{r['spGN']:+8.3f}{r['spGS']:+8.3f} | "
              f"{r['del_N']:6.3f}{r['del_S']:6.3f}{r['del_G']:6.3f}{r['del_rand']:6.3f} | "
              f"{r['ins_N']:6.3f}{r['ins_S']:6.3f}{r['ins_G']:6.3f}", flush=True)
        json.dump(res, open(os.path.join(OUT, 'probe7_andor.json'), 'w'), indent=1)
    print('\nwrote outputs/probe7_andor.json')


if __name__ == '__main__':
    main()
