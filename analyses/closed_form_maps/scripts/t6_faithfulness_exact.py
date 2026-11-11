"""T6 -- the Discussion's within-model faithfulness comparison (analyses/theory_sweeps,
t4_faithfulness.py part A) with exact necessity / sufficiency.

Same protocol as probe7 / C4: ResNet-50 on Imagenette, LSE head refitted per T
(probe1 recipe, seed 0), the same 32 images and the same per-image reference.
Within each model, float64 deletion and insertion AUCs (49 single-cell steps,
softmax probability) of
  NS  = exact necessity (= exact sufficiency in rank; both are evaluated)
  C0  = s_t(A) - s_t(B), the collapsed map (what N and S equal at T = 0)
  CAM = s_t(A);  G = Grad-CAM (float64)
and paired image-bootstrap 95% CIs of NS - C0.
"""
import os, json, copy, numpy as np, torch
import closed_form_lib as C
import probe1_aggregator as P1
from probe1_aggregator import load, batch, train, interaction

TS = [-8.0, -4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0, 8.0]


def pboot(x, y, n=2000, seed=0):
    d = np.asarray(x) - np.asarray(y); rs = np.random.RandomState(seed)
    b = np.array([d[rs.randint(0, len(d), len(d))].mean() for _ in range(n)])
    return [float(d.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def main():
    Ftr, Ytr = load('train'); Fva, Yva = load('val')
    rs = np.random.RandomState(0)
    sel = rs.permutation(len(Yva))[:96]
    Xi = batch(Fva, np.sort(sel[:32])); bank = batch(Fva, np.sort(sel[48:80]))
    out = {}
    for T in TS:
        name = f'lse{T}'
        P1.GROUP.setdefault(name, 'nonlin-agg')
        h, accu = train(name, Ftr, Ytr, Fva, Yva)
        h64 = copy.deepcopy(h).double().eval()
        rs2 = np.random.RandomState(1)
        D = {k: [] for k in ('N', 'S', 'C0', 'CAM', 'G')}; I = {k: [] for k in D}
        for k in range(Xi.shape[0]):
            A = Xi[k]
            with torch.no_grad():
                c = int(h(A[None]).argmax(1))
            B = bank[rs2.randint(bank.shape[0])]
            with torch.no_grad():
                interaction(h, A, B, c, rs=rs2)
            sa, sb = C.scores(h, A, c), C.scores(h, B, c)
            ne, se = C.ns_exact(sa, sb, T)
            maps = dict(N=ne, S=se, C0=sa - sb, CAM=sa, G=C.gradcam64(h64, A, c))
            for t, m in maps.items():
                D[t].append(C.auc64(h64, A, B, c, m, 'del')); I[t].append(C.auc64(h64, A, B, c, m, 'ins'))
        r = dict(T=T, acc=accu)
        for t in D:
            r['del_' + t] = float(np.mean(D[t])); r['ins_' + t] = float(np.mean(I[t]))
        r['del_NS_minus_C0'] = pboot(D['N'], D['C0']); r['ins_NS_minus_C0'] = pboot(I['N'], I['C0'])
        r['del_NS_minus_CAM'] = pboot(D['N'], D['CAM']); r['ins_NS_minus_CAM'] = pboot(I['N'], I['CAM'])
        r['N_equals_S'] = bool(np.allclose(D['N'], D['S'], atol=0, rtol=0) and np.allclose(I['N'], I['S'], atol=0, rtol=0))
        best_del = min(('NS', 'C0', 'CAM', 'G'), key=lambda t: r['del_' + ('N' if t == 'NS' else t)])
        best_ins = max(('NS', 'C0', 'CAM', 'G'), key=lambda t: r['ins_' + ('N' if t == 'NS' else t)])
        r['best_del'], r['best_ins'] = best_del, best_ins
        out[name] = r
        print(f"T={T:+5.1f} del NS {r['del_N']:.3f} C0 {r['del_C0']:.3f} CAM {r['del_CAM']:.3f} G {r['del_G']:.3f} "
              f"[NS-C0 {r['del_NS_minus_C0'][0]:+.3f} ({r['del_NS_minus_C0'][1]:+.3f},{r['del_NS_minus_C0'][2]:+.3f})] | "
              f"ins NS {r['ins_N']:.3f} C0 {r['ins_C0']:.3f} CAM {r['ins_CAM']:.3f} G {r['ins_G']:.3f} "
              f"[NS-C0 {r['ins_NS_minus_C0'][0]:+.3f} ({r['ins_NS_minus_C0'][1]:+.3f},{r['ins_NS_minus_C0'][2]:+.3f})] "
              f"N==S {r['N_equals_S']}", flush=True)
        json.dump(out, open(os.path.join(C.OUTD, 't6_faithfulness_exact.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
