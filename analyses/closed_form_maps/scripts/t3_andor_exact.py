"""T3 -- Fig. 2(b) recomputed: probe7_andor.py (ResNet-50 / Imagenette, LSE head
refitted at each of eleven temperatures, 32 validation images).

Reference construction is probe7's, reproduced exactly: ONE reference per image,
B = bank[rs2.randint(32)], rs2 = RandomState(1) reset per temperature, with
probe1_aggregator.interaction called in between (it consumes rs2).  Heads are
refitted with probe7's own call (probe1_aggregator.train, seed 0); the archived
accuracy is printed beside ours as the reproduction check.
Per image: f32 = the paper's estimator (probe4_utility.maps / auc, float32);
ex = closed form (closed_form_lib.ns_exact) with float64 curves and Grad-CAM.
"""
import os, sys, json, copy, time, numpy as np, torch
import closed_form_lib as C
import probe1_aggregator as P1
from probe1_aggregator import load, batch, train, interaction
from probe4_utility import maps, gradcam_head, auc

TS = [-8.0, -4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0, 8.0]


def main():
    Ftr, Ytr = load('train'); Fva, Yva = load('val')
    rs = np.random.RandomState(0)
    sel = rs.permutation(len(Yva))[:96]
    Xi = batch(Fva, np.sort(sel[:32])); bank = batch(Fva, np.sort(sel[48:80]))
    arch = json.load(open(os.path.join(C.ROOT, 'outputs', 'probe7_andor.json')))
    res, per = {'meta': dict(source='scripts/probe7_andor.py', n_img=32,
                             reference='one bank image per image (bank of 32)')}, {}
    for T in TS:
        t0 = time.time()
        name = f'lse{T}'
        P1.GROUP.setdefault(name, 'nonlin-agg')
        h, accu = train(name, Ftr, Ytr, Fva, Yva)
        h64 = copy.deepcopy(h).double().eval()
        rs2 = np.random.RandomState(1)
        acc = {k: [] for k in ('spNS_f32', 'spGN_f32', 'spGS_f32', 'insN_f32', 'insS_f32', 'delN_f32',
                               'delS_f32', 'zeroN_f32', 'spNS_ex', 'spGN_ex', 'spGS_ex', 'insN_ex',
                               'insS_ex', 'delN_ex', 'delS_ex', 'insG_ex', 'spNS_bank32',
                               'spGN_bank32', 'spGS_bank32')}
        for k in range(Xi.shape[0]):
            A = Xi[k]
            with torch.no_grad():
                c = int(h(A[None]).argmax(1))
            B = bank[rs2.randint(bank.shape[0])]
            with torch.no_grad():
                interaction(h, A, B, c, rs=rs2)
            N, S = maps(h, A, B, c); G = gradcam_head(h, A, c)
            n, s, g = [v.cpu().numpy().astype(np.float64) for v in (N, S, G)]
            acc['spNS_f32'].append(C.sp(n, s) if n.std() > 1e-9 and s.std() > 1e-9 else np.nan)
            acc['spGN_f32'].append(C.sp(g, n) if n.std() > 1e-9 and g.std() > 1e-9 else np.nan)
            acc['spGS_f32'].append(C.sp(g, s) if s.std() > 1e-9 and g.std() > 1e-9 else np.nan)
            acc['zeroN_f32'].append(float((n == 0).mean()))
            acc['insN_f32'].append(auc(h, A, B, c, N, 'ins')); acc['insS_f32'].append(auc(h, A, B, c, S, 'ins'))
            acc['delN_f32'].append(auc(h, A, B, c, N, 'del')); acc['delS_f32'].append(auc(h, A, B, c, S, 'del'))
            sa, sb = C.scores(h, A, c), C.scores(h, B, c)
            ne, se = C.ns_exact(sa, sb, T); g64 = C.gradcam64(h64, A, c)
            acc['spNS_ex'].append(C.sp(ne, se)); acc['spGN_ex'].append(C.sp(g64, ne)); acc['spGS_ex'].append(C.sp(g64, se))
            acc['insN_ex'].append(C.auc64(h64, A, B, c, ne, 'ins')); acc['insS_ex'].append(C.auc64(h64, A, B, c, se, 'ins'))
            acc['delN_ex'].append(C.auc64(h64, A, B, c, ne, 'del')); acc['delS_ex'].append(C.auc64(h64, A, B, c, se, 'del'))
            acc['insG_ex'].append(C.auc64(h64, A, B, c, g64, 'ins'))
            NE, SE = zip(*[C.ns_exact(sa, C.scores(h, bank[q], c), T) for q in range(bank.shape[0])])
            nb, sb_ = np.mean(NE, 0), np.mean(SE, 0)
            acc['spNS_bank32'].append(C.sp(nb, sb_)); acc['spGN_bank32'].append(C.sp(g64, nb))
            acc['spGS_bank32'].append(C.sp(g64, sb_))
        a = arch[name]
        r = {'T': T, 'acc': accu, 'acc_archive': a['acc'],
             'archive': {k: a[k] for k in ('spNS', 'spGN', 'spGS', 'ins_N', 'ins_S', 'ins_G', 'del_N', 'del_S')}}
        for kk, v in acc.items():
            r[kk] = C.boot(v)
        r['max_abs_insN_minus_insS_ex'] = float(np.max(np.abs(np.array(acc['insN_ex']) - np.array(acc['insS_ex']))))
        res[name] = r; per[name] = acc
        print(f"T={T:+5.1f} acc {accu:.4f} (arch {a['acc']:.4f}) | ins N/S f32 {r['insN_f32'][0]:.3f}/{r['insS_f32'][0]:.3f} "
              f"(arch {a['ins_N']:.3f}/{a['ins_S']:.3f}) ex {r['insN_ex'][0]:.3f}/{r['insS_ex'][0]:.3f} | "
              f"spNS f32 {r['spNS_f32'][0]:.3f} ex {r['spNS_ex'][0]:.6f} bank32 {r['spNS_bank32'][0]:.3f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        json.dump(res, open(os.path.join(C.OUTD, 't3_andor_exact.json'), 'w'), indent=1)
        json.dump(per, open(os.path.join(C.OUTD, 't3_andor_exact_per_image.json'), 'w'))


if __name__ == '__main__':
    main()
