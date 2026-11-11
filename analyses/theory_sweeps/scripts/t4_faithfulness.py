"""TASK 4 -- deletion / insertion AUC normalised by each model's random
ordering, and the within-model comparison the across-T figure cannot make.

Part A re-runs the probe-7 protocol (scripts/probe7_andor.py: ResNet-50 on
Imagenette, cached layer4, LSE head refit at each T with the probe-1 recipe,
the same 32 images and the same per-image bank reference, AUC of the softmax
probability over 49 single-cell steps as in probe4_utility.auc) and adds:
  * an insertion random baseline (probe 7 recorded only the deletion one),
    averaged over 16 random orderings per image instead of 1;
  * two collapsed maps evaluated INSIDE each T-model:
      C0  = s_t(A) - s_t(B) for the explained class, i.e. the map that
            necessity and sufficiency both equal at T = 0, formed from the same
            head's weights (it does not depend on T);
      CAM = s_t(A), the class-activation map of the same head.
Part B normalises outputs/probe29_fan200.json (1000 classes, 224 images,
shipped fc and refit heads), which already carries both random baselines.
"""
import os, sys, json, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import theory_sweeps_lib as L
sys.path.insert(0, L.SCRIPTS)
from probe4_utility import maps, gradcam_head, auc   # noqa: E402

TS = [-8.0, -4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
NRAND = 16
TAGS = ('N', 'S', 'G', 'C0', 'CAM')


def boot(v, n=2000, seed=0):
    v = np.asarray(v, float); rs = np.random.RandomState(seed)
    b = np.array([v[rs.randint(0, len(v), len(v))].mean() for _ in range(n)])
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def part_a():
    _, _, sel = L.probe1_sample()
    Xi = L.batch('val', np.sort(sel[:32])); bank = L.batch('val', np.sort(sel[48:80]))
    refs = L.probe7_refs()
    a7 = json.load(open(os.path.join(L.OUTP, 'probe7_andor.json')))
    out = {}
    g = torch.Generator(device=L.dev).manual_seed(0)
    for T in TS:
        name = f'lse{T}'
        h, acc = L.get_head(name)
        D = {k: [] for k in TAGS + ('rand',)}; I = {k: [] for k in TAGS + ('rand',)}
        for k in range(Xi.shape[0]):
            A = Xi[k]; B = bank[refs[k]]
            with torch.no_grad():
                c = int(h(A[None]).argmax(1))
                sA = h.fc(A.T)[:, c]; sB = h.fc(B.T)[:, c]
            N, S = maps(h, A, B, c); G = gradcam_head(h, A, c)
            od = {'N': N, 'S': S, 'G': G, 'C0': sA - sB, 'CAM': sA}
            for t, o in od.items():
                D[t].append(auc(h, A, B, c, o, 'del')); I[t].append(auc(h, A, B, c, o, 'ins'))
            dr, ir = [], []
            for _ in range(NRAND):
                r = torch.randn(A.shape[1], device=A.device, generator=g)
                dr.append(auc(h, A, B, c, r, 'del')); ir.append(auc(h, A, B, c, r, 'ins'))
            D['rand'].append(np.mean(dr)); I['rand'].append(np.mean(ir))
        D = {k: np.array(v) for k, v in D.items()}; I = {k: np.array(v) for k, v in I.items()}
        r = dict(T=T, acc=acc, archived=dict(del_N=a7[name]['del_N'], del_S=a7[name]['del_S'],
                                             del_G=a7[name]['del_G'], del_rand=a7[name]['del_rand'],
                                             ins_N=a7[name]['ins_N'], ins_S=a7[name]['ins_S'],
                                             ins_G=a7[name]['ins_G']))
        for t in TAGS + ('rand',):
            r[f'del_{t}'] = float(D[t].mean()); r[f'ins_{t}'] = float(I[t].mean())
        for t in TAGS:
            # ratio of means (deletion: lower is better; insertion: higher is better)
            r[f'delnorm_{t}'] = float(D[t].mean() / D['rand'].mean())
            r[f'insnorm_{t}'] = float(I[t].mean() / I['rand'].mean())
            # per-image ratio, with a bootstrap interval over the 32 images
            dn = D[t] / D['rand']; inn = I[t] / I['rand']
            r[f'delnorm_img_{t}'] = float(dn.mean()); r[f'delnorm_img_{t}_ci'] = boot(dn)
            r[f'insnorm_img_{t}'] = float(inn.mean()); r[f'insnorm_img_{t}_ci'] = boot(inn)
        # within-model paired contrasts: best separated map minus collapsed map C0
        for t in ('N', 'S', 'G', 'CAM'):
            d = D[t] / D['rand'] - D['C0'] / D['rand']; i_ = I[t] / I['rand'] - I['C0'] / I['rand']
            r[f'within_del_{t}_minus_C0'] = float(d.mean()); r[f'within_del_{t}_minus_C0_ci'] = boot(d)
            r[f'within_ins_{t}_minus_C0'] = float(i_.mean()); r[f'within_ins_{t}_minus_C0_ci'] = boot(i_)
        out[str(T)] = r
        print(f"T={T:+5.1f} acc={acc:.4f} | del N {r['del_N']:.3f} S {r['del_S']:.3f} G {r['del_G']:.3f} "
              f"C0 {r['del_C0']:.3f} CAM {r['del_CAM']:.3f} R {r['del_rand']:.3f} | ins N {r['ins_N']:.3f} "
              f"S {r['ins_S']:.3f} G {r['ins_G']:.3f} C0 {r['ins_C0']:.3f} CAM {r['ins_CAM']:.3f} "
              f"R {r['ins_rand']:.3f} || norm del N {r['delnorm_N']:.3f} S {r['delnorm_S']:.3f} "
              f"C0 {r['delnorm_C0']:.3f} | norm ins N {r['insnorm_N']:.3f} S {r['insnorm_S']:.3f} "
              f"C0 {r['insnorm_C0']:.3f}", flush=True)
        L.save('t4_faithfulness.json', {'probe7_rerun': out})
    return out


def part_b():
    d = json.load(open(os.path.join(L.OUTP, 'probe29_fan200.json')))
    out = {}
    for pre, lab in (('ship_', 'shipped_fc_no_training'), ('', 'refit_per_T')):
        rows = {}
        for T in (-8.0, -1.0, 0.0, 1.0, 8.0):
            r = d[f'{pre}lse{T}']
            rr = dict(T=T, acc=r['acc'])
            for t in ('N', 'S', 'G'):
                rr[f'del_{t}'] = r[f'del_{t}']; rr[f'ins_{t}'] = r[f'ins_{t}']
                rr[f'delnorm_{t}'] = r[f'del_{t}'] / r['del_rand']
                rr[f'insnorm_{t}'] = r[f'ins_{t}'] / r['ins_rand']
            rr['del_rand'] = r['del_rand']; rr['ins_rand'] = r['ins_rand']
            rows[str(T)] = rr
        out[lab] = rows
    return out


def main():
    a = part_a()
    b = part_b()
    L.save('t4_faithfulness.json', {'probe7_rerun': a, 'probe29_normalised': b})
    for lab, rows in b.items():
        print('probe29', lab)
        for T, r in rows.items():
            print(f"  T={float(T):+4.0f} acc={r['acc']:.4f} delnorm N {r['delnorm_N']:.3f} S {r['delnorm_S']:.3f} "
                  f"G {r['delnorm_G']:.3f} | insnorm N {r['insnorm_N']:.3f} S {r['insnorm_S']:.3f} "
                  f"G {r['insnorm_G']:.3f}")


if __name__ == '__main__':
    main()
