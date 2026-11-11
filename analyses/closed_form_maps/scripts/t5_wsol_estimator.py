"""T5 -- is the localisation arm's N / S estimator (scripts/wsol/wlib.ns_lse,
float64, 'cancellation-safe' by holding out the largest term, maps averaged
over 8 references) exact enough for what the localisation metrics read?

The WSOL cell features are not cached on disk, so the estimator is exercised on
real ResNet-50 layer4 activations (the 272 ImageNet images of T2, shipped fc,
class fixed at T = 0) with a bank of 8 references, at every temperature the
localisation arm used (tau = +1 primary; +-2, +-8 sensitivity), 1x1 and 3x3.
Reported: argmax agreement with the closed form, and the largest difference of
the min-max normalised maps (the extent metrics threshold that normalised map
at steps of 0.01; pointing reads its argmax).
"""
import os, sys, json, numpy as np, torch
import closed_form_lib as C
sys.path.insert(0, os.path.join(C.SCRIPTS, 'wsol'))
import wlib as W
import probe20_in1k as P20

CACHE = os.path.join(os.environ.get('TEMP', '/tmp'), '<repo>', 'c5cache')


def nrm(m):
    return (m - m.min()) / (m.max() - m.min() + 1e-300)


def main():
    F = np.load(os.path.join(CACHE, 'in1k_sel_272.npy')).astype(np.float64)
    X = F.reshape(len(F), 2048, -1).transpose(0, 2, 1)            # (n, T, C)
    Xi, bank = X[:224], X[224:232]
    Wt, bt = [v.double().numpy() for v in P20.shipped_fc()]
    cls = np.argmax(Xi.mean(1) @ Wt.T + bt, 1)
    G = 7
    blocks = [list(map(int, b)) for b in W._blocks_3x3(G, G)]
    out = {}
    for tau in (1.0, -1.0, 2.0, -2.0, 8.0, -8.0):
        for blk in (False, True):
            agN = agS = 0; dN = dS = 0.0; tinyN = []
            for k in range(len(Xi)):
                c = cls[k]
                sa = Xi[k] @ Wt[c] + bt[c]
                Nw = np.zeros(49); Sw = np.zeros(49); Ne = np.zeros(49); Se = np.zeros(49)
                for b in bank:
                    sb = b @ Wt[c] + bt[c]
                    n1, s1 = W.ns_lse(sa, sb, tau, G, G, blk)
                    n2, s2 = (C.ns_exact_units(sa, sb, tau, blocks) if blk else C.ns_exact(sa, sb, tau))
                    Nw += n1 / 8; Sw += s1 / 8; Ne += n2 / 8; Se += s2 / 8
                agN += int(np.argmax(Nw) == np.argmax(Ne)); agS += int(np.argmax(Sw) == np.argmax(Se))
                dN = max(dN, float(np.abs(nrm(Nw) - nrm(Ne)).max())); dS = max(dS, float(np.abs(nrm(Sw) - nrm(Se)).max()))
                tinyN.append(float((np.abs(Ne) < 1e-12 * np.abs(Ne).max()).mean()))
            tag = 'tau%+g_%s' % (tau, '3x3' if blk else '1x1')
            out[tag] = dict(argmax_agree_N=agN / len(Xi), argmax_agree_S=agS / len(Xi),
                            max_normalised_diff_N=dN, max_normalised_diff_S=dS,
                            frac_cells_below_1em12_of_max_N=float(np.mean(tinyN)))
            print(tag, out[tag], flush=True)
    json.dump(out, open(os.path.join(C.OUTD, 't5_wsol_estimator.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
