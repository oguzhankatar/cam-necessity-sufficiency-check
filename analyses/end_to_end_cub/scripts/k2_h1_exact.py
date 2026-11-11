"""H1 separation with the cancellation-free closed form (analyses/closed_form_maps/scripts/closed_form_lib.ns_exact).

wlib.ns_lse evaluates N as log(Z_A) - log(Z_A with t <- B); for cells whose share of Z_A is below
float64 resolution this gives exact zeros and ties in N, which lower the Spearman correlation of
N and S although the two maps share one ordering against any single reference. ns_exact uses a
log1p branch for small changes and resolves them. Localisation maps are unchanged (relative
difference to wlib.ns_lse reported). Same images, references and class as k2_eval.py.

    python k2_h1_exact.py <seed>
"""
import os, sys, json
import numpy as np
import torch
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import k2lib as L
import k2_eval as E
sys.path.insert(0, os.path.join(L.ROOT, 'analyses', 'closed_form_maps', 'scripts'))
import closed_form_lib  # noqa: E402


def main():
    seed = int(sys.argv[1])
    torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False
    ck = torch.load(L.ckpt_path('H1', seed), map_location='cpu', weights_only=False)
    net = L.Net('H1', pretrained=False); net.load_state_dict(ck['state_dict']); net = net.cuda().eval()
    H = E.Head64('H1', net.head)
    man = L.W.cub_manifest(); _, tf = L.tfs()

    @torch.no_grad()
    def cells(x):
        return net.cells(x.cuda()).transpose(1, 2).double().cpu().numpy()

    bidx = L.bank_indices(len(man['fit']))
    B = cells(torch.stack([tf(Image.open(man['fit'][i]['path']).convert('RGB')) for i in bidx]))
    RQ = [H.prep_ref(b) for b in B]
    recs = man['ev']
    dl = torch.utils.data.DataLoader(L.DS(recs, tf), batch_size=32, shuffle=False, num_workers=4)
    s1p, s1s, s8p, s8s, tie_ex, tie_w = [], [], [], [], [], []
    dmax = 0.0
    for x, y, idx in dl:
        Xb = cells(x)
        for j in range(Xb.shape[0]):
            i = int(idx[j]); c = int(recs[i]['label'])
            a = Xb[j] @ H.Wf[c] + H.bf[c]
            per = []
            for q in RQ:
                N, S = closed_form_lib.ns_exact(a, q[:, c], 1.0)
                N0, S0 = L.W.ns_lse(a, q[:, c], 1.0, 14, 14)
                dmax = max(dmax, np.abs(N - N0).max() / np.abs(N0).max(),
                           np.abs(S - S0).max() / np.abs(S0).max())
                tie_ex.append(1 - len(np.unique(N)) / len(N)); tie_w.append(1 - len(np.unique(N0)) / len(N0))
                per.append((N, S))
            Nm = np.mean([p[0] for p in per], 0); Sm = np.mean([p[1] for p in per], 0)
            s1p.append(np.nanmean([1 - E.pear(*p) for p in per]))
            s1s.append(np.nanmean([1 - E.spear(*p) for p in per]))
            s8p.append(1 - E.pear(Nm, Sm)); s8s.append(1 - E.spear(Nm, Sm))
    res = dict(seed=seed, n_test=len(s1p),
               separation=dict(sep1_pearson=float(np.nanmean(s1p)), sep1_spearman=float(np.nanmean(s1s)),
                               sep8_pearson=float(np.nanmean(s8p)), sep8_spearman=float(np.nanmean(s8s))),
               max_rel_diff_vs_wlib_ns_lse=float(dmax),
               N_tie_share_exact=float(np.mean(tie_ex)), N_tie_share_wlib=float(np.mean(tie_w)))
    np.savez_compressed(os.path.join(L.OUT, 'per_image', 'H1_s%d_exact_sep.npz' % seed),
                        sep1_pearson=np.array(s1p), sep1_spearman=np.array(s1s),
                        sep8_pearson=np.array(s8p), sep8_spearman=np.array(s8s))
    L.save_json(os.path.join(L.OUT, 'h1_exact_sep_s%d.json' % seed), res)
    print(json.dumps(res, indent=1), flush=True)


if __name__ == '__main__':
    main()
