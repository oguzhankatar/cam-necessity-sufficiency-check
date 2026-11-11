"""TASK 2 -- the locus claim, by region size and reference scale.

All heads of probe 1 plus a post-mean LayerNorm head (ln_post: fc(LN(mean))),
fitted with the identical recipe; float64 evaluation; the probe-1 sample
(48 images x 3 bank references, drawn as probe 1 draws them).

For each head and each region family R (single cells, all 25 3x3 windows of the
7x7 map, 64 random half-maps of 24 cells):
    nec(R) = z(A) - z(A with R <- B),   suf(R) = z(B with R <- A) - z(B)
    separation = 1 - mean_pairs corr_R(nec, suf)
    eps1       = max_R |nec - suf| / max_R max(|nec|, |suf|)   (mean over pairs)
    nonadd     = mean_R |nec(R) - sum_{t in R} nec(t)| / mean_R |nec(R)|
Reference scale: B_s = A + s (B - A), s in {1, .5, .25, .1, .05, .02, .01}.
"""
import os, sys, json, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import theory_sweeps_lib as L

HEADS = ['gap_fc', 'ln_gap', 'tok_mlp', 'gap_mlp', 'ln_post', 'lse0.25', 'lse0.5', 'lse1.0',
         'lse2.0', 'lse4.0', 'lse8.0', 'maxpool', 'attn', 'bilin']
SCALES = [1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01]


def measure(h, X, Bk, refs, masks, scales=(1.0,)):
    cell = masks['cell']
    out = {s: {nm: {'corr': [], 'eps1': [], 'nonadd': []} for nm in masks} for s in scales}
    for k in range(X.shape[0]):
        A = X[k]
        with torch.no_grad():
            c = int(h(A[None]).argmax(1))
        f = lambda Z, c=c: h(Z)[:, c]
        for b in refs[k]:
            B0 = Bk[b]
            for s in scales:
                B = A + s * (B0 - A)
                n1, s1 = L.region_ns(f, A, B, cell)
                for nm, M in masks.items():
                    n, su = (n1, s1) if nm == 'cell' else L.region_ns(f, A, B, M)
                    nn_, ss_ = n.cpu().numpy(), su.cpu().numpy()
                    corr, _, e1 = L.sep_stats(nn_, ss_)
                    add = (M.double() @ n1).cpu().numpy()
                    na = float(np.abs(nn_ - add).mean() / (np.abs(nn_).mean() + 1e-300))
                    d = out[s][nm]; d['corr'].append(corr); d['eps1'].append(e1); d['nonadd'].append(na)
    res = {}
    for s in scales:
        res[str(s)] = {nm: dict(sep=float(1 - np.nanmean(d['corr'])), corr=float(np.nanmean(d['corr'])),
                                eps1=float(np.mean(d['eps1'])), nonadd=float(np.mean(d['nonadd'])),
                                n=len(d['corr']))
                       for nm, d in out[s].items()}
    return res


def main():
    a1 = json.load(open(os.path.join(L.OUTP, 'probe1_aggregator.json')))
    archived = {k: dict(group=v['group'], acc=v['acc'], sep=1 - v['corrNS'], corrNS=v['corrNS'],
                        Rhalf=v['Rhalf'], I2=v['I2']) for k, v in a1.items()}
    imgs, bank_idx, _ = L.probe1_sample()
    refs = L.probe1_refs()
    X = L.batch('val', imgs).double(); Bk = L.batch('val', bank_idx).double()
    masks = {'cell': L.cell_masks(), 'win3x3': L.window_masks(), 'half': L.half_masks()}
    res = {'archived_probe1': archived, 'region': {}, 'scale': {}, 'acc': {}}
    for nm in HEADS:
        t0 = time.time()
        h, acc = L.get_head(nm, double=True)
        res['acc'][nm] = acc
        r = measure(h, X, Bk, refs, masks, scales=(1.0,))
        res['region'][nm] = r['1.0']
        # reference-scale sweep: first reference of each image (48 pairs)
        rs = measure(h, X, Bk, [[r_[0]] for r_ in refs], {'cell': masks['cell'], 'half': masks['half']},
                     scales=SCALES)
        res['scale'][nm] = {s: {'cell_sep': v['cell']['sep'], 'half_sep': v['half']['sep'],
                                'cell_eps1': v['cell']['eps1'], 'half_nonadd': v['half']['nonadd']}
                            for s, v in rs.items()}
        g = res['region'][nm]
        print(f"{nm:8s} acc={acc:.4f} sep cell={g['cell']['sep']:.4g} 3x3={g['win3x3']['sep']:.4g} "
              f"half={g['half']['sep']:.4g} | eps1 cell={g['cell']['eps1']:.3g} 3x3={g['win3x3']['eps1']:.3g} "
              f"half={g['half']['eps1']:.3g} | nonadd half={g['half']['nonadd']:.3g} | scale sep "
              + ' '.join(f"{s}:{v['cell_sep']:.3g}" for s, v in res['scale'][nm].items())
              + f"  ({time.time()-t0:.0f}s)", flush=True)
        L.save('t2_locus.json', res)


if __name__ == '__main__':
    main()
