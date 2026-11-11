"""TASK 1 -- is nec == suf a property of the class LOGIT only?

GAP + linear head, Imagenette ResNet-50 layer4 (cached), the probe-1 setting:
48 validation images, three bank references each drawn exactly as probe 1
draws them. Per cell, nec and suf are computed in float64 on
  (a) the class logit z_c                       (the paper's quantity)
  (b) the softmax probability p_c of the predicted class
  (c) the log-probability log p_c   (shown for completeness)
with the same reference. Two heads: the refit 10-class gap_fc head (probe-1
recipe) and the distributed torchvision ResNet-50 fc (1000 classes, no
training), which reads the same layer4 features.
"""
import os, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import theory_sweeps_lib as L


def run(h, tag):
    imgs, bank_idx, _ = L.probe1_sample()
    refs = L.probe1_refs()
    X = L.batch('val', imgs).double(); Bk = L.batch('val', bank_idx).double()
    eye = L.cell_masks()
    R = {q: {'corr': [], 'rho': [], 'eps1': [], 'scale': []} for q in ('logit', 'prob', 'logprob')}
    pA = []
    for k in range(X.shape[0]):
        A = X[k]
        with torch.no_grad():
            c = int(h(A[None]).argmax(1))
            pA.append(float(torch.softmax(h(A[None]), 1)[0, c]))
        fs = {'logit': lambda Z, c=c: h(Z)[:, c],
              'prob': lambda Z, c=c: torch.softmax(h(Z), 1)[:, c],
              'logprob': lambda Z, c=c: torch.log_softmax(h(Z), 1)[:, c]}
        for b in refs[k]:
            B = Bk[b]
            for q, f in fs.items():
                n, s = L.region_ns(f, A, B, eye)
                n, s = n.cpu().numpy(), s.cpu().numpy()
                corr, rho, e1 = L.sep_stats(n, s)
                R[q]['corr'].append(corr); R[q]['rho'].append(rho); R[q]['eps1'].append(e1)
                R[q]['scale'].append(float(max(np.abs(n).max(), np.abs(s).max())))
    out = {'n_pairs': len(R['logit']['corr']),
           'pA_median': float(np.median(pA)), 'pA_frac_above_0.99': float(np.mean(np.array(pA) > 0.99))}
    for q, d in R.items():
        c = np.array(d['corr']); e = np.array(d['eps1'])
        out[q] = dict(corr_mean=float(np.nanmean(c)), sep_1_minus_corr=float(1 - np.nanmean(c)),
                      corr_min=float(np.nanmin(c)), corr_median=float(np.nanmedian(c)),
                      rho_mean=float(np.nanmean(d['rho'])),
                      eps1_mean=float(e.mean()), eps1_median=float(np.median(e)), eps1_max=float(e.max()),
                      n_nan=int(np.isnan(c).sum()), scale_median=float(np.median(d['scale'])))
        print(f"  {tag:12s} {q:8s} 1-corr={out[q]['sep_1_minus_corr']:.6g}  corr_min={out[q]['corr_min']:.4f}  "
              f"eps1 mean={out[q]['eps1_mean']:.3g} med={out[q]['eps1_median']:.3g} max={out[q]['eps1_max']:.3g}",
              flush=True)
    return out


def main():
    res = {}
    h, acc = L.get_head('gap_fc', double=True)
    res['gap_fc_10class'] = dict(acc=acc, **run(h, 'gap_fc'))
    from torchvision import models
    tv = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    hs = L.build('gap_fc', k=1000).to(L.dev).double()
    hs.fc.weight.data.copy_(tv.fc.weight.data.double()); hs.fc.bias.data.copy_(tv.fc.bias.data.double())
    hs.eval()
    res['shipped_fc_1000class'] = run(hs, 'shipped_fc')
    L.save('t1_softmax.json', res)


if __name__ == '__main__':
    main()
