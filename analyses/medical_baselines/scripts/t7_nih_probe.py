"""C2 task 6, diagnostic -- is the NIH mean AUC of 0.667 a property of the
distributed weights or of the images/labels?

(a) forward check: xrv's own model.forward (op_threshs disabled) on xrv-
    preprocessed inputs equals the cached GAP-head logits.
(b) the same 5,720 images through two other distributed xrv DenseNet-121s
    whose label spaces include the NIH findings: `densenet121-res224-all`
    (trained on NIH + 6 other sets) and, for contrast, our cached nih weights.
(c) AUC split by the reconstructed xrv train / held-out partition.
"""
import os
import sys
import json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import medical_baselines_lib as L
from medical_baselines_lib import mlib, dev
from mlib import NIH_FINDINGS
import t5_nih as T5


def main():
    import torchxrayvision as xrv
    vn, vY, _, _ = T5.nih_names()
    P = T5.xrv_pp_cache(vn)
    held = set(json.load(open(os.path.join(L.OUT, 'nih_xrv_heldout_images.json'))))
    hm = np.array([n in held for n in vn])
    res = {}
    for w in ('densenet121-res224-nih', 'densenet121-res224-all'):
        m = xrv.models.DenseNet(weights=w).eval().to(dev)
        m.op_threshs = None
        Z = []
        with torch.no_grad():
            for i in range(0, len(vn), 64):
                Z.append(m(torch.from_numpy(np.asarray(P[i:i + 64]))[:, None].to(dev)).cpu().numpy())
        Z = np.concatenate(Z)
        idx = {f: list(m.targets).index(f) for f in NIH_FINDINGS if f in m.targets}
        res[w] = dict(all=T5.aucs(Z, vY, idx), xrv_heldout=T5.aucs(Z[hm], vY[hm], idx),
                      xrv_train=T5.aucs(Z[~hm], vY[~hm], idx))
        if w.endswith('nih'):
            Fb = np.load(os.path.join(L.CACHE, 'F_nih_xrv_xrvpp.npy'), mmap_mode='r')
            W = m.classifier.weight.detach().cpu().numpy()
            b = m.classifier.bias.detach().cpu().numpy()
            Zc = np.asarray(Fb[:256]).mean(1) @ W.T + b
            res['forward_vs_cached_head_maxabs_logit'] = float(np.abs(Zc - Z[:256]).max())
        print(w, 'mean AUC all %.4f  heldout %.4f  train %.4f'
              % (res[w]['all']['mean_auc'], res[w]['xrv_heldout']['mean_auc'],
                 res[w]['xrv_train']['mean_auc']), flush=True)
    L.save('t7_nih_probe.json', res)


if __name__ == '__main__':
    main()
