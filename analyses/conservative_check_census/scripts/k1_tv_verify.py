"""Check that the forward_features / forward_head split of k1_tv.TV reproduces torchvision's
own forward (random weights, eval, float64, one N(0,1) image). Output: ../out/k1_tv_verify.json"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import torch
import k1_tv as TVM
torch.set_num_threads(12)
rows = {}
for n in TVM.tv_names():
    try:
        w = TVM.TV(n).eval().double()
        with torch.no_grad():                      # zero-initialised heads (ViT) would give 0/0
            for pp in w.parameters():
                if pp.dim() >= 2 and not bool(pp.any()):
                    pp.normal_(0, 0.02)
        x = torch.randn(1, *w.pretrained_cfg['input_size'], generator=torch.Generator().manual_seed(0), dtype=torch.float64)
        with torch.no_grad():
            a = w.m(x); b = w(x)
        rows[n] = float((a - b).abs().max() / a.abs().max())
    except Exception as e:
        rows[n] = 'error: %s' % e
    print(n, rows[n], flush=True)
num = [v for v in rows.values() if isinstance(v, float)]
res = dict(n=len(rows), max_rel_diff=max(num), n_exact=sum(v == 0.0 for v in num), rows=rows)
json.dump(res, open(os.path.join(HERE, '..', 'out', 'k1_tv_verify.json'), 'w'), indent=1)
print(res['n'], res['max_rel_diff'], res['n_exact'])
