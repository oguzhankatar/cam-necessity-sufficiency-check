"""TASK 8 complement -- why probe 30 prints rho(Grad-CAM, necessity) = 0.9495 at
T = 0 while probe 29 prints spGN = 0.9406 for the same shipped ResNet-50 fc.

Both are the same quantity (Spearman over the 49 cells between the Grad-CAM map
and the necessity map, GAP + shipped fc, ImageNet val mirror, 1000 classes);
they differ only in WHICH images and WHICH references:
  probe 29: 224 images = sorted(perm[:224]), bank of 32 = sorted(perm[240:272]),
            perm = RandomState(0).permutation(10000); reference per image
            rs2.randint(32), with probe 29's rs2 also consumed by
            nslib.interaction between images.
  probe 30: 128 images = sorted(RandomState(0).choice(10000, 128)), bank of 16 =
            sorted(next choice(10000, 16)); reference rs2.randint(16), no other
            consumption.
Here both configurations are rebuilt and their per-image values computed, so
the 0.009 gap can be attributed to images vs references. Features are
extracted for the needed images only (in memory, fp16 as in the archived
cache); nothing is written outside theory_sweeps/out."""
import os, sys, numpy as np, torch
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import theory_sweeps_lib as L
import nslib as R  # noqa: E402 (scripts/ on sys.path via theory_sweeps_lib)

dev = 'cuda'


def main():
    from torchvision import models
    ds, ncls = R.get_dataset('in1k', 'val', 224)
    n = len(ds)
    perm = np.random.RandomState(0).permutation(n)
    i29 = np.sort(perm[:224]); b29 = np.sort(perm[240:272])
    rs = np.random.RandomState(0)
    i30 = np.sort(rs.choice(n, 128, replace=False)); b30 = np.sort(rs.choice(n, 16, replace=False))
    need = np.unique(np.concatenate([i29, b29, i30, b30]))
    body, C, _ = R.get_body('resnet50'); body = body.to(dev)
    feats = {}
    with torch.no_grad():
        for s in range(0, len(need), 32):
            ids = need[s:s + 32]
            x = torch.stack([ds[int(i)][0] for i in ids]).to(dev)
            f = body(x).half().float()
            for i, fi in zip(ids, f):
                feats[int(i)] = fi.reshape(fi.shape[0], -1)
    del body; torch.cuda.empty_cache()
    tv = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    W = tv.fc.weight.data.to(dev); bb = tv.fc.bias.data.to(dev)
    h = lambda X: X.mean(-1) @ W.T + bb           # LSE T = 0 == GAP + fc, (B,C,T) -> (B,1000)

    def rho_gn(A, B):
        with torch.no_grad():
            c = int(h(A[None]).argmax(1))
        N, _ = R.maps(h, A, B, c); G = R.gradcam_head(h, A, c)
        g, nn_ = G.cpu().numpy(), N.cpu().numpy()
        return float(spearmanr(g, nn_).statistic) if g.std() > 1e-9 and nn_.std() > 1e-9 else np.nan, c

    # probe-29 configuration (rs2 consumed by interaction, as in measure_ci)
    rs2 = np.random.RandomState(1); v29 = {}; ref29 = {}
    for i in i29:
        A = feats[int(i)]; bi = b29[rs2.randint(32)]; B = feats[int(bi)]
        r, c = rho_gn(A, B); v29[int(i)] = r; ref29[int(i)] = int(bi)
        R.interaction(h, A, B, c, rs2)
    # probe-30 configuration
    rs2 = np.random.RandomState(1); v30 = {}; ref30 = {}
    for i in i30:
        bi = b30[rs2.randint(16)]; r, _ = rho_gn(feats[int(i)], feats[int(bi)]); v30[int(i)] = r; ref30[int(i)] = int(bi)
    # probe-30 images, each with the reference probe 29 gave it
    v30_ref29 = {int(i): rho_gn(feats[int(i)], feats[ref29[int(i)]])[0] for i in i30}
    m = lambda d, keys=None: float(np.nanmean([d[k] for k in (keys if keys is not None else d)]))
    sub = [int(i) for i in i30]
    res = dict(
        probe30_images_subset_of_probe29=bool(set(i30) <= set(i29)),
        bank_overlap=int(len(set(b29) & set(b30))),
        same_reference_for_shared_images=int(sum(ref29[k] == ref30[k] for k in sub)),
        probe29_config_224=m(v29), archived_probe29_spGN=0.9405762923732166,
        probe29_config_restricted_to_probe30_images=m(v29, sub),
        probe30_config_128=m(v30), archived_probe30_rho=0.9495285454647051,
        probe30_images_with_probe29_refs=m(v30_ref29),
        probe29_other_96_images=m(v29, [k for k in v29 if k not in set(sub)]))
    for k, v in res.items():
        print(f'{k:45s} {v}', flush=True)
    L.save('t8b_gradcam_rho_decomp.json', res)


if __name__ == '__main__':
    main()
