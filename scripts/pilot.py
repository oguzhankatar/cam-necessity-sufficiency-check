"""Pilot of the necessity/sufficiency maps (src/nscam.py).

Hypothesis under test (H1): the N-map and the S-map DISSOCIATE --
  N beats S and Grad-CAM on DELETION, S beats N and Grad-CAM on INSERTION,
  and neither wins both.
A method that won both would REFUTE the theory (it would mean the two
metrics are measuring one thing after all).
"""
import argparse, os, sys, tarfile, urllib.request, json, time
import numpy as np, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from nscam import NSCAM

URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"


def get_data(root):
    d = os.path.join(root, "imagenette2-160")
    if not os.path.isdir(d):
        os.makedirs(root, exist_ok=True)
        tgz = os.path.join(root, "imagenette2-160.tgz")
        if not os.path.exists(tgz):
            print("downloading imagenette2-160 (~99MB) ...")
            urllib.request.urlretrieve(URL, tgz)
        with tarfile.open(tgz) as t:
            t.extractall(root)
    return d


class R50:
    """Splits ResNet50 into features (through layer4) and head (avgpool+fc)."""
    def __init__(self, device):
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.m = m.eval().to(device)
        self.body = torch.nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool,
                                        m.layer1, m.layer2, m.layer3, m.layer4)

    def features(self, x):
        return self.body(x)

    def head(self, A):
        return self.m.fc(torch.flatten(self.m.avgpool(A), 1))


def gradcam(bb, x, c):
    A = bb.features(x).detach().requires_grad_(True)
    z = bb.head(A)[0, c]
    g, = torch.autograd.grad(z, A)
    w = g.mean(dim=(2, 3), keepdim=True)
    m = F.relu((w * A).sum(1))[0]
    m = m - m.min()
    return (m / (m.max() + 1e-8)).detach()


def auc_curve(bb, x, c, sal, mode, steps=28, blur=None):
    """Standard deletion / insertion AUC on the input image."""
    H = x.shape[-1]
    s = F.interpolate(sal[None, None], size=(H, H), mode="bilinear",
                      align_corners=False)[0, 0]
    order = torch.argsort(s.flatten(), descending=True)
    n = order.numel()
    k = int(np.ceil(n / steps))
    start = x.clone() if mode == "del" else blur.clone()
    finish = torch.zeros_like(x) if mode == "del" else x
    probs, cur = [], start.clone()
    with torch.no_grad():
        probs.append(F.softmax(bb.head(bb.features(cur)), 1)[0, c].item())
        for i in range(steps):
            idx = order[i * k:(i + 1) * k]
            yy, xx = idx // H, idx % H
            cur[0, :, yy, xx] = finish[0, :, yy, xx]
            probs.append(F.softmax(bb.head(bb.features(cur)), 1)[0, c].item())
    p = np.array(probs)
    return float((p.sum() - p[0] / 2 - p[-1] / 2) / (len(p) - 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--bank", type=int, default=200)
    ap.add_argument("--out", default="outputs/pilot.json")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0); np.random.seed(0)

    root = get_data(os.path.join(os.path.dirname(__file__), "..", "data"))
    tf = transforms.Compose([transforms.Resize(160), transforms.CenterCrop(160),
                             transforms.Resize(224), transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406],
                                                  [0.229, 0.224, 0.225])])
    val = datasets.ImageFolder(os.path.join(root, "val"), tf)
    bb = R50(dev)

    bank_ld = DataLoader(torch.utils.data.Subset(val, range(a.bank)),
                         batch_size=16)
    ns = NSCAM(bb, rank=a.rank).fit_bank(bank_ld, dev)
    print(f"bank: {len(ns.bank)} classes")

    # gaussian blur baseline for insertion
    kk = torch.arange(-10, 11, dtype=torch.float32)
    g1 = torch.exp(-kk ** 2 / (2 * 5. ** 2)); g1 /= g1.sum()
    ker = (g1[:, None] * g1[None, :]).to(dev).expand(3, 1, 21, 21).contiguous()

    idx = np.random.RandomState(0).permutation(len(val))[:a.n * 3]
    rows, t_ns, t_gc, done = [], 0., 0., 0
    for j in idx:
        if done >= a.n:
            break
        x, _ = val[int(j)]
        x = x[None].to(dev)
        with torch.no_grad():
            A = bb.features(x); c = int(bb.head(A).argmax(1))
        blur = F.conv2d(F.pad(x, (10,) * 4, mode="reflect"), ker, groups=3)

        t0 = time.time(); out = ns(x, c); t_ns += time.time() - t0
        t0 = time.time(); gc = gradcam(bb, x, c); t_gc += time.time() - t0

        r = {"i": int(j), "rank": out["rank"]}
        for name, m in (("N", out["N"]), ("S", out["S"]), ("GradCAM", gc)):
            r[f"del_{name}"] = auc_curve(bb, x, c, m, "del")
            r[f"ins_{name}"] = auc_curve(bb, x, c, m, "ins", blur=blur)
        rows.append(r); done += 1
        if done % 20 == 0:
            print(f"  {done}/{a.n}")

    def agg(k):
        v = np.array([r[k] for r in rows])
        b = np.array([np.mean(np.random.choice(v, len(v))) for _ in range(2000)])
        return float(v.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))

    res = {"n": done, "rank": a.rank,
           "sec_per_image": {"NS-CAM": t_ns / done, "GradCAM": t_gc / done},
           "partial_forwards_nscam": 2 * a.rank}
    print(f"\n{'':10s} {'deletion (lower better)':>28s}   {'insertion (higher better)':>28s}")
    for name in ("GradCAM", "N", "S"):
        d = agg(f"del_{name}"); i = agg(f"ins_{name}")
        res[name] = {"deletion": d, "insertion": i}
        print(f"{name:10s} {d[0]:.4f} [{d[1]:.4f},{d[2]:.4f}]   "
              f"{i[0]:.4f} [{i[1]:.4f},{i[2]:.4f}]")
    print(f"\nsec/image  NS-CAM {t_ns/done:.4f}   Grad-CAM {t_gc/done:.4f}")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"summary": res, "rows": rows}, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
