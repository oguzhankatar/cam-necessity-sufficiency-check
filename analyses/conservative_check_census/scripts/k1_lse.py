"""K1 theory check: channel-wise log-sum-exp / generalised-mean pooling followed by a linear
layer, against score-then-pool log-sum-exp, on real ResNet-50 layer4 features.

Backbone: torchvision ResNet-50, IMAGENET1K_V1 weights (local torch-hub cache, no download).
Images: 64 ImageNet-validation-like images, data/imagenette2-160/val, the first 6-7 files of
each of the 10 class folders in sorted order; resize 256, centre crop 224, ImageNet
normalisation -> layer4 features 2048 x 7 x 7 (T = 49 cells), float64 from here on.
Linear layer: the ResNet-50 fc (2048 -> 1000) for every head.
Heads (z = class logit; LSE_T(u) = (logsumexp(T u) - log n) / T, the article's form):
  ch-LSE_T   z = w_c . [LSE_T over cells of each channel] + b_c       T = 1, 4, 8
  ch-GeM_p   z = w_c . [(mean over cells of max(x,1e-6)^p)^(1/p)] + b_c p = 1, 3, 4, 8
  stp-LSE_T  z = LSE_T over cells of (w_c . x_t + b_c)                 T = 1, 4, 8
Maps (single reference): image i, reference image j = (i + 32) mod 64, class c = argmax of
the head on image i;  N_t = z(A) - z(A with cell t from B),  S_t = z(B with cell t from A) - z(B),
literal per-cell substitution. Reported: Spearman rho(N, S) over the 49 cells, mean and minimum
over the 64 images. For stp-LSE both N_t and S_t are increasing functions of
d_t = exp(T s_A,t) - exp(T s_B,t), so rho = 1 in closed form; the literal value is reported with
the closed-form check (rho(N, d) and rho(S, d)).
Output: ../out/k1_lse.json, ../out/k1_lse.log
"""
import os, sys, json, glob
import numpy as np
import torch
import torchvision
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
OUT = os.path.join(HERE, '..', 'out')
torch.set_num_threads(8)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
if dev == 'cuda':
    torch.cuda.set_per_process_memory_fraction(2.0e9 / torch.cuda.get_device_properties(0).total_memory)

VAL = os.path.join(ROOT, 'data', 'imagenette2-160', 'val')
cls = sorted(os.listdir(VAL))
files = []
k = 0
per = [7, 7, 7, 7, 6, 6, 6, 6, 6, 6]
for c, nper in zip(cls, per):
    files += sorted(glob.glob(os.path.join(VAL, c, '*.JPEG')))[:nper]
assert len(files) == 64

hub = os.path.join(torch.hub.get_dir(), 'checkpoints', 'resnet50-0676ba61.pth')
assert os.path.exists(hub), 'IMAGENET1K_V1 weights not in the local cache'
W = ResNet50_Weights.IMAGENET1K_V1
net = resnet50()
net.load_state_dict(torch.load(hub, map_location='cpu'))
net.eval().to(dev)
tf = W.transforms()
x = torch.stack([tf(Image.open(f).convert('RGB')) for f in files]).to(dev)
with torch.no_grad():
    h = net.maxpool(net.relu(net.bn1(net.conv1(x))))
    F4 = net.layer4(net.layer3(net.layer2(net.layer1(h))))
    acc_top1 = None
F4 = F4.double().cpu()                              # (64, 2048, 7, 7)
X = F4.flatten(2).transpose(1, 2).contiguous()      # (64, 49, 2048)
Wfc = net.fc.weight.detach().double().cpu(); bfc = net.fc.bias.detach().double().cpu()
n = X.shape[1]
logn = float(np.log(n))


def ch_lse(T):
    return lambda Xb: (torch.logsumexp(T * Xb, dim=-2) - logn) / T @ Wfc.T + bfc


def ch_gem(p):
    return lambda Xb: Xb.clamp(min=1e-6).pow(p).mean(-2).pow(1.0 / p) @ Wfc.T + bfc


def stp_lse(T):
    return lambda Xb: (torch.logsumexp(T * (Xb @ Wfc.T + bfc), dim=-2) - logn) / T


def maps(head, XA, XB, c):
    """literal per-cell substitution, batched over t (each row of the batch is one substituted map)."""
    idx = torch.arange(n)
    An = XA.expand(n, n, -1).clone(); An[idx, idx] = XB
    Bs = XB.expand(n, n, -1).clone(); Bs[idx, idx] = XA
    zA = head(XA[None])[0, c]; zB = head(XB[None])[0, c]
    N = (zA - head(An)[:, c]).numpy()
    S = (head(Bs)[:, c] - zB).numpy()
    return N, S


heads = [('ch-LSE', T, ch_lse(T)) for T in (1, 4, 8)] + [('ch-GeM', p, ch_gem(p)) for p in (1, 3, 4, 8)] + \
        [('stp-LSE', T, stp_lse(T)) for T in (1, 4, 8)]
res = dict(images=[os.path.relpath(f, ROOT) for f in files], n_cells=n, backbone='torchvision resnet50 IMAGENET1K_V1',
           torch=torch.__version__, torchvision=torchvision.__version__, heads=[])
res['reference'] = 'image (i + 32) mod 64'


def stable(name, T, XA, XB, c):
    """the same N_t, S_t by exact algebra, evaluated without subtracting two logits
    (a one-cell substitution changes each pooled value by a log1p / expm1 term)."""
    w, b = Wfc[c], bfc[c]
    if name == 'ch-LSE':
        dA = lse_delta(XA, XB, T)      # A with cell t from B
        dB = lse_delta(XB, XA, T)      # B with cell t from A
        return (-(dA @ w)).numpy(), (dB @ w).numpy()
    if name == 'ch-GeM':
        a = XA.clamp(min=1e-6).pow(T); bb = XB.clamp(min=1e-6).pow(T)
        MA = a.mean(0); MB = bb.mean(0)
        dA = MA.pow(1 / T) * torch.expm1(torch.log1p((bb - a) / n / MA) / T)
        dB = MB.pow(1 / T) * torch.expm1(torch.log1p((a - bb) / n / MB) / T)
        return (-(dA @ w)).numpy(), (dB @ w).numpy()
    sA = (XA @ w + b)[:, None]; sB = (XB @ w + b)[:, None]
    N = -lse_delta(sA, sB, T)[:, 0]
    S = lse_delta(sB, sA, T)[:, 0]
    return N.numpy(), S.numpy()


def lse_delta(Y, Ynew, T):
    """(n, C): change of LSE_T over cells (per column) when cell t of Y is replaced by Ynew[t].
    log1p form when the removed cell holds less than half of the mass; otherwise the
    leave-one-out log-sum-exp, where the change is large and the direct difference is exact."""
    L = torch.logsumexp(T * Y, 0)
    arg = torch.exp(T * Ynew - L) - torch.exp(T * Y - L)
    small = torch.log1p(arg.clamp(min=-0.5)) / T
    M = (T * Y)[None].expand(n, n, -1).clone()
    idx = torch.arange(n)
    M[idx, idx] = -float('inf')
    Lm = torch.logsumexp(M, 1)                                    # (n, C): without cell t
    direct = (torch.logaddexp(Lm, T * Ynew) - L) / T
    return torch.where(arg > -0.5, small, direct)


def ties(v):
    return int(len(v) - len(np.unique(v)))


lines = ['Channel-wise vs score-then-pool pooling, ResNet-50 layer4 (64 images, 49 cells, float64)',
         'rho = Spearman(N, S) over the 49 cells; exact = exact algebra evaluated stably; literal = per-cell loop on logits', '']
with torch.no_grad():
    for name, par, head in heads:
        rho, rho_lit, tie_lit, tie_ex, agree = [], [], 0, 0, []
        for i in range(64):
            j = (i + 32) % 64
            XA, XB = X[i], X[j]
            c = int(head(XA[None])[0].argmax())
            N, S = maps(head, XA, XB, c)
            Ne, Se = stable(name, par, XA, XB, c)
            rho.append(float(spearmanr(Ne, Se).statistic))
            rho_lit.append(float(spearmanr(N, S).statistic))
            tie_lit += ties(N) + ties(S); tie_ex += ties(Ne) + ties(Se)
            sc = max(np.abs(Ne).max(), np.abs(Se).max())
            agree.append(float(max(np.abs(N - Ne).max(), np.abs(S - Se).max()) / sc))
        r = dict(head=name, param=par, rho_mean=float(np.mean(rho)), rho_min=float(np.min(rho)),
                 rho_max=float(np.max(rho)), rho=rho, literal_rho_mean=float(np.mean(rho_lit)),
                 literal_rho_min=float(np.min(rho_lit)), literal_tied_values=tie_lit, exact_tied_values=tie_ex,
                 literal_vs_exact_max_rel_diff=float(np.max(agree)))
        if name == 'stp-LSE':
            r['closed_form_rho'] = 1.0
        res['heads'].append(r)
        lines.append('%-8s %s=%g  exact: rho mean %.4f min %.4f (tied values %d) | literal: rho mean %.4f min %.4f '
                     '(tied values %d) | max |literal - exact| / scale %.1e'
                     % (name, 'p' if name == 'ch-GeM' else 'T', par, r['rho_mean'], r['rho_min'], tie_ex,
                        r['literal_rho_mean'], r['literal_rho_min'], tie_lit, r['literal_vs_exact_max_rel_diff']))
txt = '\n'.join(lines) + '\n'
print(txt)
open(os.path.join(OUT, 'k1_lse.log'), 'w', encoding='utf-8').write(txt)
json.dump(res, open(os.path.join(OUT, 'k1_lse.json'), 'w'), indent=1)
