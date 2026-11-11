"""TASK 3 -- the Grad-CAM invariance condition, float64, real features.

Fixed head (no fitting): the distributed torchvision ResNet-50 fc (1000 classes)
applied to the cached ResNet-50 layer4 features of the probe-1 sample (48
Imagenette validation images, 7x7 cells). Every operator below uses that same
W, b; only the pooling differs. For each operator, image and class c (the
predicted class plus nine random classes):

    alpha_c' = (1/N) sum_t dz_c / da_{c',t}     (the Grad-CAM channel weight)
    report   max_{c'} |alpha_c' - W[c,c']/N|, absolute and relative to max|W[c]|/N

Shift-equivariance check: sum_t dz/da_{c',t} = W[c,c'] for all c' <=> the
readout satisfies z(a + eps e_c' 1^T) = z(a) + eps W[c,c'].

Score-then-pool operators (s_t = W a_t + b, z = A(s)):
  mean; LSE at T in {+-0.25, +-1, +-8}; top-k and bottom-k mean (k-of-N) for
  k in {1, 2, 4, 8, 16, 24, 32, 48}; max; min; median; linear-gate attention
  z = sum_t softmax(u.a)_t s_t; tanh-sigmoid gated attention (Ilse et al.)
  z = sum_t softmax(w_a.(tanh(V a_t) * sigm(U a_t)))_t s_t.
Pool-then-score operators (z = W pool(a) + b): GeM p in {2, 3}; channel-wise
max; channel-wise LSE (T = 1).

Also: Grad-CAM == CAM (up to ReLU and scale) for every operator that passes,
and how far Grad-CAM moves for those that do not; and Grad-CAM across REFITTED
heads (gap_fc vs lse+-8, lse0.25, maxpool; probe-1 recipe), where the weights
differ.
"""
import os, sys, numpy as np, torch, torch.nn.functional as Fn
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import theory_sweeps_lib as L

torch.manual_seed(0)


def score_pool(agg):
    def z(A, W, b):                      # A (C,T) -> (K,)
        s = W @ A + b[:, None]           # (K,T)
        return agg(s)
    return z


def lse(T):
    return lambda s: (torch.logsumexp(T * s, -1) - np.log(s.shape[-1])) / T


def topk(k):
    return lambda s: s.topk(k, -1).values.mean(-1)


def botk(k):
    return lambda s: -(-s).topk(k, -1).values.mean(-1)


def make_ops(C, X):
    ops = {'mean': ('score-then-pool', score_pool(lambda s: s.mean(-1)))}
    for T in (-8.0, -1.0, -0.25, 0.25, 1.0, 8.0):
        ops[f'lse T={T:+g}'] = ('score-then-pool', score_pool(lse(T)))
    for k in (1, 2, 4, 8, 16, 24, 32, 48):
        ops[f'top-{k} of 49'] = ('score-then-pool', score_pool(topk(k)))
        ops[f'bottom-{k} of 49'] = ('score-then-pool', score_pool(botk(k)))
    ops['max'] = ('score-then-pool', score_pool(lambda s: s.max(-1).values))
    ops['min'] = ('score-then-pool', score_pool(lambda s: s.min(-1).values))
    ops['median'] = ('score-then-pool', score_pool(lambda s: s.median(-1).values))
    # gates scaled so the gate logits have a cross-cell s.d. of about 2 on real features
    g = torch.Generator(device='cpu').manual_seed(0)
    u = torch.randn(C, generator=g, dtype=torch.float64).to(L.dev)
    sd = (u @ X[0]).std(); u = u * (2.0 / sd)

    def lin_gate(A, W, b):
        p = torch.softmax(u @ A, -1)                    # (T,)
        return (W @ A + b[:, None]) @ p
    ops['linear-gate attention'] = ('score-then-pool', lin_gate)
    H = 128
    V = (torch.randn(H, C, generator=g, dtype=torch.float64) / np.sqrt(C)).to(L.dev)
    U = (torch.randn(H, C, generator=g, dtype=torch.float64) / np.sqrt(C)).to(L.dev)
    wa = torch.randn(H, generator=g, dtype=torch.float64).to(L.dev)
    sd = (wa @ (torch.tanh(V @ X[0]) * torch.sigmoid(U @ X[0]))).std(); wa = wa * (2.0 / sd)

    def tanh_gate(A, W, b):
        p = torch.softmax(wa @ (torch.tanh(V @ A) * torch.sigmoid(U @ A)), -1)
        return (W @ A + b[:, None]) @ p
    ops['tanh-sigmoid gated attention (Ilse)'] = ('score-then-pool', tanh_gate)
    for p_ in (2.0, 3.0):
        ops[f'GeM p={p_:g}'] = ('pool-then-score',
                                lambda A, W, b, p_=p_: W @ (A.clamp_min(1e-6) ** p_).mean(-1) ** (1 / p_) + b)
    ops['channel-wise max'] = ('pool-then-score', lambda A, W, b: W @ A.max(-1).values + b)
    ops['channel-wise LSE T=1'] = ('pool-then-score',
                                   lambda A, W, b: W @ (torch.logsumexp(A, -1) - np.log(A.shape[-1])) + b)
    return ops


def gradcam(zf, A, W, b, c):
    X = A.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        g, = torch.autograd.grad(zf(X, W, b)[c], X)
    alpha = g.sum(-1) / A.shape[-1]
    return alpha, Fn.relu(alpha @ A)


def nrm(m):
    m = m - m.min(); return m / (m.max() + 1e-300)


def main():
    from torchvision import models
    tv = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    W = tv.fc.weight.data.double().to(L.dev); b = tv.fc.bias.data.double().to(L.dev)
    imgs, _, _ = L.probe1_sample()
    X = L.batch('val', imgs).double()
    C, N = X.shape[1], X.shape[2]
    ops = make_ops(C, X)
    rs = np.random.RandomState(0)
    res = {'meta': dict(head='torchvision ResNet-50 fc (IMAGENET1K_V1), fixed', n_images=X.shape[0],
                        classes_per_image=10, cells=N, dtype='float64')}
    per = {}
    for nm, (kind, zf) in ops.items():
        dev_abs, dev_rel, shift, gc_cam, mv, rho = [], [], [], [], [], []
        for k in range(X.shape[0]):
            A = X[k]
            with torch.no_grad():
                cls = [int((W @ A.mean(-1) + b).argmax())] + list(rs.choice(1000, 9, replace=False))
            for c in cls:
                alpha, gcam = gradcam(zf, A, W, b, c)
                d = (alpha - W[c] / N).abs().max().item()
                dev_abs.append(d); dev_rel.append(d / (W[c].abs().max().item() / N))
                # direct shift-equivariance, finite difference with eps = 1 on 5 random channels
                with torch.no_grad():
                    z0 = zf(A, W, b)[c]
                    for ch in rs.choice(C, 2, replace=False):
                        Ap = A.clone(); Ap[ch] += 1.0
                        shift.append(abs((zf(Ap, W, b)[c] - z0).item() - W[c, ch].item())
                                     / (W[c].abs().max().item()))
                cam = Fn.relu(W[c] @ A) / N
                gc_cam.append((gcam - cam).abs().max().item() / (cam.abs().max().item() + 1e-300))
                # movement relative to the mean-pool Grad-CAM (== CAM)
                mv.append((nrm(gcam) - nrm(cam)).abs().max().item())
                if gcam.std() > 0 and cam.std() > 0:
                    rho.append(spearmanr(gcam.cpu().numpy(), cam.cpu().numpy()).statistic)
        per[nm] = dict(kind=kind, max_abs_alpha_minus_w_over_N=float(np.max(dev_abs)),
                       max_rel_alpha_minus_w_over_N=float(np.max(dev_rel)),
                       median_rel=float(np.median(dev_rel)),
                       max_rel_shift_residual=float(np.max(shift)),
                       max_rel_gradcam_minus_cam=float(np.max(gc_cam)),
                       max_normalised_movement_vs_CAM=float(np.max(mv)),
                       mean_normalised_movement_vs_CAM=float(np.mean(mv)),
                       mean_spearman_with_CAM=float(np.mean(rho)) if rho else None,
                       n=len(dev_abs))
        p = per[nm]
        print(f"{nm:38s} {kind:15s} max|a-w/N|={p['max_abs_alpha_minus_w_over_N']:.2e} "
              f"rel={p['max_rel_alpha_minus_w_over_N']:.2e} shift={p['max_rel_shift_residual']:.2e} "
              f"|GC-CAM|={p['max_rel_gradcam_minus_cam']:.2e} move={p['max_normalised_movement_vs_CAM']:.3g} "
              f"rho={p['mean_spearman_with_CAM']}", flush=True)
    res['fixed_head'] = per

    # ---------------- refitted heads: the weights differ, so the maps differ
    refit = {}
    Hs = {nm: L.get_head(nm, double=True)[0] for nm in ('gap_fc', 'lse-8.0', 'lse8.0', 'lse0.25', 'maxpool')}

    def gc_head(h, A, c):
        Xg = A[None].detach().clone().requires_grad_(True)
        with torch.enable_grad():
            g, = torch.autograd.grad(h(Xg)[0, c], Xg)
        return Fn.relu((g.mean(2, keepdim=True) * Xg).sum(1))[0].detach()
    base = Hs['gap_fc']
    for nm, h in Hs.items():
        if nm == 'gap_fc':
            continue
        mv, rho, mvfix = [], [], []
        # same weights dropped into the other operator (fixed head) for contrast
        hf = L.build(nm).to(L.dev).double(); hf.fc.load_state_dict(base.fc.state_dict()); hf.eval()
        for k in range(X.shape[0]):
            A = X[k]
            with torch.no_grad():
                c = int(base(A[None]).argmax(1))
            m0, m1, mf = gc_head(base, A, c), gc_head(h, A, c), gc_head(hf, A, c)
            mv.append((nrm(m0) - nrm(m1)).abs().max().item())
            mvfix.append((nrm(m0) - nrm(mf)).abs().max().item())
            if m0.std() > 0 and m1.std() > 0:
                rho.append(spearmanr(m0.cpu().numpy(), m1.cpu().numpy()).statistic)
        refit[nm] = dict(refit_mean_max_normalised_diff_vs_gap_fc=float(np.mean(mv)),
                         refit_mean_spearman_vs_gap_fc=float(np.mean(rho)),
                         fixed_weights_max_normalised_diff=float(np.max(mvfix)))
        print(f"refit {nm:8s} vs refit gap_fc: mean max|diff|={np.mean(mv):.3f} rho={np.mean(rho):.3f}; "
              f"same weights: max|diff|={np.max(mvfix):.2e}", flush=True)
    res['refit_heads'] = refit
    L.save('t3_gradcam.json', res)


if __name__ == '__main__':
    main()
