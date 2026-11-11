"""TASK 3 complement -- clean movement statistics for the operators that FAIL
the invariance condition (tanh-sigmoid gated attention, GeM p=2, 3).

t3_gradcam.json reports max_rel_gradcam_minus_cam = 1.8e298 and max movement
= 1 for these operators: both maxima come from random (non-predicted) classes
whose CAM is identically zero after the ReLU (division by ~1e-300). Here the
same operators (make_ops from t3_gradcam.py, same gate weights) are evaluated
on the PREDICTED class only, plus a count of the zero-CAM cases among the
480 (image, class) pairs of t3."""
import os, sys, numpy as np, torch, torch.nn.functional as Fn
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import theory_sweeps_lib as L
import t3_gradcam as T3


def main():
    from torchvision import models
    tv = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    W = tv.fc.weight.data.double().to(L.dev); b = tv.fc.bias.data.double().to(L.dev)
    imgs, _, _ = L.probe1_sample()
    X = L.batch('val', imgs).double()
    C, N = X.shape[1], X.shape[2]
    ops = T3.make_ops(C, X)
    rs = np.random.RandomState(0)
    # zero-CAM count over the t3 class draw (predicted + 9 random per image)
    zero = 0; tot = 0
    cls_all = []
    for k in range(X.shape[0]):
        A = X[k]
        with torch.no_grad():
            cls = [int((W @ A.mean(-1) + b).argmax())] + list(rs.choice(1000, 9, replace=False))
        cls_all.append(cls)
        for c in cls:
            cam = Fn.relu(W[c] @ A); tot += 1; zero += int(cam.max().item() == 0)
    res = {'zero_cam_pairs': zero, 'pairs': tot, 'pred_class_only': {}}
    for nm in ('mean', 'linear-gate attention', 'tanh-sigmoid gated attention (Ilse)', 'GeM p=2', 'GeM p=3',
               'channel-wise max', 'max', 'top-8 of 49'):
        kind, zf = ops[nm]
        rel, mv, rho, gcr = [], [], [], []
        for k in range(X.shape[0]):
            A = X[k]; c = cls_all[k][0]
            alpha, gcam = T3.gradcam(zf, A, W, b, c)
            rel.append((alpha - W[c] / N).abs().max().item() / (W[c].abs().max().item() / N))
            cam = Fn.relu(W[c] @ A) / N
            mv.append((T3.nrm(gcam) - T3.nrm(cam)).abs().max().item())
            gcr.append((gcam - cam).abs().max().item() / cam.abs().max().item())
            rho.append(spearmanr(gcam.cpu().numpy(), cam.cpu().numpy()).statistic)
        res['pred_class_only'][nm] = dict(
            kind=kind, n=len(mv), median_rel_alpha_dev=float(np.median(rel)), max_rel_alpha_dev=float(np.max(rel)),
            max_rel_gradcam_minus_cam=float(np.max(gcr)), median_rel_gradcam_minus_cam=float(np.median(gcr)),
            mean_normalised_movement=float(np.mean(mv)), max_normalised_movement=float(np.max(mv)),
            mean_spearman=float(np.nanmean(rho)), min_spearman=float(np.nanmin(rho)))
        p = res['pred_class_only'][nm]
        print(f"{nm:38s} rel-alpha med={p['median_rel_alpha_dev']:.3g} | GC-CAM rel max={p['max_rel_gradcam_minus_cam']:.3g} "
              f"| move mean={p['mean_normalised_movement']:.3g} max={p['max_normalised_movement']:.3g} "
              f"| rho mean={p['mean_spearman']:.4f} min={p['min_spearman']:.4f}", flush=True)
    print('zero-CAM pairs', zero, '/', tot)
    L.save('t3b_gradcam_predclass.json', res)


if __name__ == '__main__':
    main()
