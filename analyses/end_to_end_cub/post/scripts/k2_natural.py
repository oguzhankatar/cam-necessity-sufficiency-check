"""POST HOC (written after K2 was finished): each head's own "natural" map,
scored with exactly the localisation pipeline of analyses/end_to_end_cub/scripts/k2_eval.py.

    python k2_natural.py <H0|H1|H2|H3> <seed> [--nmax N]

Nothing in analyses/end_to_end_cub/scripts or analyses/end_to_end_cub/out is modified; everything is imported. Same test images
(wlib.cub_manifest()['ev'], in order), same preprocessing (k2lib.tfs eval transform: whole image
resized to 448 x 448, bilinear, no crop), same ground-truth class, same box rescaling to the 224 x 224
evaluation frame, same wlib.upsample (bilinear to 224, min-max), wlib.box_curve (100 thresholds),
wlib.maxboxacc_v2, strict / 15 px pointing on the argmax, best IoU = max of the curve.

Backbone in float32 (TF32 off, as in k2_eval); the 14 x 14 x 2048 feature map is cast to float64 and
every natural map, including Grad-CAM (autograd through a float64 copy of the trained head), is
computed in float64. Class c = ground truth; x_t = feature vector of cell t; w_c, b_c = row c of the
final linear layer.

  H0  cam        <w_c, x_t>                                  (= stored 'cam'; sanity check)
      gradcam    ReLU(sum_k mean_t(dz_c/dx_tk) x_tk)         (= ReLU(<w_c,x_t>)/196: sanity)
  H1  cam        s_t = <w_c, x_t> + b_c (the per-cell class score the LSE pools)
      gradcam    standard Grad-CAM of z_c (LSE logit)
  H2  attn       a_t = softmax_t(e_t), gated-attention weights (class independent)
      attn_ev    a_t <w_c, x_t>: z_c = <w_c, sum_t a_t x_t> + b_c = sum_t a_t <w_c, x_t> + b_c,
                 so a_t <w_c, x_t> is the cell's additive contribution to the logit (bias excluded:
                 it is not attributable to a cell)
      gradcam    standard Grad-CAM of z_c
  H3  gradcam    standard Grad-CAM of z_c = <w_c, LN(mean_t x_t)> + b_c, i.e.
                 ReLU(<J^T w_c, x_t>)/196 with J the LayerNorm Jacobian at the pooled vector
      cam        <w_c, x_t>, LayerNorm ignored (reference)
"""
import os, sys, time, json, argparse
import numpy as np
import torch
import cv2
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
cv2.setNumThreads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
POST = os.path.dirname(HERE)
K2S = os.path.join(os.path.dirname(POST), 'scripts')
sys.path.insert(0, K2S)
import k2lib as L          # noqa: E402
import k2_eval as KE       # noqa: E402
W = L.W

POUT = os.path.join(POST, 'out')
os.makedirs(os.path.join(POUT, 'per_image'), exist_ok=True)
os.makedirs(os.path.join(POUT, 'logs'), exist_ok=True)
EVAL_RES, THR = KE.EVAL_RES, KE.THR
DELTAS = KE.DELTAS
NAT = {'H0': ['cam', 'gradcam'], 'H1': ['cam', 'gradcam'],
       'H2': ['attn', 'attn_ev', 'gradcam'], 'H3': ['gradcam', 'cam']}


def gradcam64(mod64, Xb, cs):
    """standard Grad-CAM at the head input (= last feature map), float64.
    Xb (B,T,C) float64 numpy, cs class per image -> (B,T) numpy."""
    A = torch.from_numpy(Xb).cuda().transpose(1, 2).contiguous().requires_grad_(True)   # (B,C,T)
    z = mod64(A)                                                                        # (B,K)
    ci = torch.as_tensor(cs, device=z.device)
    z.gather(1, ci[:, None]).sum().backward()      # images independent: per-image gradients
    G = A.grad                                     # (B,C,T)
    alpha = G.mean(-1, keepdim=True)               # channel weights = spatial mean of gradient
    cam = torch.relu((alpha * A).sum(1))           # (B,T)
    return cam.detach().cpu().numpy(), z.detach().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('head'); ap.add_argument('seed', type=int); ap.add_argument('--nmax', type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()

    def log(*x):
        print('[%6.0fs]' % (time.time() - t0), *x, flush=True)

    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    h = a.head
    tg = L.tag(h, a.seed)
    ck = torch.load(L.ckpt_path(h, a.seed), map_location='cpu', weights_only=False)
    net = L.Net(h, pretrained=False)
    net.load_state_dict(ck['state_dict']); net = net.cuda().eval()
    H = KE.Head64(h, net.head)
    mod64 = H.mod64.cuda()
    man = W.cub_manifest()
    recs = man['ev'][:a.nmax] if a.nmax else man['ev']
    _, ev_tf = L.tfs()
    MAPS = NAT[h]
    log('%s natural maps %s  n_test %d' % (tg, MAPS, len(recs)))

    curves = {k: [] for k in MAPS}; pts = {k: [] for k in MAPS}; ptt = {k: [] for k in MAPS}
    mstore = {k: [] for k in MAPS}
    zero_maps = {k: 0 for k in MAPS}
    chk = dict(gradcam_vs_closed_form_rel=0.0, head64_logit_vs_float32_abs=0.0)
    POOL = ThreadPoolExecutor(8)
    dl = torch.utils.data.DataLoader(L.DS(recs, ev_tf), batch_size=32, shuffle=False, num_workers=4)
    n = 0
    for x, y, idx in dl:
        with torch.no_grad():
            Af = net.cells(x.cuda())
            Zf = net.head(Af).float().cpu().numpy()
            Xb = Af.transpose(1, 2).double().cpu().numpy()      # (B,T,C) float64, as in k2_eval
        cs = [int(recs[int(i)]['label']) for i in idx]
        GC, Z64 = gradcam64(mod64, Xb, cs)
        chk['head64_logit_vs_float32_abs'] = max(chk['head64_logit_vs_float32_abs'],
                                                 float(np.abs(Z64 - Zf).max()))
        jobs = []
        for j in range(Xb.shape[0]):
            i = int(idx[j]); r = recs[i]; X = Xb[j]; c = cs[j]
            assert c == int(y[j])
            g = X @ H.Wf[c]                                  # <w_c, x_t>, float64
            mp = {}
            if h == 'H0':
                mp = {'cam': g, 'gradcam': GC[j]}
                ref = np.maximum(g, 0) / X.shape[0]          # closed form of Grad-CAM for GAP+linear
            elif h == 'H1':
                s = g + H.bf[c]
                mp = {'cam': s, 'gradcam': GC[j]}
                ref = np.maximum(g, 0) / X.shape[0]          # alpha = mean_t softmax_t w_c = w_c/T
            elif h == 'H2':
                e = H.att(X); al = np.exp(e - e.max()); al /= al.sum()
                mp = {'attn': al, 'attn_ev': al * g, 'gradcam': GC[j]}
                ref = None
            else:
                mu = X.mean(0); v = mu - mu.mean(); sd = np.sqrt((v * v).mean() + H.eps)
                xh = v / sd; u = H.lg * H.Wf[c]
                Jw = (u - u.mean() - xh * (u * xh).mean()) / sd    # J^T w_c for LayerNorm
                ref = np.maximum(X @ Jw, 0) / X.shape[0]
                mp = {'gradcam': GC[j], 'cam': g}
            if ref is not None:
                sc = max(np.abs(ref).max(), 1e-300)
                chk['gradcam_vs_closed_form_rel'] = max(chk['gradcam_vs_closed_form_rel'],
                                                        float(np.abs(GC[j] - ref).max() / sc))
            w0, h0 = Image.open(r['path']).size
            sx, sy = EVAL_RES / w0, EVAL_RES / h0
            gb = [[b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy] for b in r['boxes']]
            for k in MAPS:
                u = W.upsample(mp[k], 14, 14, EVAL_RES)
                zero_maps[k] += int(not (u.max() > 0))
                pk = np.unravel_index(np.argmax(u), u.shape)
                pts[k].append(any(b[0] <= pk[1] <= b[2] and b[1] <= pk[0] <= b[3] for b in gb))
                ptt[k].append(any(b[0] - 15 <= pk[1] <= b[2] + 15 and
                                  b[1] - 15 <= pk[0] <= b[3] + 15 for b in gb))
                mstore[k].append(mp[k])
                jobs.append((k, u, gb))
            n += 1
        for (k, _, _), cv in zip(jobs, POOL.map(lambda jb: W.box_curve(jb[1], jb[2], THR), jobs)):
            curves[k].append(cv)
        if (n // 32) % 20 == 0:
            log('  %d / %d' % (n, len(recs)))

    res = dict(head=h, seed=a.seed, n_test=n, maps=MAPS, post_hoc=True, checks=chk,
               all_zero_maps=zero_maps, per_map={},
               precision='backbone float32 (TF32 off); maps and Grad-CAM float64')
    arrs = {}
    for k in MAPS:
        cv = np.stack(curves[k]).astype(np.float32)
        v, byd = W.maxboxacc_v2(cv, DELTAS)
        res['per_map'][k] = dict(MaxBoxAccV2=v, per_delta=dict(zip([str(d) for d in DELTAS], byd)),
                                 pointing_strict=float(np.mean(pts[k])),
                                 pointing_tol15=float(np.mean(ptt[k])),
                                 best_iou=float(cv.max(1).mean()))
        arrs['curve_' + k] = cv
        arrs['pt_strict_' + k] = np.array(pts[k], bool)
        arrs['pt_tol15_' + k] = np.array(ptt[k], bool)
        arrs['best_iou_' + k] = cv.max(1)
        arrs['map_' + k] = np.stack(mstore[k])
    sfx = '' if not a.nmax else '_n%d' % a.nmax
    np.savez_compressed(os.path.join(POUT, 'per_image', '%s_natural%s.npz' % (tg, sfx)), **arrs)
    L.save_json(os.path.join(POUT, 'natural_%s%s.json' % (tg, sfx)), res)
    log('checks %s  all-zero maps %s' % (json.dumps(chk), json.dumps(zero_maps)))
    for k in MAPS:
        log('%-8s %s' % (k, json.dumps(res['per_map'][k])))


if __name__ == '__main__':
    main()
