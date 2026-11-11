"""Evaluate one K2 model: top-1, necessity / sufficiency maps of every 1x1 cell for the
ground-truth class (float64, 8 training-split references, averaged), localisation metrics,
separation of the two maps, and the substitution checks.

    python k2_eval.py <H0|H1|H2|H3> <seed> [--ckpt PATH] [--nmax N] [--suffix S]

Maps (cell grid 14x14, T = 196), per reference B and image A, class c:
  N(t) = z_c(A) - z_c(A with cell t <- B's cell t),  S(t) = z_c(B with cell t <- A's cell t) - z_c(B)
  H0  wlib.ns_gap   (N == S identically; the collapsed map) and plain CAM = <w_c, x_t>  (wlib w2 convention)
  H1  wlib.ns_lse   (closed form, T = +1, cancellation-safe)
  H2  substitution on the cell-local quantities (attention logit e_t, class projection g_t):
      z = sum_t softmax(e)_t g_t + b, exclusion sums by an explicit off-diagonal mask (no subtraction)
  H3  wlib.ns_pooled (substitution moves the pooled vector by (x^B_t - x^A_t)/T exactly)
Every map is checked against rtlib.NS_literal (literal float64 loop over the head module) on the
first NVER test images x the first 2 references.
Metrics: wlib.upsample (224, bilinear, min-max), wlib.box_curve (100 thresholds), MaxBoxAccV2
(0.3/0.5/0.7), strict pointing (peak inside the box), tolerant pointing (15 px), best box IoU
(max over thresholds) -- the conventions of scripts/wsol/w2_run.py.
"""
import os, sys, time, json, copy, argparse
import numpy as np
import torch
import cv2
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from scipy.stats import rankdata
cv2.setNumThreads(1)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import k2lib as L
W = L.W
import rtlib  # noqa: E402

EVAL_RES = W.PROTOCOL['eval_res']
NTHR = W.PROTOCOL['n_thresholds']
THR = np.arange(NTHR) / NTHR
DELTAS = tuple(W.PROTOCOL['deltas'])
NVER = 3


def pear(n, s):
    if n.std() > 1e-12 and s.std() > 1e-12:
        return float(np.corrcoef(n, s)[0, 1])
    return np.nan


def spear(n, s):
    if n.std() > 1e-12 and s.std() > 1e-12:
        rx, ry = rankdata(n), rankdata(s)
        return float(np.corrcoef(rx, ry)[0, 1])
    return np.nan


def ns_attn(eA, gA, eB, gB):
    """gated-attention head, substitution of single cells, float64.
    z(X) = sum_t w_t g_t / sum_t w_t (+ bias, which cancels), w = exp(e)."""
    T = len(eA)
    M = max(eA.max(), eB.max())
    wA, wB = np.exp(eA - M), np.exp(eB - M)
    off = 1.0 - np.eye(T)
    ZA, ZB = off @ wA, off @ wB                  # sum over u != t, no subtraction
    PA, PB = off @ (wA * gA), off @ (wB * gB)
    zA = (wA * gA).sum() / wA.sum()
    zB = (wB * gB).sum() / wB.sum()
    N = zA - (PA + wB * gB) / (ZA + wB)
    S = (PB + wA * gA) / (ZB + wA) - zB
    return N, S


class Head64:
    """float64 numpy view of a trained head; per-cell quantities and the map estimators."""

    def __init__(s, h, mod):
        s.h = h
        g = lambda t: t.detach().double().cpu().numpy()   # noqa: E731
        s.Wf, s.bf = g(mod.fc.weight), g(mod.fc.bias)
        if h == 'H2':
            s.V, s.bV, s.U, s.bU = g(mod.V.weight), g(mod.V.bias), g(mod.U.weight), g(mod.U.bias)
            s.w, s.bw = g(mod.w.weight)[0], float(g(mod.w.bias)[0])
        if h == 'H3':
            s.lg, s.lb, s.eps = g(mod.ln.weight), g(mod.ln.bias), float(mod.ln.eps)
        s.mod64 = copy.deepcopy(mod).double().eval()

    def att(s, X):
        a = np.tanh(X @ s.V.T + s.bV) * (1.0 / (1.0 + np.exp(-(X @ s.U.T + s.bU))))
        return a @ s.w + s.bw

    def ln(s, V):
        mu = V.mean(1, keepdims=True); var = V.var(1, keepdims=True)
        return (V - mu) / np.sqrt(var + s.eps) * s.lg + s.lb

    def logits(s, X):
        if s.h == 'H0':
            return X.mean(0) @ s.Wf.T + s.bf
        if s.h == 'H1':
            Sc = X @ s.Wf.T + s.bf
            m = Sc.max(0)
            return m + np.log(np.exp(Sc - m).sum(0)) - np.log(len(X))
        if s.h == 'H2':
            e = s.att(X); a = np.exp(e - e.max()); a /= a.sum()
            return (a @ X) @ s.Wf.T + s.bf
        return s.ln(X.mean(0)[None])[0] @ s.Wf.T + s.bf

    def prep_ref(s, XB):
        """per-reference quantities reused for every image."""
        if s.h in ('H0', 'H1'):
            return XB @ s.Wf.T + s.bf                    # (T,K) per-cell scores
        if s.h == 'H2':
            return (s.att(XB), XB @ s.Wf.T)              # (T,), (T,K)
        return None

    def maps(s, X, c, refs, refq):
        """-> list of (N_k, S_k) over the references."""
        out = []
        if s.h in ('H0', 'H1'):
            a = X @ s.Wf[c] + s.bf[c]
            for q in refq:
                be = q[:, c]
                out.append(W.ns_gap(a, be) if s.h == 'H0' else W.ns_lse(a, be, 1.0, 14, 14))
        elif s.h == 'H2':
            eA, gA = s.att(X), X @ s.Wf[c]
            for (eB, gB) in refq:
                out.append(ns_attn(eA, gA, eB, gB[:, c]))
        else:
            Wc, bc = s.Wf[c], s.bf[c]
            fn = lambda V: s.ln(V) @ Wc + bc            # noqa: E731
            for XB in refs:
                out.append(W.ns_pooled(X, XB, fn))
        return out

    def literal(s, X, XB, c):
        f = lambda Z: s.mod64(Z.transpose(1, 2))         # (B,T,C) -> (B,K)  # noqa: E731
        N, S, _, _ = rtlib.NS_literal(f, torch.from_numpy(X).cuda(), torch.from_numpy(XB).cuda(), c)
        return N, S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('head'); ap.add_argument('seed', type=int)
    ap.add_argument('--ckpt', default=''); ap.add_argument('--nmax', type=int, default=0)
    ap.add_argument('--suffix', default='')
    a = ap.parse_args()
    t0 = time.time()

    def log(*x):
        print('[%6.0fs]' % (time.time() - t0), *x, flush=True)

    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    tg = L.tag(a.head, a.seed) + a.suffix
    ck = torch.load(a.ckpt or L.ckpt_path(a.head, a.seed), map_location='cpu', weights_only=False)
    net = L.Net(a.head, pretrained=False)
    net.load_state_dict(ck['state_dict']); net = net.cuda().eval()
    H = Head64(a.head, net.head)
    man = W.cub_manifest()
    recs = man['ev'][:a.nmax] if a.nmax else man['ev']
    _, ev_tf = L.tfs()
    log('%s  ckpt steps %d  n_test %d' % (tg, ck['steps'], len(recs)))

    @torch.no_grad()
    def cells_and_logits(x):
        A = net.cells(x.cuda())
        return A.transpose(1, 2).double().cpu().numpy(), net.head(A).float().cpu().numpy()

    bidx = L.bank_indices(len(man['fit']))
    xb = torch.stack([ev_tf(Image.open(man['fit'][i]['path']).convert('RGB')) for i in bidx])
    BANK, _ = cells_and_logits(xb)                       # (8,T,C) float64
    REFQ = [H.prep_ref(BANK[k]) for k in range(len(bidx))]
    log('references: fit-split indices %s' % bidx)

    MAPS = ['collapsed', 'cam'] if a.head == 'H0' else ['N', 'S']
    curves = {k: [] for k in MAPS}; pts = {k: [] for k in MAPS}; ptt = {k: [] for k in MAPS}
    mstore = {k: [] for k in MAPS}
    sep = {k: [] for k in ('sep1_pearson', 'sep1_spearman', 'sep8_pearson', 'sep8_spearman')}
    correct, correct64, logit_err = [], [], []
    ver = []
    POOL = ThreadPoolExecutor(8)
    dl = torch.utils.data.DataLoader(L.DS(recs, ev_tf), batch_size=32, shuffle=False, num_workers=4)
    n = 0
    for x, y, idx in dl:
        Xb, Zm = cells_and_logits(x)
        for j in range(Xb.shape[0]):
            i = int(idx[j]); r = recs[i]; X = Xb[j]; c = int(r['label'])
            z64 = H.logits(X)
            correct.append(int(np.argmax(Zm[j]) == c)); correct64.append(int(np.argmax(z64) == c))
            logit_err.append(float(np.abs(z64 - Zm[j]).max()))
            per = H.maps(X, c, BANK, REFQ)
            Nm = np.mean([p[0] for p in per], 0); Sm = np.mean([p[1] for p in per], 0)
            sep['sep1_pearson'].append(np.nanmean([1 - pear(p[0], p[1]) for p in per]))
            sep['sep1_spearman'].append(np.nanmean([1 - spear(p[0], p[1]) for p in per]))
            sep['sep8_pearson'].append(1 - pear(Nm, Sm))
            sep['sep8_spearman'].append(1 - spear(Nm, Sm))
            if i < NVER:
                for k in range(2):
                    Nl, Sl = H.literal(X, BANK[k], c)
                    Nc, Sc = per[k]
                    sc = max(np.abs(Nl).max(), np.abs(Sl).max())
                    v = dict(image=i, ref=k, scale=float(sc),
                             N_closed_vs_literal_rel=float(np.abs(Nc - Nl).max() / sc),
                             S_closed_vs_literal_rel=float(np.abs(Sc - Sl).max() / sc),
                             N_vs_S_literal_rel=float(np.abs(Nl - Sl).max() / sc),
                             N_vs_S_literal_abs=float(np.abs(Nl - Sl).max()),
                             spearman_literal=spear(Nl, Sl), pearson_literal=pear(Nl, Sl))
                    ver.append(v)
                    log('  check img %d ref %d: |N_cf-N_lit|/s %.2e |S_cf-S_lit|/s %.2e '
                        '|N_lit-S_lit|/s %.2e (abs %.2e) rho %.6f'
                        % (i, k, v['N_closed_vs_literal_rel'], v['S_closed_vs_literal_rel'],
                           v['N_vs_S_literal_rel'], v['N_vs_S_literal_abs'], v['spearman_literal']))
            mp = ({'collapsed': Nm, 'cam': X @ H.Wf[c]} if a.head == 'H0' else {'N': Nm, 'S': Sm})
            w0, h0 = Image.open(r['path']).size
            sx, sy = EVAL_RES / w0, EVAL_RES / h0
            gb = [[b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy] for b in r['boxes']]
            U = {k: W.upsample(mp[k], 14, 14, EVAL_RES) for k in MAPS}
            for k, cv in zip(MAPS, POOL.map(lambda kk: W.box_curve(U[kk], gb, THR), MAPS)):
                curves[k].append(cv)
            for k in MAPS:
                u = U[k]
                pk = np.unravel_index(np.argmax(u), u.shape)
                pts[k].append(any(b[0] <= pk[1] <= b[2] and b[1] <= pk[0] <= b[3] for b in gb))
                ptt[k].append(any(b[0] - 15 <= pk[1] <= b[2] + 15 and
                                  b[1] - 15 <= pk[0] <= b[3] + 15 for b in gb))
                mstore[k].append(mp[k])
            n += 1
        if (n // 32) % 20 == 0:
            log('  %d / %d' % (n, len(recs)))

    res = dict(head=a.head, seed=a.seed, head_desc=L.HEAD_DESC[a.head], n_test=n,
               ckpt_steps=ck['steps'], micro=ck.get('micro'), accum=ck.get('accum'),
               references=bidx, maps=MAPS,
               top1=float(np.mean(correct)), top1_float64_head=float(np.mean(correct64)),
               max_logit_abs_err_float64_vs_module=float(np.max(logit_err)),
               separation={k: float(np.nanmean(v)) for k, v in sep.items()},
               separation_n_finite={k: int(np.isfinite(v).sum()) for k, v in sep.items()},
               checks=ver, per_map={})
    arrs = dict(correct=np.array(correct, bool), **{k: np.array(v) for k, v in sep.items()})
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
    if ver:
        res['check_worst'] = {f: float(max(v[f] for v in ver)) for f in
                              ('N_closed_vs_literal_rel', 'S_closed_vs_literal_rel',
                               'N_vs_S_literal_rel', 'N_vs_S_literal_abs')}
    np.savez_compressed(os.path.join(L.OUT, 'per_image', '%s.npz' % tg), **arrs)
    L.save_json(os.path.join(L.OUT, 'eval_%s.json' % tg), res)
    log('top-1 %.4f (float64 head %.4f, max logit err %.2e)'
        % (res['top1'], res['top1_float64_head'], res['max_logit_abs_err_float64_vs_module']))
    log('separation %s' % json.dumps(res['separation']))
    for k in MAPS:
        log('%-10s %s' % (k, json.dumps(res['per_map'][k])))
    if ver:
        log('worst check %s' % json.dumps(res['check_worst']))


if __name__ == '__main__':
    main()
