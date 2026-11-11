"""W1 -- step 1 and step 2 of the stage-7A protocol.

  1. Does each backbone sit at the DEGENERATE point as shipped?  Frozen stage-5
     rule (structural) + the binding float64 certificate (rtlib literal loop).
  2. At the degenerate point, are the N-map and the S-map identical?  (control)
  3. Do the minimal edits actually leave the degenerate point?
     - LSE head at tau = +1.0   (all backbones)
     - LayerNorm relocation     (ViT: head(LN(mean_t z_t)) vs head(mean_t LN(z_t)))
  4. Are the fast closed forms used by the evaluation loop equal to rtlib's
     LITERAL loop?  (they must be, to 1e-12 relative, at BOTH perturbation
     scales: 1x1 cells and 3x3 blocks)
  5. Is Grad-CAM able to tell the two models apart at all?

Pretrained weights, real ImageNet val images, float64 throughout.  Shipped
ImageNet-1k classifier weights, so there is ZERO refitting in this file.
"""
import os, sys, json, math, time
import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import wlib as W
sys.path.insert(0, os.path.join(W.ROOT, 'scripts', 'heldout'))
sys.path.insert(0, os.path.join(W.ROOT, 'scripts', 'crosscheck'))
import frozen_rule as FR
import hlib as H
import rtlib
import timm

IDX = [7, 4211, 15123, 3007, 40555, 21001]     # 6 val images = 3 probe pairs
res = dict(env=dict(torch=torch.__version__, timm=timm.__version__),
           images=IDX, note='shipped ImageNet-1k weights, zero refit')


def cellview_of(m, f, npfx):
    if f.dim() == 4:
        return f.flatten(2).transpose(1, 2), f.shape[2], f.shape[3]
    n = f.shape[1] - npfx
    g = int(round(math.sqrt(n)))
    return f[:, npfx:], g, g


for bk in ['resnet50', 'vitb16']:
    nm = W.BACKBONES[bk]['timm']
    print('=== %s (%s)' % (bk, nm), flush=True)
    r = dict(timm=nm)
    m = timm.create_model(nm, pretrained=True).eval().cuda().double()
    cfg = m.pretrained_cfg
    x = H.val_images(cfg, IDX).cuda().double()
    with torch.no_grad():
        f = m.forward_features(x)
        zref = m(x)
    cls = zref.argmax(1).tolist()
    r['pred_classes'] = cls

    # ---- 1a. frozen stage-5 rule on the SHIPPED readout
    st = FR.structural(m, f)
    r['frozen_rule'] = dict(verdict=st['verdict'], agg=st['agg'],
                            agg_name=st['agg_name'], blocker=st['blocker'],
                            post=[list(p) for p in st['post']])
    print('   frozen rule:', st['verdict'], st['agg_name'], flush=True)

    # ---- 1b. float64 certificate on the SHIPPED readout (m.forward_head)
    cv = H.CellView(m, f)
    shipped = W.certify_head(H.make_head(m, cv), cv.cells(f), cls,
                             'shipped:%s' % bk)
    shipped.pop('per_pair')
    r['shipped_certificate'] = shipped
    print('   shipped  ADDITIVE=%s maxNS=%.2e Rhalf=%.2e I2=%.2e n_active=%d/%d'
          % (shipped['ADDITIVE'], shipped['maxNS'], shipped['Rhalf'],
             shipped['I2'], shipped['n_active'], shipped['T']), flush=True)

    # ---- the spatial cell matrix this project evaluates maps on
    npfx = int(getattr(m, 'num_prefix_tokens', 0) or 0)
    X, Hh, Ww = cellview_of(m, f, npfx)
    T, C = X.shape[1], X.shape[2]
    r.update(grid=[Hh, Ww], T=T, C=C, npfx=npfx)

    lin = m.fc if hasattr(m, 'fc') and isinstance(m.fc, nn.Linear) else m.head
    Wt, bt = lin.weight.data.double(), lin.bias.data.double()

    heads = {'gap': W.PoolHead(Wt, bt, 'gap'),
             'lse_tau1': W.PoolHead(Wt, bt, 'lse', tau=1.0)}
    if bk == 'vitb16':
        # LayerNorm relocation: the SAME LN, moved outside the mean.  Needs the
        # PRE-norm tokens, so re-run forward_features with m.norm disabled.
        ln = m.norm
        saved = m.norm
        m.norm = nn.Identity()
        with torch.no_grad():
            f_pre = m.forward_features(x)
        m.norm = saved
        Xpre, _, _ = cellview_of(m, f_pre, npfx)
        r['ln_relocation'] = dict(available=True, ln=type(ln).__name__)
        heads['lnpost'] = W.PoolHead(Wt, bt, 'lnpost', ln=ln)

    cert = {}
    for k, hd in heads.items():
        cells = Xpre if k == 'lnpost' else X
        c = W.certify_head(hd, cells, cls, '%s:%s' % (bk, k))
        c.pop('per_pair')
        cert[k] = c
        print('   %-9s ADDITIVE=%-5s maxNS=%.2e Rhalf=%.2e I2=%.2e corrNS=%+.8f'
              % (k, c['ADDITIVE'], c['maxNS'], c['Rhalf'], c['I2'],
                 c['corrNS_recorded_never_used']), flush=True)
    r['head_certificates'] = cert

    # ---- 2 + 4. N == S control, and closed form == literal loop, BOTH scales
    grp = W._blocks_3x3(Hh, Ww)
    chk = {}
    XA, XB = X[0], X[1]
    c0 = cls[0]
    aA = (XA.cpu().numpy() @ Wt.cpu().numpy()[c0] + float(bt[c0]))
    aB = (XB.cpu().numpy() @ Wt.cpu().numpy()[c0] + float(bt[c0]))
    for scale, groups in [('cell1x1', None), ('block3x3', grp)]:
        blk = groups is not None
        e = {}
        for k, hd in heads.items():
            A = Xpre[0] if k == 'lnpost' else XA
            B = Xpre[1] if k == 'lnpost' else XB
            if groups is None:
                Nl, Sl, _, _ = rtlib.NS_literal(hd, A, B, c0)
            else:
                Nl, Sl = W.ns_literal_groups(hd, A, B, c0, groups)
            An = A.cpu().numpy(); Bn = B.cpu().numpy()
            if k == 'gap':
                Nf, Sf = W.ns_gap(An @ Wt.cpu().numpy()[c0] + float(bt[c0]),
                                  Bn @ Wt.cpu().numpy()[c0] + float(bt[c0]),
                                  Hh, Ww, blk)
            elif k == 'lse_tau1':
                Nf, Sf = W.ns_lse(An @ Wt.cpu().numpy()[c0] + float(bt[c0]),
                                  Bn @ Wt.cpu().numpy()[c0] + float(bt[c0]),
                                  1.0, Hh, Ww, blk)
            else:
                lnc = ln.double()

                def fn(V, lnc=lnc, c0=c0):
                    with torch.no_grad():
                        v = torch.from_numpy(np.ascontiguousarray(V)).cuda().double()
                        return (lnc(v) @ Wt.T + bt)[:, c0].cpu().numpy()
                Nf, Sf = W.ns_pooled(An, Bn, fn, Hh, Ww, blk)
            sc = max(np.abs(Nl).max(), np.abs(Sl).max()) + 1e-300
            e[k] = dict(closed_vs_literal_N=float(np.abs(Nf - Nl).max() / sc),
                        closed_vs_literal_S=float(np.abs(Sf - Sl).max() / sc),
                        maxNS_literal=float(np.abs(Nl - Sl).max() / sc),
                        corr_literal=float(np.corrcoef(Nl, Sl)[0, 1]),
                        n_units=int(len(Nl)))
        chk[scale] = e
        print('   scale %-9s ' % scale + ' | '.join(
            '%s: N==S %.1e, closed-vs-literal %.1e' %
            (k, v['maxNS_literal'], max(v['closed_vs_literal_N'], v['closed_vs_literal_S']))
            for k, v in e.items()), flush=True)
    r['scale_checks'] = chk

    # ---- 5. can Grad-CAM tell the two heads apart?  alpha_c = mean_t dz/dx_tc
    gc = {}
    for k, hd in heads.items():
        A = (Xpre[0] if k == 'lnpost' else XA).clone()[None].requires_grad_(True)
        z = hd(A)[0, c0]
        g, = torch.autograd.grad(z, A)
        alpha = g[0].mean(0)                       # (C,)
        cam = torch.relu((A[0].detach() * alpha).sum(-1))
        gc[k] = cam.cpu().numpy()
    ks = list(gc)
    r['gradcam_pairwise_max_rel_diff'] = {
        '%s_vs_%s' % (ks[i], ks[j]):
            float(np.abs(gc[ks[i]] - gc[ks[j]]).max()
                  / (max(np.abs(gc[ks[i]]).max(), np.abs(gc[ks[j]]).max()) + 1e-300))
        for i in range(len(ks)) for j in range(i + 1, len(ks))}
    print('   gradcam diffs:', r['gradcam_pairwise_max_rel_diff'], flush=True)

    res[bk] = r
    del m
    torch.cuda.empty_cache()

# ---------------------------------------------------------------- VGG16 shipped
m = timm.create_model(W.BACKBONES['vgg16gap']['timm'], pretrained=True).eval().cuda().double()
x = H.val_images(m.pretrained_cfg, IDX).cuda().double()
with torch.no_grad():
    f = m.forward_features(x)
st = FR.structural(m, f)
res['vgg16_shipped'] = dict(timm=W.BACKBONES['vgg16gap']['timm'],
                            frozen_rule=dict(verdict=st['verdict'], agg=st['agg'],
                                             agg_name=st['agg_name'],
                                             blocker=st['blocker']),
                            note=('timm vgg16 has no average pool: forward_head starts '
                                  'with Conv2d(512,4096,k=7).  The WSOL literature uses '
                                  'VGG16-GAP, i.e. the conv trunk + a fitted GAP head; '
                                  'that is what this round builds, and it is certified '
                                  'in the per-dataset runs.'))
print('vgg16 shipped frozen rule:', st['verdict'], st['agg_name'], st['blocker'])
del m; torch.cuda.empty_cache()

W.save('w1_certify.json', res)
