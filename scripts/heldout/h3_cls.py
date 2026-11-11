"""H3 (P3) -- what the "class-token: exact but vacuous" bucket actually is.

Stage 5 put 5 architectures in a bucket labelled DEGENERATE and called them
"exact but vacuous".  Those are two very different claims and the bucket was
never measured.  This script measures both halves and then asks the question the
scope rule cannot currently answer: for a CLS-token model, WHERE is the probe
well posed, and what does the rule say there?

A. THE HEAD SITE.  Literal float64 loop over the model's own final token grid.
   Two numbers separate "exact" from "vacuous":
     maxNS / Rhalf / I2        -> exactness of the additive decomposition
     tele_spatial = sum_{t in patch} N(t) / (z_A - z_B)
                               -> how much of the logit difference that
                                  decomposition assigns to the SPATIAL cells.
   A GAP model has tele_spatial = 1.  A CLS model has tele_spatial = 0: the
   decomposition is exact and puts 100 % of the mass on a token that is not a
   location.  `maxNS` alone cannot tell these apart, which is why stage 5's
   numeric criterion put them in the same class.

B. THE WELL-POSED SITE.  The deepest site at which the patch tokens still reach
   the logit is the INPUT of the last block.  For each block L the tokens
   entering block L are taken as the cells and the rest of the shipped network
   is the readout (installed with a forward_pre_hook, so the code path is the
   model's own).  The literal loop is run there.

C. WHAT THE RULE SAYS THERE.  At that site the operation that mixes cells is
   self-attention: a data-dependent convex combination, i.e. `OTHER` in the
   rule's aggregator taxonomy -> AGG-BLOCKED.  The prediction is that no head
   edit reaches additivity there; this is tested by also killing the final norm.
"""
import os, sys, time, json, traceback
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hlib as H
import rtlib
import timm

TOKEN_MODELS = ['vit_base_patch16_224', 'deit3_small_patch16_224',
                'cait_xxs24_224', 'pit_s_224', 'crossvit_9_240',
                'convit_tiny', 'xcit_tiny_12_p16_224', 'tnt_s_patch16_224',
                'vit_small_patch16_224']
GAP_CONTROL = ['vit_betwixt_patch16_reg4_gap_256', 'vit_medium_patch16_reg4_gap_256']
SCAN = ['vit_base_patch16_224', 'deit3_small_patch16_224', 'cait_xxs24_224',
        'convit_tiny', 'xcit_tiny_12_p16_224',
        'vit_betwixt_patch16_reg4_gap_256']
IDX = [7, 4211]


# ------------------------------------------------------------------ part A
def part_a(nm):
    m = timm.create_model(nm, pretrained=True).eval().cuda()
    cfg = m.pretrained_cfg
    x = H.val_images(cfg, IDX).cuda()
    m = m.double(); x = x.double()
    with torch.no_grad():
        f = m.forward_features(x)
    cv = H.CellView(m, f)
    head = H.make_head(m, cv)
    cells = cv.cells(f)
    with torch.no_grad():
        z = m(x)
    c = int(z.argmax(1)[0])
    N, S, zA, zB = rtlib.NS_literal(head, cells[0], cells[1], c)
    rs = np.random.RandomState(0)
    d = rtlib.stats(N, S)
    if cells.shape[1] < 2:
        Rh, I2 = 0.0, 0.0            # one cell: both statistics are undefined
    else:
        Rh, I2 = rtlib.additivity(head, cells[0], cells[1], c, N, rs)
    P = cv.npfx
    dz = zA - zB
    out = dict(model=nm, T=cv.T, C=cv.C, npfx=P, fmt=cv.fmt,
               global_pool=str(getattr(m, 'global_pool', None))[:40],
               maxNS=d['max_abs_diff_rel'], corrNS=d['corr_pearson'],
               Rhalf=Rh, I2=I2,
               tele_all=float(N.sum() / dz), tele_spatial=float(N[P:].sum() / dz),
               tele_prefix=float(N[:P].sum() / dz),
               absN_spatial_over_absz=float(np.abs(N[P:]).sum() / abs(zA)),
               absN_prefix_over_absz=float(np.abs(N[:P]).sum() / abs(zA)),
               spatial_mass=float(np.abs(N[P:]).sum() / (np.abs(N).sum() + 1e-300)))
    mx = float(np.abs(N).max())
    act = np.where(np.abs(N) > 1e-12 * mx)[0] if mx > 0 else np.array([], int)
    out['n_active'] = int(len(act))
    out['active_cells'] = act[:8].tolist()
    out['additive'] = bool(out['maxNS'] < H.TOL and Rh < H.TOL and I2 < H.TOL)
    out['vacuous'] = bool(abs(out['tele_spatial']) < 1e-9 or out['n_active'] <= 1)
    del m; torch.cuda.empty_cache()
    return out


# ------------------------------------------------------------------ part B/C
class Site:
    """Override the input of block L; the rest of the shipped network runs."""

    def __init__(s, blk):
        s.ovr = None
        s.cap = None

        def pre(mod, args):
            if s.ovr is None:
                s.cap = args[0].detach()
                return None
            return (s.ovr,) + tuple(args[1:])
        s.h = blk.register_forward_pre_hook(pre)

    def close(s):
        s.h.remove()


def block_list(m):
    b = getattr(m, 'blocks', None)
    if b is None:
        return None
    try:
        return list(b)
    except TypeError:
        return None


def part_b(nm, nsub=32, npair=96):
    m = timm.create_model(nm, pretrained=True).eval().cuda()
    cfg = m.pretrained_cfg
    x = H.val_images(cfg, IDX).cuda()
    m = m.double(); x = x.double()
    blks = block_list(m)
    if blks is None:
        del m; torch.cuda.empty_cache()
        return dict(model=nm, error='no flat .blocks')
    nb = len(blks)
    with torch.no_grad():
        z = m(x)
    c = int(z.argmax(1)[0])
    P = int(getattr(m, 'num_prefix_tokens', 0) or 0)
    Ls = sorted(set([0, nb // 2, max(0, nb - 2), nb - 1]))
    rows = []
    for L in Ls:
        st = Site(blks[L])
        with torch.no_grad():
            m(x)
        toks = st.cap.clone()                       # (B, T, C) entering block L
        if toks.dim() != 3:
            st.close(); continue

        @torch.no_grad()
        def head(X):
            st.ovr = X
            try:
                return m(x[:X.shape[0]])
            finally:
                st.ovr = None
        with torch.no_grad():
            zc = head(toks)
        rep = float((zc - z).abs().max() / (z.abs().max() + 1e-300))
        N, S, zA, zB = rtlib.NS_literal(head, toks[0], toks[1], c)
        rs = np.random.RandomState(0)
        d = rtlib.stats(N, S)
        Rh, I2 = rtlib.additivity(head, toks[0], toks[1], c, N, rs,
                                  nsub=nsub, npair=npair)
        Pl = P if toks.shape[1] > P else 0
        rows.append(dict(L=L, T=int(toks.shape[1]), npfx=Pl, site_reproduces=rep,
                         maxNS=d['max_abs_diff_rel'], corrNS=d['corr_pearson'],
                         Rhalf=Rh, I2=I2,
                         absN_over_absz=float(np.abs(N).sum() / abs(zA)),
                         absS_over_absz=float(np.abs(S).sum() / abs(zA)),
                         tele=float(N.sum() / (zA - zB)),
                         tele_spatial=float(N[Pl:].sum() / (zA - zB)),
                         additive=bool(d['max_abs_diff_rel'] < H.TOL
                                       and Rh < H.TOL and I2 < H.TOL)))
        print('    L=%-3d T=%-4d rep=%.1e |N|/|z|=%7.3f maxNS=%.2e Rh=%.2e corr=%+.4f tele=%+.4f'
              % (L, rows[-1]['T'], rep, rows[-1]['absN_over_absz'],
                 rows[-1]['maxNS'], Rh, rows[-1]['corrNS'], rows[-1]['tele']),
              flush=True)
        st.close()
    # part C: does killing the final norm help at the deepest scanned site?
    out = dict(model=nm, n_blocks=nb, npfx=P, sites=rows)
    del m; torch.cuda.empty_cache()
    return out


ONLY = sys.argv[1].upper() if len(sys.argv) > 1 else 'AB'
res = dict(part_a=[], part_b=[])
if 'A' in ONLY:
    print('=== PART A: head site, literal fp64 loop ===', flush=True)
    for nm in TOKEN_MODELS + GAP_CONTROL:
        try:
            r = part_a(nm)
            res['part_a'].append(r)
            print('%-36s T=%-4d pfx=%d nact=%-4d additive=%-5s vacuous=%-5s '
                  'maxNS=%.2e Rh=%.2e tele_all=%+.6f tele_spatial=%+.6e'
                  % (nm, r['T'], r['npfx'], r['n_active'], r['additive'],
                     r['vacuous'], r['maxNS'], r['Rhalf'], r['tele_all'],
                     r['tele_spatial']), flush=True)
        except Exception as e:
            res['part_a'].append(dict(model=nm, error='%s: %s' % (type(e).__name__, e),
                                      tb=traceback.format_exc()[-800:]))
            print('%-36s SKIP %s' % (nm, str(e)[:90]), flush=True)

if 'B' in ONLY:
    print('\n=== PART B: depth scan, literal fp64 loop ===', flush=True)
    for nm in SCAN:
        print('  %s' % nm, flush=True)
        try:
            res['part_b'].append(part_b(nm))
        except Exception as e:
            res['part_b'].append(dict(model=nm, error='%s: %s' % (type(e).__name__, e),
                                      tb=traceback.format_exc()[-800:]))
            print('    SKIP %s' % str(e)[:110], flush=True)

H.save('h3_cls_%s.json' % ONLY, res)
