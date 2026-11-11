"""Task 1 (ISIC 2018, ResNet-50 of this study): can the ISIC primary (H1, Dice at a
two-fold cross-tuned top-q threshold, 1x1, T = +8) be recomputed in closed form?

Not for the archived run: m5_isic.py kept its checkpoint (isic_r50.pt) and feature
caches (F_isic_val.npy, F_isic_train.npy) in the system temp directory, and did not
write its per-item maps to outputs/medical (m5_isic_maps.json is absent); the C2
replicate's checkpoint (isic_r50_replicate.pt) was in the same temp cache.  None of
these files exists any more, so the archived maps cannot be rebuilt.

What is done instead: a new replicate is trained with m5_isic.train (imported,
same split RandomState(0), same recipe, seed 0), and on THIS model the maps are
computed twice, with the archive's estimator (m5_isic.NS_batched: forward
differences of the float64 head on the GPU) and with the closed form of Eq. 4.
Both are scored with the archive's Dice / hit / AP protocol (m5_isic.py lines
343-369, the same 2-fold q selection), so the difference isolates the estimator.
Deviation: timm's ImageNet weights for 'resnet50' are not in the local cache, so
the ImageNet initialisation is torchvision's cached resnet50-0676ba61.pth (same
architecture and key names; fc dropped) instead of timm's default tag; no file is
downloaded.
Outputs: out/t1_isic_cf.json, cache/isic_r50_r4.pt, cache/t1_isic_items.npz.
"""
import os
import time
import numpy as np
import torch
import medical_extra_lib as R
from medical_extra_lib import L, mlib

TV_R50 = os.path.join(os.path.expanduser('~'), '.cache', 'torch', 'hub', 'checkpoints', 'resnet50-0676ba61.pth')


def patch_timm():
    import timm
    orig = timm.create_model

    def create(name, pretrained=False, **kw):
        if name == 'resnet50' and pretrained:
            m = orig(name, pretrained=False, **kw)
            sd = torch.load(TV_R50, map_location='cpu')
            sd = {k: v for k, v in sd.items() if not k.startswith('fc.')}
            miss, unexp = m.load_state_dict(sd, strict=False)
            assert set(miss) == {'fc.weight', 'fc.bias'} and not unexp, (miss, unexp)
            print('init: torchvision resnet50-0676ba61.pth (fc re-initialised)', flush=True)
            return m
        return orig(name, pretrained=pretrained, **kw)
    timm.create_model = create


def main():
    import timm
    import m5_isic as m5
    patch_timm()
    names, y = mlib.isic_frame()
    rs = np.random.RandomState(0)
    perm = rs.permutation(len(names))
    ntr = int(.7 * len(names))
    tr, va = np.sort(perm[:ntr]), np.sort(perm[ntr:])
    ck = os.path.join(R.CACHE, 'isic_r50_r4.pt')
    t0 = time.time()
    if os.path.exists(ck):
        m = timm.create_model('resnet50', pretrained=False, num_classes=3).to(L.dev)
        m.load_state_dict(torch.load(ck))
    else:
        m, _, _ = m5.train(names, y, tr, va)
        torch.save(m.state_dict(), ck)
    m.eval()
    print('model ready %.0fs' % (time.time() - t0), flush=True)
    vnames = [names[i] for i in va]
    X = torch.stack([mlib.load_rgb(os.path.join(m5.IMG, n + '.jpg'), 224) for n in vnames])
    F = L.feats(lambda x: m.forward_features(x), X)
    W = m.fc.weight.detach().double().cpu()
    b = m.fc.bias.detach().double().cpu()
    Z = F.mean(1) @ W.numpy().T + b.numpy()
    auc = m5.aucs(Z, y[va])
    print('replicate AUC', auc, flush=True)
    yv = y[va]
    rsb = np.random.RandomState(0)
    bankidx = {}
    for c in range(3):
        cand = [i for i in range(len(vnames)) if yv[i] != c]
        bankidx[c] = sorted(rsb.choice(cand, 8, replace=False).tolist())
    heads = {'gap': mlib.head_of('gap', W, b).double().to(L.dev)}
    for k in R.OFF:
        heads[k] = mlib.head_of(k, W, b).double().to(L.dev)
    u1 = m5.unit_mask([np.array([t]) for t in range(49)], 49, L.dev)
    u3 = m5.unit_mask(mlib.blocks_of(7, 3), 49, L.dev)
    Wn, bn = W.numpy(), b.numpy()
    F64 = F.astype(np.float64)
    fd, cf = {}, {}
    n = len(vnames)
    for k in ['collapsed_1x1', 'collapsed_3x3'] + ['%s_%s_%s' % (p, t, u) for p in 'NS' for t in R.OFF for u in ('1x1', '3x3')]:
        fd[k] = np.zeros((n, 49))
        if not k.startswith('coll'):
            cf[k] = np.zeros((n, 49))
    for i in range(n):
        c = int(yv[i])
        XA = torch.from_numpy(F64[i]).to(L.dev)
        Xb = [torch.from_numpy(F64[j]).to(L.dev) for j in bankidx[c]]
        for hk, h in heads.items():
            for un, um in (('1x1', u1), ('3x3', u3)):
                Ns, Ss = [], []
                for XB in Xb:
                    N_, S_ = m5.NS_batched(h, XA, XB, c, um)
                    Ns.append(N_)
                    Ss.append(S_)
                if hk == 'gap':
                    fd['collapsed_%s' % un][i] = np.mean(Ns, 0)
                else:
                    fd['N_%s_%s' % (hk, un)][i] = np.mean(Ns, 0)
                    fd['S_%s_%s' % (hk, un)][i] = np.mean(Ss, 0)
        sa = F64[i] @ Wn[c] + bn[c]
        SB = [F64[j] @ Wn[c] + bn[c] for j in bankidx[c]]
        for t in R.OFF:
            T = float(t[3:])
            for un in ('1x1', '3x3'):
                Nm, Sm = R.cf_maps(sa, SB, T, un)
                cf['N_%s_%s' % (t, un)][i] = Nm
                cf['S_%s_%s' % (t, un)][i] = Sm
        if i % 100 == 0:
            print('maps %d/%d' % (i, n), flush=True)
    diag = {}
    for key in cf:
        A, E = fd[key], cf[key]
        sc = np.abs(E).max(1) + 1e-300
        diag[key] = dict(frac_exact_zero_fd=float((A == 0).mean()),
                         frac_exact_zero_cf=float((E == 0).mean()),
                         argmax_agree=float(np.mean(A.argmax(1) == E.argmax(1))),
                         max_rel_abs_diff=float((np.abs(A - E).max(1) / sc).max()),
                         median_rel_abs_diff=float(np.median(np.abs(A - E).max(1) / sc)))
        print(key, diag[key], flush=True)
    gts = [m5.gt_of(nm) for nm in vnames]
    mets = ['hit', 'ap', 'dice']
    S_f, keep, ch_f = R.score(fd, gts, mets)
    S_c, keep2, ch_c = R.score(cf, gts, mets)
    assert (keep == keep2).all() and len(keep) == n
    for u in ('1x1', '3x3'):
        S_c['collapsed_%s' % u] = S_f['collapsed_%s' % u]
    g_f, g_c = R.grid(S_f, mets), R.grid(S_c, mets)
    O = R.jload(os.path.join(R.MOUT, 'm5_isic.json'))
    P = R.jload(os.path.join(R.C2O, 't6_score.json'))['isic']['grid']
    cells = {}
    for key in g_f:
        row = dict(replicate_r4_fd=R.brief(g_f[key]), replicate_r4_closed_form=R.brief(g_c[key]))
        if key in O['prespecified']:
            row['original'] = R.brief(R.orig_cell(O, key))
        if key in P:
            pc = dict(P[key])
            pc['P'] = pc['p_mcnemar'] if key.endswith('_hit') else pc['p_wilcoxon']
            row['replicate_c2_fd'] = R.brief(pc)
        h, tp, un, mt = key.split('_')
        mn = ('S_%s_%s' if h == 'Sminus' else 'N_%s_%s') % (tp, un)
        row['items_changed_%s' % mn] = int(np.sum(np.abs(np.asarray(S_f[mn][mt]) - np.asarray(S_c[mn][mt])) > 1e-12))
        cells[key] = row
        print('%-28s O %s | r4-FD %s | r4-CF %s' % (key, R.fmt(row['original']) if 'original' in row else '-',
                                                     R.fmt(g_f[key]), R.fmt(g_c[key])), flush=True)
    R.save('t1_isic_cf.json', dict(
        arm='ISIC (pre-specified); NEW replicate model, archived maps not recoverable',
        why_not_archived=('m5_isic.py kept isic_r50.pt and F_isic_*.npy in the temp directory '
                          '(mlib.SCRATCH) and saved no per-item maps; C2 replicate checkpoint '
                          'isic_r50_replicate.pt likewise in temp (medical_baselines_lib.CACHE); all absent'),
        init='torchvision resnet50-0676ba61.pth (timm default weights not cached; no download)',
        auc_replicate_r4=auc, auc_original=O['gap_val'],
        n_scored=int(len(keep)), diagnostics=diag, cells=cells,
        dice_q_chosen_per_fold=dict(fd=ch_f, closed_form=ch_c),
        primary=cells['H1_lse8.0_1x1_dice'], h2_primary=cells['H2_lse8.0_1x1_dice'],
        sources=dict(original='outputs/medical/m5_isic.json',
                     replicate_c2='analyses/medical_baselines/out/t6_score.json [isic][grid]',
                     closed_form='analyses/closed_form_maps/scripts/closed_form_lib.ns_exact_units')))
    np.savez_compressed(os.path.join(R.CACHE, 't1_isic_items.npz'), names=np.array(vnames),
                        **{'FD__%s__%s' % (mn, mt): S_f[mn][mt] for mn in S_f for mt in mets},
                        **{'CF__%s__%s' % (mn, mt): S_c[mn][mt] for mn in S_c for mt in mets})


if __name__ == '__main__':
    main()
