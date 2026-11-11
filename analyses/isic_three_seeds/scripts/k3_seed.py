"""K3, one seed: ISIC arm re-run end to end with an archived model and closed-form maps.

Protocol: protocols/K3_isic_closed_form_protocol.md (not edited).
Everything is imported from the original arm and the closed-form analysis of analyses/medical_extra:
  data / split / preprocessing / training recipe : scripts/medical/m5_isic.py (isic_frame, split
      RandomState(0) 70/30, load_rgb 224, train(): ResNet-50, class-weighted CE, AdamW 2e-4, 12 ep)
  reference bank (8 other-class validation images, RandomState(0)), heads, units : m5_isic / mlib
  forward-difference maps : m5_isic.NS_batched (float64 head on the GPU)
  closed-form maps         : medical_extra_lib.cf_maps / cf_collapsed -> closed_form_lib.ns_exact_units (float64)
  scoring and grid         : medical_extra_lib.score / medical_extra_lib.grid (hit, AP, Dice at 2-fold cross-tuned top-q)
Archived: checkpoints/isic_r50_k3_seed{S}.pt, out/k3_seed{S}_maps.npz (every per-item map, both
estimators, per-cell class scores, bank), out/k3_seed{S}_items.npz (per-item scores),
out/k3_seed{S}.json (full grid, both estimators).
Usage: python k3_seed.py SEED
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
K3 = os.path.abspath(os.path.join(HERE, '..'))
CK = os.path.join(K3, 'checkpoints')
OUTD = os.path.join(K3, 'out')
os.makedirs(CK, exist_ok=True)
os.makedirs(OUTD, exist_ok=True)
# keep every cache of the imported libraries inside K3 (medical_extra_lib only setdefaults these)
os.environ['MEDCACHE'] = CK
os.environ['C2CACHE'] = CK
ROOT = os.path.abspath(os.path.join(K3, *['..'] * 2))
sys.path.insert(0, os.path.join(ROOT, 'analyses', 'medical_extra', 'scripts'))

import json                     # noqa: E402
import numpy as np              # noqa: E402
import torch                    # noqa: E402

GPU_CAP_GB = 5.5
if torch.cuda.is_available():
    tot = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(GPU_CAP_GB * 2 ** 30 / tot, 0)

import medical_extra_lib as R               # noqa: E402
from medical_extra_lib import L, mlib       # noqa: E402
import t1_isic_cf as T1         # noqa: E402  (patch_timm: torchvision ImageNet init, as in analyses/medical_extra)


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    return o


def main(seed):
    import timm
    import m5_isic as m5
    T1.patch_timm()
    t0 = time.time()
    names, y = mlib.isic_frame()
    rs = np.random.RandomState(0)
    perm = rs.permutation(len(names))
    ntr = int(.7 * len(names))
    tr, va = np.sort(perm[:ntr]), np.sort(perm[ntr:])
    print('seed %d  n=%d train=%d val=%d classes=%s' % (seed, len(names), len(tr), len(va),
          np.bincount(y, minlength=3).tolist()), flush=True)
    ck = os.path.join(CK, 'isic_r50_k3_seed%d.pt' % seed)
    if os.path.exists(ck):
        m = timm.create_model('resnet50', pretrained=False, num_classes=3).to(L.dev)
        m.load_state_dict(torch.load(ck))
        print('loaded archived checkpoint', ck, flush=True)
    else:
        m, _, _ = m5.train(names, y, tr, va, seed=seed)
        torch.save(m.state_dict(), ck)
        print('saved checkpoint', ck, flush=True)
    m.eval()
    print('model ready %.0fs; peak GPU alloc %.2f GB, reserved %.2f GB' % (
        time.time() - t0, torch.cuda.max_memory_allocated() / 2 ** 30,
        torch.cuda.max_memory_reserved() / 2 ** 30), flush=True)

    vnames = [names[i] for i in va]
    yv = y[va]
    X = torch.stack([mlib.load_rgb(os.path.join(m5.IMG, n + '.jpg'), 224) for n in vnames])
    F = L.feats(lambda x: m.forward_features(x), X)
    del X
    W = m.fc.weight.detach().double().cpu()
    b = m.fc.bias.detach().double().cpu()
    Z = F.mean(1) @ W.numpy().T + b.numpy()
    auc = m5.aucs(Z, yv)
    print('AUC', auc, flush=True)

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
    SC = np.einsum('ntc,kc->ntk', F64, Wn) + bn            # per-cell class scores (n,49,3)
    n = len(vnames)
    keys = ['collapsed_1x1', 'collapsed_3x3'] + ['%s_%s_%s' % (p, t, u) for p in 'NS'
                                                 for t in R.OFF for u in ('1x1', '3x3')]
    fd = {k: np.zeros((n, 49)) for k in keys}
    cf = {k: np.zeros((n, 49)) for k in keys}
    t1 = time.time()
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
        sa = SC[i, :, c]
        SB = [SC[j, :, c] for j in bankidx[c]]
        for un in ('1x1', '3x3'):
            cf['collapsed_%s' % un][i] = R.cf_collapsed(sa, SB, un)
            for t in R.OFF:
                Nm, Sm = R.cf_maps(sa, SB, float(t[3:]), un)
                cf['N_%s_%s' % (t, un)][i] = Nm
                cf['S_%s_%s' % (t, un)][i] = Sm
        if i % 100 == 0:
            print('maps %d/%d %.0fs' % (i, n, time.time() - t1), flush=True)
    del F, F64

    diag = {}
    for key in keys:
        A, E = fd[key], cf[key]
        sc = np.abs(E).max(1) + 1e-300
        diag[key] = dict(frac_exact_zero_fd=float((A == 0).mean()),
                         frac_exact_zero_cf=float((E == 0).mean()),
                         argmax_agree=float(np.mean(A.argmax(1) == E.argmax(1))),
                         max_rel_abs_diff=float((np.abs(A - E).max(1) / sc).max()),
                         median_rel_abs_diff=float(np.median(np.abs(A - E).max(1) / sc)))
        print(key, diag[key], flush=True)

    np.savez_compressed(os.path.join(OUTD, 'k3_seed%d_maps.npz' % seed),
                        names=np.array(vnames), cls=yv, val_index=va,
                        bank=np.array([bankidx[c] for c in range(3)]),
                        cell_scores=SC, W=Wn, b=bn,
                        **{'FD__%s' % k: v for k, v in fd.items()},
                        **{'CF__%s' % k: v for k, v in cf.items()})

    gts = [m5.gt_of(nm) for nm in vnames]
    mets = ['hit', 'ap', 'dice']
    S_f, keep, ch_f = R.score(fd, gts, mets)
    S_c, keep2, ch_c = R.score(cf, gts, mets)
    assert (keep == keep2).all()
    g_f, g_c = R.grid(S_f, mets), R.grid(S_c, mets)
    np.savez_compressed(os.path.join(OUTD, 'k3_seed%d_items.npz' % seed), names=np.array(vnames)[keep],
                        **{'FD__%s__%s' % (mn, mt): S_f[mn][mt] for mn in S_f for mt in mets},
                        **{'CF__%s__%s' % (mn, mt): S_c[mn][mt] for mn in S_c for mt in mets})
    summ = {est: {mn: {mt: mlib.boot_ci(SS[mn][mt]) for mt in mets} for mn in SS}
            for est, SS in (('forward_difference', S_f), ('closed_form', S_c))}
    for key in ('H1_lse8.0_1x1_dice',):
        print('PRIMARY %s  FD %s | CF %s' % (key, R.fmt(g_f[key]), R.fmt(g_c[key])), flush=True)
    res = dict(seed=seed, n_val=n, n_scored=int(len(keep)), auc=auc,
               checkpoint=os.path.relpath(ck, K3),
               init='torchvision resnet50-0676ba61.pth (ImageNet; timm default weights not cached)',
               primary=dict(key='H1_lse8.0_1x1_dice', closed_form=g_c['H1_lse8.0_1x1_dice'],
                            forward_difference=g_f['H1_lse8.0_1x1_dice']),
               grid=dict(closed_form=g_c, forward_difference=g_f),
               map_means=summ,
               dice_q_chosen_per_fold=dict(closed_form=ch_c, forward_difference=ch_f),
               fd_vs_cf=diag, bank=bankidx,
               peak_gpu_gb=dict(allocated=torch.cuda.max_memory_allocated() / 2 ** 30,
                                reserved=torch.cuda.max_memory_reserved() / 2 ** 30),
               seconds=time.time() - t0)
    p = os.path.join(OUTD, 'k3_seed%d.json' % seed)
    json.dump(jsonable(res), open(p, 'w'), indent=1)
    print('wrote', p, flush=True)


if __name__ == '__main__':
    main(int(sys.argv[1]))
