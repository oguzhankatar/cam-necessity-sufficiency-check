"""T2 -- Fig. 3(a)-(c) / Sec. 4.4 recomputed: probe29_fan200.py condition A
(shipped ResNet-50 fc in each LSE operator, 224 ImageNet-1k validation images).

Reference construction is probe29's, reproduced exactly: ONE reference per image,
B = bank[rs2.randint(32)] with rs2 = RandomState(1) reset per temperature, and
nslib.interaction called in between so that the RNG stream is consumed as in
the archive.  For every image and temperature we compute
  f32  : the paper's estimator (nslib.maps / gradcam_head / auc, float32)
  fd64 : the same forward differences with head and features in float64
  ex   : the closed form (closed_form_lib.ns_exact), float64, no cancellation;
         Grad-CAM and the deletion / insertion curves in float64
plus, as the 'genuine' comparison, exact maps averaged over the 32-image bank
and over its first 8 images before ranking.
Features of the 256 images needed are recomputed with probe29's own body and
fp16 cache convention (the archive cache is not on disk).
"""
import os, sys, json, copy, time, numpy as np, torch
import closed_form_lib as C
import nslib as L
import probe20_in1k as P20

NIMG, TS = 224, [-8.0, -1.0, 0.0, 1.0, 8.0]
dev = 'cuda'
CACHE = os.path.join(os.environ.get('TEMP', '/tmp'), '<repo>', 'c5cache')
os.makedirs(CACHE, exist_ok=True)


def feats():
    fp = os.path.join(CACHE, 'in1k_sel_272.npy')
    ds, _ = L.get_dataset('in1k', 'val', 224)
    assert len(ds) == 10000, len(ds)
    rs = np.random.RandomState(0)
    sel = rs.permutation(len(ds))[:NIMG + 48]
    ii, bi = np.sort(sel[:NIMG]), np.sort(sel[NIMG + 16:NIMG + 48])
    want = np.concatenate([ii, bi])
    if not os.path.exists(fp):
        body, _, _ = L.get_body('resnet50'); body = body.to(dev)
        F = []
        with torch.no_grad():
            for s in range(0, len(want), 64):
                x = torch.stack([ds[int(j)][0] for j in want[s:s + 64]]).to(dev)
                F.append(body(x).half().cpu().numpy())
        np.save(fp, np.concatenate(F))
    F = np.load(fp)
    X = torch.from_numpy(F).to(dev).float().reshape(len(F), 2048, -1)
    return X[:NIMG], X[NIMG:], want


def main():
    Xi, bank, want = feats()
    W, b = P20.shipped_fc()
    arch = json.load(open(os.path.join(C.ROOT, 'outputs', 'probe29_fan200.json')))
    res = {'meta': dict(source='scripts/probe29_fan200.py condition A', n_img=NIMG,
                        reference='one bank image per image (bank of 32), rs2=RandomState(1)',
                        images_val_index=[int(v) for v in want[:NIMG]])}
    per_image = {}
    for T in TS:
        t0 = time.time()
        h = P20.make_head(T, 1000, 2048, W, b).eval()
        h64 = copy.deepcopy(h).double().eval()
        rs2 = np.random.RandomState(1)
        acc = {k: [] for k in ('corrNS_f32', 'spNS_f32', 'spGN_f32', 'spGS_f32', 'delN_f32',
                               'delS_f32', 'insN_f32', 'insS_f32', 'zeroN_f32', 'zeroS_f32',
                               'spNS_fd64', 'spGN_fd64', 'spGS_fd64', 'zeroN_fd64',
                               'corrNS_ex', 'spNS_ex', 'spGN_ex', 'spGS_ex', 'delN_ex',
                               'delS_ex', 'insN_ex', 'insS_ex', 'delG_ex',
                               'spNS_bank32', 'spGN_bank32', 'spGS_bank32',
                               'spNS_bank8', 'spGN_bank8', 'spGS_bank8', 'ref')}

        for k in range(NIMG):
            A = Xi[k]
            with torch.no_grad():
                c = int(h(A[None]).argmax(1))
            j = rs2.randint(bank.shape[0]); B = bank[j]
            i2, rh, cc, mg, zm = L.interaction(h, A, B, c, rs2)
            # ---- paper's estimator, float32
            N, S = L.maps(h, A, B, c); G = L.gradcam_head(h, A, c)
            n, s, g = [v.cpu().numpy().astype(np.float64) for v in (N, S, G)]
            acc['ref'].append(int(j))
            acc['corrNS_f32'].append(cc)
            acc['spNS_f32'].append(C.sp(n, s) if n.std() > 1e-9 and s.std() > 1e-9 else np.nan)
            acc['spGN_f32'].append(C.sp(g, n) if n.std() > 1e-9 and g.std() > 1e-9 else np.nan)
            acc['spGS_f32'].append(C.sp(g, s) if s.std() > 1e-9 and g.std() > 1e-9 else np.nan)
            acc['zeroN_f32'].append(float((n == 0).mean())); acc['zeroS_f32'].append(float((s == 0).mean()))
            acc['delN_f32'].append(L.auc(h, A, B, c, N, 'del')); acc['delS_f32'].append(L.auc(h, A, B, c, S, 'del'))
            acc['insN_f32'].append(L.auc(h, A, B, c, N, 'ins')); acc['insS_f32'].append(L.auc(h, A, B, c, S, 'ins'))
            # ---- same forward differences in float64
            n64, s64 = C.maps_fd64(h64, A, B, c); g64 = C.gradcam64(h64, A, c)
            acc['spNS_fd64'].append(C.sp(n64, s64)); acc['spGN_fd64'].append(C.sp(g64, n64))
            acc['spGS_fd64'].append(C.sp(g64, s64)); acc['zeroN_fd64'].append(float((n64 == 0).mean()))
            # ---- exact closed form
            sa = C.scores(h, A, c); sb = C.scores(h, B, c)
            ne, se = C.ns_exact(sa, sb, T)
            acc['corrNS_ex'].append(float(np.corrcoef(ne, se)[0, 1]) if T != 0 else 1.0)
            acc['spNS_ex'].append(C.sp(ne, se)); acc['spGN_ex'].append(C.sp(g64, ne))
            acc['spGS_ex'].append(C.sp(g64, se))
            acc['delN_ex'].append(C.auc64(h64, A, B, c, ne, 'del'))
            acc['delS_ex'].append(C.auc64(h64, A, B, c, se, 'del'))
            acc['insN_ex'].append(C.auc64(h64, A, B, c, ne, 'ins'))
            acc['insS_ex'].append(C.auc64(h64, A, B, c, se, 'ins'))
            acc['delG_ex'].append(C.auc64(h64, A, B, c, g64, 'del'))
            # ---- genuine: exact maps averaged over references BEFORE ranking
            SB = np.stack([C.scores(h, bank[q], c) for q in range(bank.shape[0])])
            NE = []; SE = []
            for q in range(bank.shape[0]):
                a_, b_ = C.ns_exact(sa, SB[q], T); NE.append(a_); SE.append(b_)
            NE, SE = np.array(NE), np.array(SE)
            for tag, m in (('bank32', 32), ('bank8', 8)):
                nb, sb_ = NE[:m].mean(0), SE[:m].mean(0)
                acc['spNS_' + tag].append(C.sp(nb, sb_))
                acc['spGN_' + tag].append(C.sp(g64, nb)); acc['spGS_' + tag].append(C.sp(g64, sb_))
        r = {'T': T}
        for kk, v in acc.items():
            if kk == 'ref':
                continue
            r[kk] = C.boot(v)
        a = arch.get(f'ship_lse{T}', {})
        r['archive'] = {k: a.get(k) for k in ('corrNS', 'spNS', 'spGN', 'spGS', 'del_N', 'del_S', 'ins_N', 'ins_S')}
        r['min_spNS_ex_over_images'] = float(np.nanmin(acc['spNS_ex']))
        r['max_abs_delN_minus_delS_ex'] = float(np.max(np.abs(np.array(acc['delN_ex']) - np.array(acc['delS_ex']))))
        r['sec'] = time.time() - t0
        res[f'ship_lse{T}'] = r
        per_image[str(T)] = {k: [float(x) for x in v] for k, v in acc.items()}
        print(f"T={T:+.0f}  spNS f32 {r['spNS_f32'][0]:.3f} (arch {a.get('spNS', np.nan):.3f}) ex {r['spNS_ex'][0]:.6f} | "
              f"spGN/spGS f32 {r['spGN_f32'][0]:.3f}/{r['spGS_f32'][0]:.3f} (arch {a.get('spGN', np.nan):.3f}/{a.get('spGS', np.nan):.3f}) "
              f"ex {r['spGN_ex'][0]:.3f}/{r['spGS_ex'][0]:.3f} bank32 {r['spGN_bank32'][0]:.3f}/{r['spGS_bank32'][0]:.3f} | "
              f"delN/S f32 {r['delN_f32'][0]:.3f}/{r['delS_f32'][0]:.3f} (arch {a.get('del_N', np.nan):.3f}/{a.get('del_S', np.nan):.3f}) "
              f"ex {r['delN_ex'][0]:.3f}/{r['delS_ex'][0]:.3f} | zeroN f32 {r['zeroN_f32'][0]:.2f} fd64 {r['zeroN_fd64'][0]:.2f} "
              f"| spNS bank32 {r['spNS_bank32'][0]:.3f} ({r['sec']:.0f}s)", flush=True)
        json.dump(res, open(os.path.join(C.OUTD, 't2_in1k_fan.json'), 'w'), indent=1)
        json.dump(per_image, open(os.path.join(C.OUTD, 't2_in1k_fan_per_image.json'), 'w'))


if __name__ == '__main__':
    main()
