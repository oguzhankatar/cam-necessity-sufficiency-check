"""PROBE 27b (P2, explanation (a)) -- TRAIN A ViT FROM SCRATCH UNDER EACH
ORDERING.

MoCo v3 chose the ordering BEFORE pre-training; we relocate it AFTER training.
If the 6.6-point gap is caused by "the encoder was optimised under that head",
then training from scratch under `pre` must cost much more than relocating to
`pre` costs in a model trained under `post`.

Three architectures that differ by ONE line, everything else identical (same
seed, same augmentation, same schedule, same parameter count up to the LN's
2C affine parameters):

    post : timm fc_norm=True   -> head( LN( mean_t z_t ) )   (what timm ships)
    pre  : timm fc_norm=False  -> head( mean_t LN( z_t ) )   (MoCo v3's LN+pool)
    none : fc_norm=False and final norm replaced by Identity -> head(mean_t z_t)
                                                               (MoCo v3's pool)

Paired over seeds.  After training we re-measure corrNS at the aggregation
point, so the table carries both the cost and the identifiability of each
ordering in the SAME run.  `none` is again the internal control: it is GAP +
linear and must return corrNS = +1.000000.

Usage:  DS=imagenette SEEDS=3 EPOCHS=60 python scripts/probe27b_scratch.py
"""
import os, sys, time, numpy as np, torch, torch.nn as nn
import torch.nn.functional as Fn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nslib as L

dev = 'cuda'
DS = os.environ.get('DS', 'imagenette')
SEEDS = int(os.environ.get('SEEDS', 3))
EPOCHS = int(os.environ.get('EPOCHS', 60))
MODES = ('post', 'pre', 'none')


def build(mode, ncls, img, patch, seed):
    import timm
    torch.manual_seed(seed); np.random.seed(seed)
    m = timm.create_model('vit_tiny_patch16_224', pretrained=False,
                          num_classes=ncls, global_pool='avg',
                          class_token=False, img_size=img, patch_size=patch,
                          fc_norm=(mode == 'post'))
    if mode == 'none':
        m.norm = nn.Identity()
    return m.to(dev)


def data():
    from torchvision import datasets, transforms
    if DS == 'imagenette':
        img, patch = 128, 16
        tr = transforms.Compose([
            transforms.RandomResizedCrop(img, scale=(0.35, 1.0)),
            transforms.RandomHorizontalFlip(), transforms.ToTensor(),
            transforms.Normalize(L.MEAN, L.STD)])
        te = transforms.Compose([
            transforms.Resize(int(img * 1.14)), transforms.CenterCrop(img),
            transforms.ToTensor(), transforms.Normalize(L.MEAN, L.STD)])
        p = os.path.join(L.ROOT, 'data', 'imagenette2-160')
        return (datasets.ImageFolder(os.path.join(p, 'train'), tr),
                datasets.ImageFolder(os.path.join(p, 'val'), te), 10, img, patch)
    if DS == 'cifar100':
        img, patch = 32, 4
        tr = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(), transforms.ToTensor(),
            transforms.Normalize(L.MEAN, L.STD)])
        te = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize(L.MEAN, L.STD)])
        r = os.path.join(L.ROOT, 'data')
        return (datasets.CIFAR100(r, True, tr), datasets.CIFAR100(r, False, te),
                100, img, patch)
    raise ValueError(DS)


@torch.no_grad()
def corrns(m, mode, xs, xb, nimg=32):
    """corrNS over patch tokens at the aggregation point, after training."""
    from scipy.stats import spearmanr
    def toks(x):
        z = m.patch_embed(x); z = m._pos_embed(z); z = m.norm_pre(z)
        for b in m.blocks:
            z = b(z)
        return m.norm(z)                        # Identity for mode='none'
    ln = m.fc_norm if mode == 'post' else nn.Identity()
    def head(v):                                # v: pooled (B,C)
        return m.head(ln(v))
    cs, rh = [], []
    rs = np.random.RandomState(1)
    TA = toks(xs); TB = toks(xb)
    for q in range(min(nimg, len(xs))):
        A, B = TA[q], TB[rs.randint(len(TB))]
        T = A.shape[0]
        mA, mB = A.mean(0), B.mean(0)
        c = int(head(mA[None]).argmax())
        d = (A - B) / T
        zA = head(mA[None])[0, c]; zB = head(mB[None])[0, c]
        N = zA - head(mA[None] - d)[:, c]
        S = head(mB[None] + d)[:, c] - zB
        n, s = N.cpu().numpy().astype(np.float64), S.cpu().numpy().astype(np.float64)
        if n.std() > 1e-14 and s.std() > 1e-14:
            cs.append(float(np.corrcoef(n, s)[0, 1]))
        H = torch.zeros(32, T, device=dev)
        for i in range(32):
            H[i, torch.from_numpy(rs.choice(T, T // 2, replace=False)).to(dev)] = 1
        D = zA - head(mA[None] - H @ d)[:, c]
        rh.append(float(((D - H @ N).abs().mean() / (D.abs().mean() + 1e-12)).item()))
    return float(np.mean(cs)), float(np.mean(rh))


def run(mode, seed, dtr, dte, ncls, img, patch):
    from torch.utils.data import DataLoader
    m = build(mode, ncls, img, patch, seed)
    ltr = DataLoader(dtr, batch_size=128, shuffle=True, num_workers=8,
                     drop_last=True, persistent_workers=True)
    lte = DataLoader(dte, batch_size=256, num_workers=8, persistent_workers=True)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.05)
    sch = torch.optim.lr_scheduler.OneCycleLR(
        opt, 1e-3, epochs=EPOCHS, steps_per_epoch=len(ltr), pct_start=0.15)
    sc = torch.amp.GradScaler('cuda')
    t0 = time.time()
    for ep in range(EPOCHS):
        m.train()
        for x, y in ltr:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.amp.autocast('cuda'):
                loss = Fn.cross_entropy(m(x), y, label_smoothing=0.1)
            opt.zero_grad(); sc.scale(loss).backward(); sc.step(opt); sc.update()
            sch.step()
    m.eval(); ok = tot = 0
    with torch.no_grad():
        for x, y in lte:
            ok += (m(x.to(dev)).argmax(1).cpu() == y).sum().item(); tot += len(y)
    xs = torch.stack([dte[i][0] for i in range(0, 32)]).to(dev)
    xb = torch.stack([dte[i][0] for i in range(len(dte) - 8, len(dte))]).to(dev)
    c, r = corrns(m, mode, xs, xb)
    return ok / tot, c, r, time.time() - t0


def main():
    dtr, dte, ncls, img, patch = data()
    print(f'[{DS}] train={len(dtr)} val={len(dte)} classes={ncls} img={img} '
          f'patch={patch} tokens={(img//patch)**2} epochs={EPOCHS} '
          f'seeds={SEEDS}', flush=True)
    res = {'meta': dict(ds=DS, epochs=EPOCHS, seeds=SEEDS, ncls=ncls, img=img,
                        patch=patch)}
    A = {k: [] for k in MODES}; C = {k: [] for k in MODES}; R = {k: [] for k in MODES}
    for s in range(SEEDS):
        for mode in MODES:
            a, c, r, sec = run(mode, s, dtr, dte, ncls, img, patch)
            A[mode].append(a); C[mode].append(c); R[mode].append(r)
            print(f'  seed {s} {mode:>5s}: top-1 {a:.4f}  corrNS {c:+.6f}  '
                  f'Rhalf {r:.2e}  ({sec:.0f}s)', flush=True)
            res[f'{mode}_s{s}'] = dict(acc=a, corrNS=c, Rhalf=r, sec=sec)
        L.save(f'probe27b_scratch_{DS}.json', res)
    print(f"\n{'ordering':>8s}{'top-1 mean':>12s}{'[95% CI]':>20s}"
          f"{'corrNS':>12s}{'Rhalf':>11s}", flush=True)
    for mode in MODES:
        mu, lo, hi = L.boot_ci(A[mode])
        res[mode] = dict(acc=mu, acc_ci=[lo, hi], accs=A[mode],
                         corrNS=float(np.mean(C[mode])),
                         Rhalf=float(np.mean(R[mode])))
        print(f"{mode:>8s}{mu:12.4f}  [{lo:.4f},{hi:.4f}]"
              f"{np.mean(C[mode]):+12.6f}{np.mean(R[mode]):11.2e}", flush=True)
    d = np.array(A['pre']) - np.array(A['none'])
    mu, lo, hi = L.boot_ci(d)
    res['pre_minus_none_paired'] = dict(mean=100 * mu, ci=[100 * lo, 100 * hi],
                                        per_seed=[100 * float(x) for x in d])
    d2 = np.array(A['post']) - np.array(A['none'])
    mu2, lo2, hi2 = L.boot_ci(d2)
    res['post_minus_none_paired'] = dict(mean=100 * mu2, ci=[100 * lo2, 100 * hi2],
                                         per_seed=[100 * float(x) for x in d2])
    print(f"\nPAIRED, trained from scratch under each ordering:")
    print(f"  pre  - none = {100*mu:+.2f} pts  [{100*lo:+.2f},{100*hi:+.2f}]"
          f"   (MoCo v3 linear probe: -6.6)")
    print(f"  post - none = {100*mu2:+.2f} pts  [{100*lo2:+.2f},{100*hi2:+.2f}]")
    ok = abs(res['none']['corrNS'] - 1) < 1e-5
    res['control_none_exact'] = bool(ok)
    print(f"INTERNAL CONTROL (none == GAP+linear == LSE T=0): "
          f"{'PASS' if ok else 'FAIL'} corrNS={res['none']['corrNS']:.6f}")
    L.save(f'probe27b_scratch_{DS}.json', res)


if __name__ == '__main__':
    main()
