"""PROBE 22b -- where does GRADIENT DESCENT put the aggregator on the AND/OR
axis, when it is free to choose?

Probe 22 condition B learned `alpha` at the same learning rate as the weights
(1e-3) and it barely moved from whatever it was initialised to -- except from
alpha = 0, where every class moved to +0.06 .. +0.14 and validation accuracy
rose above GAP's.  That is suggestive but confounded by the tiny step size, so
this probe gives `alpha` its own learning rate and sweeps it, with 5 seeds, and
asks three separable questions:

  1. Is GAP (alpha = 0) a stationary point of the loss?  ->  initialise AT 0
     and report the signed displacement and its seed CI.
  2. Is the axis position identifiable at all, or is the loss flat along it?
     ->  initialise far away on both sides and see whether the runs converge.
  3. Does the movement buy accuracy?  ->  paired comparison against alpha frozen
     at 0 with the same seed.

`alpha` is NOT identifiable on its own (alpha*z is invariant under
(alpha,w) -> (alpha/s, s*w)), so every row also reports the scale-free
participation ratio PR = 1/(T * sum_t w_t^2) of the pooling weights: PR = 1 is a
uniform average (GAP), PR = 1/T is a hard max.
"""
import os, sys, numpy as np, torch, torch.nn.functional as Fn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nslib as L
from probe22_autopool import AutoPool, part_ratio, measure

BODY = os.environ.get('BODY', 'resnet50')
DS = os.environ.get('DS', 'imagenette')
EPOCHS = int(os.environ.get('EPOCHS', 8))
NIMG = int(os.environ.get('NIMG', 24))
ALR = [float(x) for x in os.environ.get('ALR', '0.001,0.01,0.1').split(',')]
INITS = [float(x) for x in os.environ.get('INITS', '-2,0,2').split(',')]
SEEDS = [int(s) for s in os.environ.get('SEEDS', '0,1,2,3,4').split(',')]
dev = 'cuda'


def train2(h, Ftr, Ytr, Fva, Yva, epochs, alr, bs=96, seed=0, lr=1e-3):
    ps = [p for n, p in h.named_parameters() if n != 'alpha']
    opt = torch.optim.AdamW([{'params': ps, 'lr': lr, 'weight_decay': 1e-4},
                             {'params': [h.alpha], 'lr': alr,
                              'weight_decay': 0.0}])
    n = len(Ytr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(1, epochs*(n//bs)))
    yt = torch.from_numpy(Ytr).to(dev); rs = np.random.RandomState(seed)
    traj = []
    for ep in range(epochs):
        perm = rs.permutation(n)
        for i in range(0, n - bs + 1, bs):
            j = np.sort(perm[i:i + bs])
            loss = Fn.cross_entropy(h(L.batch(Ftr, j)), yt[j])
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        traj.append(float(h.alpha.detach().mean()))
    h.eval()
    with torch.no_grad():
        ok = 0
        for i in range(0, len(Yva), 128):
            j = np.arange(i, min(i + 128, len(Yva)))
            ok += (h(L.batch(Fva, j)).argmax(1).cpu().numpy() == Yva[j]).sum()
    return h, float(ok / len(Yva)), traj


def main():
    ftr, ytr, C, T = L.cache_feats(BODY, DS, 'train')
    fva, yva, _, _ = L.cache_feats(BODY, DS, 'val')
    Ftr, Ytr = L.load_cached(ftr, ytr); Fva, Yva = L.load_cached(fva, yva)
    ncl = int(Ytr.max()) + 1
    rs = np.random.RandomState(0)
    sel = rs.permutation(len(Yva))[:NIMG + 48]
    Xi = L.batch(Fva, np.sort(sel[:NIMG]))
    bank = L.batch(Fva, np.sort(sel[NIMG + 16:NIMG + 48]))
    res = {'meta': dict(body=BODY, ds=DS, cells=T, nclass=ncl, seeds=SEEDS,
                        epochs=EPOCHS, alr=ALR, inits=INITS)}
    print(f'[{BODY}/{DS}] AutoPool alpha learning, {len(SEEDS)} seeds', flush=True)

    # --- baseline: alpha frozen at 0 (= GAP+FC), same seeds ------------------
    base = []
    for sd in SEEDS:
        torch.manual_seed(sd)
        h = AutoPool(ncl, C, alpha=0.0, learn=False).to(dev)
        h, a, _ = train2(h, Ftr, Ytr, Fva, Yva, EPOCHS, 0.0, seed=sd)
        base.append(a)
    m, lo, hi = L.boot_ci(base)
    res['frozen0'] = dict(acc=m, ci=[lo, hi], per_seed=base)
    print(f'  alpha FROZEN at 0 (= GAP+FC): acc {m:.4f} [{lo:.4f},{hi:.4f}]',
          flush=True)

    print(f"\n{'alpha_lr':>9s}{'init':>7s}{'alpha_end':>11s}{'CI':>22s}"
          f"{'PR':>7s}{'acc':>9s}{'d_acc vs frozen0':>18s}", flush=True)
    for alr in ALR:
        for a0 in INITS:
            ends, accs, prs, last = [], [], [], None
            for sd in SEEDS:
                torch.manual_seed(sd)
                h = AutoPool(ncl, C, alpha=a0, learn=True).to(dev)
                h, a, traj = train2(h, Ftr, Ytr, Fva, Yva, EPOCHS, alr, seed=sd)
                ends.append(float(h.alpha.detach().mean()))
                accs.append(a); prs.append(part_ratio(h, Xi)); last = (h, traj)
            em, el, eh = L.boot_ci(ends)
            am, al, ah = L.boot_ci(accs)
            d = [x - y for x, y in zip(accs, base)]
            dm, dl, dh = L.boot_ci(d)
            key = f'alr{alr}_init{a0}'
            res[key] = dict(alpha_end=em, alpha_ci=[el, eh], acc=am,
                            acc_ci=[al, ah], dacc=dm, dacc_ci=[dl, dh],
                            PR=float(np.mean(prs)), per_seed_alpha=ends,
                            per_seed_acc=accs, traj=last[1])
            print(f"{alr:9.3f}{a0:+7.1f}{em:+11.4f}"
                  f"  [{el:+.4f},{eh:+.4f}]{np.mean(prs):7.3f}{am:9.4f}"
                  f"{dm:+10.4f} [{dl:+.4f},{dh:+.4f}]", flush=True)
            if a0 == 0.0:
                res[key + '_maps'] = measure(last[0], Xi, bank)
            L.save(f'probe22b_learnalpha_{BODY}_{DS}.json', res)

    print('\n--- maps of the model that STARTED at GAP and was allowed to move '
          '---')
    for alr in ALR:
        k = f'alr{alr}_init0.0_maps'
        if k in res:
            r = res[k]
            print(f"  alpha_lr={alr:<6}: I2={r['I2']:.4f} Rhalf={r['Rhalf']:.4f} "
                  f"corrNS={r['corrNS']:+.4f} spNS={r['spNS']:+.3f} "
                  f"spGN={r['spGN']:+.3f} spGS={r['spGS']:+.3f}", flush=True)
    L.save(f'probe22b_learnalpha_{BODY}_{DS}.json', res)


if __name__ == '__main__':
    main()
