"""PROBE 13 -- the AND/OR fan under END-TO-END training.

Stage-1 Probe 7 (the fan) froze the body, so the standing objection is that the
features never got a chance to adapt to the pooling operator: maybe a co-adapted
backbone can undo the AND/OR asymmetry.  Stage-1 Probe 6 trained end to end but
only measured I2 / Rhalf / corrNS -- it never ran the Grad-CAM and
deletion/insertion columns that carry the fan's actual claim.

Here the whole network is trained from scratch, per temperature, and the FULL
fan measurement is run on the resulting model with baselines drawn from THAT
model's own feature bank.

4-block CNN (3-64-128-256-256), imagenette at 96 px -> 6x6 = 36 cells,
RandomResizedCrop + flip, AdamW + OneCycle.
"""
import os, sys, json, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nslib as L
from heads import build
from pilot import get_data

dev = 'cuda'
CH = 256
EP = int(os.environ.get('EP', 20))
NIMG = int(os.environ.get('NIMG', 32))
TS = [float(t) for t in os.environ.get('TS', '-4,-2,-1,0,1,2,4').split(',')]


def blk(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, 2, 1, bias=False), nn.BatchNorm2d(o),
                         nn.ReLU(True), nn.Conv2d(o, o, 3, 1, 1, bias=False),
                         nn.BatchNorm2d(o), nn.ReLU(True))


class Net(nn.Module):
    def __init__(s, head):
        super().__init__()
        s.body = nn.Sequential(blk(3, 64), blk(64, 128), blk(128, CH), blk(CH, CH))
        s.head = head

    def feat(s, x):
        A = s.body(x); return A.reshape(A.shape[0], A.shape[1], -1)

    def forward(s, x): return s.head(s.feat(x))


def main():
    root = get_data(os.path.join(L.ROOT, 'data'))
    ttf = transforms.Compose([transforms.RandomResizedCrop(96, scale=(0.5, 1.0)),
                              transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                              transforms.Normalize(L.MEAN, L.STD)])
    vtf = transforms.Compose([transforms.Resize(110), transforms.CenterCrop(96),
                              transforms.ToTensor(),
                              transforms.Normalize(L.MEAN, L.STD)])
    tr = datasets.ImageFolder(os.path.join(root, 'train'), ttf)
    va = datasets.ImageFolder(os.path.join(root, 'val'), vtf)
    res = {'meta': dict(epochs=EP, cells=36, n_img=NIMG, mode='end-to-end')}
    print(f"{'T':>6s}{'acc':>8s}{'I2':>8s}{'Rhalf':>8s}{'corrNS':>9s}{'spNS':>8s}"
          f"{'spGN':>8s}{'spGS':>8s} | {'delN':>6s}{'delS':>6s}{'delG':>6s}"
          f"{'delR':>6s} | {'insN':>6s}{'insS':>6s}{'insG':>6s}{'insR':>6s}",
          flush=True)
    for Tt in TS:
        t0 = time.time(); torch.manual_seed(0)
        net = Net(build(f'lse{Tt}', k=10, c=CH)).to(dev)
        ld = DataLoader(tr, batch_size=64, shuffle=True, num_workers=4, drop_last=True)
        vl = DataLoader(va, batch_size=64, num_workers=4)
        opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=5e-4)
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, EP * len(ld))
        for ep in range(EP):
            net.train()
            for x, y in ld:
                loss = Fn.cross_entropy(net(x.to(dev)), y.to(dev))
                opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        net.eval()
        with torch.no_grad():
            ok = tot = 0; P = []
            for x, y in vl:
                z = net(x.to(dev))
                ok += (z.argmax(1).cpu() == y).sum().item(); tot += len(y)
                P.append(torch.softmax(z, 1).cpu().numpy())
            acc = ok / tot; P = np.concatenate(P)
            Yv = np.array(va.targets)
            rs = np.random.RandomState(0)
            idx = rs.permutation(len(va))[:NIMG + 32]
            Fm = torch.cat([net.feat(torch.stack(
                [va[int(j)][0] for j in idx[i:i + 32]]).to(dev))
                for i in range(0, len(idx), 32)])
            Xi, bank = Fm[:NIMG], Fm[NIMG:]
        h = net.head
        rs2 = np.random.RandomState(1)
        I2s, Rhs, cs, sns, sgn, sgs = [], [], [], [], [], []
        D = {k: [] for k in ('N', 'S', 'G', 'rand')}
        I = {k: [] for k in ('N', 'S', 'G', 'rand')}
        for k in range(Xi.shape[0]):
            A = Xi[k]
            with torch.no_grad():
                c = int(h(A[None]).argmax(1))
            B = bank[rs2.randint(bank.shape[0])]
            i2, rh, cc, _, _ = L.interaction(h, A, B, c, rs2)
            I2s.append(i2); Rhs.append(rh)
            if not np.isnan(cc):
                cs.append(cc)
            N, S = L.maps(h, A, B, c); Gm = L.gradcam_head(h, A, c)
            n, s, g = [v.cpu().numpy() for v in (N, S, Gm)]
            if n.std() > 1e-9 and s.std() > 1e-9: sns.append(spearmanr(n, s).statistic)
            if n.std() > 1e-9 and g.std() > 1e-9: sgn.append(spearmanr(g, n).statistic)
            if s.std() > 1e-9 and g.std() > 1e-9: sgs.append(spearmanr(g, s).statistic)
            rnd = torch.randn(A.shape[1], device=A.device)
            for tag, o in (('N', N), ('S', S), ('G', Gm), ('rand', rnd)):
                D[tag].append(L.auc(h, A, B, c, o, 'del'))
                I[tag].append(L.auc(h, A, B, c, o, 'ins'))
        r = dict(T=Tt, acc=acc, I2=float(np.mean(I2s)), Rhalf=float(np.mean(Rhs)),
                 corrNS=float(np.mean(cs)), spNS=float(np.mean(sns)),
                 spGN=float(np.mean(sgn)), spGS=float(np.mean(sgs)),
                 ece=L.ece(P, Yv), sec=time.time() - t0)
        for t in D:
            r[f'del_{t}'] = float(np.mean(D[t])); r[f'ins_{t}'] = float(np.mean(I[t]))
            r[f'del_{t}_ci'] = L.boot_ci(D[t])[1:]; r[f'ins_{t}_ci'] = L.boot_ci(I[t])[1:]
        res[f'lse{Tt}'] = r
        print(f"{Tt:6.2f}{acc:8.4f}{r['I2']:8.4f}{r['Rhalf']:8.4f}{r['corrNS']:+9.4f}"
              f"{r['spNS']:+8.3f}{r['spGN']:+8.3f}{r['spGS']:+8.3f} | "
              f"{r['del_N']:6.3f}{r['del_S']:6.3f}{r['del_G']:6.3f}{r['del_rand']:6.3f}"
              f" | {r['ins_N']:6.3f}{r['ins_S']:6.3f}{r['ins_G']:6.3f}"
              f"{r['ins_rand']:6.3f}  ({r['sec']:.0f}s)", flush=True)
        L.save('probe13_e2e_fan.json', res)
    z = res.get('lse0.0')
    if z is not None:
        ok = (abs(z['I2']) < 1e-4 and abs(z['Rhalf']) < 1e-4
              and abs(z['corrNS'] - 1) < 1e-6
              and abs(z['del_N'] - z['del_S']) < 1e-9)
        res['control_T0_exact'] = bool(ok)
        print(f"\nCONTROL T=0 exact-GAP degeneracy: {'PASS' if ok else 'FAIL'} "
              f"(I2={z['I2']:.2e} Rhalf={z['Rhalf']:.2e} corrNS={z['corrNS']:.6f})")
        L.save('probe13_e2e_fan.json', res)


if __name__ == '__main__':
    main()
