"""TASK 9 side check, part 3 -- seeds 2 and 5 (an unarchived seed), predictions saved.
Copy of t9b_check27b_none_preds.py with SEEDS changed.

Original docstring: part 2 -- why does probe27b report the `none` ordering at
exactly 2853/3925 for seeds 0, 1 and 2?

t9_check27b_none.py re-ran seed 0 and reproduced 0.7268789808917198 exactly.
Here we (i) check that the seed changes the initial weights of the `none`
model, and (ii) re-train `none` for seeds 0 and 1 with the archived recipe
(scripts/probe27b_scratch: build, data, corrns; the training loop of
probe27b_scratch.run copied verbatim) while keeping the per-image predictions,
so we can tell "same model" from "different models, same count".
Nothing is written outside theory_sweeps/out."""
import os, sys, json, time, hashlib, numpy as np, torch
import torch.nn.functional as Fn
os.environ.setdefault('EPOCHS', '30')
os.environ.setdefault('DS', 'imagenette')
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
OUTF = os.path.join(os.path.dirname(HERE), 'out', 't9c_check27b_none_seed25.json')


def main():
    import probe27b_scratch as P
    from torch.utils.data import DataLoader
    dtr, dte, ncls, img, patch = P.data()
    res = {'init_checksums': {}}
    for s in (2, 5):
        m = P.build('none', ncls, img, patch, s)
        w = m.patch_embed.proj.weight.detach().double()
        res['init_checksums'][f'none_s{s}'] = dict(sum=float(w.sum()), abs_sum=float(w.abs().sum()))
        del m
    print(res['init_checksums'], flush=True)
    preds = {}
    for s in (2, 5):
        m = P.build('none', ncls, img, patch, s)
        ltr = DataLoader(dtr, batch_size=128, shuffle=True, num_workers=8,
                         drop_last=True, persistent_workers=True)
        lte = DataLoader(dte, batch_size=256, num_workers=8, persistent_workers=True)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.05)
        sch = torch.optim.lr_scheduler.OneCycleLR(
            opt, 1e-3, epochs=P.EPOCHS, steps_per_epoch=len(ltr), pct_start=0.15)
        sc = torch.amp.GradScaler('cuda')
        t0 = time.time()
        for ep in range(P.EPOCHS):
            m.train()
            for x, y in ltr:
                x, y = x.to(P.dev, non_blocking=True), y.to(P.dev, non_blocking=True)
                with torch.amp.autocast('cuda'):
                    loss = Fn.cross_entropy(m(x), y, label_smoothing=0.1)
                opt.zero_grad(); sc.scale(loss).backward(); sc.step(opt); sc.update()
                sch.step()
        m.eval(); pr = []; ys = []
        with torch.no_grad():
            for x, y in lte:
                pr.append(m(x.to(P.dev)).argmax(1).cpu()); ys.append(y)
        pr = torch.cat(pr).numpy(); ys = torch.cat(ys).numpy()
        preds[s] = pr
        np.save(os.path.join(os.path.dirname(OUTF), f't9c_none_s{s}_preds.npy'), np.stack([pr, ys]))
        acc = float((pr == ys).mean())
        res[f'none_s{s}'] = dict(acc=acc, n_correct=int((pr == ys).sum()), n=int(len(ys)),
                                 pred_md5=hashlib.md5(pr.astype(np.int64).tobytes()).hexdigest(),
                                 sec=time.time() - t0)
        print(f'none seed {s}: acc={acc:.6f} ({(pr == ys).sum()}/{len(ys)}) md5={res[f"none_s{s}"]["pred_md5"]}',
              flush=True)
        json.dump(res, open(OUTF, 'w'), indent=1)
        del m, ltr, lte
    res['n_pred_disagree_s2_s5'] = int((preds[2] != preds[5]).sum())
    print('disagreements seed2 vs seed5:', res['n_pred_disagree_s2_s5'], flush=True)
    json.dump(res, open(OUTF, 'w'), indent=1)


if __name__ == '__main__':
    main()
