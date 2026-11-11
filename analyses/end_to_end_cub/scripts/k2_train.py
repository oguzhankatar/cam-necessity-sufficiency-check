"""Train one K2 model end to end.

    python k2_train.py <H0|H1|H2|H3> <seed> [--micro 32] [--max_steps N] [--out PATH]

Recipe fixed by the protocol (k2lib.RECIPE). --micro < 32 keeps the effective batch at 32 by
gradient accumulation (a recorded deviation; BatchNorm then sees micro-batches).
"""
import os, sys, time, json, math, argparse, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import k2lib as L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('head'); ap.add_argument('seed', type=int)
    ap.add_argument('--micro', type=int, default=32)
    ap.add_argument('--max_steps', type=int, default=0)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--out', default='')
    a = ap.parse_args()
    R = L.RECIPE
    assert R['batch'] % a.micro == 0
    accum = R['batch'] // a.micro
    t0 = time.time()

    def log(*s):
        print('[%6.0fs]' % (time.time() - t0), *s, flush=True)

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    torch.backends.cudnn.benchmark = True

    man = L.W.cub_manifest()
    tr_tf, _ = L.tfs()
    ds = L.DS(man['fit'], tr_tf)
    g = torch.Generator(); g.manual_seed(a.seed)
    dl = torch.utils.data.DataLoader(ds, batch_size=a.micro, shuffle=True, drop_last=True,
                                     num_workers=a.workers, pin_memory=True, generator=g,
                                     persistent_workers=True, prefetch_factor=4)
    net = L.Net(a.head).cuda().to(memory_format=torch.channels_last)
    pb = list(net.backbone.parameters()); ph = list(net.head.parameters())
    opt = torch.optim.SGD([{'params': pb, 'lr': R['lr_backbone']},
                           {'params': ph, 'lr': R['lr_head']}],
                          momentum=R['momentum'], weight_decay=R['weight_decay'])
    steps_per_epoch = len(ds) // R['batch']
    total = R['epochs'] * steps_per_epoch
    if a.max_steps:
        total = a.max_steps
    sch = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda it: 0.5 * (1.0 + math.cos(math.pi * min(it, total) / total)))
    scaler = torch.amp.GradScaler('cuda')
    log('head %s (%s) seed %d  n_train %d  micro %d x accum %d  steps/epoch %d  total %d'
        % (a.head, L.HEAD_DESC[a.head], a.seed, len(ds), a.micro, accum, steps_per_epoch, total))
    hist = []
    step = 0
    torch.cuda.reset_peak_memory_stats()
    done = False
    for ep in range(R['epochs']):
        net.train()
        tl, tc, tn = 0.0, 0, 0
        it = iter(dl)
        for k in range(steps_per_epoch):
            opt.zero_grad(set_to_none=True)
            for m in range(accum):
                x, y, _ = next(it)
                x = x.cuda(non_blocking=True).to(memory_format=torch.channels_last)
                y = y.cuda(non_blocking=True)
                with torch.autocast('cuda', dtype=torch.float16):
                    z = net(x)
                    loss = F.cross_entropy(z, y)
                scaler.scale(loss / accum).backward()
                tl += float(loss) * len(y); tc += int((z.argmax(1) == y).sum()); tn += len(y)
            scaler.step(opt); scaler.update(); sch.step(); step += 1
            if not math.isfinite(tl):
                raise RuntimeError('non-finite loss')
            if a.max_steps and step >= a.max_steps:
                done = True
                break
        mem = torch.cuda.max_memory_allocated() / 2 ** 30
        res = torch.cuda.max_memory_reserved() / 2 ** 30
        hist.append(dict(epoch=ep + 1, loss=tl / max(tn, 1), train_acc=tc / max(tn, 1),
                         lr_backbone=opt.param_groups[0]['lr'], lr_head=opt.param_groups[1]['lr'],
                         peak_alloc_gib=mem, peak_reserved_gib=res, t=time.time() - t0))
        log('epoch %2d  loss %.4f  train acc %.4f  lr %.2e/%.2e  peak alloc %.2f GiB reserved %.2f GiB'
            % (ep + 1, hist[-1]['loss'], hist[-1]['train_acc'], hist[-1]['lr_backbone'],
               hist[-1]['lr_head'], mem, res))
        if done:
            break
    out = a.out or L.ckpt_path(a.head, a.seed)
    torch.save(dict(state_dict=net.state_dict(), head=a.head, seed=a.seed, recipe=R,
                    micro=a.micro, accum=accum, steps=step, history=hist,
                    torch=torch.__version__), out)
    L.save_json(out[:-3] + '_train.json', dict(head=a.head, seed=a.seed, recipe=R, micro=a.micro,
                                               accum=accum, steps=step, history=hist))
    log('saved %s' % out)


if __name__ == '__main__':
    main()
