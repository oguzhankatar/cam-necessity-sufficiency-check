"""C1 task 6 -- paired uncertainty for Table 2's accuracy changes.

outputs/probe25_lnplace_*.json store only aggregate accuracies (acc.post /
acc.pre / acc.none, n_eval = 50000); per-image correctness is NOT stored.  This
script re-creates it with probe25's own adapter (read-only import of
scripts/probe25_lnplace.py: Site, transform), on the full 50 000-image val set
in the same order and with the same three orderings, checks that the recomputed
counts reproduce the archived accuracies, and then computes for every edit
(relocation and deletion, relative to the shipped ordering):
   discordant counts b (shipped right, edited wrong) and c (edited right,
   shipped wrong); exact two-sided McNemar P (binomial on b+c, p=1/2);
   95% Wald CI for the paired difference using the paired variance
   ((b+c)/n - d^2)/n.
Also a binomial-approximation interval that uses only the archived counts
(unpaired, worst case), for comparison.
"""
import os, sys, json, glob, time
import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'out')
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from scipy.stats import binom, beta, norm


def exact_mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return float(min(1.0, 2 * binom.cdf(min(b, c), n, 0.5)))


def paired_ci(b, c, n, z=1.959964):
    d = (c - b) / n
    var = ((b + c) / n - d ** 2) / n
    return d, d - z * np.sqrt(var), d + z * np.sqrt(var)


def unpaired_ci(p1, p0, n, z=1.959964):
    d = p1 - p0
    se = np.sqrt(p1 * (1 - p1) / n + p0 * (1 - p0) / n)
    return d - z * se, d + z * se


def main():
    import probe25_lnplace as P
    from torchvision import datasets
    from torch.utils.data import DataLoader
    files = sorted(glob.glob(os.path.join(ROOT, 'outputs', 'probe25_lnplace_*.json')))
    only = set(sys.argv[1].split(',')) if len(sys.argv) > 1 else None
    res = {}
    for fp in files:
        arch = json.load(open(fp))
        name = arch['meta']['model']
        if only and name.split('.')[0] not in only:
            continue
        npz = os.path.join(OUT, 'c6_correct_%s.npz' % name.split('.')[0])
        t0 = time.time()
        if os.path.exists(npz):
            z = np.load(npz); corr = {k: z[k] for k in P.MODES}
        else:
            st = P.Site(name)
            ds = datasets.ImageFolder(os.path.join(ROOT, 'data', 'in1k', 'val'), P.transform(st.cfg))
            ld = DataLoader(ds, batch_size=64, num_workers=8, pin_memory=True)
            corr = {k: [] for k in P.MODES}
            with torch.no_grad():
                for x, y in ld:
                    X = st.cells(x.to('cuda', non_blocking=True))
                    for k in P.MODES:
                        corr[k].append((st.head(X, k).argmax(1).cpu() == y).numpy())
            corr = {k: np.concatenate(v) for k, v in corr.items()}
            np.savez_compressed(npz, **corr)
            del st
            torch.cuda.empty_cache()
        n = len(corr['post'])
        shipped = arch['meta']['shipped']
        rec = dict(model=name, shipped=shipped, n=n,
                   acc_archived=arch['acc'],
                   acc_recomputed={k: float(corr[k].mean()) for k in P.MODES},
                   hits_recomputed={k: int(corr[k].sum()) for k in P.MODES})
        rec['reproduces_archived'] = all(int(round(arch['acc'][k] * n)) == rec['hits_recomputed'][k]
                                         for k in P.MODES)
        rec['max_count_diff_vs_archived'] = max(abs(int(round(arch['acc'][k] * n)) - rec['hits_recomputed'][k])
                                                for k in P.MODES)
        edits = {}
        relocate = 'pre' if shipped == 'post' else 'post'
        for lab, tgt in (('relocate', relocate), ('delete', 'none')):
            if shipped == 'pre' and tgt == 'none':
                continue        # Table 2: n/a for Swin (deletion leaves it degenerate)
            s, e = corr[shipped], corr[tgt]
            b = int((s & ~e).sum()); c = int((~s & e).sum())
            d, lo, hi = paired_ci(b, c, n)
            lo_u, hi_u = unpaired_ci(e.mean(), s.mean(), n)
            # archived-count-only interval (no pairing): what is possible without per-image data
            pa1, pa0 = arch['acc'][tgt], arch['acc'][shipped]
            lo_a, hi_a = unpaired_ci(pa1, pa0, n)
            edits[lab] = dict(target=tgt, b_shipped_only=b, c_edited_only=c, discordant=b + c,
                              dacc_pp=100 * d, ci95_pp=[100 * lo, 100 * hi],
                              mcnemar_exact_p=exact_mcnemar(b, c),
                              unpaired_ci95_pp=[100 * lo_u, 100 * hi_u],
                              archived_count_unpaired_ci95_pp=[100 * lo_a, 100 * hi_a],
                              archived_dacc_pp=100 * (pa1 - pa0))
        rec['edits'] = edits
        res[name] = rec
        print('%-50s repro=%s (maxdiff %d) %.0fs' % (name, rec['reproduces_archived'],
                                                   rec['max_count_diff_vs_archived'], time.time() - t0))
        for lab, e in edits.items():
            print('   %-9s d=%+.3f pp  CI[%+.3f,%+.3f]  b=%d c=%d  P=%.3g' % (
                lab, e['dacc_pp'], e['ci95_pp'][0], e['ci95_pp'][1], e['b_shipped_only'],
                e['c_edited_only'], e['mcnemar_exact_p']), flush=True)
    tag = '' if not only else '_partial'
    json.dump(res, open(os.path.join(OUT, 'c6_mcnemar%s.json' % tag), 'w'), indent=1)


if __name__ == '__main__':
    main()
