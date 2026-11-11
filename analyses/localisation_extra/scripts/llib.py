"""Shared helpers for analyses/localisation_extra (localisation analyses).

Reads the archived per-record arrays in outputs/wsol READ-ONLY and runs the
tests of scripts/wsol/w3_stats.py exactly (same wlib functions, same seeds,
same resample counts).  Nothing under scripts/ or outputs/ is written; every
output goes to analyses/localisation_extra/out.
"""
import os
import sys
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LDIR = os.path.dirname(HERE)
ROOT = os.path.abspath(os.path.join(LDIR, '..', '..'))
WS = os.path.join(ROOT, 'scripts', 'wsol')
WO = os.path.join(ROOT, 'outputs', 'wsol')
OUT = os.path.join(LDIR, 'out')
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, WS)
import wlib as W  # noqa: E402  (pure functions; importing it writes nothing)

DATASETS = ['cub', 'voc12', 'voc12seg', 'coco17']
BACKBONES = ['resnet50', 'vgg16gap', 'vitb16']
METRICS = {'cub': ['MaxBoxAccV2', 'best_iou', 'pointing_strict', 'pointing_tol15'],
           'voc12': ['pointing_strict', 'pointing_tol15'],
           'voc12seg': ['pointing_strict', 'pointing_tol15', 'peak_mask_iou'],
           'coco17': ['pointing_strict', 'pointing_tol15', 'peak_mask_iou']}
EXTENT = {'MaxBoxAccV2', 'best_iou', 'peak_mask_iou'}
MAXLOC = {'pointing_strict', 'pointing_tol15'}
MNAME = {'MaxBoxAccV2': 'MaxBoxAccV2', 'best_iou': 'best IoU',
         'pointing_strict': 'pointing, strict', 'pointing_tol15': 'pointing, 15 px',
         'peak_mask_iou': 'peak-mask IoU'}
DSNAME = {'cub': 'CUB', 'voc12': 'VOC boxes', 'voc12seg': 'VOC masks', 'coco17': 'COCO'}
BKNAME = {'resnet50': 'ResNet-50', 'vgg16gap': 'VGG-16', 'vitb16': 'ViT-B/16'}
ALPHA = 0.05
PRIMARY_PAIRS = [('N_lse', 'collapsed'), ('S_lse', 'collapsed'),
                 ('N_lse', 'center'), ('S_lse', 'center'),
                 ('N_ln', 'collapsed'), ('S_ln', 'collapsed'),
                 ('N_ln', 'center'), ('S_ln', 'center')]


def mclass(metric):
    return 'extent' if metric in EXTENT else 'maxloc'


def arm_of(a):
    return 'relocation' if '_ln' in a else 'aggregator'


def w2(ds, bk, suf=''):
    return json.load(open(os.path.join(WO, 'w2_%s_%s%s.json' % (ds, bk, suf))))


def deltas(ds, bk, suf=''):
    return tuple(w2(ds, bk, suf)['deltas'])


def arr(ds, bk, kind, mp, suf=''):
    p = os.path.join(WO, '%s_%s_%s%s_%s.npy' % (kind, ds, bk, suf, mp))
    return np.load(p) if os.path.exists(p) else None


def per_record(ds, bk, metric, mp, suf=''):
    """exactly the vector (or curve matrix) w3_stats.py tests for this metric."""
    if metric == 'MaxBoxAccV2':
        return arr(ds, bk, 'curve', mp, suf)
    if metric == 'best_iou':
        c = arr(ds, bk, 'curve', mp, suf)
        return None if c is None else c.max(1).astype(np.float64)
    if metric.startswith('pointing'):
        p = arr(ds, bk, 'pt', mp, suf)
        return None if p is None else p[0 if metric == 'pointing_strict' else 1]
    return arr(ds, bk, 'iou', mp, suf)


def test(ds, bk, metric, a, b, suf=''):
    """the test w3_stats.py runs for (metric, a, b): same functions, seed, n."""
    xa, xb = per_record(ds, bk, metric, a, suf), per_record(ds, bk, metric, b, suf)
    if xa is None or xb is None:
        return None
    P = W.PROTOCOL
    if metric == 'MaxBoxAccV2':
        dl = deltas(ds, bk, suf)
        d, lo, hi, p = W.boot_mbav2_diff(xa, xb, n=P['n_boot_mbav2'], seed=P['seed'],
                                         deltas=dl)
        return dict(va=W.maxboxacc_v2(xa, dl)[0], vb=W.maxboxacc_v2(xb, dl)[0],
                    diff=d, ci=[lo, hi], p=p, test='paired bootstrap', n=int(len(xa)))
    if metric.startswith('pointing'):
        xa, xb = xa.astype(bool), xb.astype(bool)
        d, lo, hi = W.boot_diff(xa.astype(float), xb.astype(float), n=P['n_boot'],
                                seed=P['seed'])
        mc = W.mcnemar(xa, xb)
        return dict(va=float(xa.mean()), vb=float(xb.mean()), diff=d, ci=[lo, hi],
                    p=mc['p'], test='mcnemar', n10=mc['n10'], n01=mc['n01'],
                    n=int(len(xa)))
    xa, xb = np.asarray(xa, np.float64), np.asarray(xb, np.float64)
    d, lo, hi = W.boot_diff(xa, xb, n=P['n_boot'], seed=P['seed'])
    w = W.wilcoxon(xa, xb)
    return dict(va=float(xa.mean()), vb=float(xb.mean()), diff=d, ci=[lo, hi],
                p=w['p'], test='wilcoxon', rank_biserial=w['rank_biserial'],
                n=int(len(xa)))


def holm(p):
    return W.holm(p)


def verdict(diff, p_holm):
    if p_holm < ALPHA:
        return 'better' if diff > 0 else 'worse'
    return 'no difference'


def ci_excludes_zero(ci):
    return ci[0] > 0 or ci[1] < 0


def ci_disagrees(r, pkey='p_holm'):
    """Holm verdict vs unadjusted 95% bootstrap CI of the mean difference."""
    sig = r[pkey] < ALPHA
    return sig != ci_excludes_zero(r['ci'])


def load_w3():
    return json.load(open(os.path.join(WO, 'w3_stats.json')))


def save(name, obj):
    p = os.path.join(OUT, name)
    json.dump(obj, open(p, 'w'), indent=1, default=W._jd)
    print('  -> %s' % p, flush=True)
    return p


# ------------------------------------------------------------ LaTeX helpers
def f4(x):
    return '%.4f' % x


def fs(x):
    s = '%+.4f' % x
    return s.replace('-', '$-$')


def fp(p):
    if p == 0:
        return '$<10^{-300}$'
    if p < 1e-3:
        m, e = ('%.1e' % p).split('e')
        return '$%s\\times10^{%d}$' % (m, int(e))
    return '%.3f' % p


def tex_escape(s):
    return s.replace('_', '\\_')


def longtable(path, colspec, header, rows, caption, label, note=None):
    """write a self-contained longtable (needs \\usepackage{longtable,booktabs})."""
    L = ['% generated by analyses/localisation_extra/scripts -- do not edit by hand',
         '\\begin{longtable}{%s}' % colspec,
         '\\caption{%s}\\label{%s}\\\\' % (caption, label),
         '\\toprule', header + ' \\\\', '\\midrule', '\\endfirsthead',
         '\\toprule', header + ' \\\\', '\\midrule', '\\endhead',
         '\\midrule \\multicolumn{%d}{r}{\\emph{continued}} \\\\' % (colspec.count('l') +
                                                                    colspec.count('r') +
                                                                    colspec.count('c')),
         '\\endfoot', '\\bottomrule',
         ('\\multicolumn{%d}{p{0.95\\linewidth}}{\\footnotesize %s} \\\\'
          % (colspec.count('l') + colspec.count('r') + colspec.count('c'), note))
         if note else '', '\\endlastfoot']
    L += [r + ' \\\\' for r in rows]
    L.append('\\end{longtable}')
    open(path, 'w', encoding='utf-8').write('\n'.join(l for l in L if l != '') + '\n')
    print('  -> %s' % path, flush=True)
