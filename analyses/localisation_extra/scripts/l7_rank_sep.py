"""Task 7 -- rank-based separation (1 - Spearman) beside 1 - Pearson for the
readouts of Figure 2(d).

Setting = analyses/theory_sweeps/scripts/t2b_interaction64.py, the source of the
Fig. 2(d) values: ResNet-50 layer4 features of Imagenette (frozen), the probe-1
heads cached in analyses/theory_sweeps/out/heads, the same 48 validation images
x 3 references drawn from the same bank of 32 with the same RNG stream
(theory_sweeps_lib.probe1_refs), the explained class = the head's predicted class, single
cell (1x1) substitution, one reference per pair.  Separation per pair, averaged
over the 144 pairs (pairs with a constant map are skipped, as in t2b).

Estimators:
  fd64   literal forward differences in float64 (the t2b estimator); Pearson
         from it must reproduce t2b's one_minus_corr.
  fd32   the same in float32: a rounding check (if fd32 and fd64 agree, the
         rank statistic is not a rounding artefact).
  exact  log-sum-exp heads only: the closed form of analyses/closed_form_maps
         (closed_form_lib.ns_exact, float64, no cancellation).  Against one reference the
         closed form orders the cells identically for N and S, so Spearman = 1.
"""
import os
import sys
import json
import numpy as np
import torch
from scipy.stats import spearmanr
import llib as L

C4S = os.path.join(L.ROOT, 'analyses', 'theory_sweeps', 'scripts')
C5S = os.path.join(L.ROOT, 'analyses', 'closed_form_maps', 'scripts')
sys.path.insert(0, C4S)
sys.path.insert(0, C5S)
import theory_sweeps_lib as L4  # noqa: E402
import closed_form_lib as L5  # noqa: E402

HEADS = ['gap_fc', 'ln_gap', 'tok_mlp', 'gap_mlp', 'bilin', 'ln_post', 'lse0.25', 'lse0.5',
         'lse1.0', 'lse2.0', 'lse4.0', 'lse8.0', 'maxpool', 'attn']
LABEL = {'gap_fc': 'linear control', 'ln_gap': 'per-cell LayerNorm', 'tok_mlp': 'per-cell MLP',
         'gap_mlp': 'two-layer perceptron after the mean', 'bilin': 'bilinear head',
         'ln_post': 'LayerNorm after the mean (not in Fig. 2(d))',
         'lse0.25': 'log-sum-exp T=0.25', 'lse0.5': 'log-sum-exp T=0.5',
         'lse1.0': 'log-sum-exp T=1', 'lse2.0': 'log-sum-exp T=2', 'lse4.0': 'log-sum-exp T=4',
         'lse8.0': 'log-sum-exp T=8', 'maxpool': 'max pooling', 'attn': 'gated attention'}
AGG8 = ['lse0.25', 'lse0.5', 'lse1.0', 'lse2.0', 'lse4.0', 'lse8.0', 'maxpool', 'attn']


def pear(n, s):
    if n.std() > 1e-12 and s.std() > 1e-12:
        return float(np.corrcoef(n, s)[0, 1])
    return np.nan


def spear(n, s):
    if n.std() > 1e-12 and s.std() > 1e-12:
        return float(spearmanr(n, s).statistic)
    return np.nan


def ns_fd(h, A, B, c):
    T = A.shape[1]
    eye = torch.eye(T, dtype=torch.bool, device=A.device)
    f = lambda X: h(X)[:, c]          # noqa: E731
    n, s = L4.region_ns(f, A, B, eye)
    return n.double().cpu().numpy(), s.double().cpu().numpy()


def main():
    t2b = json.load(open(os.path.join(L.ROOT, 'analyses', 'theory_sweeps', 'out',
                                      't2b_interaction64.json')))
    imgs, bank_idx, _ = L4.probe1_sample()
    refs = L4.probe1_refs()
    X64 = L4.batch('val', imgs).double(); B64 = L4.batch('val', bank_idx).double()
    X32 = L4.batch('val', imgs).float(); B32 = L4.batch('val', bank_idx).float()
    res = {}
    for nm in HEADS:
        h64, acc = L4.get_head(nm, double=True)
        h32, _ = L4.get_head(nm, double=False)
        rec = {k: [] for k in ('p64', 's64', 'p32', 's32', 'pex', 'sex', 'zero64', 'zeroex',
                               'ties64')}
        for k in range(X64.shape[0]):
            A = X64[k]
            with torch.no_grad():
                c = int(h64(A[None]).argmax(1))
            for b in refs[k]:
                B = B64[b]
                n, s = ns_fd(h64, A, B, c)
                rec['p64'].append(pear(n, s)); rec['s64'].append(spear(n, s))
                rec['zero64'].append(float(np.mean(n == 0)))
                rec['ties64'].append(float(1 - len(np.unique(n)) / len(n)))
                n3, s3 = ns_fd(h32, X32[k], B32[b], c)
                rec['p32'].append(pear(n3, s3)); rec['s32'].append(spear(n3, s3))
                if nm.startswith('lse'):
                    sa = L5.scores(h64, A, c); sb = L5.scores(h64, B, c)
                    ne, se = L5.ns_exact(sa, sb, float(nm[3:]))
                    rec['pex'].append(pear(ne, se)); rec['sex'].append(spear(ne, se))
                    rec['zeroex'].append(float(np.mean(ne == 0)))

        def m(v):
            v = [x for x in v if np.isfinite(x)]
            return (float(np.mean(v)), len(v)) if v else (None, 0)
        r = dict(label=LABEL[nm], acc=acc)
        for key in ('p64', 's64', 'p32', 's32', 'pex', 'sex'):
            mv, nn = m(rec[key])
            r[key] = mv; r['n_' + key] = nn
        r['one_minus_pearson'] = 1 - r['p64']
        r['one_minus_spearman_fd64'] = 1 - r['s64']
        r['one_minus_spearman_fd32'] = 1 - r['s32']
        r['frac_zero_N_fd64'] = float(np.mean(rec['zero64']))
        r['frac_tied_N_fd64'] = float(np.mean(rec['ties64']))
        if rec['pex']:
            r['one_minus_pearson_exact'] = 1 - r['pex']
            r['one_minus_spearman_exact'] = 1 - r['sex']
            r['frac_zero_N_exact'] = float(np.mean(rec['zeroex']))
            r['min_spearman_exact'] = float(np.nanmin(rec['sex']))
        r['t2b_one_minus_corr'] = t2b[nm]['one_minus_corr']
        r['repro_abs_err'] = abs(r['one_minus_pearson'] - t2b[nm]['one_minus_corr'])
        res[nm] = r
        print('%-8s 1-Pearson %.4f (t2b %.4f, err %.1e) | 1-Spearman fd64 %.4f fd32 %.4f%s'
              ' | zero N fd64 %.3f tied %.3f'
              % (nm, r['one_minus_pearson'], r['t2b_one_minus_corr'], r['repro_abs_err'],
                 r['one_minus_spearman_fd64'], r['one_minus_spearman_fd32'],
                 (' exact %.2e (1-Pearson exact %.4f, min rho %.12f)'
                  % (r['one_minus_spearman_exact'], r['one_minus_pearson_exact'],
                     r['min_spearman_exact'])) if 'one_minus_spearman_exact' in r else '',
                 r['frac_zero_N_fd64'], r['frac_tied_N_fd64']), flush=True)
    # summary of the 8 aggregators as Fig. 2(d) groups them
    def best(k):
        r = res[k]
        return r.get('one_minus_spearman_exact', r['one_minus_spearman_fd64'])
    agg = dict(one_minus_pearson=[res[k]['one_minus_pearson'] for k in AGG8],
               one_minus_spearman=[best(k) for k in AGG8])
    agg['range_pearson'] = [min(agg['one_minus_pearson']), max(agg['one_minus_pearson'])]
    agg['range_spearman'] = [min(agg['one_minus_spearman']), max(agg['one_minus_spearman'])]
    print('8 aggregators: 1-Pearson %.4f..%.4f, 1-Spearman (exact for LSE) %.4f..%.4f'
          % (*agg['range_pearson'], *agg['range_spearman']))

    # ---------------------------------------------------- task 7b: sweep files
    sweep = {}
    p7 = os.path.join(L.ROOT, 'outputs', 'probe7_andor.json')
    j7 = json.load(open(p7))
    sweep['outputs/probe7_andor.json'] = dict(
        producer='scripts/probe7_andor.py', temps=[v['T'] for v in j7.values()],
        fields=sorted(next(iter(j7.values())).keys()),
        one_minus_corrNS={k: 1 - v['corrNS'] for k, v in j7.items()},
        spNS_float32={k: v['spNS'] for k, v in j7.items()})
    p5 = os.path.join(L.ROOT, 'analyses', 'closed_form_maps', 'out', 't3_andor_exact.json')
    j5 = json.load(open(p5))
    sweep['analyses/closed_form_maps/out/t3_andor_exact.json'] = dict(
        producer='analyses/closed_form_maps/scripts/t3_andor_exact.py (closed form)',
        spNS_exact={k: v['spNS_ex'][0] for k, v in j5.items() if k != 'meta'},
        spNS_bank32={k: v['spNS_bank32'][0] for k, v in j5.items()
                     if k != 'meta' and 'spNS_bank32' in v})
    hd = os.path.join(L.ROOT, 'analyses', 'theory_sweeps', 'out', 'heads')
    sweep['analyses/theory_sweeps/out/heads'] = sorted(f[:-3] for f in os.listdir(hd)
                                                       if f.startswith('lse') and f.endswith('.pt'))
    pf = sorted(f for f in os.listdir(os.path.join(L.ROOT, 'outputs'))
                if f.startswith('probe10_fan_'))
    sweep['probe10_fan_files'] = pf
    print('\nLSE temperature sweeps, ResNet-50 / Imagenette, frozen backbone:')
    print(json.dumps(sweep, indent=1))
    L.save('l7_rank_sep.json', dict(design=__doc__, heads=res, aggregators8=agg, sweeps=sweep))

    rows = []
    for k in ['gap_fc', 'ln_gap', 'tok_mlp', 'gap_mlp', 'bilin'] + AGG8:
        r = res[k]
        ex = r.get('one_minus_spearman_exact')
        rows.append('%s & %.4f & %.4f & %.4f & %s' % (
            LABEL[k].replace('T=', '$T = $'),
            r['one_minus_pearson'], r['one_minus_spearman_fd64'], r['one_minus_spearman_fd32'],
            ('%.4f' % ex) if ex is not None else '--'))
    L.longtable(os.path.join(L.OUT, 'tab_l7_rank_sep.tex'), 'lrrrr',
                'Readout & $1-$Pearson & $1-$Spearman (float64) & $1-$Spearman (float32) & '
                '$1-$Spearman (closed form)', rows,
                'Correlation-based and rank-based separation of necessity and sufficiency for the '
                'readouts of Figure 2(d).', 'tab:rank_sep',
                'ResNet-50 features on Imagenette, 48 images $\\times$ 3 single references, '
                'single-cell substitution, mean over pairs. Forward differences in float64 and '
                'float32; for log-sum-exp heads also the closed form, under which the two maps '
                'share one ordering against any single reference.')


if __name__ == '__main__':
    main()
