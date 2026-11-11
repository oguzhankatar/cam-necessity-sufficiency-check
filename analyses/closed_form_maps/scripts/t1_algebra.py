"""T1 -- verify the closed-form algebra and its scope in 500-digit arithmetic.

(a) LSE, one reference, single cell, T in {-8,-1,-0.5,+0.5,+1,+8}: closed form
    (closed_form_lib.ns_exact, float64) vs literal forward differences in mpmath; rank
    correlation of nec and suf (must be exactly 1); both monotone in d_t.
(b) LSE, one reference, 3x3 BLOCK substitution: nec(R), suf(R) monotone in
    d_R = sum_{t in R} d_t, so rank correlation over blocks is again 1.
(c) LSE, maps AVERAGED over K = 8 references before ranking: rank correlation
    below 1 in 500-digit arithmetic (genuine, not rounding).
(d) max pooling and k-of-N (mean of top-k / bottom-k scores), one reference:
    rank correlation below 1 in exact arithmetic (no shared scalar).
Scores are drawn with realistic spread (sd 3 and sd 8 logits on 49 cells).
"""
import json, os, numpy as np, mpmath as mp
from scipy.stats import spearmanr, rankdata
import closed_form_lib as C

mp.mp.dps = 500   # > |T| * score range / ln 10 + 50: literal forward differences cannot cancel
rs = np.random.RandomState(0)
G, N = 7, 49
blocks = []
for t in range(N):
    r, c = divmod(t, G)
    blocks.append([rr * G + cc for rr in range(max(0, r - 1), min(G, r + 2))
                   for cc in range(max(0, c - 1), min(G, c + 2))])


def spm(x, y):
    """Spearman on mpmath numbers via exact ranks (ties averaged)."""
    rx, ry = _rank(x), _rank(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def _rank(v):
    idx = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v); i = 0
    while i < len(idx):
        j = i
        while j + 1 < len(idx) and v[idx[j + 1]] == v[idx[i]]:
            j += 1
        for k in range(i, j + 1):
            r[idx[k]] = (i + j) / 2.0
        i = j + 1
    return np.array(r)


def lse_z(v, T):
    T = mp.mpf(T)
    return mp.log(mp.fsum(mp.exp(T * x) for x in v) / len(v)) / T


def ns_units(z, sa, sb, units):
    za, zb = z(sa), z(sb)
    Nn, Ss = [], []
    for u in units:
        a2 = list(sa); b2 = list(sb)
        for t in u:
            a2[t] = sb[t]; b2[t] = sa[t]
        Nn.append(za - z(a2)); Ss.append(z(b2) - zb)
    return Nn, Ss


out = {'a_single_ref_cell': [], 'b_single_ref_block': [], 'c_bank_avg': [], 'd_other_pools': []}
TS = [-8.0, -1.0, -0.5, 0.5, 1.0, 8.0]
for sd in (3.0, 8.0):
    for trial in range(3):
        sa = [mp.mpf(float(v)) for v in rs.randn(N) * sd]
        sb = [mp.mpf(float(v)) for v in rs.randn(N) * sd]
        for T in TS:
            z = lambda v, T=T: lse_z(v, T)
            Nm, Sm = ns_units(z, sa, sb, [[t] for t in range(N)])
            d = [mp.exp(T * sa[t]) - mp.exp(T * sb[t]) for t in range(N)]
            ne, se = C.ns_exact([float(v) for v in sa], [float(v) for v in sb], T)
            relN = max(abs(float((mp.mpf(ne[t]) - Nm[t]) / Nm[t])) for t in range(N) if Nm[t] != 0)
            relS = max(abs(float((mp.mpf(se[t]) - Sm[t]) / Sm[t])) for t in range(N) if Sm[t] != 0)
            sgn = 1 if T > 0 else -1
            # monotone in d: rank(nec) == rank(sgn*d) and rank(suf) == rank(sgn*d)
            mon = bool((_rank(Nm) == _rank([sgn * x for x in d])).all()
                       and (_rank(Sm) == _rank([sgn * x for x in d])).all())
            Bm, Bs = ns_units(z, sa, sb, blocks)
            out['a_single_ref_cell'].append(dict(
                sd=sd, trial=trial, T=T, spearman_mp=spm(Nm, Sm),
                spearman_closedform_f64=float(spearmanr(ne, se).statistic),
                pearson_mp=float(np.corrcoef([float(v) for v in Nm], [float(v) for v in Sm])[0, 1]),
                monotone_in_d=mon, closedform_max_rel_err_N=relN, closedform_max_rel_err_S=relS,
                min_abs_nec=float(min(abs(v) for v in Nm))))
            out['b_single_ref_block'].append(dict(sd=sd, trial=trial, T=T, spearman_mp=spm(Bm, Bs)))
        # (c) averaged over a bank of 8 references
        for T in (-8.0, 1.0, 8.0):
            z = lambda v, T=T: lse_z(v, T)
            Nacc = [mp.mpf(0)] * N; Sacc = [mp.mpf(0)] * N
            for k in range(8):
                sbk = [mp.mpf(float(v)) for v in rs.randn(N) * sd]
                Nm, Sm = ns_units(z, sa, sbk, [[t] for t in range(N)])
                Nacc = [x + y / 8 for x, y in zip(Nacc, Nm)]
                Sacc = [x + y / 8 for x, y in zip(Sacc, Sm)]
            out['c_bank_avg'].append(dict(sd=sd, trial=trial, T=T, spearman_mp=spm(Nacc, Sacc)))
        # (d) max pooling and k-of-N, single reference
        for name, k in (('max', 1), ('top5', 5), ('top10', 10), ('bottom5', -5), ('bottom10', -10)):
            def z(v, k=k):
                s = sorted(v)
                return mp.fsum(s[-k:]) / k if k > 0 else mp.fsum(s[:(-k)]) / (-k)
            Nm, Sm = ns_units(z, sa, sb, [[t] for t in range(N)])
            out['d_other_pools'].append(dict(sd=sd, trial=trial, pool=name, spearman_mp=spm(Nm, Sm),
                                             n_nonzero_nec=int(sum(1 for v in Nm if v != 0)),
                                             n_nonzero_suf=int(sum(1 for v in Sm if v != 0))))

summ = {}
a = out['a_single_ref_cell']
summ['a_min_spearman'] = min(r['spearman_mp'] for r in a)
summ['a_all_monotone_in_d'] = all(r['monotone_in_d'] for r in a)
summ['a_min_pearson'] = min(r['pearson_mp'] for r in a)
summ['a_closedform_max_rel_err'] = max(max(r['closedform_max_rel_err_N'], r['closedform_max_rel_err_S']) for r in a)
summ['a_closedform_min_spearman'] = min(r['spearman_closedform_f64'] for r in a)
summ['a_smallest_abs_nec'] = min(r['min_abs_nec'] for r in a)
summ['b_min_spearman'] = min(r['spearman_mp'] for r in out['b_single_ref_block'])
summ['c_spearman_range'] = [min(r['spearman_mp'] for r in out['c_bank_avg']),
                            max(r['spearman_mp'] for r in out['c_bank_avg'])]
summ['d_spearman_by_pool'] = {p: [round(r['spearman_mp'], 4) for r in out['d_other_pools'] if r['pool'] == p]
                              for p in ('max', 'top5', 'top10', 'bottom5', 'bottom10')}
out['summary'] = summ
json.dump(out, open(os.path.join(C.OUTD, 't1_algebra.json'), 'w'), indent=1)
print(json.dumps(summ, indent=1))
