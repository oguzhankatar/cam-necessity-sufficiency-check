"""C5 (exact necessity and sufficiency of log-sum-exp heads) -- shared machinery.

Exact, cancellation-free necessity / sufficiency for a score-then-pool
log-sum-exp head  z = (1/T) log((1/N) sum_u exp(T s_u)),  s_u = w_c.a_u + b_c,
against ONE reference b, single-cell substitution:

    nec(t) = -(1/T) log(1 - r_t),  r_t = d_t / Z_a
    suf(t) =  (1/T) log(1 + q_t),  q_t = d_t / Z_b
    d_t = exp(T s_t(a)) - exp(T s_t(b)),  Z_x = sum_u exp(T s_u(x)).

Evaluated in float64 in the log domain:
  |r| < 1/2 :  r = exp(x_a,t - LA) * (-expm1(x_b,t - x_a,t)),  nec = -log1p(-r)/T
  otherwise :  nec = (LA - logaddexp(LSE_{u!=t} x_a,u, x_b,t)) / T
and the same for suf with the roles of a and b exchanged.  No difference of two
nearly equal logits is ever formed.  T = 0 gives (s_t(a) - s_t(b)) / N.

Nothing in scripts/ or outputs/ is modified; the original estimators are
imported read-only so that reference draws and RNG consumption are identical.
"""
import os, sys, numpy as np, torch
from scipy.special import logsumexp

HERE = os.path.dirname(os.path.abspath(__file__))
C5 = os.path.abspath(os.path.join(HERE, '..'))
ROOT = os.path.abspath(os.path.join(C5, '..', '..'))
SCRIPTS = os.path.join(ROOT, 'scripts')
OUTD = os.path.join(C5, 'out')
os.makedirs(OUTD, exist_ok=True)
sys.path.insert(0, SCRIPTS)


def scores(h, A, c):
    """per-cell class score s_t = fc(a_t)[c] in float64. A: (C,T) tensor."""
    W = h.fc.weight[c].detach().double()
    b = h.fc.bias[c].detach().double()
    return (A.double().T @ W + b).cpu().numpy()


def _lse_excl(x):
    """LSE over u != t, for every t, float64 (N small)."""
    n = len(x)
    M = np.broadcast_to(x, (n, n)).copy()
    np.fill_diagonal(M, -np.inf)
    return logsumexp(M, axis=1)


def ns_exact(sa, sb, T):
    """closed-form necessity and sufficiency (float64, no cancellation)."""
    sa = np.asarray(sa, np.float64); sb = np.asarray(sb, np.float64)
    n = len(sa)
    if T == 0:
        d = (sa - sb) / n
        return d.copy(), d.copy()
    xa, xb = T * sa, T * sb
    LA, LB = logsumexp(xa), logsumexp(xb)
    ra, rb = _lse_excl(xa), _lse_excl(xb)
    # necessity
    with np.errstate(over='ignore', invalid='ignore'):
        r = np.exp(xa - LA) * (-np.expm1(xb - xa))
        nec_small = -np.log1p(-r) / T
        nec_big = (LA - np.logaddexp(ra, xb)) / T
        nec = np.where(np.abs(r) < 0.5, nec_small, nec_big)
        q = np.exp(xb - LB) * np.expm1(xa - xb)
        suf_small = np.log1p(q) / T
        suf_big = (np.logaddexp(rb, xa) - LB) / T
        suf = np.where(np.abs(q) < 0.5, suf_small, suf_big)
    return nec, suf


def ns_exact_units(sa, sb, T, units):
    """closed form for substitution of a UNIT (list of cells) at once:
    nec(R) = -(1/T) log(1 - d_R/Z_a), suf(R) = (1/T) log(1 + d_R/Z_b),
    d_R = sum_{t in R} d_t.  Same two-branch evaluation as ns_exact."""
    sa = np.asarray(sa, np.float64); sb = np.asarray(sb, np.float64)
    n = len(sa)
    if T == 0:
        d = (sa - sb) / n
        return (np.array([d[u].sum() for u in units]),) * 2
    xa, xb = T * sa, T * sb
    LA, LB = logsumexp(xa), logsumexp(xb)
    N = np.empty(len(units)); S = np.empty(len(units))
    for i, u in enumerate(units):
        u = np.asarray(u); m = np.ones(n, bool); m[u] = False
        r = float(np.sum(np.exp(xa[u] - LA) * (-np.expm1(xb[u] - xa[u]))))
        q = float(np.sum(np.exp(xb[u] - LB) * np.expm1(xa[u] - xb[u])))
        ra = logsumexp(xa[m]) if m.any() else -np.inf
        rb = logsumexp(xb[m]) if m.any() else -np.inf
        N[i] = -np.log1p(-r) / T if abs(r) < 0.5 else (LA - np.logaddexp(ra, logsumexp(xb[u]))) / T
        S[i] = np.log1p(q) / T if abs(q) < 0.5 else (np.logaddexp(rb, logsumexp(xa[u])) - LB) / T
    return N, S


def ns_exact_mp(sa, sb, T, dps=60):
    """the same quantities by literal forward differences in mpmath (reference)."""
    import mpmath as mp
    mp.mp.dps = dps
    sa = [mp.mpf(float(v)) for v in sa]; sb = [mp.mpf(float(v)) for v in sb]
    n = len(sa); T = mp.mpf(T)

    def z(v):
        return mp.log(mp.fsum(mp.exp(T * x) for x in v) / n) / T
    za, zb = z(sa), z(sb)
    N, S = [], []
    for t in range(n):
        a2 = list(sa); a2[t] = sb[t]
        b2 = list(sb); b2[t] = sa[t]
        N.append(za - z(a2)); S.append(z(b2) - zb)
    return N, S


@torch.no_grad()
def auc64(h64, A, B, c, order, mode, steps=None):
    """nslib.auc / probe4_utility.auc, verbatim in definition, float64 head."""
    T = A.shape[1]
    steps = T if steps is None else steps
    order = torch.as_tensor(np.asarray(order, np.float64), device=A.device)
    rank = torch.argsort(order, descending=True)
    A, B = A.double(), B.double()
    start, fin = (A, B) if mode == 'del' else (B, A)
    cur = start.clone(); ps = [torch.softmax(h64(cur[None]), 1)[0, c].item()]
    for i in range(steps):
        cur[:, rank[i]] = fin[:, rank[i]]
        ps.append(torch.softmax(h64(cur[None]), 1)[0, c].item())
    p = np.array(ps)
    return float((p.sum() - p[0] / 2 - p[-1] / 2) / (len(p) - 1))


@torch.no_grad()
def maps_fd64(h64, A, B, c):
    """forward differences, float64 (the paper's estimator, higher precision)."""
    import nslib as L
    return [v.cpu().numpy() for v in L.maps(h64, A.double(), B.double(), c)]


def gradcam64(h64, A, c):
    import nslib as L
    return L.gradcam_head(h64, A.double(), c).cpu().numpy()


def sp(x, y):
    from scipy.stats import spearmanr
    x = np.asarray(x, np.float64); y = np.asarray(y, np.float64)
    if x.std() <= 1e-300 or y.std() <= 1e-300:
        return np.nan
    return float(spearmanr(x, y).statistic)


def boot(v, n=2000, seed=0):
    v = np.asarray([z for z in v if np.isfinite(z)], float)
    rs = np.random.RandomState(seed)
    b = np.array([v[rs.randint(0, len(v), len(v))].mean() for _ in range(n)])
    return [float(v.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)), int(len(v))]
