"""C6 task 3.  Constructed-head probe set: the frozen check's error envelope.

Each probe is a small module with a `forward_head(f)` on a random feature map
(C = 16 channels, 7 x 7 = 49 cells; token probes read the same 49 cells as a
(B, 49, C) sequence, class-token probes prepend 3 class tokens).  For each probe:

  frozen   = scripts/heldout/frozen_rule.structural(model, f)   (read-only import)
  measured = the paper's float64 certificate (Sec. 2.2): eps1 (max |nec - suf|,
             relative), eps2 (half-subset residual), eps3 (pair residual), three
             probe pairs, threshold 1e-9; VACUOUS if additive with zero spatial
             mass; NORM-ONLY if additive only after the post-aggregator
             normalisations (known by construction) are replaced by Identity.
  agree    = the paper's scoring (h1_run / hlib.GT_OF): frozen code -> label
             predicted, compared with the measured label.
  reason   = whether a blocked verdict also names the operation that actually
             blocks (by construction).

Random weights (seed 0), float64, CPU.  Outputs: out/a3_probes.json,
out/a3_probes.md, out/a3_probes.tex.
"""
import os, sys, json, math
import torch
import torch.nn as nn
import torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import check_audit_lib as L
import frozen_rule as FR
import rule_v2 as R2                      # later operator-level variant, for comparison only
import timm.layers as TL

C, H, K = 16, 7, 10
torch.set_default_dtype(torch.float64)


class Probe(nn.Module):
    """forward_head(f) -> logits; `fmt` / `prefix` tell the certificate where the cells are."""
    fmt, prefix, post_norms = 'BCHW', (), ()

    def forward_head(self, f):
        return self.fh(f)


def probe(name, desc, design, fmt='BCHW', prefix=(), post_norms=(), global_pool=None):
    def deco(cls):
        cls.pname, cls.desc, cls.design = name, desc, design
        cls.fmt, cls.prefix, cls.post_norms = fmt, tuple(prefix), tuple(post_norms)
        cls.gp = global_pool
        return cls
    return deco


PROBES = []


def reg(cls):
    PROBES.append(cls)
    return cls


# ------------------------------------------------------------ mean + linear
@reg
@probe('P01', 'mean as a module (AdaptiveAvgPool2d) -> Flatten -> Linear', 'degenerate')
class P01(Probe):
    def __init__(s):
        super().__init__(); s.pool = nn.AdaptiveAvgPool2d(1); s.flat = nn.Flatten(); s.head = nn.Linear(C, K)
    def fh(s, f): return s.head(s.flat(s.pool(f)))


@reg
@probe('P02', 'mean as a tensor op x.mean((2,3)) -> Linear, no declared pool', 'degenerate')
class P02(Probe):
    def __init__(s):
        super().__init__(); s.head = nn.Linear(C, K)
    def fh(s, f): return s.head(f.mean((2, 3)))


# ------------------------------------------------------------ max pooling
@reg
@probe('P03', 'max pool as a module (AdaptiveMaxPool2d) -> Linear', 'aggregator-blocked')
class P03(Probe):
    def __init__(s):
        super().__init__(); s.pool = nn.AdaptiveMaxPool2d(1); s.head = nn.Linear(C, K)
    def fh(s, f): return s.head(s.pool(f).flatten(1))


@reg
@probe('P04', "max pool as timm SelectAdaptivePool2d(pool_type='max') -> Linear", 'aggregator-blocked')
class P04(Probe):
    def __init__(s):
        super().__init__(); s.global_pool = TL.SelectAdaptivePool2d(pool_type='max', flatten=True); s.head = nn.Linear(C, K)
    def fh(s, f): return s.head(s.global_pool(f))


@reg
@probe('P05', 'max as a tensor op x.amax((2,3)) -> Linear, no declared pool', 'aggregator-blocked')
class P05(Probe):
    def __init__(s):
        super().__init__(); s.head = nn.Linear(C, K)
    def fh(s, f): return s.head(f.amax((2, 3)))


@reg
@probe('P06', "max as a tensor op, model declares global_pool='max'", 'aggregator-blocked', global_pool='max')
class P06(P05):
    pass


# ------------------------------------------------------------ attention pooling
def _apl(norm=False, mlp=False):
    ap = TL.AttentionPoolLatent(C, num_heads=2, mlp_ratio=2.0,
                                norm_layer=nn.LayerNorm if norm else None)
    if not mlp:
        ap.mlp = None
    return ap


@reg
@probe('P07', "attention pool module without MLP (timm AttentionPoolLatent, mlp=None) -> Linear; global_pool='map'",
       'aggregator-blocked', fmt='BTC', global_pool='map')
class P07(Probe):
    def __init__(s):
        super().__init__(); s.attn_pool = _apl(); s.head = nn.Linear(C, K)
    def fh(s, f): return s.head(s.attn_pool(f))


@reg
@probe('P08', "attention pool + LayerNorm + MLP (timm AttentionPoolLatent as in SigLIP 'map') -> Linear",
       'aggregator-blocked', fmt='BTC', global_pool='map', post_norms=('attn_pool.norm',))
class P08(Probe):
    def __init__(s):
        super().__init__(); s.attn_pool = _apl(norm=True, mlp=True); s.head = nn.Linear(C, K)
    def fh(s, f): return s.head(s.attn_pool(f))


@reg
@probe('P09', 'attention pool written as tensor ops (Linear k, v; softmax over cells) -> Linear, no declared pool',
       'aggregator-blocked', fmt='BTC')
class P09(Probe):
    def __init__(s):
        super().__init__(); s.k = nn.Linear(C, C); s.v = nn.Linear(C, C)
        s.q = nn.Parameter(torch.randn(C)); s.head = nn.Linear(C, K)
    def fh(s, f):
        a = torch.softmax(s.k(f) @ s.q / math.sqrt(C), dim=1)
        return s.head((a[..., None] * s.v(f)).sum(1))


@reg
@probe('P10', 'attention pool (tensor ops) followed by an MLP (Linear-GELU-Linear)', 'aggregator-blocked', fmt='BTC')
class P10(P09):
    def __init__(s):
        super().__init__(); s.fc1 = nn.Linear(C, 2 * C); s.act = nn.GELU(); s.head = nn.Linear(2 * C, K)
    def fh(s, f):
        a = torch.softmax(s.k(f) @ s.q / math.sqrt(C), dim=1)
        return s.head(s.act(s.fc1((a[..., None] * s.v(f)).sum(1))))


# ------------------------------------------------------------ class-token readouts
@reg
@probe('P11', 'stack of 3 class tokens reduced by Conv1d(3,1,1) -> Linear (CoaT head)', 'class-token',
       fmt='BTC', prefix=(0, 1, 2))
class P11(Probe):
    def __init__(s):
        super().__init__(); s.aggregate = nn.Conv1d(3, 1, 1); s.head = nn.Linear(C, K)
    def fh(s, f): return s.head(s.aggregate(f[:, :3]).squeeze(1))


@reg
@probe('P12', 'stack of 3 class tokens reduced by Linear(3,1) over the token axis -> Linear', 'class-token',
       fmt='BTC', prefix=(0, 1, 2))
class P12(Probe):
    def __init__(s):
        super().__init__(); s.mix = nn.Linear(3, 1); s.head = nn.Linear(C, K)
    def fh(s, f): return s.head(s.mix(f[:, :3].transpose(1, 2)).squeeze(-1))


@reg
@probe('P13', "single class token x[:, 0] -> Linear, global_pool='token'", 'class-token',
       fmt='BTC', prefix=(0, 1, 2), global_pool='token')
class P13(Probe):
    def __init__(s):
        super().__init__(); s.head = nn.Linear(C, K)
    def fh(s, f): return s.head(f[:, 0])


# ------------------------------------------------------------ normalisations
@reg
@probe('P14', 'LayerNorm after the mean (AdaptiveAvgPool2d -> LayerNorm -> Linear)', 'normalisation-blocked',
       post_norms=('norm',))
class P14(Probe):
    def __init__(s):
        super().__init__(); s.pool = nn.AdaptiveAvgPool2d(1); s.norm = nn.LayerNorm(C); s.head = nn.Linear(C, K)
    def fh(s, f): return s.head(s.norm(s.pool(f).flatten(1)))


@reg
@probe('P15', 'LayerNorm before the mean, per cell (LayerNorm2d -> AdaptiveAvgPool2d -> Linear)', 'degenerate')
class P15(Probe):
    def __init__(s):
        super().__init__(); s.norm_pre = TL.LayerNorm2d(C); s.pool = nn.AdaptiveAvgPool2d(1); s.head = nn.Linear(C, K)
    def fh(s, f): return s.head(s.pool(s.norm_pre(f)).flatten(1))


@reg
@probe('P16', 'LayerNorm before the mean, per token, mean as a tensor op (LayerNorm -> x.mean(1) -> Linear)',
       'degenerate', fmt='BTC')
class P16(Probe):
    def __init__(s):
        super().__init__(); s.norm_pre = nn.LayerNorm(C); s.head = nn.Linear(C, K)
    def fh(s, f): return s.head(s.norm_pre(f).mean(1))


@reg
@probe('P17', 'RMSNorm after the mean (timm RmsNorm)', 'normalisation-blocked', post_norms=('norm',))
class P17(Probe):
    def __init__(s):
        super().__init__(); s.pool = nn.AdaptiveAvgPool2d(1); s.norm = TL.RmsNorm(C); s.head = nn.Linear(C, K)
    def fh(s, f): return s.head(s.norm(s.pool(f).flatten(1)))


@reg
@probe('P18', 'RMSNorm after the mean (torch nn.RMSNorm)', 'normalisation-blocked', post_norms=('norm',))
class P18(P17):
    def __init__(s):
        super().__init__(); s.norm = nn.RMSNorm(C)


@reg
@probe('P19', 'SimpleNorm2d after the mean (ConvNeXt-Zepto-RMS head: pool -> SimpleNorm2d -> Linear)',
       'normalisation-blocked', post_norms=('norm',))
class P19(Probe):
    def __init__(s):
        super().__init__(); s.pool = nn.AdaptiveAvgPool2d(1); s.norm = TL.SimpleNorm2d(C); s.head = nn.Linear(C, K)
    def fh(s, f): return s.head(s.norm(s.pool(f)).flatten(1))


# ------------------------------------------------------------ per-cell maps before the mean
@reg
@probe('P20', 'per-cell MLP before the mean (Conv1x1-GELU-Conv1x1 -> AdaptiveAvgPool2d -> Linear)', 'degenerate')
class P20(Probe):
    def __init__(s):
        super().__init__(); s.c1 = nn.Conv2d(C, 2 * C, 1); s.act = nn.GELU(); s.c2 = nn.Conv2d(2 * C, C, 1)
        s.pool = nn.AdaptiveAvgPool2d(1); s.head = nn.Linear(C, K)
    def fh(s, f): return s.head(s.pool(s.c2(s.act(s.c1(f)))).flatten(1))


@reg
@probe('P21', 'per-token MLP before a tensor-op mean (Linear-GELU-Linear -> x.mean(1) -> Linear)', 'degenerate', fmt='BTC')
class P21(Probe):
    def __init__(s):
        super().__init__(); s.fc1 = nn.Linear(C, 2 * C); s.act = nn.GELU(); s.fc2 = nn.Linear(2 * C, C); s.head = nn.Linear(C, K)
    def fh(s, f): return s.head(s.fc2(s.act(s.fc1(f))).mean(1))


# ------------------------------------------------------------ GeM, mixed pools, LSE
class GeM(nn.Module):
    def __init__(s, p=3.0):
        super().__init__(); s.p = p
    def forward(s, x):
        return x.clamp(min=1e-6).pow(s.p).mean((2, 3)).pow(1.0 / s.p)


@reg
@probe('P22', 'GeM pooling as a module (p = 3) -> Linear', 'aggregator-blocked')
class P22(Probe):
    def __init__(s):
        super().__init__(); s.pool = GeM(); s.head = nn.Linear(C, K)
    def fh(s, f): return s.head(s.pool(f))


@reg
@probe('P23', 'GeM pooling as tensor ops -> Linear, no declared pool', 'aggregator-blocked')
class P23(Probe):
    def __init__(s):
        super().__init__(); s.head = nn.Linear(C, K)
    def fh(s, f): return s.head(f.clamp(min=1e-6).pow(3).mean((2, 3)).pow(1 / 3))


@reg
@probe('P24', "mixed avg+max concat as timm SelectAdaptivePool2d('catavgmax') -> Linear", 'aggregator-blocked')
class P24(Probe):
    def __init__(s):
        super().__init__(); s.global_pool = TL.SelectAdaptivePool2d(pool_type='catavgmax', flatten=True)
        s.head = nn.Linear(2 * C, K)
    def fh(s, f): return s.head(s.global_pool(f))


@reg
@probe('P25', 'mixed avg+max concat as tensor ops -> Linear, no declared pool', 'aggregator-blocked')
class P25(Probe):
    def __init__(s):
        super().__init__(); s.head = nn.Linear(2 * C, K)
    def fh(s, f): return s.head(torch.cat([f.mean((2, 3)), f.amax((2, 3))], 1))


@reg
@probe('P26', 'per-cell class scores (Conv1x1) pooled by log-sum-exp over cells (tensor op), T = 1', 'aggregator-blocked')
class P26(Probe):
    def __init__(s):
        super().__init__(); s.score = nn.Conv2d(C, K, 1)
    def fh(s, f): return torch.logsumexp(s.score(f).flatten(2), dim=2)


# ------------------------------------------------------------ cell mixing before the aggregator
@reg
@probe('P27', 'linear cell mixing before the mean (Conv3x3, no activation -> AdaptiveAvgPool2d -> Linear)', 'degenerate')
class P27(Probe):
    def __init__(s):
        super().__init__(); s.mix = nn.Conv2d(C, C, 3, padding=1); s.pool = nn.AdaptiveAvgPool2d(1); s.head = nn.Linear(C, K)
    def fh(s, f): return s.head(s.pool(s.mix(f)).flatten(1))


@reg
@probe('P28', 'nonlinear cell mixing before the mean (Conv3x3 -> ReLU -> AdaptiveAvgPool2d -> Linear)',
       'nonlinearity-blocked (before the aggregator)')
class P28(Probe):
    def __init__(s):
        super().__init__(); s.mix = nn.Conv2d(C, C, 3, padding=1); s.act = nn.ReLU(); s.pool = nn.AdaptiveAvgPool2d(1)
        s.head = nn.Linear(C, K)
    def fh(s, f): return s.head(s.pool(s.act(s.mix(f))).flatten(1))


@reg
@probe('P29', 'self-attention over cells before the mean (nn.MultiheadAttention -> AdaptiveAvgPool2d -> Linear)',
       'nonlinearity-blocked (before the aggregator)')
class P29(Probe):
    def __init__(s):
        super().__init__(); s.attn = nn.MultiheadAttention(C, 2, batch_first=True); s.pool = nn.AdaptiveAvgPool2d(1)
        s.head = nn.Linear(C, K)
    def fh(s, f):
        t = f.flatten(2).transpose(1, 2)
        t = s.attn(t, t, t, need_weights=False)[0]
        return s.head(s.pool(t.transpose(1, 2).reshape(f.shape)).flatten(1))


# ------------------------------------------------------------ tails
@reg
@probe('P30', 'MLP tail (AdaptiveAvgPool2d -> Linear -> GELU -> Linear)', 'nonlinearity-blocked')
class P30(Probe):
    def __init__(s):
        super().__init__(); s.pool = nn.AdaptiveAvgPool2d(1); s.fc1 = nn.Linear(C, 2 * C); s.act = nn.GELU()
        s.head = nn.Linear(2 * C, K)
    def fh(s, f): return s.head(s.act(s.fc1(s.pool(f).flatten(1))))


@reg
@probe('P31', 'BatchNorm1d (inference) after the mean -> Linear', 'degenerate')
class P31(Probe):
    def __init__(s):
        super().__init__(); s.pool = nn.AdaptiveAvgPool2d(1); s.bn = nn.BatchNorm1d(C); s.head = nn.Linear(C, K)
        with torch.no_grad():
            s.bn.running_mean.normal_(); s.bn.running_var.uniform_(0.5, 2.0)
    def fh(s, f): return s.head(s.bn(s.pool(f).flatten(1)))


@reg
@probe('P32', 'ReLU in the tail that is never crossed on the reachable set (Linear bias +50 -> ReLU -> Linear)',
       'degenerate on the reachable set (inactive nonlinearity)')
class P32(Probe):
    def __init__(s):
        super().__init__(); s.pool = nn.AdaptiveAvgPool2d(1); s.fc1 = nn.Linear(C, 2 * C); s.act = nn.ReLU()
        s.head = nn.Linear(2 * C, K)
        with torch.no_grad():
            s.fc1.bias.fill_(50.0)
    def fh(s, f): return s.head(s.act(s.fc1(s.pool(f).flatten(1))))


# ------------------------------------------------------------ run
def make_feat(P, n=6, seed=1):
    g = torch.Generator().manual_seed(seed)
    if P.fmt == 'BCHW':
        return torch.randn(n, C, H, H, generator=g)
    T = H * H + len(P.prefix)
    return torch.randn(n, T, C, generator=g)


def run_one(P):
    torch.manual_seed(0)
    m = P().eval()
    if P.gp is not None:
        m.global_pool = P.gp
    f = make_feat(P)
    st = FR.structural(m, f)
    v2b = R2.structural_v2(m, f)['v2b']
    cv = L.Cells(f, fmt=P.fmt, prefix=P.prefix)
    ship, nonorm, lab = L.measure(m, f, cv, P.post_norms)
    want_paper = {'ADDITIVE': 'degenerate', 'VACUOUS': 'class-token', 'NORM-ONLY': 'normalisation-blocked',
                  'HARD-BLOCKED': 'blocked (hard)'}[lab]
    agree = L.GT_OF[st['verdict']] == lab
    # does a blocked verdict name the actual cause?
    if lab == 'HARD-BLOCKED' and agree:
        cause_agg = P.design.startswith('aggregator')
        reason_ok = (st['verdict'] == 'AGG-BLOCKED') == cause_agg
    else:
        reason_ok = agree
    return dict(id=P.pname, probe=P.desc, design=P.design, frozen=st['verdict'],
                frozen_paper=L.PAPER[st['verdict']], agg=st['agg'], agg_name=st['agg_name'],
                blocker=st['blocker'], post=[list(p) for p in st['post']],
                eps1=ship['eps1'], eps2=ship['eps2'], eps3=ship['eps3'],
                spatial_mass=ship['spatial_mass'], n_active=ship['n_active'], T=ship['T'],
                nonorm=(None if nonorm is None else dict(eps1=nonorm['eps1'], eps2=nonorm['eps2'],
                                                         eps3=nonorm['eps3'])),
                measured=lab, measured_paper=want_paper, agree=bool(agree), reason_ok=bool(reason_ok),
                v2b=v2b, agree_v2b=bool(L.GT_OF[v2b] == lab))


def fmt_e(x):
    return '%.1e' % x


def tex_e(x):
    if x == 0:
        return '$0$'
    m, e = ('%.1e' % x).split('e')
    return r'$%s\times10^{%d}$' % (m, int(e))


if __name__ == '__main__':
    OUT = os.path.join(HERE, '..', 'out'); os.makedirs(OUT, exist_ok=True)
    rows = []
    for P in PROBES:
        r = run_one(P)
        rows.append(r)
        print('%s %-12s frozen %-14s measured %-12s eps1 %.1e eps2 %.1e eps3 %.1e %s%s v2b %s  | %s'
              % (r['id'], r['agg'], r['frozen'], r['measured'], r['eps1'], r['eps2'], r['eps3'],
                 'agree' if r['agree'] else 'DISAGREE', '' if r['reason_ok'] else ' (wrong reason)',
                 r['v2b'], r['probe']), flush=True)
    n = len(rows); na = sum(r['agree'] for r in rows); nr = sum(r['reason_ok'] for r in rows)
    fp = [r['id'] for r in rows if r['frozen'] == 'IN-SCOPE' and r['measured'] != 'ADDITIVE']
    fn = [r['id'] for r in rows if r['frozen'] != 'IN-SCOPE' and r['measured'] == 'ADDITIVE']
    summ = dict(n=n, agree=na, agree_with_reason=nr, agree_v2b=sum(r['agree_v2b'] for r in rows),
                v2b_disagree=[r['id'] for r in rows if not r['agree_v2b']], false_degenerate=fp, missed_degenerate=fn,
                disagree=[r['id'] for r in rows if not r['agree']],
                right_label_wrong_reason=[r['id'] for r in rows if r['agree'] and not r['reason_ok']])
    print(json.dumps(summ))
    json.dump(dict(summary=summ, rows=rows), open(os.path.join(OUT, 'a3_probes.json'), 'w'), indent=1)

    md = ['| ID | Constructed head | Frozen verdict (code) | eps1 | eps2 | eps3 | Measured | Agree | Operator-level variant (v2b) |',
          '|---|---|---|---|---|---|---|---|---|']
    tex = [r'\begin{tabular}{lp{0.40\textwidth}lllll}', r'\toprule',
           r'ID & Head (on a $16\times7\times7$ map) & Check & $\varepsilon_1$ & $\varepsilon_2$ & Measured & Agree\\',
           r'\midrule']
    for r in rows:
        ag = 'yes' if r['agree'] and r['reason_ok'] else ('yes (wrong cause)' if r['agree'] else '**no**')
        md.append('| %s | %s | %s (`%s`) | %s | %s | %s | %s | %s | %s |' % (
            r['id'], r['probe'], r['frozen_paper'], r['frozen'], fmt_e(r['eps1']), fmt_e(r['eps2']),
            fmt_e(r['eps3']), r['measured_paper'], ag,
            '%s (%s)' % (L.PAPER[r['v2b']], 'agrees' if r['agree_v2b'] else 'disagrees')))
        tag = 'yes' if r['agree'] and r['reason_ok'] else ('yes$^{\\dagger}$' if r['agree'] else r'\textbf{no}')
        e1, e2 = tex_e(r['eps1']), tex_e(r['eps2'])
        desc = r['probe'].replace('_', r'\_').replace('->', r'$\rightarrow$').replace("'", "'")
        tex.append('%s & %s & %s & %s & %s & %s & %s\\\\' % (
            r['id'], desc, r['frozen_paper'], e1, e2, r['measured_paper'], tag))
    tex += [r'\bottomrule', r'\end{tabular}',
            r'% \dagger: blocked verdict agrees with measurement but names the wrong operation.']
    md.append('')
    md.append('Agreement (paper scoring): %d/%d; with the blocking operation also correct: %d/%d. '
              'False "degenerate": %s. Missed degenerate: %s. Operator-level variant (v2b, not the frozen '
              'check): %d/%d.' % (na, n, nr, n, ', '.join(fp) or 'none', ', '.join(fn) or 'none',
                                   summ['agree_v2b'], n))
    open(os.path.join(OUT, 'a3_probes.md'), 'w', encoding='utf-8').write('\n'.join(md) + '\n')
    open(os.path.join(OUT, 'a3_probes.tex'), 'w', encoding='utf-8').write('\n'.join(tex) + '\n')
