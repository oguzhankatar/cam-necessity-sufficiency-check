"""Interchangeable spatial aggregators on a FROZEN ResNet50 layer4.

All heads map A:(B,C,T) -> logits:(B,K).  They differ ONLY in where the
nonlinearity sits relative to the spatial sum:

  POST-pool nonlinearity   gap_fc, gap_mlp          (sum first, then f)
  PRE-pool  nonlinearity   ln_gap, tok_mlp          (f per token, then sum)
  NONLINEAR aggregator     lse{T}, maxpool, attn, bilin

Working theory under test: additivity over spatial positions is caused by the
LINEAR aggregation, so anything in the first group stays exactly additive and
anything in the last two groups does not.
"""
import torch, torch.nn as nn, torch.nn.functional as F

C = 2048   # default; build(name, c=...) overrides


class GapFC(nn.Module):
    def __init__(s, k=10, c=C):
        super().__init__(); s.fc = nn.Linear(c, k)

    def forward(s, A): return s.fc(A.mean(-1))


class GapMLP(nn.Module):
    """POST-pool nonlinearity: g(mean_t a_t). Still additive in spatial parts
    of a *linear* readout? No -- but additive in the GAP summary."""
    def __init__(s, k=10, h=512, c=C):
        super().__init__()
        s.net = nn.Sequential(nn.Linear(c, h), nn.GELU(), nn.Linear(h, h),
                              nn.GELU(), nn.Linear(h, k))

    def forward(s, A): return s.net(A.mean(-1))


class LNGap(nn.Module):
    """PRE-pool per-token LayerNorm, then mean (Swin / ViT-with-final-norm)."""
    def __init__(s, k=10, c=C):
        super().__init__(); s.ln = nn.LayerNorm(c); s.fc = nn.Linear(c, k)

    def forward(s, A): return s.fc(s.ln(A.transpose(1, 2)).mean(1))


class TokMLP(nn.Module):
    """PRE-pool per-token MLP, then mean. Same parameter budget as GapMLP."""
    def __init__(s, k=10, h=512, c=C):
        super().__init__()
        s.net = nn.Sequential(nn.Linear(c, h), nn.GELU(), nn.Linear(h, h), nn.GELU())
        s.fc = nn.Linear(h, k)

    def forward(s, A): return s.fc(s.net(A.transpose(1, 2)).mean(1))


class LSE(nn.Module):
    """Per-token logits, then log-sum-exp pooling.
       T -> 0  : exactly GAP+FC.      T -> inf : max over tokens."""
    def __init__(s, k=10, T=1.0, c=C):
        super().__init__(); s.fc = nn.Linear(c, k); s.T = T

    def forward(s, A):
        z = s.fc(A.transpose(1, 2))                       # (B,T,K)
        if s.T == 0: return z.mean(1)
        n = z.shape[1]
        return (torch.logsumexp(s.T * z, 1) - torch.log(torch.tensor(
            float(n), device=z.device))) / s.T


class MaxPool(nn.Module):
    def __init__(s, k=10, c=C):
        super().__init__(); s.fc = nn.Linear(c, k)

    def forward(s, A): return s.fc(A.transpose(1, 2)).max(1).values


class Attn(nn.Module):
    """Gated-attention pooling (Ilse et al. 2018): softmax weights that depend
    on the content -> a genuinely nonlinear aggregator."""
    def __init__(s, k=10, h=256, c=C):
        super().__init__()
        s.V = nn.Linear(c, h); s.U = nn.Linear(c, h); s.w = nn.Linear(h, 1)
        s.fc = nn.Linear(c, k)

    def forward(s, A):
        X = A.transpose(1, 2)
        a = s.w(torch.tanh(s.V(X)) * torch.sigmoid(s.U(X)))      # (B,T,1)
        al = torch.softmax(a, 1)
        return s.fc((al * X).sum(1))


class Bilin(nn.Module):
    """Low-rank second-order (bilinear) pooling: mean_t (P1 a)(P2 a)^T."""
    def __init__(s, k=10, d=48, c=C):
        super().__init__()
        s.P1 = nn.Linear(c, d, bias=False); s.P2 = nn.Linear(c, d, bias=False)
        s.fc = nn.Linear(d * d, k); s.d = d

    def forward(s, A):
        X = A.transpose(1, 2)
        u, v = s.P1(X), s.P2(X)
        p = torch.einsum('bti,btj->bij', u, v) / X.shape[1]
        p = torch.sign(p) * torch.sqrt(p.abs() + 1e-6)           # signed sqrt
        p = p.reshape(p.shape[0], -1)
        return s.fc(F.normalize(p, dim=1))


def build(name, k=10, c=C):
    if name == 'gap_fc':   return GapFC(k, c)
    if name == 'gap_mlp':  return GapMLP(k, c=c)
    if name == 'ln_gap':   return LNGap(k, c)
    if name == 'tok_mlp':  return TokMLP(k, c=c)
    if name == 'maxpool':  return MaxPool(k, c)
    if name == 'attn':     return Attn(k, c=c)
    if name == 'bilin':    return Bilin(k, c=c)
    if name.startswith('lse'): return LSE(k, T=float(name[3:]), c=c)
    raise ValueError(name)


GROUP = {'gap_fc': 'linear-agg', 'gap_mlp': 'post-pool-nl', 'ln_gap': 'pre-pool-nl',
         'tok_mlp': 'pre-pool-nl', 'maxpool': 'nonlin-agg', 'attn': 'nonlin-agg',
         'bilin': 'nonlin-agg'}
for _t in ('0.25', '0.5', '1.0', '2.0', '4.0', '8.0'):
    GROUP['lse' + _t] = 'nonlin-agg'
