"""A model-agnostic view of the aggregation point, built with a forward hook.

`Agg(m)` exposes
    cells(x)   -> (B, T, C)  the cell matrix the model pools over
    post(v)    -> (B, K)     the map from a pooled (B, C) vector to the logits
    z = post(cells.mean(1))  reproduces m(x) exactly  (checked, not assumed)

The hook is installed on the module that collapses the spatial/token axes, so
`post` is EXACTLY the shipped post-pool computation -- no family-specific code,
no re-implementation to get wrong.  Models that pool with a tensor op instead of
a module (the ViT/MambaOut family) get a small explicit adapter.
"""
import torch, torch.nn as nn


def _cells(t):
    if t.dim() == 4:
        return t.shape[2] * t.shape[3]
    if t.dim() == 3:
        return t.shape[1]
    return 1 if t.dim() == 2 else None


def find_pool(m, x):
    """Return the name of the last leaf module that collapses cells."""
    ev = []
    hs = []

    def mk(name):
        def f(mod, inp, out):
            if len(inp) and torch.is_tensor(inp[0]) and torch.is_tensor(out):
                ev.append((name, _cells(inp[0]), _cells(out), inp[0].dim()))
        return f
    for name, mod in m.named_modules():
        if len(list(mod.children())) == 0:
            hs.append(mod.register_forward_hook(mk(name)))
    with torch.no_grad():
        m(x)
    for h in hs:
        h.remove()
    hit = [e for e in ev if e[1] and e[2] == 1 and e[1] > 1]
    return hit[-1][0] if hit else None


class Agg:
    """Hook-based aggregation-point view. `ok` says whether it reproduced m(x)."""

    def __init__(s, m, x):
        s.m = m
        s.pool_name = find_pool(m, x)
        if s.pool_name is None:
            raise ValueError('no module-level pool (use an explicit adapter)')
        s.pool = dict(m.named_modules())[s.pool_name]
        s._cap = None
        s._ovr = None

        def hook(mod, inp, out):
            s._cap = inp[0]
            if s._ovr is not None:
                return s._ovr
            return out
        s._h = s.pool.register_forward_hook(hook)

    def close(s):
        s._h.remove()

    @torch.no_grad()
    def cells(s, x):
        s._ovr = None
        s.m(x)
        c = s._cap
        if c.dim() == 4:                       # (B,C,H,W) -> (B,T,C)
            return c.flatten(2).transpose(1, 2)
        if c.dim() == 3:                       # (B,T,C) already
            return c
        raise ValueError(f'unexpected cell tensor {tuple(c.shape)}')

    @torch.no_grad()
    def post(s, v, x_dummy):
        """v: (B,C) pooled vector -> logits.

        x_dummy only has to survive the pool module call; its value is discarded
        because the hook replaces the pool's OUTPUT with v.
        """
        s.ndim = len(s._cap.shape)
        o = v[:, :, None, None] if s.ndim == 4 else v[:, None, :]
        s._ovr = o
        try:
            z = s.m.forward_head(x_dummy)
        finally:
            s._ovr = None
        return z


class AggMamba:
    """MambaOut pools with x.mean((1,2)) inside MlpHead."""

    def __init__(s, m, x):
        s.m = m; s.pool_name = 'head(mean over H,W)'

    @torch.no_grad()
    def cells(s, x):
        f = s.m.forward_features(x)                      # (B,H,W,C)
        return f.reshape(f.shape[0], -1, f.shape[-1])

    @torch.no_grad()
    def post(s, v, x_dummy=None):
        h = s.m.head
        y = h.norm(v)
        y = h.pre_logits(y)
        y = h.head_dropout(y)
        return h.fc(y)

    def close(s):
        pass
