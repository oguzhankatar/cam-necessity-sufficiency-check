"""NS-CAM: Necessity-Sufficiency Class Activation Mapping.

Core idea: a saliency map conflates two causally distinct quantities.
  - NECESSITY (PN): does the prediction break when this evidence is removed?
  - SUFFICIENCY (PS): does this evidence alone preserve the prediction?
Deletion AUC probes the former, insertion AUC the latter. NS-CAM estimates
them separately.

Perturbation unit = rank-r SVD subspace of the target activation tensor
(not channels: channels are correlated and polysemantic; not clusters:
cluster count is hand-tuned in the literature). Deterministic by construction.

Intervention fills removed content with a class-conditional mean activation
from a feature bank, so the probe stays on-manifold and, being in feature
space, leaks no mask-shape information (cf. ROAD, Rong et al. ICML 2022).
"""
import torch
import torch.nn.functional as F


class NSCAM:
    def __init__(self, backbone, rank=16, energy=None):
        """backbone: an object exposing .features(x) -> A (B,C,H,W)
                     and .head(A) -> logits (B,K)."""
        self.bb = backbone
        self.rank = rank
        self.energy = energy          # if set, rank chosen by explained variance
        self.bank = {}                # class id -> mean activation (C,H,W)

    # ---- feature bank -------------------------------------------------
    @torch.no_grad()
    def fit_bank(self, loader, device, max_per_class=32):
        sums, counts = {}, {}
        for x, _ in loader:
            x = x.to(device)
            A = self.bb.features(x)
            pred = self.bb.head(A).argmax(1)
            for i, c in enumerate(pred.tolist()):
                if counts.get(c, 0) >= max_per_class:
                    continue
                sums[c] = sums.get(c, 0) + A[i]
                counts[c] = counts.get(c, 0) + 1
        self.bank = {c: sums[c] / counts[c] for c in sums}
        self.global_mu = torch.stack(list(self.bank.values())).mean(0)
        return self

    def _mu(self, c):
        return self.bank.get(c, self.global_mu)

    # ---- core ---------------------------------------------------------
    @torch.no_grad()
    def __call__(self, x, target=None):
        """x: (1,3,H,W). Returns dict with 'N' and 'S' maps (H,W) and scores."""
        A = self.bb.features(x)                    # (1,C,h,w)
        _, C, h, w = A.shape
        logits = self.bb.head(A)
        c = int(logits.argmax(1)) if target is None else int(target)

        M = A[0].reshape(C, h * w)                 # (C, hw)
        U, S, Vh = torch.linalg.svd(M, full_matrices=False)

        r = self.rank
        if self.energy is not None:               # rank by explained variance
            ev = torch.cumsum(S ** 2, 0) / (S ** 2).sum()
            r = int((ev < self.energy).sum().item()) + 1
        r = min(r, S.numel())

        mu = self._mu(c)                           # (C,h,w)
        Mmu = mu.reshape(C, h * w)

        # component i content, and the same subspace applied to the bank mean
        comps, comps_mu = [], []
        for i in range(r):
            u = U[:, i:i + 1]                      # (C,1)
            comps.append(S[i] * u @ Vh[i:i + 1, :])          # (C,hw)
            comps_mu.append(u @ (u.T @ Mmu))                 # proj of mu (C,hw)
        comps = torch.stack(comps)                 # (r,C,hw)
        comps_mu = torch.stack(comps_mu)

        # NECESSITY: drop component i, fill its subspace with the bank mean
        A_neg = (M.unsqueeze(0) - comps + comps_mu).reshape(r, C, h, w)
        # SUFFICIENCY: keep only component i, fill the rest with the bank mean
        A_pos = (comps + (Mmu.unsqueeze(0) - comps_mu)).reshape(r, C, h, w)

        z_full = logits[0, c]
        z_mu = self.bb.head(mu.unsqueeze(0))[0, c]
        z_neg = self.bb.head(A_neg)[:, c]          # r partial forwards
        z_pos = self.bb.head(A_pos)[:, c]          # r partial forwards

        PN = z_full - z_neg                        # logit lost by removal
        PS = z_pos - z_mu                          # logit held by the part alone

        # spatial energy of each component: ||A_i[:,x,y]||_2 = S_i * |V_i(x,y)|
        E = (S[:r, None] * Vh[:r].abs()).reshape(r, h, w)

        N = (F.relu(PN)[:, None, None] * E).sum(0)
        Sm = (F.relu(PS)[:, None, None] * E).sum(0)
        return {"N": self._norm(N), "S": self._norm(Sm), "PN": PN, "PS": PS,
                "rank": r, "class": c,
                "n_partial_forwards": 2 * r}

    @staticmethod
    def _norm(m):
        m = m - m.min()
        return m / (m.max() + 1e-8)
