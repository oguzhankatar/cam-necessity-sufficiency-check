# K1 protocol: a conservative (sound) structural check, fixed before any of its verdicts is computed

Date written: 2026-11-10 (before any K1 code is run). Author-designated; not externally registered.

## Question
Can the structural check be made *sound* for the verdict that matters to a user, "the cell-wise necessity and
sufficiency maps cannot differ" (degenerate), so that it never returns degenerate when the float64 certificate
says the readout is not additive, and how much coverage does that cost?

## The conservative check (wrapper; the hash-frozen rule file is not edited)
`check_model_sound(model, x)` returns one of: degenerate, blocked (normalisation / nonlinearity / aggregator),
class-token, **unknown**. It calls the existing module-level check and the existing operator-level trace and returns
**degenerate only if all of the following hold**; otherwise it returns the module-level blocked/class-token verdict
when that verdict is not degenerate, and **unknown** when the module-level verdict is degenerate but a condition
below fails:
1. A hooked module performs the spatial reduction, or the operator-level trace finds exactly one reduction over the
   spatial axes and it is a mean (no reliance on a declared pool type alone).
2. The aggregator module's class is in a whitelist of additive pools (adaptive/global average pooling and their
   timm wrappers), or the operator-level reduction is a plain mean.
3. Every module that the head executes before the aggregator is in a whitelist of per-cell module classes
   (pointwise convolutions, batch normalisation in eval mode, activations, dropout, identity, per-token norms and
   per-token linear layers applied over the channel axis), so no cell-mixing operation precedes the pool.
4. The reduced axes are the spatial axes of the feature map (checked from tensor shapes in the operator trace),
   so a reduction over a stack of class tokens is not mistaken for spatial pooling.
5. The module-level and operator-level verdicts agree.
Whitelists are written before evaluation, from the module classes of the 66 development architectures only.

## Evaluation (all fixed now)
- Sets: the 148 development + held-out architectures (measured labels exist), the 32 constructed heads (float64
  certificate labels exist), and the census of 895 timm 1.0.27 ImageNet-1k entries (random weights; audited labels
  exist for the 30 unmeasured traces).
- Primary outcome: the number of **false degenerate** verdicts (degenerate returned while the certificate says not
  additive) on the 32 constructed heads and on the 147 measured architectures. Target: zero.
- Secondary: coverage (share of entries with a decided verdict, i.e. not unknown) on each set and on the census;
  binary accuracy for the question "can the two maps differ?" (degenerate or vacuous = cannot; blocked = can) for the
  original and the conservative check; the cost of each check in seconds per model; the cost of the direct float64
  certificate on the same models for comparison.
- Reported in full whatever the result.
