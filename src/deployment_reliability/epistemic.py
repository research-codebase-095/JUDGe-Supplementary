"""MC-dropout epistemic uncertainty estimate (STUDY_PLAN.md 3.6 item 5) -
ViT-B/16 ONLY, an explicit, disclosed opt-in with a real extra cost (T
additional forward passes), never the default path anywhere in this project.

STUDY_PLAN.md 3.6's own comparison table (Uncertainty quantification)
named this as an outright capability gap: "Cannot be done at all here - an
explicit non-goal (§2, Objective 6)." This module closes that gap for the
one backbone where it can be closed without adding any new, never-trained
parameters: `backbone.py`'s `load_frozen_vit_b16_with_dropout` loads the
IDENTICAL frozen `IMAGENET1K_V1` checkpoint with dropout/attention-dropout
modules present (they have zero learnable parameters, so this changes
nothing about the loaded weights - verified directly, see that function's
docstring), and `mc_dropout_predict` below runs T stochastic passes with
ONLY those modules in `.train()` mode, everything else (BatchNorm-free ViT
has none anyway; LayerNorm, the patch embedding conv, attention projections)
staying exactly as `.eval()` left it.

Scope, stated precisely and only once here (not re-litigated per call site):
- ResNet-50: zero `nn.Dropout` modules in torchvision's standard weights -
  not available without splicing in new, untrained layers.
- ConvNeXt-Tiny: regularized by `StochasticDepth`, not `nn.Dropout` - and,
  checked directly against torchvision's own `stochastic_depth` source
  (`torchvision/ops/stochastic_depth.py`), that op is an unconditional no-op
  whenever `training=False`. So ConvNeXt-Tiny's stochastic depth is fully
  inert during this project's existing `.eval()`-mode forward passes - not
  a hidden source of stochasticity already present, and not a mechanism
  this module repurposes for MC-dropout-style sampling (a structurally
  different technique - identical-architecture Monte Carlo path sampling,
  not weight/activation dropout - so it is not treated as equivalent here).
- ViT-B/16: has both plain `nn.Dropout` (post-attention, post-MLP) and
  `attention_dropout`, the latter applied *functionally* inside
  `nn.MultiheadAttention.forward`, gated on THAT module's own `.training`
  flag (verified against `torch.nn.functional.multi_head_attention_forward`
  and `nn.MultiheadAttention.forward`'s source - it is not a nested
  `nn.Dropout` submodule, so a naive "only toggle `nn.Dropout` instances"
  implementation would silently leave `attention_dropout` inert). This
  module's `_dropout_submodule_types` toggles both module types together.
"""

from __future__ import annotations

import torch

# Both module types gate their internal dropout on their OWN .training flag,
# not a shared global switch - nn.Dropout directly, nn.MultiheadAttention
# functionally inside its forward (see module docstring above). Toggling
# only nn.Dropout would silently leave attention_dropout inert.
_DROPOUT_SUBMODULE_TYPES = (torch.nn.Dropout, torch.nn.MultiheadAttention)


def _set_dropout_submodules_train(model: torch.nn.Module, mode: bool) -> list[torch.nn.Module]:
    """Set every dropout-bearing submodule's own train()/eval() flag,
    independent of the rest of the model. Returns the affected modules so a
    caller can restore them afterward without calling model.train() (which
    would also re-enable BatchNorm running-stats updates etc. - out of
    scope, and not needed since ViT-B/16 has no BatchNorm layers anyway)."""
    affected = []
    for module in model.modules():
        if isinstance(module, _DROPOUT_SUBMODULE_TYPES):
            module.train(mode)
            affected.append(module)
    return affected


def mc_dropout_predict(
    model: torch.nn.Module, preprocess, images, passes: int = 10
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run `passes` stochastic forward passes over `images` with only the
    dropout/attention-dropout submodules in `.train()` mode - everything
    else stays exactly however the caller's already-`.eval()`'d `model` left
    it (raises if `model.training` is True, since this function's whole
    point is a *controlled*, partial deviation from eval mode, not a full
    return to training behavior).

    An explicit, disclosed extra cost: `passes` full forward passes over the
    same batch, not one - never call this on the default/primary inference
    path (DESIGN.md 3.2's single-forward-pass constraint is about the
    REQUIRED path; this is an opt-in diagnostic addition on top of it, the
    same "extra cost, explicitly named" status STUDY_PLAN.md 3.6's own
    comparison table gives MC-dropout/Deep Ensembles generally).

    Returns (mean_logits, predictive_variance):
    - `mean_logits`: (N, C) - the average logit vector across the `passes`
      stochastic forward passes. NOT the same as a single deterministic
      `.eval()` forward pass's logits (dropout perturbs each pass), though
      it should be close for a well-trained model with modest dropout rates.
    - `predictive_variance`: (N,) - total variance of the softmax
      probability vector across passes, summed over classes
      (`sum_c Var[p_c]`), one non-negative scalar per input - the epistemic-
      uncertainty field this project did not previously have any way to
      compute (STUDY_PLAN.md 3.6's comparison table's named gap).

      A real, checked, counter-intuitive property of this specific
      quantity, not assumed away: it is NOT a monotonically increasing
      function of the dropout rate. Verified directly (`tests/test_epistemic.py`):
      raw LOGIT-space variance across passes does increase with dropout
      rate, as expected, but this PROBABILITY-space variance can decrease
      at higher dropout rates - a softmax-saturation effect. Heavy dropout
      drives the mean predictive distribution toward near-uniform faster
      than it grows logit-space spread, and a near-uniform distribution
      sits in a low-curvature region of the probability simplex where
      further perturbation produces LESS probability variance than milder
      dropout perturbing a still-peaked distribution. Read this quantity as
      "how much the predictive DISTRIBUTION moved across passes," not as a
      simple, monotonic dial on how much stochasticity was injected.
    """
    if model.training:
        raise RuntimeError(
            "mc_dropout_predict expects `model` in .eval() mode already (as every backbone.py "
            "loader returns it) - it temporarily flips ONLY dropout-bearing submodules back to "
            ".train(), not the whole model."
        )
    if passes < 2:
        raise ValueError(f"passes must be >= 2 to compute a meaningful variance, got {passes}")

    # Fail loudly rather than silently returning a zero-variance estimate that
    # looks like a real (if boring) MC-dropout result. Checked BEFORE
    # flipping anything to .train() mode, so an invalid model is rejected
    # with zero side effects. Two distinct ways this can go wrong, both
    # checked directly rather than assumed: (1) no dropout-bearing
    # submodules exist at all (a ResNet-50/ConvNeXt-Tiny model - see this
    # module's docstring for why plain nn.Dropout wouldn't even be present);
    # (2) dropout-bearing submodules exist but are configured at p=0.0 (e.g.
    # `backbone.load_frozen_vit_b16()`, whose nn.Dropout/nn.MultiheadAttention
    # instances are present in the architecture but numerically inert) -
    # .train() mode alone does NOT make either of those stochastic, so
    # checking only "were any modules found" would silently pass this second
    # case through as if it were a real MC-dropout result.
    has_active_dropout = any(
        (isinstance(m, torch.nn.Dropout) and m.p > 0.0)
        or (isinstance(m, torch.nn.MultiheadAttention) and m.dropout > 0.0)
        for m in model.modules()
    )
    if not has_active_dropout:
        raise RuntimeError(
            "no ACTIVE (p > 0) nn.Dropout/nn.MultiheadAttention submodules found on this model - "
            "mc_dropout_predict needs a model loaded with dropout enabled "
            "(backbone.load_frozen_vit_b16_with_dropout(dropout=..., attention_dropout=...)), "
            "not a plain frozen backbone or a ViT loaded with dropout=attention_dropout=0.0"
        )

    affected = _set_dropout_submodules_train(model, True)
    try:
        with torch.inference_mode():
            batch = torch.stack([preprocess(img) for img in images], dim=0)
            all_logits = [model(batch) for _ in range(passes)]
    finally:
        for module in affected:
            module.train(False)

    stacked_logits = torch.stack(all_logits, dim=0)  # (T, N, C)
    stacked_probs = torch.softmax(stacked_logits, dim=-1)
    mean_logits = stacked_logits.mean(dim=0)
    predictive_variance = stacked_probs.var(dim=0, unbiased=False).sum(dim=-1)
    return mean_logits, predictive_variance
