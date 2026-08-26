"""Frozen backbone loading, per DESIGN.md's validation-backbone role (0, 1) and
cross-architecture invariance analysis (7). Every loader here returns the same
(model, preprocess, categories) shape regardless of architecture (CNN or ViT),
which is what lets features.py/combiner.py/router.py stay backbone-agnostic:
they only ever see the logit tensor, never the model itself.

This module never trains or modifies a backbone - it exists purely to produce
the logit vectors the rest of the package consumes (DESIGN.md 4, 14.1).
"""

from __future__ import annotations

import torch
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    ResNet50_Weights,
    ViT_B_16_Weights,
    WeightsEnum,
    convnext_tiny,
    resnet50,
    vit_b_16,
)


def _load_frozen(model_fn, weights: WeightsEnum):
    """Shared loader: build the model, freeze it, and pull its matching preprocess
    + category list from `weights` - no architecture-specific logic here, so
    adding a new backbone is a one-line entry in the registries below, not a
    new code path.
    """
    model = model_fn(weights=weights)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    preprocess = weights.transforms()
    categories = list(weights.meta["categories"])
    return model, preprocess, categories


def load_frozen_resnet50(weights: ResNet50_Weights = ResNet50_Weights.IMAGENET1K_V2):
    """Load pretrained ResNet-50 in eval mode with gradients disabled (DESIGN.md 3.2: inference-only).

    Returns (model, preprocess, categories) - the frozen model, its matching
    input transform, and the ordered list of 1000 ImageNet class names.
    """
    return _load_frozen(resnet50, weights)


def load_frozen_vit_b16(weights: ViT_B_16_Weights = ViT_B_16_Weights.IMAGENET1K_V1):
    """Load pretrained ViT-B/16 in eval mode with gradients disabled.

    Same 1000-class ImageNet head as ResNet-50 - DESIGN.md 7/14.2's direct
    transfer case for phi(z)/combiner, since output shape/semantics match.
    """
    return _load_frozen(vit_b_16, weights)


def load_frozen_vit_b16_with_dropout(
    weights: ViT_B_16_Weights = ViT_B_16_Weights.IMAGENET1K_V1,
    dropout: float = 0.1,
    attention_dropout: float = 0.1,
):
    """Load the SAME frozen, pretrained ViT-B/16 weights as `load_frozen_vit_b16`,
    but constructed with `dropout`/`attention_dropout` > 0 - STUDY_PLAN.md
    3.6 item 5's prerequisite for MC-dropout (`epistemic.py`).

    This does NOT retrain or modify anything: dropout/attention-dropout
    layers have zero learnable parameters, so building the model with these
    enabled and then loading the identical `IMAGENET1K_V1` checkpoint is
    valid and produces byte-identical weights to `load_frozen_vit_b16` -
    verified directly (`state_dict()` keys and tensors both compared equal
    between the two constructions before this function was written, not
    assumed from torchvision's docs). At `dropout=attention_dropout=0.0`
    (torchvision's own default), this is exactly `load_frozen_vit_b16`.

    Still returned in `.eval()` with gradients disabled, exactly like every
    other loader in this module - the dropout/attention modules stay inert
    until `epistemic.mc_dropout_predict` explicitly, temporarily flips them
    back to `.train()` for a bounded number of extra forward passes. Loading
    this way alone changes nothing about a normal forward pass.

    ViT-B/16-only: ResNet-50 has zero `nn.Dropout` modules in torchvision's
    standard weights, and ConvNeXt-Tiny's regularization is `StochasticDepth`,
    not `nn.Dropout` - a structurally different mechanism (confirmed by
    reading torchvision's own `stochastic_depth` source: it is a no-op
    whenever `training=False`, i.e. it is already fully inert during a
    normal `.eval()` forward pass for both backbones, not a hidden
    stochasticity this project was silently already exposed to). Splicing
    equivalent dropout into either would mean adding new, never-trained
    layers, which is out of scope for this item.
    """
    return _load_frozen(
        lambda weights: vit_b_16(weights=weights, dropout=dropout, attention_dropout=attention_dropout), weights
    )


def load_frozen_convnext_tiny(weights: ConvNeXt_Tiny_Weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1):
    """Load pretrained ConvNeXt-Tiny in eval mode with gradients disabled.

    A third, structurally distinct backbone: a modern convolutional design
    (depthwise convs, LayerNorm, GELU - closer in several respects to a ViT's
    building blocks than to ResNet-50's) with a different training recipe from
    both ResNet-50 and ViT-B/16 above. Same 1000-class ImageNet head, so
    phi(z)/combiner transfer directly per DESIGN.md 7's prediction - a third
    data point for the cross-architecture invariance claim, distinguishing
    "generalizes across CNN/ViT" from "generalizes across ResNet/ViT specifically."
    """
    return _load_frozen(convnext_tiny, weights)


@torch.inference_mode()
def logits_for_images(model, preprocess, images) -> torch.Tensor:
    """Run a frozen backbone forward pass on a batch of PIL images, returning raw logits.

    `images` is a list of PIL.Image objects; returns a tensor of shape (N, C).
    This is the single forward pass DESIGN.md 3.2 requires the whole confidence
    layer to live within - no additional passes, no perturbation, no ensembling.
    Backbone-agnostic: works identically for ResNet-50 or any ViT above, since
    both just implement `model(batch) -> logits`.

    Unchanged by the optional penultimate-feature path below
    (`logits_and_features_for_images`) - every existing caller of this
    function keeps working identically; that function is a separate, additive
    entry point, not a modification of this one.
    """
    batch = torch.stack([preprocess(img) for img in images], dim=0)
    return model(batch)


# Dotted module path to each architecture's final linear classifier layer -
# whatever feeds this layer IS the penultimate feature representation, by
# definition, regardless of architecture. Used only by
# logits_and_features_for_images below; logits_for_images never touches this.
PENULTIMATE_CLASSIFIER_PATH = {
    "resnet50": "fc",
    "vit_b16": "heads.head",
    "convnext_tiny": "classifier.2",
}


def _resolve_module(model, dotted_path: str):
    module = model
    for part in dotted_path.split("."):
        module = module[int(part)] if part.isdigit() else getattr(module, part)
    return module


@torch.inference_mode()
def logits_and_features_for_images(model, preprocess, images, architecture: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Like `logits_for_images`, but also captures penultimate-layer features
    (everything up to, but not including, the final linear classifier) via a
    forward pre-hook on that layer - STUDY_PLAN.md 3.6 item 2's optional
    feature-space signal (Mahalanobis-lite, `mahalanobis.py`), additive
    alongside the existing logit-only path.

    Deliberately does NOT change `logits_for_images` or any of its callers -
    this is a new, separate entry point only used by code that explicitly
    opts into the feature-space signal (`scripts/collect_features_mahalanobis.py`,
    `mahalanobis.py`'s callers). `architecture` must be a key of
    `PENULTIMATE_CLASSIFIER_PATH` (one of "resnet50", "vit_b16",
    "convnext_tiny") - it selects which module is that backbone's final
    classifier, since each architecture names its head differently (`fc` vs.
    `heads.head` vs. `classifier[2]`), a detail `logits_for_images` never had
    to know because it only ever needed the model to be callable end to end.

    Returns (logits, features): `features` has shape (N, D) - the exact
    tensor the final linear layer received as input, D=2048 for ResNet-50,
    D=768 for ViT-B/16 and ConvNeXt-Tiny. The forward pre-hook is registered
    and removed within this single call (`try`/`finally`), so it never leaks
    onto the model for any later, unrelated call to `logits_for_images` or
    this function again.
    """
    if architecture not in PENULTIMATE_CLASSIFIER_PATH:
        raise ValueError(
            f"unknown architecture {architecture!r}; choose from {sorted(PENULTIMATE_CLASSIFIER_PATH)}"
        )
    classifier = _resolve_module(model, PENULTIMATE_CLASSIFIER_PATH[architecture])

    captured: dict[str, torch.Tensor] = {}

    def _capture_input(module, inputs):
        captured["features"] = inputs[0]

    handle = classifier.register_forward_pre_hook(_capture_input)
    try:
        batch = torch.stack([preprocess(img) for img in images], dim=0)
        logits = model(batch)
    finally:
        handle.remove()
    return logits, captured["features"]
