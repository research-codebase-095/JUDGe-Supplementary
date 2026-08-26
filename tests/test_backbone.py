import torch
from PIL import Image

from deployment_reliability.backbone import (
    load_frozen_convnext_tiny,
    load_frozen_resnet50,
    load_frozen_vit_b16,
    logits_for_images,
)


def test_load_frozen_resnet50_is_frozen_and_in_eval_mode():
    model, preprocess, categories = load_frozen_resnet50()
    assert model.training is False
    assert all(not p.requires_grad for p in model.parameters())
    assert len(categories) == 1000
    assert preprocess is not None


def test_logits_for_images_shape_and_no_grad():
    model, preprocess, _ = load_frozen_resnet50()
    image = Image.new("RGB", (224, 224), color=(120, 80, 200))
    logits = logits_for_images(model, preprocess, [image, image])
    assert logits.shape == (2, 1000)
    assert not logits.requires_grad


def test_load_frozen_vit_b16_is_frozen_and_in_eval_mode():
    model, preprocess, categories = load_frozen_vit_b16()
    assert model.training is False
    assert all(not p.requires_grad for p in model.parameters())
    assert len(categories) == 1000
    assert preprocess is not None


def test_vit_b16_logits_same_shape_as_resnet50_confirming_direct_transfer():
    # DESIGN.md 7/14.2: ViT's classification head is claimed to be a direct
    # transfer target for phi(z)/combiner because the output shape/semantics
    # match ResNet-50's. This is the concrete, checkable form of that claim.
    model, preprocess, _ = load_frozen_vit_b16()
    image = Image.new("RGB", (224, 224), color=(120, 80, 200))
    logits = logits_for_images(model, preprocess, [image, image])
    assert logits.shape == (2, 1000)
    assert not logits.requires_grad


def test_load_frozen_convnext_tiny_is_frozen_and_in_eval_mode():
    model, preprocess, categories = load_frozen_convnext_tiny()
    assert model.training is False
    assert all(not p.requires_grad for p in model.parameters())
    assert len(categories) == 1000
    assert preprocess is not None


def test_convnext_tiny_logits_same_shape_confirming_third_architecture_transfer():
    model, preprocess, _ = load_frozen_convnext_tiny()
    image = Image.new("RGB", (224, 224), color=(120, 80, 200))
    logits = logits_for_images(model, preprocess, [image, image])
    assert logits.shape == (2, 1000)
    assert not logits.requires_grad
