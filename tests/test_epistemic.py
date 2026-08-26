import pytest
import torch
from PIL import Image

from deployment_reliability.backbone import (
    load_frozen_convnext_tiny,
    load_frozen_resnet50,
    load_frozen_vit_b16,
    load_frozen_vit_b16_with_dropout,
    logits_for_images,
)
from deployment_reliability.epistemic import mc_dropout_predict

_IMG_A = Image.new("RGB", (224, 224), color=(120, 80, 200))
_IMG_B = Image.new("RGB", (224, 224), color=(10, 200, 50))


def test_load_frozen_vit_b16_with_dropout_has_identical_weights_to_the_plain_loader():
    # The load-bearing claim in backbone.py's docstring: enabling dropout at
    # construction time does not change which weights get loaded, since
    # dropout layers carry zero learnable parameters. Checked directly.
    plain_model, _, _ = load_frozen_vit_b16()
    dropout_model, _, _ = load_frozen_vit_b16_with_dropout()
    sd_plain, sd_dropout = plain_model.state_dict(), dropout_model.state_dict()
    assert set(sd_plain.keys()) == set(sd_dropout.keys())
    assert all(torch.equal(sd_plain[k], sd_dropout[k]) for k in sd_plain)


def test_load_frozen_vit_b16_with_dropout_is_frozen_and_in_eval_mode():
    model, preprocess, categories = load_frozen_vit_b16_with_dropout()
    assert model.training is False
    assert all(not p.requires_grad for p in model.parameters())
    assert len(categories) == 1000


def test_mc_dropout_predict_returns_shape_matched_mean_and_variance():
    model, preprocess, _ = load_frozen_vit_b16_with_dropout(dropout=0.2, attention_dropout=0.2)
    mean_logits, variance = mc_dropout_predict(model, preprocess, [_IMG_A, _IMG_B], passes=6)
    assert mean_logits.shape == (2, 1000)
    assert variance.shape == (2,)
    assert torch.isfinite(mean_logits).all()
    assert torch.isfinite(variance).all()
    assert (variance >= 0.0).all()


def test_mc_dropout_predict_restores_eval_mode_on_every_dropout_submodule_afterward():
    model, preprocess, _ = load_frozen_vit_b16_with_dropout(dropout=0.2, attention_dropout=0.2)
    mc_dropout_predict(model, preprocess, [_IMG_A], passes=4)
    assert model.training is False
    assert all(not m.training for m in model.modules() if isinstance(m, (torch.nn.Dropout, torch.nn.MultiheadAttention)))


def test_mc_dropout_predict_restores_eval_mode_even_if_the_forward_pass_raises():
    model, preprocess, _ = load_frozen_vit_b16_with_dropout(dropout=0.2, attention_dropout=0.2)

    def _broken_preprocess(img):
        raise ValueError("simulated preprocessing failure")

    with pytest.raises(ValueError):
        mc_dropout_predict(model, _broken_preprocess, [_IMG_A], passes=4)
    assert all(not m.training for m in model.modules() if isinstance(m, (torch.nn.Dropout, torch.nn.MultiheadAttention)))


def test_mc_dropout_predict_raises_if_model_is_already_in_train_mode():
    model, preprocess, _ = load_frozen_vit_b16_with_dropout()
    model.train()
    with pytest.raises(RuntimeError):
        mc_dropout_predict(model, preprocess, [_IMG_A], passes=4)
    model.eval()


def test_mc_dropout_predict_requires_at_least_two_passes():
    model, preprocess, _ = load_frozen_vit_b16_with_dropout()
    with pytest.raises(ValueError):
        mc_dropout_predict(model, preprocess, [_IMG_A], passes=1)


def test_mc_dropout_predict_raises_on_a_plain_zero_dropout_vit_instead_of_silently_returning_zero_variance():
    # dropout=attention_dropout=0.0 (load_frozen_vit_b16's default) has the
    # SAME nn.Dropout/nn.MultiheadAttention module types present, just
    # numerically inert (p=0.0) - checked directly that this is rejected,
    # not silently treated as a valid (if boring) all-zero-variance result.
    model, preprocess, _ = load_frozen_vit_b16()
    with pytest.raises(RuntimeError, match="ACTIVE"):
        mc_dropout_predict(model, preprocess, [_IMG_A], passes=4)


def test_mc_dropout_predict_raises_on_resnet50_which_has_no_dropout_modules_at_all():
    model, preprocess, _ = load_frozen_resnet50()
    with pytest.raises(RuntimeError, match="ACTIVE"):
        mc_dropout_predict(model, preprocess, [_IMG_A], passes=4)


def test_mc_dropout_predict_raises_on_convnext_tiny_stochastic_depth_is_not_equivalent():
    # ConvNeXt-Tiny's StochasticDepth modules are not nn.Dropout/
    # nn.MultiheadAttention instances at all (a structurally different
    # mechanism - see epistemic.py's module docstring), so this correctly
    # falls into the "no active dropout" rejection too.
    model, preprocess, _ = load_frozen_convnext_tiny()
    with pytest.raises(RuntimeError, match="ACTIVE"):
        mc_dropout_predict(model, preprocess, [_IMG_A], passes=4)


def test_mc_dropout_predict_mean_logits_are_close_to_but_not_identical_to_deterministic_logits():
    # A real, checked property, not assumed: MC-dropout's mean over T
    # stochastic passes should be IN THE NEIGHBORHOOD of the deterministic
    # eval-mode forward pass (same underlying weights), but not exactly
    # equal (dropout genuinely perturbs each pass - if it were exactly
    # equal, dropout wouldn't be doing anything).
    model, preprocess, _ = load_frozen_vit_b16_with_dropout(dropout=0.1, attention_dropout=0.1)
    det_logits = logits_for_images(model, preprocess, [_IMG_A, _IMG_B])
    mean_logits, variance = mc_dropout_predict(model, preprocess, [_IMG_A, _IMG_B], passes=15)
    assert not torch.allclose(mean_logits, det_logits, atol=1e-4)
    # But not wildly different either - same underlying weights, modest dropout rate.
    assert (mean_logits - det_logits).abs().mean().item() < 5.0


def test_mc_dropout_predict_logit_variance_increases_with_dropout_rate_but_probability_variance_does_not():
    # A genuinely counter-intuitive, checked-not-assumed finding: this
    # module's predictive_variance is defined over softmax PROBABILITIES
    # (not raw logits), and probability-space variance does NOT increase
    # monotonically with dropout rate the way an initial "more dropout ->
    # more per-pass stochasticity -> higher predictive variance" intuition
    # would suggest. Directly measured (see the mechanistic check this test
    # locks in): raw LOGIT variance across passes does increase with dropout
    # rate as expected, but softmax PROBABILITY variance can decrease at
    # higher dropout rates - a softmax-saturation effect. Probability mass
    # is bounded and sums to 1, and heavy dropout drives the mean predictive
    # distribution toward near-uniform faster than it grows logit-space
    # spread; a near-uniform distribution sits close to the flat interior of
    # the probability simplex, where further perturbation moves it around a
    # low-curvature region and produces LESS probability-space variance than
    # milder dropout perturbing a still-peaked (still confident-looking)
    # distribution. This means `predictive_variance` should be read as
    # "how much the predictive DISTRIBUTION moved," not as a monotonic proxy
    # for "how much dropout was applied" - a real, disclosed limitation of
    # reading it as a simple epistemic-uncertainty dial.
    torch.manual_seed(0)
    model_low, preprocess, _ = load_frozen_vit_b16_with_dropout(dropout=0.05, attention_dropout=0.05)
    torch.manual_seed(0)
    batch = torch.stack([preprocess(_IMG_A)], dim=0)
    with torch.inference_mode():
        from deployment_reliability.epistemic import _set_dropout_submodules_train

        _set_dropout_submodules_train(model_low, True)
        logits_low = torch.stack([model_low(batch) for _ in range(20)], dim=0)
        _set_dropout_submodules_train(model_low, False)

    torch.manual_seed(0)
    model_high, _, _ = load_frozen_vit_b16_with_dropout(dropout=0.4, attention_dropout=0.4)
    torch.manual_seed(0)
    with torch.inference_mode():
        _set_dropout_submodules_train(model_high, True)
        logits_high = torch.stack([model_high(batch) for _ in range(20)], dim=0)
        _set_dropout_submodules_train(model_high, False)

    logit_var_low = logits_low.var(dim=0).mean().item()
    logit_var_high = logits_high.var(dim=0).mean().item()
    prob_var_low = torch.softmax(logits_low, dim=-1).var(dim=0).sum().item()
    prob_var_high = torch.softmax(logits_high, dim=-1).var(dim=0).sum().item()

    assert logit_var_high > logit_var_low, "expected raw logit-space variance to increase with dropout rate"
    assert prob_var_high < prob_var_low, (
        "expected the counter-intuitive softmax-saturation effect: probability-space variance "
        "LOWER at higher dropout rate, despite higher logit-space variance"
    )
