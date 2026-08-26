import math

import torch

from deployment_reliability.evidence import Evidence
from deployment_reliability.features import energy_score, logit_l2_norm, logit_margin, msp, normalized_entropy

C = 1000


def peaked_logits(peak_value, second_value=0.0, batch_shape=()):
    z = torch.zeros(*batch_shape, C)
    z[..., 0] = peak_value
    z[..., 1] = second_value
    return z


def test_concentration_matches_msp():
    z = peaked_logits(10.0)
    assert torch.allclose(Evidence(z).concentration(), msp(z))


def test_separability_matches_logit_margin():
    z = peaked_logits(5.0, 2.0)
    assert torch.allclose(Evidence(z).separability(), logit_margin(z))


def test_ambiguity_matches_normalized_entropy():
    z = peaked_logits(3.0)
    assert torch.allclose(Evidence(z).ambiguity(), normalized_entropy(z))


def test_plausibility_matches_energy_score():
    z = peaked_logits(7.0)
    assert torch.allclose(Evidence(z).plausibility(), energy_score(z))


def test_magnitude_matches_logit_l2_norm():
    z = peaked_logits(3.0, 4.0)
    assert torch.allclose(Evidence(z).magnitude(), logit_l2_norm(z))
    assert torch.allclose(Evidence(z).magnitude(per_class=True), logit_l2_norm(z, per_class=True))


def test_conflict_is_near_one_when_the_runner_up_owns_essentially_all_non_top_mass():
    z = torch.full((C,), -10.0)
    z[0] = 5.0
    z[1] = 5.0  # top-2 tied, everything else negligible: the runner-up IS the non-top mass
    conflict = Evidence(z).conflict()
    assert conflict.item() > 0.999


def test_conflict_is_bounded_and_batched():
    torch.manual_seed(0)
    z = torch.randn(16, C)
    conflict = Evidence(z).conflict()
    assert conflict.shape == (16,)
    assert (conflict >= 0.0).all() and (conflict <= 1.0).all()


def test_conflict_is_finite_on_extreme_logits():
    # Regression check: an earlier implementation computed the non-top
    # probability mass as "1 - p_(1)", a subtraction of two nearly-equal
    # float32 numbers when p_(1) is close to 1 - that lost enough precision
    # to make conflict blow up past 1 (observed: ~3.6e20) instead of staying
    # in (0, 1]. Fixed by summing the non-top probabilities directly instead
    # of subtracting from 1.
    z = peaked_logits(1e4)
    conflict = Evidence(z).conflict()
    assert torch.isfinite(conflict).all()
    assert (conflict >= 0.0).all() and (conflict <= 1.0).all()


def test_conflict_differs_from_separability_for_identical_margin_but_different_rival_concentration():
    # Same top-1/top-2 margin (2.0) in both cases, so `separability` can't
    # tell them apart - but the non-top mass is concentrated in a single
    # rival in one case and spread across several near-tied rivals in the
    # other, which `conflict` (unlike a bounded rescaling of margin) is
    # built to distinguish.
    concentrated_rival = torch.tensor([5.0, 3.0, -10.0, -10.0, -10.0])
    diffuse_rivals = torch.tensor([5.0, 3.0, 2.9, 2.8, 2.7])

    sep_concentrated = Evidence(concentrated_rival).separability()
    sep_diffuse = Evidence(diffuse_rivals).separability()
    assert math.isclose(sep_concentrated.item(), sep_diffuse.item(), abs_tol=1e-4)

    conflict_concentrated = Evidence(concentrated_rival).conflict()
    conflict_diffuse = Evidence(diffuse_rivals).conflict()
    assert conflict_concentrated.item() > conflict_diffuse.item() + 0.5


def test_concentration_separability_ambiguity_conflict_are_exactly_shift_invariant():
    # DESIGN.md 17.4's proposition: since these four are all built from
    # softmax(z) (or a raw difference, for separability), they must be
    # unchanged by z -> z + c*1 for any scalar c - this is the precise,
    # provable reason these operators can't detect a whole-vector magnitude
    # shift (the thing plausibility/magnitude exist to catch instead).
    torch.manual_seed(7)
    z = torch.randn(10, C) * 5
    for c in (-50.0, -1.0, 3.7, 100.0):
        shifted = Evidence(z + c)
        base = Evidence(z)
        assert torch.allclose(shifted.concentration(), base.concentration(), atol=1e-4)
        assert torch.allclose(shifted.separability(), base.separability(), atol=1e-4)
        assert torch.allclose(shifted.ambiguity(), base.ambiguity(), atol=1e-4)
        assert torch.allclose(shifted.conflict(), base.conflict(), atol=1e-4)


def test_plausibility_shifts_exactly_by_the_shift_and_magnitude_is_not_invariant():
    # The complementary half of the same proposition: plausibility
    # (features.energy_score) satisfies E(z + c*1) = E(z) + c exactly (it's
    # T*logsumexp(z/T), and logsumexp(z+c) = c + logsumexp(z)); magnitude
    # (L2 norm) is provably NOT shift-invariant at all (changes by a
    # data-dependent amount). Neither operator would be useful for
    # correctness-style ranking (which needs shift-invariance to only
    # depend on relative logit structure) - which is exactly the point:
    # these two exist to catch what the shift-invariant four cannot.
    torch.manual_seed(8)
    z = torch.randn(10, C) * 5
    for c in (-50.0, -1.0, 3.7, 100.0):
        shifted = Evidence(z + c)
        base = Evidence(z)
        assert torch.allclose(shifted.plausibility() - base.plausibility(), torch.full((10,), c), atol=1e-3)
        assert not torch.allclose(shifted.magnitude(), base.magnitude(), atol=1e-2)


def test_separability_and_magnitude_are_homogeneous_degree_one_under_scaling():
    # DESIGN.md 17.4's scale-invariance proposition: f(a*z) = a*f(z) exactly
    # for these two, for any a > 0 - separability is a raw coordinate
    # difference (linear, so this is immediate), magnitude is a norm
    # (positively homogeneous of degree 1 by definition).
    torch.manual_seed(11)
    z = torch.randn(10, C) * 5
    base = Evidence(z)
    for a in (0.1, 0.5, 2.0, 10.0):
        scaled = Evidence(a * z)
        assert torch.allclose(scaled.separability(), a * base.separability(), atol=1e-3)
        assert torch.allclose(scaled.magnitude(), a * base.magnitude(), atol=1e-3)


def test_concentration_ambiguity_conflict_plausibility_are_not_scale_invariant():
    # The complementary half: these four all change under z -> a*z, and none
    # of them satisfy the simple f(a*z) = a*f(z) law that separability/
    # magnitude do (checked explicitly, not just "they differ from f(z)").
    # This is exactly the mechanism TemperatureScaling (calibration.py)
    # exploits - temperature scaling only has something to correct because
    # these operators are scale-sensitive in the first place.
    torch.manual_seed(12)
    z = torch.randn(10, C) * 5
    base = Evidence(z)
    for a in (0.1, 0.5, 2.0, 10.0):
        scaled = Evidence(a * z)
        assert not torch.allclose(scaled.concentration(), base.concentration(), atol=1e-3)
        assert not torch.allclose(scaled.ambiguity(), base.ambiguity(), atol=1e-3)
        assert not torch.allclose(scaled.conflict(), base.conflict(), atol=1e-3)
        assert not torch.allclose(scaled.plausibility(), base.plausibility(), atol=1e-3)
        assert not torch.allclose(scaled.plausibility(), a * base.plausibility(), atol=1e-2)


def test_consistency_raises_not_implemented():
    z = peaked_logits(1.0)
    try:
        Evidence(z).consistency()
        assert False, "expected NotImplementedError"
    except NotImplementedError:
        pass


def test_stability_raises_not_implemented():
    z = peaked_logits(1.0)
    try:
        Evidence(z).stability()
        assert False, "expected NotImplementedError"
    except NotImplementedError:
        pass
