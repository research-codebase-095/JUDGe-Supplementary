import math
import os

import numpy as np
import torch

from deployment_reliability.features import (
    DEFAULT_FEATURE_DIRECTIONS,
    DEFAULT_FEATURE_NAMES,
    FEATURE_DIRECTIONS,
    aggregate_sequence_features,
    energy_score,
    featurize,
    logit_l2_norm,
    logit_margin,
    msp,
    normalized_entropy,
    sequence_correctness,
    verify_feature_directions,
)

C = 1000
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def uniform_logits(batch_shape=()):
    return torch.zeros(*batch_shape, C)


def peaked_logits(peak_value, batch_shape=()):
    z = torch.zeros(*batch_shape, C)
    z[..., 0] = peak_value
    return z


def test_msp_uniform_logits_equals_one_over_c():
    z = uniform_logits()
    assert math.isclose(msp(z).item(), 1.0 / C, rel_tol=1e-5)


def test_msp_increases_with_peak_sharpness():
    low = msp(peaked_logits(1.0))
    high = msp(peaked_logits(10.0))
    assert high.item() > low.item()
    assert 0.0 < low.item() <= 1.0
    assert 0.0 < high.item() <= 1.0


def test_logit_margin_uniform_is_zero():
    z = uniform_logits()
    assert math.isclose(logit_margin(z).item(), 0.0, abs_tol=1e-6)


def test_logit_margin_matches_hand_computed_gap():
    z = torch.zeros(C)
    z[0] = 5.0
    z[1] = 2.0
    assert math.isclose(logit_margin(z).item(), 3.0, abs_tol=1e-5)


def test_normalized_entropy_uniform_is_one():
    z = uniform_logits()
    assert math.isclose(normalized_entropy(z).item(), 1.0, rel_tol=1e-4)


def test_normalized_entropy_decreases_as_peak_sharpens():
    low_peak = normalized_entropy(peaked_logits(1.0))
    high_peak = normalized_entropy(peaked_logits(20.0))
    assert high_peak.item() < low_peak.item()
    assert 0.0 <= high_peak.item() <= 1.0


def test_energy_score_uniform_matches_closed_form():
    # logsumexp of C zeros == log(C); confidence-aligned energy == T*log(C) at T=1
    z = uniform_logits()
    expected = math.log(C)
    assert math.isclose(energy_score(z).item(), expected, rel_tol=1e-5)


def test_energy_score_increases_with_peak_sharpness():
    low = energy_score(peaked_logits(1.0))
    high = energy_score(peaked_logits(10.0))
    assert high.item() > low.item()


def test_logit_l2_norm_matches_hand_computed_value():
    z = torch.zeros(5)
    z[0] = 3.0
    z[1] = 4.0
    assert math.isclose(logit_l2_norm(z).item(), 5.0, abs_tol=1e-6)


def test_featurize_shape_and_ordering_single_prediction():
    z = peaked_logits(10.0)
    phi = featurize(z)
    assert phi.shape == (5,)
    assert DEFAULT_FEATURE_NAMES == ("msp", "logit_margin", "normalized_entropy", "energy_score", "logit_l2_norm")
    assert math.isclose(phi[0].item(), msp(z).item(), abs_tol=1e-6)
    assert math.isclose(phi[1].item(), logit_margin(z).item(), abs_tol=1e-6)
    assert math.isclose(phi[2].item(), normalized_entropy(z).item(), abs_tol=1e-6)
    assert math.isclose(phi[3].item(), energy_score(z).item(), abs_tol=1e-6)
    assert math.isclose(phi[4].item(), logit_l2_norm(z).item(), abs_tol=1e-6)


def test_featurize_is_batched():
    z = torch.stack([uniform_logits(), peaked_logits(10.0)], dim=0)
    phi = featurize(z)
    assert phi.shape == (2, 5)
    # confident (peaked) row should score higher on msp, margin, and energy
    assert phi[1, 0] > phi[0, 0]
    assert phi[1, 1] > phi[0, 1]
    assert phi[1, 3] > phi[0, 3]
    # and lower on normalized entropy
    assert phi[1, 2] < phi[0, 2]


def test_all_features_finite_on_extreme_logits():
    z = peaked_logits(1e4)
    phi = featurize(z)
    assert torch.isfinite(phi).all()


def test_logit_l2_norm_per_class_correction_matches_hand_computed_value():
    z = torch.zeros(5)
    z[0] = 3.0
    z[1] = 4.0
    raw = logit_l2_norm(z)
    corrected = logit_l2_norm(z, per_class=True)
    assert math.isclose(raw.item(), 5.0, abs_tol=1e-6)
    assert math.isclose(corrected.item(), 5.0 / math.sqrt(5), abs_tol=1e-6)


def test_logit_l2_norm_per_class_preserves_ranking_within_fixed_c():
    low = peaked_logits(1.0)
    high = peaked_logits(10.0)
    assert logit_l2_norm(low, per_class=True) < logit_l2_norm(high, per_class=True)


def test_logit_l2_norm_per_class_makes_different_c_comparable():
    # Same per-logit scale (all logits = 1.0), but C differs by 100x - raw
    # norm should differ by sqrt(100)=10x, per-class-corrected should match.
    small_c = torch.ones(10)
    large_c = torch.ones(1000)
    assert math.isclose(
        logit_l2_norm(small_c).item() * math.sqrt(100), logit_l2_norm(large_c).item(), rel_tol=1e-5
    )
    assert math.isclose(
        logit_l2_norm(small_c, per_class=True).item(), logit_l2_norm(large_c, per_class=True).item(), rel_tol=1e-5
    )


def test_featurize_normalize_l2_flag_matches_per_class_norm():
    z = peaked_logits(5.0)
    phi_raw = featurize(z, normalize_l2=False)
    phi_norm = featurize(z, normalize_l2=True)
    assert math.isclose(phi_raw[4].item(), logit_l2_norm(z).item(), abs_tol=1e-6)
    assert math.isclose(phi_norm[4].item(), logit_l2_norm(z, per_class=True).item(), abs_tol=1e-6)


def test_feature_directions_cover_all_default_features_with_valid_signs():
    assert set(FEATURE_DIRECTIONS) == set(DEFAULT_FEATURE_NAMES)
    assert all(v in (1, -1) for v in FEATURE_DIRECTIONS.values())
    assert FEATURE_DIRECTIONS["normalized_entropy"] == -1, "entropy is lower-is-better"
    # Checked against real cached data across all three evaluated backbones
    # (not assumed): incorrect predictions have a consistently HIGHER raw L2
    # norm than correct ones on every one of them, so this is also
    # lower-is-better, contrary to the feature's original +1 assumption.
    assert FEATURE_DIRECTIONS["logit_l2_norm"] == -1, "logit_l2_norm is lower-is-better on real data, not higher"
    assert DEFAULT_FEATURE_DIRECTIONS.shape == (5,)


def test_logit_l2_norm_direction_confirmed_on_real_data_across_backbones():
    # Permanent regression test for the bug found while building the
    # disagreement matrix: an earlier version of FEATURE_DIRECTIONS assumed
    # logit_l2_norm was higher-is-better ("stronger backbone reaction =
    # more confident"), which silently degraded WeightedLinearCombiner
    # (the only consumer of this fixed sign) on real data. Locks in the
    # corrected direction against all three cached backbones so it can't
    # silently flip back.
    for backbone in ("resnet50", "vit_b16", "convnext_tiny"):
        cache_path = os.path.join(REPO_ROOT, "data", f"logit_cache_{backbone}.pt")
        assert os.path.exists(cache_path), f"run scripts/collect_logits.py {backbone} first"
        cache = torch.load(cache_path)
        logits, labels = cache["logits"], cache["labels"]
        splits_arr = np.array(cache["splits"])
        m_test = torch.from_numpy(splits_arr == "id_test")
        correct = logits.argmax(dim=-1) == labels

        norm = logit_l2_norm(logits)
        mean_correct = norm[m_test][correct[m_test]].mean().item()
        mean_incorrect = norm[m_test][~correct[m_test]].mean().item()
        assert mean_incorrect > mean_correct, (
            f"{backbone}: expected incorrect predictions to have higher raw L2 norm "
            f"than correct ones (mean_correct={mean_correct:.3f}, mean_incorrect={mean_incorrect:.3f})"
        )


def test_verify_feature_directions_flags_a_deliberately_wrong_direction():
    torch.manual_seed(9)
    n = 500
    phi = torch.randn(n, 5)
    # Column 0 is genuinely higher-is-better; claim it's the opposite.
    correct = (phi[:, 0] + 0.05 * torch.randn(n) > 0.0)
    wrong_directions = dict(FEATURE_DIRECTIONS)
    wrong_directions["msp"] = -1  # deliberately backwards
    result = verify_feature_directions(phi, correct, directions=wrong_directions)
    assert result["msp"] is False
    assert result["logit_margin"] is True  # untouched, still assumed correctly


def test_verify_feature_directions_passes_for_a_correctly_oriented_feature():
    torch.manual_seed(10)
    n = 500
    phi = torch.randn(n, 5)
    correct = (phi[:, 1] + 0.05 * torch.randn(n) > 0.0)  # column 1 = logit_margin, direction +1
    result = verify_feature_directions(phi, correct)
    assert result["logit_margin"] is True


def test_all_default_feature_directions_confirmed_across_all_backbones():
    # Generalizes the logit_l2_norm-specific regression test above: every
    # one of the 5 default features, not just the one we happened to trip
    # over, is checked against real data on all three backbones. This is
    # the permanent, reusable version of the manual investigation that
    # caught the logit_l2_norm bug - so a similarly-backwards direction on
    # any other feature (now, or after a future backbone is added) fails a
    # test instead of waiting to be noticed by hand again.
    for backbone in ("resnet50", "vit_b16", "convnext_tiny"):
        cache_path = os.path.join(REPO_ROOT, "data", f"logit_cache_{backbone}.pt")
        assert os.path.exists(cache_path), f"run scripts/collect_logits.py {backbone} first"
        cache = torch.load(cache_path)
        logits, labels = cache["logits"], cache["labels"]
        splits_arr = np.array(cache["splits"])
        m_fit = torch.from_numpy(splits_arr == "combiner_fit")
        correct = logits.argmax(dim=-1) == labels

        phi = featurize(logits)
        result = verify_feature_directions(phi[m_fit], correct[m_fit])
        failed = [name for name, ok in result.items() if not ok]
        assert not failed, f"{backbone}: FEATURE_DIRECTIONS assumption contradicted by real data for {failed}"


def test_aggregate_sequence_features_mean_matches_hand_computed_value():
    phi_sequence = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 0.0]])
    result = aggregate_sequence_features(phi_sequence, method="mean")
    assert torch.allclose(result, torch.tensor([3.0, 2.0]))


def test_aggregate_sequence_features_min_matches_hand_computed_value():
    phi_sequence = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 0.0]])
    result = aggregate_sequence_features(phi_sequence, method="min")
    assert torch.allclose(result, torch.tensor([1.0, 0.0]))


def test_aggregate_sequence_features_default_is_mean():
    phi_sequence = torch.rand(4, 5)
    assert torch.equal(aggregate_sequence_features(phi_sequence), aggregate_sequence_features(phi_sequence, method="mean"))


def test_aggregate_sequence_features_rejects_unknown_method():
    phi_sequence = torch.rand(4, 5)
    try:
        aggregate_sequence_features(phi_sequence, method="max")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_aggregate_sequence_features_reduces_seq_len_dimension_only():
    phi_sequence = torch.rand(7, 5)
    assert aggregate_sequence_features(phi_sequence, method="mean").shape == (5,)
    assert aggregate_sequence_features(phi_sequence, method="min").shape == (5,)


def test_sequence_correctness_default_threshold_matches_all_correct():
    # threshold=1.0 (the default) must exactly reproduce DESIGN.md 14.6's
    # original strict target (token_correct.all()) - a backward-compatibility
    # guarantee, not a silent behavior change.
    token_correct = torch.tensor([True, True, True, True, True])
    assert bool(sequence_correctness(token_correct)) == bool(token_correct.all())
    token_correct_one_wrong = torch.tensor([True, True, False, True, True])
    assert bool(sequence_correctness(token_correct_one_wrong)) == bool(token_correct_one_wrong.all())


def test_sequence_correctness_fractional_threshold_matches_hand_computed_value():
    # 4/5 correct = 0.8 exactly.
    token_correct = torch.tensor([True, True, False, True, True])
    assert bool(sequence_correctness(token_correct, threshold=0.8)) is True
    assert bool(sequence_correctness(token_correct, threshold=0.81)) is False


def test_sequence_correctness_boundary_is_inclusive():
    # fraction == threshold must count as correct (>=, not >).
    token_correct = torch.tensor([True, True, True, True, False])  # exactly 0.8
    assert bool(sequence_correctness(token_correct, threshold=0.8)) is True


def test_sequence_correctness_rejects_invalid_threshold():
    token_correct = torch.tensor([True, False])
    for bad in (0.0, -0.1, 1.1):
        try:
            sequence_correctness(token_correct, threshold=bad)
            assert False, f"expected ValueError for threshold={bad}"
        except ValueError:
            pass


def test_sequence_correctness_reduces_leading_dimension_for_batched_windows():
    # (seq_len, num_windows) -> (num_windows,), matching aggregate_sequence_features's
    # "reduce the seq_len axis only" convention.
    token_correct = torch.tensor(
        [[True, True], [True, False], [True, True], [True, True], [False, True]]
    )  # window 0: 4/5=0.8 correct; window 1: 4/5=0.8 correct
    result = sequence_correctness(token_correct, threshold=0.8)
    assert result.shape == (2,)
    assert bool(result[0]) is True and bool(result[1]) is True
