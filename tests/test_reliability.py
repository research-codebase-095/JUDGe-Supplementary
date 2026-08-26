import os

import numpy as np
import torch

from deployment_reliability.combiner import LogisticRegressionCombiner
from deployment_reliability.features import featurize
from deployment_reliability.normalization import ReferenceNormalizer
from deployment_reliability.reliability import ReliabilityState, estimate_reliability_state

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESNET50_CACHE = os.path.join(REPO_ROOT, "data", "logit_cache_resnet50.pt")


def _toy_combiner():
    torch.manual_seed(0)
    n = 300
    phi = torch.randn(n, 5)
    y = (phi[:, 0] + 0.1 * torch.randn(n) > 0.0).float()
    return LogisticRegressionCombiner().fit(phi, y)


def test_reliability_state_is_a_named_tuple_of_seven_fields():
    assert ReliabilityState._fields == (
        "correctness",
        "uncertainty",
        "distribution_shift",
        "familiarity",
        "calibration_confidence",
        "feature_space_trust",
        "epistemic_uncertainty",
    )
    # feature_space_trust (item 2) and epistemic_uncertainty (item 5) are the
    # only fields with a default, both added strictly after the original
    # five - existing keyword-construction call sites that never mention
    # them (pipeline.py, this file's other tests) keep working unchanged.
    assert ReliabilityState._field_defaults == {"feature_space_trust": None, "epistemic_uncertainty": None}


def test_feature_space_trust_is_nan_without_a_mahalanobis_scorer():
    combiner = _toy_combiner()
    logits = torch.randn(5, 1000)
    state = estimate_reliability_state(logits, combiner)
    assert torch.isnan(state.feature_space_trust).all()
    assert state.feature_space_trust.shape == (5,)


def test_epistemic_uncertainty_is_nan_without_it_being_passed():
    combiner = _toy_combiner()
    logits = torch.randn(5, 1000)
    state = estimate_reliability_state(logits, combiner)
    assert torch.isnan(state.epistemic_uncertainty).all()
    assert state.epistemic_uncertainty.shape == (5,)


def test_epistemic_uncertainty_passes_through_when_given():
    combiner = _toy_combiner()
    logits = torch.randn(4, 1000)
    precomputed = torch.tensor([0.01, 0.5, 0.02, 0.9])
    state = estimate_reliability_state(logits, combiner, epistemic_uncertainty=precomputed)
    assert torch.equal(state.epistemic_uncertainty, precomputed)


def test_correctness_matches_the_combiners_own_score():
    combiner = _toy_combiner()
    torch.manual_seed(1)
    logits = torch.randn(10, 1000)
    state = estimate_reliability_state(logits, combiner)
    assert torch.allclose(state.correctness, combiner.score(featurize(logits)))


def test_uncertainty_and_correctness_are_shape_matched_to_batch():
    combiner = _toy_combiner()
    logits = torch.randn(7, 1000)
    state = estimate_reliability_state(logits, combiner)
    assert state.correctness.shape == (7,)
    assert state.uncertainty.shape == (7,)


def test_distribution_shift_and_familiarity_are_nan_without_a_reference_normalizer():
    combiner = _toy_combiner()
    logits = torch.randn(5, 1000)
    state = estimate_reliability_state(logits, combiner)
    assert torch.isnan(state.distribution_shift).all()
    assert torch.isnan(state.familiarity).all()


def test_distribution_shift_and_familiarity_are_finite_and_nonnegative_with_a_reference_normalizer():
    combiner = _toy_combiner()
    torch.manual_seed(2)
    reference_phi = torch.randn(500, 5)
    normalizer = ReferenceNormalizer().fit(reference_phi)
    logits = torch.randn(5, 1000)
    state = estimate_reliability_state(logits, combiner, reference_normalizer=normalizer)
    assert torch.isfinite(state.distribution_shift).all()
    assert torch.isfinite(state.familiarity).all()
    assert (state.distribution_shift >= 0.0).all()
    assert (state.familiarity >= 0.0).all()


def test_calibration_confidence_equals_correctness_without_a_calibrator():
    combiner = _toy_combiner()
    logits = torch.randn(4, 1000)
    state = estimate_reliability_state(logits, combiner)
    assert torch.equal(state.calibration_confidence, state.correctness)


def test_calibration_confidence_uses_the_calibrator_when_given():
    class _HalvingCalibrator:
        def transform(self, s):
            return s * 0.5

    combiner = _toy_combiner()
    logits = torch.randn(4, 1000)
    state = estimate_reliability_state(logits, combiner, calibrator=_HalvingCalibrator())
    assert torch.allclose(state.calibration_confidence, state.correctness * 0.5)


def test_real_resnet50_data_correctness_separates_correct_from_incorrect_and_signals_are_sane():
    # Same "verify against real cached data" discipline as test_combiner.py's
    # L2 tests and notebooks/07 - not just synthetic sanity checks.
    assert os.path.exists(RESNET50_CACHE), "run scripts/collect_logits.py resnet50 first"
    cache = torch.load(RESNET50_CACHE)
    logits, labels = cache["logits"], cache["labels"]
    splits_arr = np.array(cache["splits"])

    def mask(name):
        return torch.from_numpy(splits_arr == name)

    m_fit, m_test = mask("combiner_fit"), mask("id_test")
    correct = logits.argmax(dim=-1) == labels

    phi = featurize(logits)
    combiner = LogisticRegressionCombiner().fit(phi[m_fit], correct[m_fit].float())
    normalizer = ReferenceNormalizer().fit(phi[m_fit])

    state = estimate_reliability_state(logits[m_test], combiner, reference_normalizer=normalizer)

    assert state.correctness.shape == (int(m_test.sum()),)
    assert torch.isfinite(state.correctness).all()
    assert torch.isfinite(state.uncertainty).all()
    assert torch.isfinite(state.distribution_shift).all()
    assert torch.isfinite(state.familiarity).all()

    mean_correctness_correct = state.correctness[correct[m_test]].mean().item()
    mean_correctness_incorrect = state.correctness[~correct[m_test]].mean().item()
    assert mean_correctness_correct > mean_correctness_incorrect

    mean_uncertainty_correct = state.uncertainty[correct[m_test]].mean().item()
    mean_uncertainty_incorrect = state.uncertainty[~correct[m_test]].mean().item()
    assert mean_uncertainty_incorrect > mean_uncertainty_correct
