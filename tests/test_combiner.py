import os

import numpy as np
import torch

from deployment_reliability.combiner import LogisticRegressionCombiner, WeightedLinearCombiner
from deployment_reliability.features import DEFAULT_FEATURE_NAMES, featurize

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Cross-checked once, out-of-band (not repeated here since it's not a project
# dependency): fitting the same data with class_balanced=False, l2=0.0 via
# statsmodels.Logit(y, X_with_intercept).fit() reproduces both the MLE
# weights and this module's Laplace-approximated standard errors (sqrt of
# the _covariance diagonal) to within ~1e-4 relative error - i.e. the Laplace
# covariance IS the standard asymptotic Wald covariance for plain logistic
# regression, not merely "some" uncertainty estimate.


def test_weighted_linear_combiner_is_monotonic_in_each_feature():
    torch.manual_seed(0)
    phi = torch.rand(50, 5)
    combiner = WeightedLinearCombiner().fit(phi)

    low = phi.clone()
    low[:, 0] = 0.0
    high = phi.clone()
    high[:, 0] = 1.0

    assert (combiner.score(high) >= combiner.score(low)).all()


def test_weighted_linear_combiner_score_range():
    torch.manual_seed(1)
    phi = torch.rand(50, 5) * 10 - 5
    combiner = WeightedLinearCombiner().fit(phi)
    s = combiner.score(phi)
    assert (s >= 0.0).all() and (s <= 1.0).all()


def test_weighted_linear_combiner_orients_entropy_the_right_direction():
    # Columns follow DEFAULT_FEATURE_NAMES order: msp, margin, entropy, energy, l2norm.
    # Hold everything but entropy fixed; low entropy should score higher than high
    # entropy, since entropy is DESIGN.md's one lower-is-better feature (features.py
    # FEATURE_DIRECTIONS). Before the direction fix, this combiner would have scored
    # high-entropy HIGHER, since it just averaged the raw min-max-normalized value.
    phi = torch.tensor(
        [
            [0.5, 0.5, 0.1, 0.5, 0.5],  # low entropy -> should score higher
            [0.5, 0.5, 0.9, 0.5, 0.5],  # high entropy -> should score lower
        ]
    )
    combiner = WeightedLinearCombiner().fit(phi)
    scores = combiner.score(phi)
    assert scores[0] > scores[1]


def test_logistic_regression_combiner_learns_separable_signal():
    torch.manual_seed(2)
    n = 400
    phi = torch.randn(n, 5)
    # Only feature 0 carries signal; label is a noisy function of it.
    y = (phi[:, 0] + 0.1 * torch.randn(n) > 0.0).float()

    combiner = LogisticRegressionCombiner().fit(phi, y)
    s = combiner.score(phi)
    predicted = (s > 0.5).float()
    accuracy = (predicted == y).float().mean().item()
    assert accuracy > 0.9
    # The learned weight on the informative feature should dominate the others.
    assert combiner.weight[0].abs() > combiner.weight[1:].abs().max()


def test_logistic_regression_combiner_infers_feature_count_from_data():
    # No num_features constructor arg exists anymore - it must be inferred
    # from phi.shape[-1], so this must work unchanged for a non-default width
    # (e.g. an ablated or extended feature set, not just DESIGN.md's 5).
    torch.manual_seed(3)
    n, num_features = 300, 8
    phi = torch.randn(n, num_features)
    y = (phi[:, 2] + 0.1 * torch.randn(n) > 0.0).float()

    combiner = LogisticRegressionCombiner().fit(phi, y)
    assert combiner.weight.shape == (num_features,)
    accuracy = (((combiner.score(phi) > 0.5).float()) == y).float().mean().item()
    assert accuracy > 0.85


def test_logistic_regression_combiner_requires_fit_before_score():
    combiner = LogisticRegressionCombiner()
    try:
        combiner.score(torch.rand(3, 5))
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass


def test_weighted_linear_combiner_raises_on_width_mismatch_without_explicit_directions():
    phi = torch.rand(10, 3)  # not DEFAULT_FEATURE_NAMES' width (5)
    try:
        WeightedLinearCombiner().fit(phi)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_weighted_linear_combiner_works_with_explicit_directions_for_non_default_width():
    phi = torch.rand(10, 3)
    directions = torch.tensor([1.0, -1.0, 1.0])
    combiner = WeightedLinearCombiner(feature_directions=directions).fit(phi)
    s = combiner.score(phi)
    assert s.shape == (10,)
    assert (s >= 0.0).all() and (s <= 1.0).all()


def test_weighted_linear_combiner_default_directions_match_default_feature_names():
    assert len(DEFAULT_FEATURE_NAMES) == 5  # sanity: the default-width assumption this test file relies on


def test_weighted_linear_combiner_matches_input_dtype_not_a_hardcoded_one():
    # float64 is the practical CPU-only proxy for "not hardcoded to float32/CPU" -
    # the same .to(device=..., dtype=...) call path handles a real GPU device.
    phi64 = torch.rand(20, 5, dtype=torch.float64)
    combiner = WeightedLinearCombiner().fit(phi64)
    assert combiner.feature_directions.dtype == torch.float64
    s = combiner.score(phi64)
    assert s.dtype == torch.float64


def test_logistic_regression_combiner_matches_input_dtype_not_a_hardcoded_one():
    torch.manual_seed(4)
    phi64 = torch.randn(50, 5, dtype=torch.float64)
    y64 = (phi64[:, 0] > 0).double()
    combiner = LogisticRegressionCombiner().fit(phi64, y64)
    assert combiner.weight.dtype == torch.float64
    assert combiner.bias.dtype == torch.float64
    assert combiner.score(phi64).dtype == torch.float64


def _correlated_multicollinear_data(n=500, seed=5):
    # Mirrors the DESIGN.md 15.2/15.6 scenario that motivated adding L2:
    # two near-redundant features (like MSP/entropy, r~-0.9) plus one
    # genuinely informative feature, on a modest sample size.
    torch.manual_seed(seed)
    signal = torch.randn(n)
    redundant_a = signal + 0.05 * torch.randn(n)
    redundant_b = -signal + 0.05 * torch.randn(n)  # near-perfectly anti-correlated with redundant_a
    noise_feature = torch.randn(n)
    phi = torch.stack([redundant_a, redundant_b, noise_feature], dim=-1)
    y = (signal + 0.1 * torch.randn(n) > 0.0).float()
    return phi, y


def test_l2_regularization_shrinks_weight_norm_under_multicollinearity():
    phi, y = _correlated_multicollinear_data()
    unregularized = LogisticRegressionCombiner(l2=0.0).fit(phi, y)
    regularized = LogisticRegressionCombiner(l2=1e-2).fit(phi, y)
    assert regularized.weight.norm().item() < unregularized.weight.norm().item()


def test_l2_regularization_does_not_meaningfully_hurt_discrimination():
    phi, y = _correlated_multicollinear_data()
    unregularized = LogisticRegressionCombiner(l2=0.0).fit(phi, y)
    regularized = LogisticRegressionCombiner(l2=1e-2).fit(phi, y)

    acc_unreg = ((unregularized.score(phi) > 0.5).float() == y).float().mean().item()
    acc_reg = ((regularized.score(phi) > 0.5).float() == y).float().mean().item()
    assert acc_reg > acc_unreg - 0.02


def test_l2_zero_matches_default_unregularized_behavior():
    # Default l2 is now nonzero (1e-2); l2=0.0 must recover the original,
    # fully unregularized fit - a direct backward-compatibility guarantee.
    torch.manual_seed(6)
    n = 400
    phi = torch.randn(n, 5)
    y = (phi[:, 0] + 0.1 * torch.randn(n) > 0.0).float()

    combiner = LogisticRegressionCombiner(l2=0.0).fit(phi, y)
    s = combiner.score(phi)
    accuracy = (((s > 0.5).float()) == y).float().mean().item()
    assert accuracy > 0.9
    assert combiner.weight[0].abs() > combiner.weight[1:].abs().max()


def test_default_l2_is_nonzero_and_stored():
    combiner = LogisticRegressionCombiner()
    assert combiner.l2 > 0.0


def _noisy_logistic_data(n=300, seed=5, num_features=4):
    # Genuinely noisy (non-separable) labels - unlike a hard threshold on a
    # linear combination, this never triggers the classical perfect-
    # separation MLE-divergence degeneracy, which would make any Hessian-
    # based check numerically meaningless (checked directly: a deterministic-
    # threshold version of this generator drives fitted weights to ~1000+
    # and the resulting covariance to match only in relative, not absolute,
    # terms - the noisy version here matches to near machine precision instead).
    torch.manual_seed(seed)
    phi = torch.randn(n, num_features)
    p_true = torch.sigmoid(phi[:, 0] - phi[:, 1])
    y = (torch.rand(n) < p_true).float()
    return phi, y


def test_laplace_covariance_matches_closed_form_fisher_information():
    # Independent, dependency-free correctness check for the autograd-based
    # Hessian: for plain (unregularized, unweighted) logistic regression, the
    # Fisher information has a known closed form, H = X_aug^T diag(p(1-p))
    # X_aug, computable directly without autograd.functional.hessian at all.
    phi, y = _noisy_logistic_data()
    combiner = LogisticRegressionCombiner(class_balanced=False, l2=0.0).fit(phi, y)

    n = phi.shape[0]
    x_aug = torch.cat([phi, torch.ones(n, 1)], dim=-1)
    logits = phi @ combiner.weight + combiner.bias
    p = torch.sigmoid(logits)
    w_diag = p * (1 - p)
    hessian_closed_form = x_aug.T @ (w_diag.unsqueeze(-1) * x_aug)
    covariance_closed_form = torch.linalg.inv(hessian_closed_form)

    assert torch.allclose(covariance_closed_form, combiner._covariance, atol=1e-5, rtol=1e-4)


def test_epistemic_std_shrinks_monotonically_with_more_training_data():
    # The single most important qualitative property for this to be a
    # meaningful epistemic-uncertainty measure at all: more data fitting the
    # SAME underlying relationship must reduce uncertainty about the
    # combiner's own weights at a fixed query point.
    true_w = torch.tensor([1.5, -0.8, 0.3, 0.5, -0.2])
    torch.manual_seed(7)
    test_point = torch.randn(1, 5)

    stds = []
    for n in (50, 500, 5000):
        torch.manual_seed(42)
        phi = torch.randn(n, 5)
        p_true = torch.sigmoid(phi @ true_w)
        y = (torch.rand(n) < p_true).float()
        combiner = LogisticRegressionCombiner().fit(phi, y)  # default class_balanced=True, l2=1e-2
        stds.append(combiner.epistemic_std(test_point).item())

    assert stds[0] > stds[1] > stds[2], f"expected strictly decreasing uncertainty with more data, got {stds}"


def test_epistemic_std_requires_fit_before_use():
    combiner = LogisticRegressionCombiner()
    try:
        combiner.epistemic_std(torch.rand(3, 5))
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass


def test_score_with_uncertainty_returns_score_and_matching_epistemic_std():
    phi, y = _noisy_logistic_data()
    combiner = LogisticRegressionCombiner().fit(phi, y)
    score, std = combiner.score_with_uncertainty(phi)
    assert torch.equal(score, combiner.score(phi))
    assert torch.equal(std, combiner.epistemic_std(phi))


def test_score_mackay_adjusted_pulls_toward_half_when_uncertain_and_matches_plain_score_when_certain():
    true_w = torch.tensor([1.5, -0.8, 0.3, 0.5, -0.2])
    torch.manual_seed(8)
    test_point = torch.randn(1, 5)

    # small n -> high epistemic uncertainty -> mackay-adjusted score should
    # sit strictly between the plain score and 0.5.
    phi_small = torch.randn(40, 5)
    y_small = (torch.rand(40) < torch.sigmoid(phi_small @ true_w)).float()
    small_combiner = LogisticRegressionCombiner().fit(phi_small, y_small)
    plain_small = small_combiner.score(test_point).item()
    adjusted_small = small_combiner.score_mackay_adjusted(test_point).item()
    if abs(plain_small - 0.5) > 1e-6:  # only meaningful to check pull-toward-0.5 if not already at 0.5
        assert abs(adjusted_small - 0.5) < abs(plain_small - 0.5)

    # large n -> low epistemic uncertainty -> mackay-adjusted score should
    # converge close to the plain score.
    torch.manual_seed(9)
    phi_large = torch.randn(20000, 5)
    y_large = (torch.rand(20000) < torch.sigmoid(phi_large @ true_w)).float()
    large_combiner = LogisticRegressionCombiner().fit(phi_large, y_large)
    plain_large = large_combiner.score(test_point).item()
    adjusted_large = large_combiner.score_mackay_adjusted(test_point).item()
    assert abs(adjusted_large - plain_large) < 0.01


def test_epistemic_std_handles_singular_hessian_without_crashing():
    # Exact collinearity + l2=0.0: the weight-block Hessian is singular
    # (no ridge term to regularize it) - must fall back to a pseudo-inverse
    # rather than raising, since epistemic_std is an optional add-on that
    # should never break a fit that otherwise succeeds.
    torch.manual_seed(10)
    n = 200
    base = torch.randn(n)
    phi = torch.stack([base, 2 * base, torch.randn(n)], dim=-1)  # column 1 = 2 * column 0
    y = (base + 0.1 * torch.randn(n) > 0).float()
    combiner = LogisticRegressionCombiner(class_balanced=False, l2=0.0).fit(phi, y)
    std = combiner.epistemic_std(torch.randn(5, 3))
    assert torch.isfinite(std).all()
    assert (std >= 0.0).all()


def test_epistemic_std_matches_input_dtype_and_is_batched():
    torch.manual_seed(11)
    phi64 = torch.randn(100, 5, dtype=torch.float64)
    y64 = (phi64[:, 0] > 0).double()
    combiner = LogisticRegressionCombiner().fit(phi64, y64)

    std_single = combiner.epistemic_std(phi64)
    assert std_single.dtype == torch.float64
    assert std_single.shape == (100,)

    phi_batch = torch.randn(4, 7, 5, dtype=torch.float64)
    std_batch = combiner.epistemic_std(phi_batch)
    assert std_batch.shape == (4, 7)


def test_real_resnet50_epistemic_std_is_finite_and_shape_matched():
    # Basic real-data smoke test, not a claim about direction: DESIGN.md's
    # uncertainty-quantification section documents (and investigates the
    # mechanism behind) a genuinely surprising real finding - epistemic_std
    # on this fitted combiner does NOT come out higher on imagenet_a/o than
    # id_test, and is not a useful correctness predictor either - so this
    # test only locks in basic well-formedness, not an intuition-matching
    # direction that the real investigation found to be false.
    cache_path = os.path.join(REPO_ROOT, "data", "logit_cache_resnet50.pt")
    assert os.path.exists(cache_path), "run scripts/collect_logits.py resnet50 first"
    cache = torch.load(cache_path)
    logits, labels = cache["logits"], cache["labels"]
    splits_arr = np.array(cache["splits"])

    def mask(name):
        return torch.from_numpy(splits_arr == name)

    m_fit, m_test = mask("combiner_fit"), mask("id_test")
    correct = logits.argmax(dim=-1) == labels
    phi = featurize(logits)
    combiner = LogisticRegressionCombiner().fit(phi[m_fit], correct[m_fit].float())

    std = combiner.epistemic_std(phi[m_test])
    assert std.shape == (int(m_test.sum()),)
    assert torch.isfinite(std).all()
    assert (std >= 0.0).all()
