import os
import time

import numpy as np
import torch

from deployment_reliability.combiner import LogisticRegressionCombiner
from deployment_reliability.features import featurize, msp
from deployment_reliability.router import auroc
from deployment_reliability.significance import (
    _rank_based_auroc,
    _structural_components,
    bootstrap_auroc_ci,
    delong_test,
)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESNET50_CACHE = os.path.join(REPO_ROOT, "data", "logit_cache_resnet50.pt")


# --------------------------------------------------------------------------
# _rank_based_auroc / _structural_components: the O(n log n) replacements for
# an initial O(n_pos*n_neg) implementation that turned out to be not just slow
# but genuinely infeasible at this project's real LLM/full-scale sample sizes
# (see significance.py's module docstring) - verified here against small,
# deliberately-naive O(n_pos*n_neg) reference implementations before either
# is trusted for a bootstrap loop or a real DeLong test.
# --------------------------------------------------------------------------


def _bruteforce_psi_matrix(pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    """The exact, naive O(n_pos*n_neg) placement matrix - deliberately not
    reused from significance.py, so this is an independent ground truth."""
    diff = pos[:, None] - neg[None, :]
    return (diff > 0).astype(np.float64) + 0.5 * (diff == 0).astype(np.float64)


def _bruteforce_auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    return float(_bruteforce_psi_matrix(pos, neg).mean())


def _bruteforce_structural_components(pos: np.ndarray, neg: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    psi = _bruteforce_psi_matrix(pos, neg)
    v10 = psi.mean(axis=1)
    v01 = psi.mean(axis=0)
    return v10, v01, float(psi.mean())


def test_rank_based_auroc_matches_bruteforce_on_random_continuous_data():
    rng = np.random.default_rng(0)
    for trial in range(20):
        n_pos, n_neg = rng.integers(2, 80), rng.integers(2, 80)
        pos = rng.normal(0.2, 1.0, size=n_pos)
        neg = rng.normal(-0.2, 1.0, size=n_neg)
        fast = _rank_based_auroc(pos, neg)
        slow = _bruteforce_auroc(pos, neg)
        assert abs(fast - slow) < 1e-9, f"trial {trial}: fast={fast} slow={slow}"


def test_rank_based_auroc_matches_bruteforce_with_heavy_ties():
    # Ties are exactly where a rank-based reformulation is most likely to
    # silently diverge from the pairwise definition (tie-breaking convention
    # mismatches) - stress that specifically, not just generic continuous data.
    rng = np.random.default_rng(1)
    discrete_values = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    for trial in range(20):
        n_pos, n_neg = rng.integers(3, 60), rng.integers(3, 60)
        pos = rng.choice(discrete_values, size=n_pos)
        neg = rng.choice(discrete_values, size=n_neg)
        fast = _rank_based_auroc(pos, neg)
        slow = _bruteforce_auroc(pos, neg)
        assert abs(fast - slow) < 1e-9, f"trial {trial}: fast={fast} slow={slow}"


def test_rank_based_auroc_matches_router_auroc():
    torch.manual_seed(2)
    for trial in range(10):
        pos_t, neg_t = torch.rand(60), torch.rand(60)
        fast = _rank_based_auroc(pos_t.numpy().astype(np.float64), neg_t.numpy().astype(np.float64))
        reference = auroc(pos_t, neg_t)
        assert abs(fast - reference) < 1e-5, f"trial {trial}: fast={fast} router.auroc={reference}"


def test_structural_components_match_bruteforce_including_ties():
    rng = np.random.default_rng(3)
    discrete_values = np.array([0.0, 0.1, 0.2, 0.3])
    for trial in range(15):
        n_pos, n_neg = rng.integers(3, 50), rng.integers(3, 50)
        pos = rng.choice(discrete_values, size=n_pos)
        neg = rng.choice(discrete_values, size=n_neg)
        v10_fast, v01_fast, theta_fast = _structural_components(pos, neg)
        v10_slow, v01_slow, theta_slow = _bruteforce_structural_components(pos, neg)
        assert np.allclose(v10_fast, v10_slow, atol=1e-9), f"trial {trial}: V10 mismatch"
        assert np.allclose(v01_fast, v01_slow, atol=1e-9), f"trial {trial}: V01 mismatch"
        assert abs(theta_fast - theta_slow) < 1e-9, f"trial {trial}: theta mismatch"


def test_rank_based_auroc_and_structural_components_scale_to_llm_sample_sizes():
    # The actual reason this module was rewritten: reproduce (at smaller but
    # still large scale, to keep the test fast) the exact situation that broke
    # the original O(n_pos*n_neg) implementation - tens of thousands of
    # positives and negatives - and confirm both the point AUROC and the
    # DeLong structural components complete quickly with no memory blowup.
    # 20,000/20,000 would be an 800M-entry (6.4GB float64) dense matrix under
    # the old approach; this must run in well under a second here.
    rng = np.random.default_rng(4)
    n_pos, n_neg = 20_000, 20_000
    pos = rng.normal(0.6, 0.3, size=n_pos)
    neg = rng.normal(0.4, 0.3, size=n_neg)

    start = time.time()
    a = _rank_based_auroc(pos, neg)
    v10, v01, theta = _structural_components(pos, neg)
    elapsed = time.time() - start

    assert 0.0 <= a <= 1.0
    assert abs(a - theta) < 1e-9
    assert v10.shape == (n_pos,) and v01.shape == (n_neg,)
    assert elapsed < 10.0, f"O(n log n) computation at n=20,000/20,000 took {elapsed:.2f}s - unexpectedly slow"


# --------------------------------------------------------------------------
# bootstrap_auroc_ci
# --------------------------------------------------------------------------


def test_bootstrap_auroc_ci_contains_the_point_estimate():
    torch.manual_seed(0)
    pos = torch.rand(200) + 0.3
    neg = torch.rand(200)
    result = bootstrap_auroc_ci(pos, neg, n_bootstrap=500, seed=1)
    assert result.ci_lo <= result.auroc <= result.ci_hi


def test_bootstrap_auroc_ci_is_narrow_for_perfect_separation():
    pos = torch.arange(100, 200).float()
    neg = torch.arange(0, 100).float()
    result = bootstrap_auroc_ci(pos, neg, n_bootstrap=500, seed=1)
    assert result.auroc == 1.0
    assert result.ci_lo > 0.95  # every bootstrap draw is still perfectly separated


def test_bootstrap_auroc_ci_widens_with_smaller_sample_size():
    # Same underlying separation, but a much smaller sample should give a
    # wider CI - basic sanity that this isn't just returning a fixed-width band.
    torch.manual_seed(0)
    pos_big = torch.randn(2000) + 0.5
    neg_big = torch.randn(2000)
    pos_small = pos_big[:30]
    neg_small = neg_big[:30]

    result_big = bootstrap_auroc_ci(pos_big, neg_big, n_bootstrap=500, seed=2)
    result_small = bootstrap_auroc_ci(pos_small, neg_small, n_bootstrap=500, seed=2)
    width_big = result_big.ci_hi - result_big.ci_lo
    width_small = result_small.ci_hi - result_small.ci_lo
    assert width_small > width_big


def test_bootstrap_auroc_ci_reproducible_with_fixed_seed():
    torch.manual_seed(0)
    pos, neg = torch.rand(100), torch.rand(100)
    r1 = bootstrap_auroc_ci(pos, neg, n_bootstrap=300, seed=42)
    r2 = bootstrap_auroc_ci(pos, neg, n_bootstrap=300, seed=42)
    assert r1.ci_lo == r2.ci_lo
    assert r1.ci_hi == r2.ci_hi


def test_bootstrap_auroc_ci_raises_on_empty_group():
    try:
        bootstrap_auroc_ci(torch.tensor([]), torch.rand(5))
        assert False, "expected ValueError"
    except ValueError:
        pass


# --------------------------------------------------------------------------
# delong_test - correctness checks against independent references
# --------------------------------------------------------------------------


def test_delong_auc_estimates_match_router_auroc():
    # _structural_components' theta must agree with router.auroc's
    # independently-defined pairwise-comparison AUROC exactly (both are the
    # same quantity computed two different ways).
    torch.manual_seed(0)
    n = 150
    correct = torch.rand(n) < 0.4
    score_a = torch.rand(n)
    score_b = torch.rand(n)
    result = delong_test(correct, score_a, score_b)
    expected_auc_a = auroc(score_a[correct], score_a[~correct])
    expected_auc_b = auroc(score_b[correct], score_b[~correct])
    # router.auroc computes in float32, delong_test's structural components in
    # float64 - a small numerical gap is expected, not a correctness issue.
    assert abs(result.auc_a - expected_auc_a) < 1e-5
    assert abs(result.auc_b - expected_auc_b) < 1e-5


def test_delong_identical_scores_gives_p_value_one():
    torch.manual_seed(0)
    n = 100
    correct = torch.rand(n) < 0.5
    score = torch.rand(n)
    result = delong_test(correct, score, score.clone())
    assert result.z == 0.0
    assert result.p_value == 1.0
    assert abs(result.auc_diff) < 1e-9


def _paired_permutation_p_value(correct: torch.Tensor, score_a: torch.Tensor, score_b: torch.Tensor, n_perm: int, seed: int) -> float:
    """Independent from-scratch check: under H0 (the two scores are
    exchangeable per-example), randomly swap score_a[i]/score_b[i] with
    probability 0.5 for each i, recompute the AUC difference, and see how
    often the permuted |difference| meets or exceeds the observed one. This
    uses none of significance.py's DeLong machinery - only router.auroc -
    so agreement with delong_test is real evidence the closed-form variance
    formula is correct, not circular.
    """
    observed_diff = auroc(score_a[correct], score_a[~correct]) - auroc(score_b[correct], score_b[~correct])
    rng = np.random.default_rng(seed)
    n = len(correct)
    count_ge = 0
    for _ in range(n_perm):
        swap = torch.from_numpy(rng.random(n) < 0.5)
        perm_a = torch.where(swap, score_b, score_a)
        perm_b = torch.where(swap, score_a, score_b)
        diff = auroc(perm_a[correct], perm_a[~correct]) - auroc(perm_b[correct], perm_b[~correct])
        if abs(diff) >= abs(observed_diff):
            count_ge += 1
    return (count_ge + 1) / (n_perm + 1)  # +1 smoothing, standard permutation-test convention


def test_delong_p_value_matches_independent_paired_permutation_test_on_a_real_difference():
    # Construct a case with a genuine, sizeable paired difference: score_a is
    # a much better ranker of `correct` than score_b, sharing the same
    # underlying noise (so they're correlated, the situation DeLong is for).
    torch.manual_seed(7)
    n = 400
    true_signal = torch.rand(n)
    correct = true_signal > 0.5
    shared_noise = torch.randn(n) * 0.5
    score_a = true_signal * 3.0 + shared_noise  # strong signal
    score_b = true_signal * 0.3 + shared_noise  # weak signal, same noise -> correlated with a

    result = delong_test(correct, score_a, score_b)
    perm_p = _paired_permutation_p_value(correct, score_a, score_b, n_perm=2000, seed=3)

    # Both methods should agree the difference is real (score_a really does
    # rank correctness better - it has 10x the signal weight) ...
    assert result.auc_diff > 0
    assert result.p_value < 0.05
    assert perm_p < 0.05
    # ... and land in the same rough ballpark, not just the same side of 0.05.
    assert abs(result.p_value - perm_p) < 0.05, f"DeLong p={result.p_value:.4f} vs. permutation p={perm_p:.4f}"


def test_delong_p_value_matches_independent_paired_permutation_test_under_the_null():
    # Same setup, but now score_b is just a relabeled copy of score_a plus
    # independent extra noise with matched marginal AUC - no real difference
    # to detect. Both methods should agree: not significant.
    torch.manual_seed(11)
    n = 300
    true_signal = torch.rand(n)
    correct = true_signal > 0.5
    score_a = true_signal + torch.randn(n) * 0.4
    score_b = true_signal + torch.randn(n) * 0.4  # same signal strength, independent noise

    result = delong_test(correct, score_a, score_b)
    perm_p = _paired_permutation_p_value(correct, score_a, score_b, n_perm=2000, seed=5)

    assert result.p_value > 0.05
    assert perm_p > 0.05


def test_delong_null_calibration_rejection_rate_matches_nominal_alpha():
    # The strongest correctness check: simulate many independent datasets
    # drawn under a TRUE null (score_a and score_b are i.i.d. draws from the
    # exact same generative process, so AUC_a == AUC_b in expectation) and
    # confirm DeLong's p-value rejects at alpha=0.05 close to 5% of the time.
    # This is a calibration check on the test statistic itself, not just a
    # one-off agreement check - a wrong variance formula (e.g. treating the
    # two AUCs as independent instead of correlated, or an off-by-factor
    # error) would show up here as a rejection rate far from 5%.
    rng = np.random.default_rng(123)
    n_sims = 400
    alpha = 0.05
    n = 150
    rejections = 0
    for i in range(n_sims):
        true_signal = rng.random(n)
        correct = torch.from_numpy(true_signal > 0.5)
        score_a = torch.from_numpy(true_signal + rng.normal(0, 0.4, size=n))
        score_b = torch.from_numpy(true_signal + rng.normal(0, 0.4, size=n))
        result = delong_test(correct, score_a, score_b)
        if result.p_value < alpha:
            rejections += 1
    rejection_rate = rejections / n_sims
    # Loose tolerance for Monte Carlo noise at n_sims=400 (binomial SE at
    # p=0.05 is ~1.1%, so 3 SE is ~3.3 points): must be in [1%, 10%], not
    # exactly 5%, but nowhere near e.g. 30-50% (which is what an independence
    # assumption or a sign/scale error in the variance would produce).
    assert 0.01 <= rejection_rate <= 0.10, f"null rejection rate {rejection_rate:.3f} far from nominal alpha={alpha}"


def test_delong_raises_on_all_correct_or_all_incorrect():
    n = 20
    all_correct = torch.ones(n, dtype=torch.bool)
    score = torch.rand(n)
    try:
        delong_test(all_correct, score, score)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_delong_raises_on_mismatched_lengths():
    correct = torch.tensor([True, False, True])
    try:
        delong_test(correct, torch.rand(3), torch.rand(4))
        assert False, "expected ValueError"
    except ValueError:
        pass


# --------------------------------------------------------------------------
# Real ResNet-50 data: the actual comparison this module exists to answer
# --------------------------------------------------------------------------


def test_real_resnet50_combiner_beats_msp_significantly_on_id_test():
    # This IS comparison #1 from the significance notebook (notebooks/18),
    # reproduced here as a locked-in regression test: the combiner's AUROC
    # improvement over raw MSP on id_test should be real, not noise.
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

    s_combiner = combiner.score(phi[m_test])
    s_msp = msp(logits[m_test])
    correct_test = correct[m_test]

    result = delong_test(correct_test, s_combiner, s_msp)
    assert result.auc_a > result.auc_b, "combiner should outrank MSP on id_test"
    assert result.p_value < 0.05, f"expected the combiner's edge over MSP to be significant, got p={result.p_value:.4f}"
