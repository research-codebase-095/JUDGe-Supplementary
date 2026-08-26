import math
import os

import numpy as np
import torch

from deployment_reliability.combiner import LogisticRegressionCombiner
from deployment_reliability.features import featurize
from deployment_reliability.router import (
    EXECUTE,
    HITL,
    VERIFY,
    aurc,
    auroc,
    bonferroni_clopper_pearson_thresholds,
    clopper_pearson_threshold,
    conformal_threshold,
    cost_sensitive_joint_policy,
    cost_sensitive_thresholds,
    decision_region_grid,
    dyadic_zero_error_threshold,
    risk_coverage_curve,
    route,
    route_joint,
    threshold_for_target_risk,
    two_threshold_risk_coverage,
)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESNET50_CACHE = os.path.join(REPO_ROOT, "data", "logit_cache_resnet50.pt")


def _perfectly_ranked_scores(n_correct=80, n_incorrect=20):
    # Descending scores 100..1; the top n_correct are correct, the rest aren't -
    # i.e. the score perfectly ranks correctness.
    n = n_correct + n_incorrect
    scores = torch.arange(n, 0, -1).float()
    correct = torch.zeros(n, dtype=torch.bool)
    correct[:n_correct] = True
    return scores, correct


def test_risk_coverage_curve_zero_risk_while_perfectly_ranked():
    scores, correct = _perfectly_ranked_scores()
    coverage, risk = risk_coverage_curve(scores, correct)
    n_correct = int(correct.sum().item())
    # Risk should be exactly 0 up through full coverage of the correct prefix.
    assert torch.allclose(risk[:n_correct], torch.zeros(n_correct), atol=1e-6)
    assert math.isclose(coverage[-1].item(), 1.0, rel_tol=1e-6)


def test_aurc_is_small_for_perfectly_ranked_scores_and_larger_for_random():
    scores, correct = _perfectly_ranked_scores()
    good_aurc = aurc(scores, correct)

    torch.manual_seed(0)
    shuffled = correct[torch.randperm(len(correct))]
    bad_aurc = aurc(scores, shuffled)

    assert good_aurc < bad_aurc


def test_threshold_for_target_risk_zero_keeps_only_correct_prefix():
    scores, correct = _perfectly_ranked_scores()
    tau = threshold_for_target_risk(scores, correct, target_risk=0.0)
    accepted = scores >= tau
    assert correct[accepted].all()


def test_two_threshold_risk_coverage_orders_thresholds_and_raises_on_bad_input():
    scores, correct = _perfectly_ranked_scores()
    tau_hi, tau_lo = two_threshold_risk_coverage(scores, correct, execute_risk=0.0, verify_risk=0.1)
    assert tau_hi >= tau_lo
    try:
        two_threshold_risk_coverage(scores, correct, execute_risk=0.2, verify_risk=0.1)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_bonferroni_clopper_pearson_thresholds_orders_thresholds_and_raises_on_bad_input():
    scores, correct = _perfectly_ranked_scores()
    tau_hi, tau_lo = bonferroni_clopper_pearson_thresholds(scores, correct, execute_risk=0.0, verify_risk=0.1)
    assert tau_hi >= tau_lo
    try:
        bonferroni_clopper_pearson_thresholds(scores, correct, execute_risk=0.2, verify_risk=0.1)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_bonferroni_clopper_pearson_thresholds_matches_manual_confidence_adjustment():
    # bonferroni_clopper_pearson_thresholds(..., confidence=c) on n scores
    # must equal calling clopper_pearson_threshold directly at the
    # Bonferroni-adjusted confidence 1-(1-c)/n - the whole function is that
    # adjustment plus two clopper_pearson_threshold calls, nothing more.
    scores, correct = _perfectly_ranked_scores(n_correct=80, n_incorrect=20)
    n = len(scores)
    confidence_adj = 1.0 - (1.0 - 0.95) / n
    tau_hi_expected = clopper_pearson_threshold(scores, correct, target_risk=0.05, confidence=confidence_adj)
    tau_lo_expected = clopper_pearson_threshold(scores, correct, target_risk=0.2, confidence=confidence_adj)
    tau_hi, tau_lo = bonferroni_clopper_pearson_thresholds(scores, correct, execute_risk=0.05, verify_risk=0.2)
    assert math.isclose(tau_hi, tau_hi_expected)
    assert math.isclose(tau_lo, tau_lo_expected)


def test_dyadic_zero_error_threshold_matches_hand_derived_boundary():
    # DESIGN.md 14.13's headline numeric result: at alpha=0.10, gamma=0.95,
    # m=5141, the dyadic-block bound clears alpha exactly at n=64 (block
    # [64,127]) but not at n=63 (block [32,63]) - hand-verified via
    # scipy.stats.beta.ppf independently before trusting the function.
    n_cal = 5141
    # A synthetic calibration split with a genuine 64-long zero-error prefix
    # from the top (scores strictly decreasing, first 64 correct, then one
    # incorrect, then arbitrary) - constructed so N* is exactly 64, not
    # dependent on tie-breaking.
    scores = torch.linspace(1.0, 0.0, n_cal)
    correct = torch.ones(n_cal, dtype=torch.bool)
    correct[64] = False  # first error appears right after the 64-long prefix
    tau = dyadic_zero_error_threshold(scores, correct, target_risk=0.10, confidence=0.95)
    assert tau is not None
    accepted = int((scores >= tau).sum())
    assert accepted == 64, f"expected the full 64-long zero-error prefix to be accepted, got {accepted}"

    # One fewer available zero-error point (N*=63) must fail to clear the
    # same target - the exact boundary DESIGN.md 14.13 reports.
    correct63 = torch.ones(n_cal, dtype=torch.bool)
    correct63[63] = False
    tau63 = dyadic_zero_error_threshold(scores, correct63, target_risk=0.10, confidence=0.95)
    assert tau63 is None, "N*=63 should not clear target_risk=0.10 at this m - the exact boundary this section reports"


def test_dyadic_zero_error_threshold_returns_none_when_top_ranked_item_is_wrong():
    # If the very top-scored item is itself incorrect, N*=0 - no zero-error
    # prefix exists at all, regardless of how loose target_risk/confidence
    # are. This is the real, checked GPT-2-at-f=1.0 finding (DESIGN.md
    # 14.13): a ranking-quality limit no amount of statistical tightening
    # can work around.
    scores = torch.tensor([0.9, 0.5, 0.4, 0.3])
    correct = torch.tensor([False, True, True, True])
    tau = dyadic_zero_error_threshold(scores, correct, target_risk=0.99, confidence=0.5)
    assert tau is None


def test_dyadic_zero_error_threshold_never_accepts_a_point_that_is_actually_incorrect():
    # Whatever threshold is returned, everything at or above it in the real
    # data must actually be correct - the zero-error contract, checked
    # directly rather than only inferred from N* bookkeeping.
    torch.manual_seed(0)
    scores = torch.rand(500)
    correct = torch.rand(500) > 0.3
    tau = dyadic_zero_error_threshold(scores, correct, target_risk=0.5, confidence=0.5)
    if tau is not None:
        accepted = scores >= tau
        assert correct[accepted].all()


def test_real_resnet50_dyadic_zero_error_threshold_never_admits_more_than_bonferroni_cp_alone():
    # dyadic_zero_error_threshold's K=ceil(log2(n)) blocks use a much less
    # severe per-test confidence adjustment than clopper_pearson_threshold's
    # single-candidate bound at the SAME (unadjusted) confidence would - but
    # it must still never accept more than the raw, uncorrected
    # clopper_pearson_threshold(..., confidence=0.95) does, since that bound
    # (evaluated once, at N* itself, un-Bonferroni'd) is an upper bound on
    # what any valid selection-aware procedure should accept.
    assert os.path.exists(RESNET50_CACHE), "run scripts/collect_logits.py resnet50 first"
    cache = torch.load(RESNET50_CACHE)
    logits, labels = cache["logits"], cache["labels"]
    splits_arr = np.array(cache["splits"])

    def mask(name):
        return torch.from_numpy(splits_arr == name)

    m_fit, m_cal = mask("combiner_fit"), mask("threshold_cal")
    correct = logits.argmax(dim=-1) == labels
    phi = featurize(logits)
    combiner = LogisticRegressionCombiner().fit(phi[m_fit], correct[m_fit].float())
    S_cal = combiner.score(phi)[m_cal]
    correct_cal = correct[m_cal]

    tau_dyadic = dyadic_zero_error_threshold(S_cal, correct_cal, target_risk=0.05, confidence=0.95)
    tau_cp_single = clopper_pearson_threshold(S_cal, correct_cal, target_risk=0.05, confidence=0.95)
    if tau_dyadic is not None:
        n_dyadic = int((S_cal >= tau_dyadic).sum())
        n_cp_single = int((S_cal >= tau_cp_single).sum())
        assert n_dyadic <= n_cp_single, (
            f"dyadic-peeling accepted more ({n_dyadic}) than the single-candidate CP bound ({n_cp_single}) - "
            "should never happen, since the single-candidate bound is itself an upper bound on valid selection"
        )


def test_cost_sensitive_thresholds_matches_hand_solved_crossover():
    # cost_execute(p)=10(1-p), cost_verify(p)=2-p, cost_hitl=3.
    # Crossover Execute vs Verify: 10-10p = 2-p -> p = 8/9. HITL is never
    # optimal here since cost_verify(p) <= 2 < 3 = cost_hitl for all p in [0,1].
    p_hi, p_lo = cost_sensitive_thresholds(
        c_execute_incorrect=10.0, c_verify_correct=1.0, c_verify_incorrect=2.0, c_hitl=3.0
    )
    assert math.isclose(p_hi, 8.0 / 9.0, abs_tol=1e-3)
    assert math.isclose(p_lo, 0.0, abs_tol=1e-3)


def test_cost_sensitive_thresholds_extreme_hitl_cost_never_selects_hitl():
    p_hi, p_lo = cost_sensitive_thresholds(
        c_execute_incorrect=1.0, c_verify_correct=0.5, c_verify_incorrect=0.5, c_hitl=1e6
    )
    assert p_lo == 0.0


def test_conformal_threshold_is_monotonic_in_alpha():
    scores, correct = _perfectly_ranked_scores(n_correct=80, n_incorrect=20)
    tau_strict = conformal_threshold(scores, correct, alpha=0.01)
    tau_loose = conformal_threshold(scores, correct, alpha=0.2)
    # A stricter (smaller) alpha must accept a same-or-smaller set, i.e. a
    # same-or-higher score threshold.
    assert tau_strict >= tau_loose


def test_route_assigns_expected_buckets():
    scores = torch.tensor([0.9, 0.6, 0.2])
    labels = route(scores, tau_hi=0.8, tau_lo=0.4)
    assert labels == [EXECUTE, VERIFY, HITL]


def test_route_raises_when_thresholds_inverted():
    try:
        route(torch.rand(3), tau_hi=0.3, tau_lo=0.5)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_risk_coverage_curve_matches_input_dtype_not_a_hardcoded_one():
    scores, correct = _perfectly_ranked_scores()
    scores64 = scores.double()
    coverage, risk = risk_coverage_curve(scores64, correct)
    assert coverage.dtype == torch.float64
    assert risk.dtype == torch.float64


def test_conformal_threshold_works_with_non_default_dtype():
    scores, correct = _perfectly_ranked_scores()
    tau = conformal_threshold(scores.double(), correct, alpha=0.1)
    assert isinstance(tau, float)


def test_clopper_pearson_threshold_matches_hand_computed_beta_ppf():
    # Hand-verified against an independent computation (root-finding the
    # definitional relationship P(X<=x | n,p=U) = 1-confidence via scipy's
    # binomial CDF, done separately - not repeated here since that needs
    # scipy.optimize) rather than trusted from the beta-ppf formula alone.
    # 100 perfectly-ranked scores, top 90 correct: at the n=90 prefix (all
    # correct, x=0), U(0,90,0.95) = Beta.ppf(0.95, 1, 90) ~= 0.03247.
    from scipy.stats import beta as beta_dist

    scores, correct = _perfectly_ranked_scores(n_correct=90, n_incorrect=10)
    expected_u_at_90 = beta_dist.ppf(0.95, 1, 90)
    assert expected_u_at_90 < 0.033  # sanity on the reference value itself

    # target_risk just above the n=90 bound should accept exactly the top 90;
    # target_risk just below it should accept fewer.
    tau_above = clopper_pearson_threshold(scores, correct, target_risk=expected_u_at_90 + 1e-4, confidence=0.95)
    tau_below = clopper_pearson_threshold(scores, correct, target_risk=expected_u_at_90 - 1e-4, confidence=0.95)
    assert int((scores >= tau_above).sum()) == 90
    assert int((scores >= tau_below).sum()) < 90


def test_clopper_pearson_threshold_is_stricter_than_approximate_conformal_threshold():
    # The whole point of the tightening: the ad hoc "+1 correction" in
    # conformal_threshold is not a proven bound and can be too permissive.
    # At the same target_risk, the properly-derived Clopper-Pearson threshold
    # must never accept a LARGER Execute band than the approximate one.
    torch.manual_seed(3)
    n = 300
    scores = torch.rand(n)
    correct = torch.rand(n) < (0.5 + 0.4 * scores)  # noisy but real ranking signal
    for alpha in (0.02, 0.05, 0.1, 0.2):
        tau_approx = conformal_threshold(scores, correct, alpha=alpha)
        tau_cp = clopper_pearson_threshold(scores, correct, target_risk=alpha, confidence=0.95)
        n_approx = int((scores >= tau_approx).sum())
        n_cp = int((scores >= tau_cp).sum())
        assert n_cp <= n_approx, f"alpha={alpha}: Clopper-Pearson accepted MORE than the approximate rule"


def test_clopper_pearson_threshold_confidence_bound_actually_holds_on_held_out_data():
    # The real payoff of the tightening: repeatedly split into cal/test,
    # calibrate the threshold on the cal split, and confirm the resulting
    # Execute-band error rate on a FRESH test split stays under target_risk
    # in at least `confidence` fraction of repeats - the actual property the
    # bound claims to guarantee, checked empirically over many trials.
    torch.manual_seed(4)
    target_risk, confidence = 0.1, 0.95
    n_trials = 300
    violations = 0
    for trial in range(n_trials):
        g = torch.Generator().manual_seed(trial)
        p_true = torch.rand(400, generator=g)  # true P(correct) per example
        scores = p_true + 0.05 * torch.randn(400, generator=g)  # noisy proxy score
        correct = torch.rand(400, generator=g) < p_true
        cal_scores, test_scores = scores[:200], scores[200:]
        cal_correct, test_correct = correct[:200], correct[200:]

        tau = clopper_pearson_threshold(cal_scores, cal_correct, target_risk=target_risk, confidence=confidence)
        accepted = test_scores >= tau
        if accepted.any():
            empirical_risk = 1.0 - test_correct[accepted].float().mean().item()
            if empirical_risk > target_risk:
                violations += 1
    violation_rate = violations / n_trials
    # Loose tolerance: this checks the guarantee is in the right ballpark
    # (violations should be rare, near the nominal 1-confidence=5% rate),
    # not exact calibration at this trial count/effect size.
    assert violation_rate < 0.20, f"violation rate {violation_rate:.3f} is far above the nominal 5% target"


def test_clopper_pearson_threshold_raises_on_invalid_confidence():
    scores, correct = _perfectly_ranked_scores()
    try:
        clopper_pearson_threshold(scores, correct, target_risk=0.1, confidence=1.0)
        assert False, "expected ValueError"
    except ValueError:
        pass
    try:
        clopper_pearson_threshold(scores, correct, target_risk=0.1, confidence=0.0)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_clopper_pearson_threshold_all_incorrect_returns_sentinel():
    scores, correct = _perfectly_ranked_scores(n_correct=0, n_incorrect=50)
    tau = clopper_pearson_threshold(scores, correct, target_risk=0.01, confidence=0.95)
    assert tau > scores.max().item()


def test_auroc_is_one_when_positives_strictly_outscore_negatives():
    pos = torch.tensor([0.9, 0.8, 0.7])
    neg = torch.tensor([0.3, 0.2, 0.1])
    assert math.isclose(auroc(pos, neg), 1.0, abs_tol=1e-6)


def test_auroc_is_zero_when_positives_strictly_underscore_negatives():
    pos = torch.tensor([0.1, 0.2])
    neg = torch.tensor([0.8, 0.9])
    assert math.isclose(auroc(pos, neg), 0.0, abs_tol=1e-6)


def test_auroc_is_half_for_identical_distributions():
    torch.manual_seed(0)
    shared = torch.rand(500)
    # Same underlying distribution for pos/neg -> expect ~0.5 up to sampling noise.
    pos = shared[:250]
    neg = shared[250:]
    assert abs(auroc(pos, neg) - 0.5) < 0.1


def test_auroc_handles_exact_ties_as_half_credit():
    pos = torch.tensor([0.5, 0.5])
    neg = torch.tensor([0.5, 0.5])
    assert math.isclose(auroc(pos, neg), 0.5, abs_tol=1e-6)


def test_cost_sensitive_joint_policy_matches_hand_solved_boundary_points():
    # Same cost table as test_cost_sensitive_thresholds_matches_hand_solved_crossover
    # (p_hi=8/9 at q=0), plus anomaly_cost_execute=5.0 tilting the Execute
    # boundary. Hand-solved: cost_execute(p,q) = -10p + 5q + 10,
    # cost_verify(p,q) = -p + 2. Boundary where they're equal:
    # -10p+5q+10 = -p+2  =>  q = (9p - 8) / 5.
    # At p=0.95, boundary q = (8.55-8)/5 = 0.11: below -> Execute, above -> Verify.
    policy = cost_sensitive_joint_policy(
        c_execute_incorrect=10.0, c_verify_correct=1.0, c_verify_incorrect=2.0, c_hitl=3.0,
        anomaly_cost_execute=5.0,
    )
    pts = [(0.95, 0.0), (0.95, 0.05), (0.95, 0.2), (0.85, 0.0), (0.0, 0.0)]
    expected = [EXECUTE, EXECUTE, VERIFY, VERIFY, VERIFY]
    p = torch.tensor([pt[0] for pt in pts])
    q = torch.tensor([pt[1] for pt in pts])
    assert route_joint(p, q, policy) == expected


def test_cost_sensitive_joint_policy_reduces_to_the_1d_scalar_rule_when_anomaly_costs_are_zero():
    # With anomaly_cost_execute/verify/hitl all 0, route_joint over (p, q) must
    # reproduce router.route(p, tau_hi, tau_lo) exactly, for ANY q - the 2D
    # policy is a strict superset of the 1D one, not a different system.
    kwargs = dict(c_execute_incorrect=10.0, c_verify_correct=1.0, c_verify_incorrect=2.0, c_hitl=3.0)
    p_hi, p_lo = cost_sensitive_thresholds(**kwargs)
    policy = cost_sensitive_joint_policy(**kwargs, anomaly_cost_execute=0.0)

    torch.manual_seed(0)
    p = torch.rand(200)
    q = torch.rand(200) * 50  # arbitrary - must not matter when the weight is 0
    joint_decisions = route_joint(p, q, policy)
    scalar_decisions = route(p, tau_hi=p_hi, tau_lo=p_lo)
    assert joint_decisions == scalar_decisions


def test_route_joint_never_selects_verify_or_hitl_boundary_differently_when_only_execute_is_tilted():
    # anomaly_cost_verify=anomaly_cost_hitl=0 (the default) keeps the
    # Verify/HITL boundary exactly at p=p_lo regardless of q - only Execute
    # can be pulled back to Verify. Construct a case where p sits strictly
    # in the Verify band and confirm high q never pushes it to HITL.
    policy = cost_sensitive_joint_policy(
        c_execute_incorrect=10.0, c_verify_correct=1.0, c_verify_incorrect=2.0, c_hitl=3.0,
        anomaly_cost_execute=5.0,
    )
    p = torch.tensor([0.5, 0.5, 0.5])
    q = torch.tensor([0.0, 10.0, 1000.0])  # increasingly anomalous
    decisions = route_joint(p, q, policy)
    assert decisions == [VERIFY, VERIFY, VERIFY]


def test_decision_region_grid_matches_route_joint_on_the_same_points():
    # The diagnostic grid helper must agree with the closed-form router it's
    # meant to numerically cross-check, not just run without error.
    policy = cost_sensitive_joint_policy(
        c_execute_incorrect=10.0, c_verify_correct=1.0, c_verify_incorrect=2.0, c_hitl=3.0,
        anomaly_cost_execute=5.0,
    )
    pp, qq, best = decision_region_grid(policy, p_range=(0.0, 1.0), q_range=(0.0, 1.0), grid_size_per_axis=21)
    p_flat, q_flat, best_flat = pp.reshape(-1), qq.reshape(-1), best.reshape(-1)
    via_route_joint = route_joint(p_flat, q_flat, policy)
    via_grid = [policy.action_names[i] for i in best_flat.tolist()]
    assert via_route_joint == via_grid


def test_real_resnet50_clopper_pearson_threshold_tracks_its_nominal_violation_rate():
    # The actual payoff of the Clopper-Pearson tightening, validated on real
    # data with a methodologically sound protocol: repeatedly draw FRESH,
    # non-overlapping cal/test re-splits from the pooled threshold_cal+id_test
    # data (holding the once-fit combiner score S fixed, matching standard
    # split-conformal practice where the score function is fixed and only the
    # calibration/test partition is randomized) and measure how often the
    # resulting Execute-band held-out error rate exceeds target_risk.
    #
    # An earlier version of this check used bootstrap resampling (with
    # replacement) instead of fresh re-splits, which inflated violation rates
    # for BOTH methods well past their nominal targets - bootstrap duplicates
    # break the IID assumption the Clopper-Pearson bound (and conformal
    # prediction generally) relies on, so it understates the true n and is
    # NOT a valid way to simulate repeated real-world calibration/deployment
    # splits. Caught by comparing bootstrap-resampling results against this
    # corrected fresh-re-split protocol before writing anything up.
    assert os.path.exists(RESNET50_CACHE), "run scripts/collect_logits.py resnet50 first"
    cache = torch.load(RESNET50_CACHE)
    logits, labels = cache["logits"], cache["labels"]
    splits_arr = np.array(cache["splits"])

    def mask(name):
        return torch.from_numpy(splits_arr == name)

    m_fit, m_cal, m_test = mask("combiner_fit"), mask("threshold_cal"), mask("id_test")
    correct = logits.argmax(dim=-1) == labels
    phi = featurize(logits)
    combiner = LogisticRegressionCombiner().fit(phi[m_fit], correct[m_fit].float())
    S = combiner.score(phi)

    pool_mask = m_cal | m_test
    S_pool, correct_pool = S[pool_mask], correct[pool_mask]
    n_pool = len(S_pool)
    n_cal_size = int(m_cal.sum())

    target_risk, confidence = 0.05, 0.95
    n_trials = 300
    torch.manual_seed(0)
    violations_approx, violations_cp = 0, 0
    for _ in range(n_trials):
        perm = torch.randperm(n_pool)
        cal_idx, test_idx = perm[:n_cal_size], perm[n_cal_size:]
        Sc, Cc = S_pool[cal_idx], correct_pool[cal_idx]
        St, Ct = S_pool[test_idx], correct_pool[test_idx]

        tau_approx = conformal_threshold(Sc, Cc, alpha=target_risk)
        tau_cp = clopper_pearson_threshold(Sc, Cc, target_risk=target_risk, confidence=confidence)

        accepted_approx = St >= tau_approx
        if accepted_approx.any() and (1.0 - Ct[accepted_approx].float().mean().item()) > target_risk:
            violations_approx += 1
        accepted_cp = St >= tau_cp
        if accepted_cp.any() and (1.0 - Ct[accepted_cp].float().mean().item()) > target_risk:
            violations_cp += 1

    rate_approx = violations_approx / n_trials
    rate_cp = violations_cp / n_trials
    # Loose tolerances (Monte Carlo noise at n_trials=300): the CP rate should
    # be in the right ballpark of its nominal 1-confidence=5% target, and
    # substantially - not marginally - better than the approximate rule's.
    assert rate_cp < 0.15, f"Clopper-Pearson violation rate {rate_cp:.3f} far exceeds its 5% nominal target"
    assert rate_cp < rate_approx, f"Clopper-Pearson ({rate_cp:.3f}) should violate less often than the approximate rule ({rate_approx:.3f})"


def test_real_resnet50_clopper_pearson_threshold_is_never_looser_than_approximate():
    # Companion to the violation-rate check: on the ACTUAL fixed
    # threshold_cal/id_test split (no resampling), the CP threshold's
    # Execute band must never be larger than the approximate rule's at the
    # same target_risk - the point of the tightening.
    assert os.path.exists(RESNET50_CACHE), "run scripts/collect_logits.py resnet50 first"
    cache = torch.load(RESNET50_CACHE)
    logits, labels = cache["logits"], cache["labels"]
    splits_arr = np.array(cache["splits"])

    def mask(name):
        return torch.from_numpy(splits_arr == name)

    m_fit, m_cal = mask("combiner_fit"), mask("threshold_cal")
    correct = logits.argmax(dim=-1) == labels
    phi = featurize(logits)
    combiner = LogisticRegressionCombiner().fit(phi[m_fit], correct[m_fit].float())
    S = combiner.score(phi)

    for target_risk in (0.02, 0.05, 0.1):
        tau_approx = conformal_threshold(S[m_cal], correct[m_cal], alpha=target_risk)
        tau_cp = clopper_pearson_threshold(S[m_cal], correct[m_cal], target_risk=target_risk, confidence=0.95)
        n_approx = int((S[m_cal] >= tau_approx).sum())
        n_cp = int((S[m_cal] >= tau_cp).sum())
        assert n_cp <= n_approx, f"target_risk={target_risk}: CP accepted more than the approximate rule on real data"


def test_real_resnet50_bonferroni_thresholds_never_looser_than_naive_or_single_candidate_cp():
    # DESIGN.md 23.6: two_threshold_risk_coverage's naive threshold_for_target_risk
    # selects from up to n candidates with no correction at all; a single-candidate
    # clopper_pearson_threshold corrects for sampling noise at ONE fixed candidate
    # but not for the selection step itself. bonferroni_clopper_pearson_thresholds
    # must never accept a larger (looser) Execute band than either, on the same
    # real threshold_cal split, at the same execute_risk/verify_risk targets.
    assert os.path.exists(RESNET50_CACHE), "run scripts/collect_logits.py resnet50 first"
    cache = torch.load(RESNET50_CACHE)
    logits, labels = cache["logits"], cache["labels"]
    splits_arr = np.array(cache["splits"])

    def mask(name):
        return torch.from_numpy(splits_arr == name)

    m_fit, m_cal = mask("combiner_fit"), mask("threshold_cal")
    correct = logits.argmax(dim=-1) == labels
    phi = featurize(logits)
    combiner = LogisticRegressionCombiner().fit(phi[m_fit], correct[m_fit].float())
    S_cal = combiner.score(phi)[m_cal]
    correct_cal = correct[m_cal]

    tau_hi_naive, tau_lo_naive = two_threshold_risk_coverage(S_cal, correct_cal, execute_risk=0.05, verify_risk=0.2)
    tau_hi_cp, tau_lo_cp = (
        clopper_pearson_threshold(S_cal, correct_cal, target_risk=0.05, confidence=0.95),
        clopper_pearson_threshold(S_cal, correct_cal, target_risk=0.2, confidence=0.95),
    )
    tau_hi_bonf, tau_lo_bonf = bonferroni_clopper_pearson_thresholds(S_cal, correct_cal, execute_risk=0.05, verify_risk=0.2)

    n_naive = int((S_cal >= tau_hi_naive).sum())
    n_cp = int((S_cal >= tau_hi_cp).sum())
    n_bonf = int((S_cal >= tau_hi_bonf).sum())
    assert n_bonf <= n_cp <= n_naive, (
        f"Execute band should shrink (or stay equal) naive({n_naive}) -> single-candidate CP({n_cp}) "
        f"-> Bonferroni CP({n_bonf}), the intended tightening order"
    )
