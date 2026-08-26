"""Threshold router: S -> {Execute, Verify, HITL}, per DESIGN.md section 10."""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
import torch
from scipy.stats import beta as _beta_dist
from scipy.stats import rankdata as _rankdata

EXECUTE, VERIFY, HITL = "Execute", "Verify", "HITL"


def risk_coverage_curve(scores: torch.Tensor, correct: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Sort by descending score; return (coverage, risk) arrays (DESIGN.md 11.3's AURC input)."""
    n = len(scores)
    order = torch.argsort(scores, descending=True)
    sorted_correct = correct[order].to(dtype=scores.dtype)
    cum_correct = torch.cumsum(sorted_correct, dim=0)
    counts = torch.arange(1, n + 1, dtype=scores.dtype, device=scores.device)
    coverage = counts / n
    risk = 1.0 - cum_correct / counts
    return coverage, risk


def aurc(scores: torch.Tensor, correct: torch.Tensor) -> float:
    """Area under the risk-coverage curve (trapezoidal), per DESIGN.md 11.3."""
    coverage, risk = risk_coverage_curve(scores, correct)
    return float(torch.trapz(risk, coverage).item())


def auroc(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> float:
    """AUROC separating two groups by score, per DESIGN.md 11.3.

    `pos_scores` is the group expected to score HIGHER (e.g. correct
    predictions, or in-distribution inputs); `neg_scores` the group expected
    to score lower. Semantically this is still the pairwise-comparison
    definition of AUROC (probability a random positive outscores a random
    negative, with ties counted as half a win) - but computed via the
    equivalent Mann-Whitney-U-via-ranks identity (`AUROC = (rank_sum(pos) -
    n_pos*(n_pos+1)/2) / (n_pos*n_neg)`, using average/midrank tie-breaking,
    the standard textbook equivalence) rather than materializing the dense
    `(n_pos, n_neg)` pairwise-difference matrix an earlier version of this
    function built directly.

    That earlier, more literal version was "exact, and cheap enough at the
    sample sizes this module deals with" for this project's original
    ~1,500-2,000-sample evaluation splits, but broke for real once this
    project's own LLM-extension tests (`tests/test_llm_extension.py`,
    `tests/test_llm_extension_pythia.py`) started calling it on the full
    ~130,000-example `id_test` split: `RuntimeError: ... not enough memory:
    you tried to allocate 16085602400 bytes` - a single dense matrix that size
    cannot fit in this machine's ~16GB of total RAM regardless of what else is
    running, not a transient contention issue. The rank-based formula computes
    the identical quantity in `O(n log n)` (dominated by one sort), with no
    dense intermediate, and is checked to agree with the original
    pairwise-matrix definition to float precision - including deliberately
    tie-heavy cases, where a rank/threshold reformulation is most likely to
    silently diverge from the pairwise definition - in
    `tests/test_significance.py` (which independently re-derives and verifies
    the same identity for its own bootstrap/DeLong use) before being trusted
    here.
    """
    pos_np = pos_scores.detach().cpu().to(dtype=torch.float64).numpy()
    neg_np = neg_scores.detach().cpu().to(dtype=torch.float64).numpy()
    n_pos, n_neg = len(pos_np), len(neg_np)
    ranks = _rankdata(np.concatenate([pos_np, neg_np]), method="average")
    u_statistic = ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u_statistic / (n_pos * n_neg))


def threshold_for_target_risk(scores: torch.Tensor, correct: torch.Tensor, target_risk: float) -> float:
    """Largest-coverage threshold whose Execute-band risk stays <= target_risk (DESIGN.md 10.2)."""
    order = torch.argsort(scores, descending=True)
    sorted_scores = scores[order]
    _, risk = risk_coverage_curve(scores, correct)
    valid = torch.nonzero(risk <= target_risk, as_tuple=True)[0]
    if len(valid) == 0:
        return float(sorted_scores.max().item()) + 1e-6  # no coverage achieves target_risk
    return float(sorted_scores[valid.max()].item())


def two_threshold_risk_coverage(
    scores: torch.Tensor, correct: torch.Tensor, execute_risk: float, verify_risk: float
) -> tuple[float, float]:
    """tau_hi/tau_lo via the risk-coverage curve, the Chow's-rule-generalizing strategy (DESIGN.md 10.2).

    Requires execute_risk <= verify_risk (Execute is the higher-trust band).
    """
    if execute_risk > verify_risk:
        raise ValueError("execute_risk must be <= verify_risk (Execute is the higher-trust band)")
    tau_hi = threshold_for_target_risk(scores, correct, execute_risk)
    tau_lo = threshold_for_target_risk(scores, correct, verify_risk)
    return tau_hi, tau_lo


def cost_sensitive_thresholds(
    c_execute_incorrect: float,
    c_verify_correct: float,
    c_verify_incorrect: float,
    c_hitl: float,
    u_execute_correct: float = 0.0,
    grid_size: int = 200_001,
) -> tuple[float, float]:
    """Bayes-optimal p_hi/p_lo boundaries on P(correct), the primary recommendation (DESIGN.md 10.3).

    Cost table: C(Execute,correct)=-u_execute_correct, C(Execute,incorrect)=c_execute_incorrect,
    C(Verify,correct)=c_verify_correct, C(Verify,incorrect)=c_verify_incorrect,
    C(HITL,*)=c_hitl (human cost, ~correctness-independent). Solved by evaluating
    all three expected-cost lines on a fine grid over p=P(correct) in [0,1] and
    locating where the argmin action switches, rather than deriving closed-form
    crossovers, so it stays correct for any cost ordering. Depends only on the
    cost table, not on data - the Bayes-optimal generalization of Chow's rule.

    `u_execute_correct` (default 0.0, fully backward-compatible with every
    existing caller) is an optional benefit/utility term for a CORRECT
    automatic execution: without it, f_E(p)=(1-p)*c_execute_incorrect has no
    reward for being right, so Verify/HITL can never look worse than a "free"
    correct execution - a structural bias toward conservatism raised as a
    reviewer concern. With u_execute_correct > 0,
    f_E(p) = (1-p)*c_execute_incorrect - p*u_execute_correct, still affine in
    p, so the same grid-argmin solves it without any other change.

    Apply to a *calibrated* score (DESIGN.md section 9) so it can be read as
    P(correct); an uncalibrated S only gives a ranking, not this probability.
    """
    p = torch.linspace(0.0, 1.0, grid_size)
    cost_execute = (1.0 - p) * c_execute_incorrect - p * u_execute_correct
    cost_verify = p * c_verify_correct + (1.0 - p) * c_verify_incorrect
    cost_hitl = torch.full_like(p, float(c_hitl))
    costs = torch.stack([cost_hitl, cost_verify, cost_execute], dim=-1)  # 0=HITL, 1=Verify, 2=Execute
    best_action = costs.argmin(dim=-1)

    execute_mask = best_action == 2
    verify_or_better_mask = best_action >= 1
    p_hi = float(p[execute_mask].min().item()) if execute_mask.any() else 1.0 + 1e-6
    p_lo = float(p[verify_or_better_mask].min().item()) if verify_or_better_mask.any() else 1.0 + 1e-6
    return p_hi, p_lo


def conformal_threshold(scores: torch.Tensor, correct: torch.Tensor, alpha: float) -> float:
    """Approximate split-conformal threshold for the Execute band (DESIGN.md 10.4).

    Finds the largest top-k-by-score prefix whose empirical error rate, with a
    "+1" finite-sample correction, stays <= alpha, and returns the score at its
    boundary. This is a practical first-pass approximation of conformal risk
    control (Bates et al., 2021) for a monotone risk function, not a formally
    proven bound - replace with a Clopper-Pearson (or similar) upper confidence
    bound before relying on it for a safety-critical guarantee.
    """
    order = torch.argsort(scores, descending=True)
    sorted_scores = scores[order]
    sorted_correct = correct[order].to(dtype=scores.dtype)
    n = len(scores)
    cum_incorrect = torch.cumsum(1.0 - sorted_correct, dim=0)
    counts = torch.arange(1, n + 1, dtype=scores.dtype, device=scores.device)
    empirical_risk = (cum_incorrect + 1.0) / (counts + 1.0)
    valid = torch.nonzero(empirical_risk <= alpha, as_tuple=True)[0]
    if len(valid) == 0:
        return float(sorted_scores.max().item()) + 1e-6  # no threshold achieves alpha
    return float(sorted_scores[valid.max()].item())


def clopper_pearson_threshold(
    scores: torch.Tensor, correct: torch.Tensor, target_risk: float, confidence: float = 0.95
) -> float:
    """Formally correct finite-sample conformal threshold for the Execute band
    (DESIGN.md's uncertainty-quantification section; STUDY_PLAN.md Objective 6),
    replacing conformal_threshold's approximate "+1 correction" with a genuine
    Clopper-Pearson upper confidence bound on the empirical error rate.

    For a top-k-by-score prefix with x observed errors out of n=k predictions,
    the one-sided Clopper-Pearson upper bound at level `confidence` is
    U(x,n) = Beta.ppf(confidence, x+1, n-x) (U=1 when x=n) - the standard
    result relating the binomial CDF to the incomplete beta function (Clopper &
    Pearson, 1934). This gives P(true error rate <= U) >= confidence, a real
    finite-sample statistical guarantee, unlike conformal_threshold's ad hoc
    correction. Verified independently against root-finding the definitional
    relationship P(X <= x | n, p=U) = 1 - confidence via scipy's binomial CDF
    (not included here since it needs scipy.optimize; see the module tests).

    Finds the largest top-k prefix whose U(x,k) stays <= target_risk, same
    "largest coverage that satisfies the bound" convention as
    conformal_threshold and threshold_for_target_risk.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    order = torch.argsort(scores, descending=True)
    sorted_scores = scores[order]
    sorted_correct = correct[order].to(dtype=scores.dtype)
    n = len(scores)
    cum_incorrect = torch.cumsum(1.0 - sorted_correct, dim=0).cpu().numpy()
    counts = np.arange(1, n + 1)

    x = cum_incorrect  # errors in each prefix
    upper = np.where(
        x >= counts,
        1.0,
        _beta_dist.ppf(confidence, x + 1, np.maximum(counts - x, 1e-12)),
    )
    valid = np.nonzero(upper <= target_risk)[0]
    if len(valid) == 0:
        return float(sorted_scores.max().item()) + 1e-6  # no threshold achieves target_risk at this confidence
    return float(sorted_scores[int(valid.max())].item())


def bonferroni_clopper_pearson_thresholds(
    scores: torch.Tensor, correct: torch.Tensor, execute_risk: float, verify_risk: float, confidence: float = 0.95
) -> tuple[float, float]:
    """tau_hi/tau_lo via Bonferroni-corrected Clopper-Pearson bounds (DESIGN.md 23.6), a
    drop-in, formally-correct replacement for two_threshold_risk_coverage.

    two_threshold_risk_coverage (and clopper_pearson_threshold called with an
    unadjusted confidence) both *select* a threshold by scanning up to n
    candidate cut-points and keeping whichever looks best; neither's guarantee
    accounts for that selection step - the same threshold-selection-multiplicity
    gap DESIGN.md 14.9 found and fixed for the LLM sequence-level threshold.
    This applies the identical fix here: Bonferroni-adjusting confidence to
    1-(1-confidence)/n before calling clopper_pearson_threshold restores a
    valid simultaneous guarantee across all n candidates (union bound), so the
    guarantee holds for whichever candidate ends up selected.

    Requires execute_risk <= verify_risk (Execute is the higher-trust band).
    """
    if execute_risk > verify_risk:
        raise ValueError("execute_risk must be <= verify_risk (Execute is the higher-trust band)")
    n = len(scores)
    confidence_adj = 1.0 - (1.0 - confidence) / n
    tau_hi = clopper_pearson_threshold(scores, correct, execute_risk, confidence=confidence_adj)
    tau_lo = clopper_pearson_threshold(scores, correct, verify_risk, confidence=confidence_adj)
    return tau_hi, tau_lo


def dyadic_zero_error_threshold(
    scores: torch.Tensor, correct: torch.Tensor, target_risk: float, confidence: float = 0.95
) -> float | None:
    """Tighter (rank-based, DESIGN.md 14.13), still-formally-valid alternative
    to bonferroni_clopper_pearson_thresholds, RESTRICTED to the zero-error
    prefix case (matching DESIGN.md 14.9's Proposition's own "best case"
    scope, not a general-x replacement for clopper_pearson_threshold).

    bonferroni_clopper_pearson_thresholds allocates confidence budget
    (1-confidence)/n across n INDIVIDUAL candidate thresholds via a plain
    union bound - correct, but wasteful, since candidates are nested
    (top-k subset of top-(k+1)) rather than independent: "the k-prefix is
    error-free" is a MONOTONE property in k, so the event "some zero-error
    prefix exists whose true risk exceeds a fixed bound U" has probability
    exactly (1-p)^(smallest k in the relevant range), which lets confidence
    be allocated across O(log n) DYADIC BLOCKS of candidate sizes instead of
    across all n individually (DESIGN.md 14.13's Proposition/Proof) - a
    union bound over far fewer, genuinely near-independent events.

    Finds N* = the largest prefix (by descending score) with zero observed
    errors, then checks it against the dyadic-block bound at confidence
    1-(1-confidence)/ceil(log2(n)). Returns the threshold at N* if it clears
    target_risk, else None (no valid zero-error-based threshold at this
    n/confidence - unlike clopper_pearson_threshold's sentinel-max-score
    fallback, since a caller should not silently execute nothing here: the
    zero-error restriction means there is no weaker fallback candidate to
    try within this function's own scope).
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    order = torch.argsort(scores, descending=True)
    sorted_scores = scores[order]
    sorted_correct = correct[order]
    n = len(scores)

    incorrect = ~sorted_correct.bool()
    first_error = int(torch.argmax(incorrect.int()).item()) if incorrect.any() else n
    n_star = first_error  # largest zero-error prefix length (prefix[:n_star])
    if n_star == 0:
        return None

    k_blocks = math.ceil(math.log2(n))
    confidence_adj = 1.0 - (1.0 - confidence) / k_blocks
    block_start = 2 ** (math.floor(math.log2(n_star)))
    bound = float(_beta_dist.ppf(confidence_adj, 1, block_start))

    if bound > target_risk:
        return None
    return float(sorted_scores[n_star - 1].item())


def route(scores: torch.Tensor, tau_hi: float, tau_lo: float) -> list[str]:
    """Map scores to {Execute, Verify, HITL} given two thresholds (DESIGN.md 10)."""
    if tau_hi < tau_lo:
        raise ValueError("tau_hi must be >= tau_lo")
    labels = []
    for s in scores.tolist():
        if s >= tau_hi:
            labels.append(EXECUTE)
        elif s >= tau_lo:
            labels.append(VERIFY)
        else:
            labels.append(HITL)
    return labels


class AffineCostPolicy(NamedTuple):
    """Per-action (weight, bias) for a cost model affine in x = (p, q, ...) in R^k.

    cost_a(x) = weights[a] @ x + biases[a]. Produced by
    cost_sensitive_joint_policy() below; consumed by route_joint().
    """

    action_names: tuple[str, ...]
    weights: torch.Tensor  # (num_actions, k)
    biases: torch.Tensor  # (num_actions,)


def cost_sensitive_joint_policy(
    c_execute_incorrect: float,
    c_verify_correct: float,
    c_verify_incorrect: float,
    c_hitl: float,
    anomaly_cost_execute: float,
    anomaly_cost_verify: float = 0.0,
    anomaly_cost_hitl: float = 0.0,
) -> AffineCostPolicy:
    """2D generalization of cost_sensitive_thresholds (DESIGN.md 10.3) to a cost
    model affine in x = (p, q), where p is a genuine posterior P(correct|features)
    but q (e.g. reliability.ReliabilityState.distribution_shift) is NOT a
    probability - see DESIGN.md's joint-policy section for the full derivation.

    IMPORTANT SCOPE: this is NOT "Bayes-optimal" in the way
    cost_sensitive_thresholds is. That function's rule is Bayes-optimal in the
    formal statistical-decision-theory sense because p is a real conditional
    probability, so cost_execute(p) = (1-p)*c really is E[cost | p]. Here,
    anomaly_cost_execute * q is a DESIGNER-CHOSEN policy penalty on q, not a
    probabilistic expectation - q carries no claim to being P(anything). What
    IS proven (DESIGN.md's joint-policy proposition): the resulting pointwise-
    argmin rule minimizes this explicitly stated affine cost surface among all
    measurable rules of (p, q), and its per-action decision regions are convex
    polyhedra (half-plane intersections) - call this "argmin-optimal for a
    stated affine cost model," never unqualified "Bayes-optimal."

    Defaults to anomaly_cost_verify = anomaly_cost_hitl = 0.0: the Verify/HITL
    boundary stays exactly p = p_lo (the 1D cost_sensitive_thresholds result),
    untouched by q; only the Execute boundary tilts based on q. This preserves
    anomaly_augmented_policy's tested guarantee that Verify/HITL decisions are
    never further downgraded - only Execute can be pulled back to Verify. This
    is a SMOOTH tradeoff (very high q does not force a downgrade if p is also
    very high), unlike anomaly_augmented_policy's hard rectangular cutoff -
    more principled, but not a hard safety floor unless anomaly_cost_execute
    is large. For a hard floor, compose route_joint's output with a separate
    cap check rather than folding it into this affine cost (a saturating
    penalty would break the polyhedral proof below).
    """
    weights = torch.tensor(
        [
            [0.0, anomaly_cost_hitl],
            [c_verify_correct - c_verify_incorrect, anomaly_cost_verify],
            [-c_execute_incorrect, anomaly_cost_execute],
        ]
    )
    biases = torch.tensor([c_hitl, c_verify_incorrect, c_execute_incorrect])
    return AffineCostPolicy((HITL, VERIFY, EXECUTE), weights, biases)


def route_joint(p: torch.Tensor, q: torch.Tensor, policy: AffineCostPolicy) -> list[str]:
    """Vectorized pointwise argmin over an AffineCostPolicy's cost surface -
    closed form, no grid search (DESIGN.md's joint-policy section)."""
    x = torch.stack([p, q], dim=-1).to(dtype=policy.weights.dtype)  # (N, k)
    costs = x @ policy.weights.T + policy.biases  # (N, num_actions)
    best = costs.argmin(dim=-1)
    return [policy.action_names[i] for i in best.tolist()]


def decision_region_grid(
    policy: AffineCostPolicy,
    p_range: tuple[float, float] = (0.0, 1.0),
    q_range: tuple[float, float] = (0.0, 10.0),
    grid_size_per_axis: int = 501,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Diagnostic-only brute-force cross-check of route_joint's closed-form
    boundary: evaluate every action's cost on a coarse (p, q) grid and return
    (p_grid, q_grid, best_action_index). NOT used for routing - route_joint's
    closed form is exact and cheap; this exists purely to numerically verify
    that closed form (or to plot the resulting decision regions), so it stays
    intentionally coarse (501^2 ~= 250k points, well under a second) rather
    than the 200_001-per-axis default cost_sensitive_thresholds uses for its
    1D grid, which would be computationally infeasible in 2D (~4*10^10 points).
    """
    p_grid = torch.linspace(*p_range, grid_size_per_axis)
    q_grid = torch.linspace(*q_range, grid_size_per_axis)
    pp, qq = torch.meshgrid(p_grid, q_grid, indexing="ij")
    x = torch.stack([pp.reshape(-1), qq.reshape(-1)], dim=-1).to(dtype=policy.weights.dtype)
    costs = x @ policy.weights.T + policy.biases
    best = costs.argmin(dim=-1).reshape(pp.shape)
    return pp, qq, best
