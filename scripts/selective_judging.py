"""Applies the cost-sensitive three-way router (R_tau, src/deployment_reliability/router.py)
to the LLM-judge task itself: given a judge's confidence score, should its
verdict be executed automatically, sent for verification, or escalated to a
human? Runs entirely over already-cached judge data (same caches
judge_characterization.py reads) - no new model inference.

Cost regime (illustrative, satisfies the paper's required ordering
0 <= c_V <= c_H <= c_V' < c_E): c_execute_incorrect=10 (an automatically
executed wrong verdict silently corrupts a benchmark/leaderboard),
c_verify_correct=1, c_verify_incorrect=3, c_hitl=2. Thresholds tau_hi/tau_lo
are computed once from this cost table alone (cost_sensitive_thresholds does
not depend on data) and then applied identically to two p-like scores per
judge config: raw MSP and the fitted combiner - so any difference in outcome
reflects the score, not the thresholds. Restricted to these two (not the
best single feature) because both are genuine [0,1]-valued probability
estimates (MSP is a softmax probability, the combiner is a fitted sigmoid
output); margin/entropy/energy/L2-norm are unbounded or differently-scaled
and are not valid direct inputs to a router whose thresholds are defined on
p = P(correct) - routing them without a calibration step would be a
different, uncontrolled experiment, not a same-thresholds comparison. No
recalibration is applied (K_theta_c is optional/untested in the paper's
Method section); MSP and the combiner both already lie in [0,1] and are used
as p directly.

Usage:
    python scripts/selective_judging.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from deployment_reliability.router import EXECUTE, HITL, VERIFY, cost_sensitive_thresholds, route  # noqa: E402

from judge_characterization import load_all_judge_configs  # noqa: E402

C_EXECUTE_INCORRECT = 10.0
C_VERIFY_CORRECT = 1.0
C_VERIFY_INCORRECT = 3.0
C_HITL = 2.0


def realized_cost(actions: list[str], correct: np.ndarray) -> np.ndarray:
    costs = np.empty(len(actions), dtype=np.float64)
    for i, (a, c) in enumerate(zip(actions, correct)):
        if a == EXECUTE:
            costs[i] = 0.0 if c else C_EXECUTE_INCORRECT
        elif a == VERIFY:
            costs[i] = C_VERIFY_CORRECT if c else C_VERIFY_INCORRECT
        else:  # HITL
            costs[i] = C_HITL
    return costs


def route_summary(
    name: str, scores: np.ndarray, correct: np.ndarray, tau_hi: float, tau_lo: float, question_id: np.ndarray
) -> dict:
    actions = route(torch.from_numpy(scores), tau_hi, tau_lo)
    actions = np.array(actions)
    n = len(actions)
    exec_mask = actions == EXECUTE
    n_exec = int(exec_mask.sum())
    exec_error_rate = float((~correct[exec_mask]).mean()) if n_exec > 0 else float("nan")
    costs = realized_cost(actions.tolist(), correct)
    ci_lo, ci_hi = cluster_bootstrap_mean_cost_ci(question_id, costs)
    return {
        "name": name,
        "n": n,
        "coverage": n_exec / n,
        "n_verify": int((actions == VERIFY).sum()),
        "n_hitl": int((actions == HITL).sum()),
        "exec_error_rate": exec_error_rate,
        "mean_cost": float(costs.mean()),
        "cost_ci": (ci_lo, ci_hi),
        "costs": costs,
    }


def always_execute_cost(correct: np.ndarray) -> float:
    return float((~correct).mean() * C_EXECUTE_INCORRECT)


def always_verify_cost(correct: np.ndarray) -> float:
    return float(correct.mean() * C_VERIFY_CORRECT + (~correct).mean() * C_VERIFY_INCORRECT)


def cluster_bootstrap_mean_cost_ci(
    question_id: np.ndarray, costs: np.ndarray, n_bootstrap: int = 1000, seed: int = 0
) -> tuple[float, float]:
    """95% CI on the mean realized cost, resampling whole question_id clusters
    with replacement (MT-Bench pairs sharing a question_id are not
    independent - same justification judge_characterization.py already uses
    for its AUROC CIs)."""
    unique_q = np.unique(question_id)
    q_to_idx = {q: np.where(question_id == q)[0] for q in unique_q}
    rng = np.random.default_rng(seed)
    draws = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        drawn = rng.choice(unique_q, size=len(unique_q), replace=True)
        idx = np.concatenate([q_to_idx[q] for q in drawn])
        draws[i] = costs[idx].mean()
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return float(lo), float(hi)


def cluster_bootstrap_paired_cost_diff_ci(
    question_id: np.ndarray, costs_a: np.ndarray, costs_b: np.ndarray, n_bootstrap: int = 1000, seed: int = 0
) -> tuple[float, float, float]:
    """95% CI on mean(costs_a) - mean(costs_b), resampling the SAME drawn
    clusters for both cost arrays each draw (paired, not two independent
    marginal CIs - the two policies are evaluated on the same examples, so
    their costs are correlated; a marginal-CI overlap check would understate
    how tight the difference actually is)."""
    unique_q = np.unique(question_id)
    q_to_idx = {q: np.where(question_id == q)[0] for q in unique_q}
    rng = np.random.default_rng(seed)
    point = costs_a.mean() - costs_b.mean()
    draws = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        drawn = rng.choice(unique_q, size=len(unique_q), replace=True)
        idx = np.concatenate([q_to_idx[q] for q in drawn])
        draws[i] = costs_a[idx].mean() - costs_b[idx].mean()
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return float(point), float(lo), float(hi)


def main() -> None:
    tau_hi, tau_lo = cost_sensitive_thresholds(
        c_execute_incorrect=C_EXECUTE_INCORRECT,
        c_verify_correct=C_VERIFY_CORRECT,
        c_verify_incorrect=C_VERIFY_INCORRECT,
        c_hitl=C_HITL,
    )
    print(f"cost regime: c_E={C_EXECUTE_INCORRECT} c_V={C_VERIFY_CORRECT} c_V'={C_VERIFY_INCORRECT} c_H={C_HITL}")
    print(f"thresholds: tau_lo={tau_lo:.4f}  tau_hi={tau_hi:.4f}")
    print(f"always-Execute cost / always-Verify cost, per config, printed below alongside routed cost")
    print()

    configs = load_all_judge_configs()
    header = f"{'Config':<32}{'Score':<14}{'Coverage':>9}{'Verify':>8}{'HITL':>6}{'ExecErr':>9}{'MeanCost':>10}{'95% CI':>18}"
    print(header)
    print("-" * len(header))
    for cfg in configs:
        correct = cfg["correct"].numpy()
        qid = cfg["question_id"]
        msp = cfg["phi"][:, 0].numpy()
        comb_scores = cfg["combiner"].score(cfg["phi"]).numpy()

        rows = [
            route_summary("MSP", msp, correct, tau_hi, tau_lo, qid),
            route_summary("combiner", comb_scores, correct, tau_hi, tau_lo, qid),
        ]
        for r in rows:
            lo, hi = r["cost_ci"]
            print(f"{cfg['name']:<32}{r['name']:<14}{r['coverage']:>9.2%}{r['n_verify']:>8}{r['n_hitl']:>6}"
                  f"{r['exec_error_rate']:>9.2%}{r['mean_cost']:>10.3f}   [{lo:.3f}, {hi:.3f}]")

        exec_costs = np.where(correct, 0.0, C_EXECUTE_INCORRECT)
        verify_costs = np.where(correct, C_VERIFY_CORRECT, C_VERIFY_INCORRECT)
        exec_lo, exec_hi = cluster_bootstrap_mean_cost_ci(qid, exec_costs)
        verify_lo, verify_hi = cluster_bootstrap_mean_cost_ci(qid, verify_costs)
        print(f"{cfg['name']:<32}{'always-Execute':<14}{'100.00%':>9}{0:>8}{0:>6}"
              f"{(~correct).mean():>9.2%}{always_execute_cost(correct):>10.3f}   [{exec_lo:.3f}, {exec_hi:.3f}]")
        print(f"{cfg['name']:<32}{'always-Verify':<14}{'0.00%':>9}{len(correct):>8}{0:>6}"
              f"{'n/a':>9}{always_verify_cost(correct):>10.3f}   [{verify_lo:.3f}, {verify_hi:.3f}]")

        combiner_costs = rows[1]["costs"]
        d_point, d_lo, d_hi = cluster_bootstrap_paired_cost_diff_ci(qid, combiner_costs, verify_costs)
        print(f"    paired diff (combiner - always-Verify): {d_point:+.3f}  95% CI [{d_lo:+.3f}, {d_hi:+.3f}]"
              f"  {'(excludes 0)' if d_lo > 0 or d_hi < 0 else '(includes 0, n.s.)'}")
        print()


if __name__ == "__main__":
    main()
