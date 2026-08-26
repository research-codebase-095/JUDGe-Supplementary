"""Two reviewer checks on Section 5.1's cost model, both reusing
scripts/selective_judging.py's existing helpers (route_summary,
cluster_bootstrap_*, always_execute_cost/always_verify_cost) with no
reimplementation:

Task A (utility term): the original cost functions have no reward for a
CORRECT automatic execution (f_E(p)=(1-p)*c_execute_incorrect), which
structurally biases toward conservatism - Verify/HITL can never look worse
than a "free" correct execution. src/deployment_reliability/router.py's
cost_sensitive_thresholds now accepts an optional u_execute_correct
(default 0.0, backward compatible): f_E(p) = (1-p)*c_execute_incorrect -
p*u_execute_correct. Reruns the full MSP/combiner/always-execute/
always-verify comparison at u_execute_correct=1.0 (same unit scale as
c_verify_correct=1.0) to check whether the qualitative finding survives.

Task B (cost-regime sweep): sweeps c_execute_incorrect in {5,7.5,10,15,20}
and c_hitl in {1,1.5,2,3,4} (c_verify_correct=1, c_verify_incorrect=3 held
fixed, skipping any (c_E,c_H) violating 0<=c_V<=c_H<=c_V'<c_E), and for each
combination + judge config reports whether the paired cluster-bootstrap CI
(combiner - always-verify) is significantly negative (combiner wins),
n.s., or significantly positive (combiner loses).

Usage:
    python scripts/selective_judging_utility_and_sweep.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from deployment_reliability.router import cost_sensitive_thresholds  # noqa: E402

from judge_characterization import load_all_judge_configs  # noqa: E402
from selective_judging import (  # noqa: E402
    C_EXECUTE_INCORRECT, C_HITL, C_VERIFY_CORRECT, C_VERIFY_INCORRECT,
    always_execute_cost, always_verify_cost, cluster_bootstrap_paired_cost_diff_ci,
    route_summary,
)


def task_a_utility_term() -> None:
    print("=" * 100)
    print("TASK A: utility term for correct execution (u_execute_correct=1.0 vs. 0.0 baseline)")
    print("=" * 100)
    for u_exec in (0.0, 1.0):
        tau_hi, tau_lo = cost_sensitive_thresholds(
            c_execute_incorrect=C_EXECUTE_INCORRECT, c_verify_correct=C_VERIFY_CORRECT,
            c_verify_incorrect=C_VERIFY_INCORRECT, c_hitl=C_HITL, u_execute_correct=u_exec,
        )
        print(f"\n--- u_execute_correct={u_exec}  ->  tau_lo={tau_lo:.4f}  tau_hi={tau_hi:.4f} ---")
        configs = load_all_judge_configs()
        for cfg in configs:
            correct = cfg["correct"].numpy()
            qid = cfg["question_id"]
            msp = cfg["phi"][:, 0].numpy()
            comb_scores = cfg["combiner"].score(cfg["phi"]).numpy()

            msp_summ = route_summary("MSP", msp, correct, tau_hi, tau_lo, qid)
            comb_summ = route_summary("Combiner", comb_scores, correct, tau_hi, tau_lo, qid)
            verify_costs = np.where(correct, C_VERIFY_CORRECT, C_VERIFY_INCORRECT)
            d, lo, hi = cluster_bootstrap_paired_cost_diff_ci(qid, comb_summ["costs"], verify_costs)
            tag = "(excludes 0)" if lo > 0 or hi < 0 else "(n.s.)"
            print(f"{cfg['name']:<32} MSP cov={msp_summ['coverage']:.1%} cost={msp_summ['mean_cost']:.3f}  "
                  f"Combiner cov={comb_summ['coverage']:.1%} cost={comb_summ['mean_cost']:.3f}  "
                  f"AlwaysVerify cost={always_verify_cost(correct):.3f}  "
                  f"paired diff(comb-verify)={d:+.3f} CI[{lo:+.3f},{hi:+.3f}] {tag}")


def task_b_cost_sweep() -> None:
    print()
    print("=" * 100)
    print("TASK B: cost-regime sweep, combiner vs always-Verify paired diff significance")
    print("=" * 100)
    c_e_grid = [5, 7.5, 10, 15, 20]
    c_h_grid = [1, 1.5, 2, 3, 4]
    c_v, c_v_prime = 1.0, 3.0

    configs = load_all_judge_configs()
    for cfg in configs:
        correct = cfg["correct"].numpy()
        qid = cfg["question_id"]
        comb_scores = cfg["combiner"].score(cfg["phi"]).numpy()
        print(f"\n--- {cfg['name']} ---")
        header = f"{'c_E \\\\ c_H':<10}" + "".join(f"{h:>8}" for h in c_h_grid)
        print(header)
        n_win = n_ns = n_lose = n_skip = 0
        for c_e in c_e_grid:
            row = f"{c_e:<10}"
            for c_h in c_h_grid:
                if not (0 <= c_v <= c_h <= c_v_prime < c_e):
                    row += f"{'--':>8}"
                    n_skip += 1
                    continue
                tau_hi, tau_lo = cost_sensitive_thresholds(
                    c_execute_incorrect=c_e, c_verify_correct=c_v, c_verify_incorrect=c_v_prime, c_hitl=c_h,
                )
                comb_summ = route_summary("Combiner", comb_scores, correct, tau_hi, tau_lo, qid)
                verify_costs = np.where(correct, c_v, c_v_prime)
                d, lo, hi = cluster_bootstrap_paired_cost_diff_ci(qid, comb_summ["costs"], verify_costs, n_bootstrap=300)
                if hi < 0:
                    sym, n_win = "WIN", n_win + 1
                elif lo > 0:
                    sym, n_lose = "LOSE", n_lose + 1
                else:
                    sym, n_ns = "ns", n_ns + 1
                row += f"{sym:>8}"
            print(row)
        total_valid = n_win + n_ns + n_lose
        print(f"summary: combiner beats always-Verify (WIN) in {n_win}/{total_valid} valid cells, "
              f"n.s. in {n_ns}/{total_valid}, loses (LOSE) in {n_lose}/{total_valid} "
              f"({n_skip} cells skipped, invalid cost ordering)")


def main() -> None:
    task_a_utility_term()
    task_b_cost_sweep()


if __name__ == "__main__":
    main()
