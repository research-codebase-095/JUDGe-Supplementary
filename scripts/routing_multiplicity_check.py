"""Reviewer follow-up: item (xi)'s Holm-Bonferroni correction covers only the
naive-DeLong combiner-vs-best-feature family (6 p-values). The routing-benefit
claims in Section 4.1 ("at 1.5B both combiner and Platt-scaled MSP beat
always-verify significantly") are a separate family -- 2 configs x 2 methods
vs always-verify, tested via cluster-bootstrap CI, not DeLong -- and were not
checked against multiplicity at all.

This script re-tests that family (4 comparisons) at a Bonferroni-adjusted
confidence level (alpha=0.05/4=0.0125, i.e. a 98.75% CI instead of 95%), with
a larger bootstrap (n=5000, same seed=0) for adequate tail resolution at that
level, and reports whether either of the two nominally-significant results
(1.5B combiner, 1.5B Platt-MSP, both vs always-verify) still excludes zero.

No new model inference -- reuses the same cached tensors and cost model as
scripts/selective_judging_calibrated_msp.py.

Usage:
    python scripts/routing_multiplicity_check.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from deployment_reliability.calibration import PlattScaling  # noqa: E402
from deployment_reliability.router import cost_sensitive_thresholds  # noqa: E402

from selective_judging import C_EXECUTE_INCORRECT, C_HITL, C_VERIFY_CORRECT, C_VERIFY_INCORRECT  # noqa: E402
from selective_judging_calibrated_msp import CACHE_FILES, load_full_config, route_summary  # noqa: E402

ALPHA = 0.05
FAMILY_SIZE = 4  # 2 configs x 2 methods (combiner, Platt-MSP) vs always-verify
ADJ_ALPHA = ALPHA / FAMILY_SIZE
N_BOOTSTRAP = 5000


def cluster_bootstrap_paired_cost_diff_ci_adj(question_id, costs_a, costs_b, n_bootstrap=N_BOOTSTRAP, seed=0):
    unique_q = np.unique(question_id)
    q_to_idx = {q: np.where(question_id == q)[0] for q in unique_q}
    rng = np.random.default_rng(seed)
    point = costs_a.mean() - costs_b.mean()
    draws = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        drawn = rng.choice(unique_q, size=len(unique_q), replace=True)
        idx = np.concatenate([q_to_idx[q] for q in drawn])
        draws[i] = costs_a[idx].mean() - costs_b[idx].mean()
    lo, hi = np.quantile(draws, [ADJ_ALPHA / 2.0, 1.0 - ADJ_ALPHA / 2.0])
    return float(point), float(lo), float(hi)


def main() -> None:
    tau_hi, tau_lo = cost_sensitive_thresholds(
        c_execute_incorrect=C_EXECUTE_INCORRECT, c_verify_correct=C_VERIFY_CORRECT,
        c_verify_incorrect=C_VERIFY_INCORRECT, c_hitl=C_HITL,
    )
    print(f"Family size {FAMILY_SIZE}, Bonferroni-adjusted alpha={ADJ_ALPHA:.4f} -> "
          f"{100*(1-ADJ_ALPHA):.2f}% CI, n_bootstrap={N_BOOTSTRAP}\n")

    for filename, name in CACHE_FILES:
        cfg = load_full_config(filename, name)
        if cfg is None or "SmolLM2" in name:
            continue
        correct_test = cfg["correct_test"].numpy()
        qid = cfg["question_id_test"]
        platt = PlattScaling().fit(cfg["msp_cal"], cfg["correct_cal"].float())
        msp_platt_test = platt.transform(cfg["msp_test"]).numpy()
        combiner_test = cfg["combiner_scores_test"].numpy()

        combiner_summ = route_summary("Combiner", combiner_test, correct_test, tau_hi, tau_lo, qid)
        platt_summ = route_summary("MSP (Platt-scaled)", msp_platt_test, correct_test, tau_hi, tau_lo, qid)
        verify_costs = np.where(correct_test, C_VERIFY_CORRECT, C_VERIFY_INCORRECT)

        for row_name, summ in (("Combiner", combiner_summ), ("MSP (Platt-scaled)", platt_summ)):
            d, lo, hi = cluster_bootstrap_paired_cost_diff_ci_adj(qid, summ["costs"], verify_costs)
            tag = "excludes 0 (survives correction)" if (lo > 0 or hi < 0) else "includes 0 (would NOT survive)"
            print(f"{name:<30} {row_name:<20} vs Always-Verify: {d:+.3f}  "
                  f"{100*(1-ADJ_ALPHA):.2f}% CI [{lo:+.3f}, {hi:+.3f}]  {tag}")
        print()


if __name__ == "__main__":
    main()
