"""Reviewer check on Section 5.1 (Selective Judging): does one/two-parameter
POST-HOC CALIBRATION of the same scalar MSP recover most of the combiner's
routing benefit, making the real story "calibrate your scalar" rather than
"use a vector"? src/deployment_reliability/calibration.py already implements
TemperatureScaling and PlattScaling (both fully working, unused anywhere in
the paper before this script). Both are fit on each judge config's own
threshold_cal split (128 examples, Qwen configs; 40, SmolLM2) - the same
split already used to fix tau_hi/tau_lo, so no extra data or extra held-out
set is invented for this check - then applied to MSP on id_test and routed
through the identical cost_sensitive_thresholds()/route() pipeline as the
existing MSP/combiner rows in Table 4.

Reuses scripts/selective_judging.py's route_summary/always_execute_cost/
always_verify_cost/cluster_bootstrap_* helpers directly - no reimplementation.

Usage:
    python scripts/selective_judging_calibrated_msp.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from deployment_reliability.calibration import PlattScaling, TemperatureScaling  # noqa: E402
from deployment_reliability.router import cost_sensitive_thresholds  # noqa: E402

from judge_characterization import load_judge_config  # noqa: E402
from selective_judging import (  # noqa: E402
    C_EXECUTE_INCORRECT, C_HITL, C_VERIFY_CORRECT, C_VERIFY_INCORRECT,
    always_execute_cost, always_verify_cost, cluster_bootstrap_mean_cost_ci,
    cluster_bootstrap_paired_cost_diff_ci, route_summary,
)

# load_judge_config (judge_characterization.py) only keeps combiner_fit/id_test
# rows; we need threshold_cal too to fit the calibrators, so reload the raw
# cache here rather than extend that function's contract for every caller.
DATA_DIR = os.path.join(REPO_ROOT, "data")

CACHE_FILES = [
    ("judge_feature_cache_mtbench.pt", "Qwen2.5-0.5B-Instruct (judge)"),
    ("judge_feature_cache_mtbench_1p5b.pt", "Qwen2.5-1.5B-Instruct (judge)"),
    ("judge_feature_cache_mtbench_smollm2_360m.pt", "SmolLM2-360M-Instruct (judge, n=400 subset)"),
]


def load_full_config(cache_filename: str, display_name: str) -> dict | None:
    path = os.path.join(DATA_DIR, cache_filename)
    if not os.path.exists(path):
        print(f"[skip] {display_name}: {cache_filename} not found")
        return None
    cache = torch.load(path)
    splits = np.array(cache["splits"])
    m_fit = torch.from_numpy(splits == "combiner_fit")
    m_cal = torch.from_numpy(splits == "threshold_cal")
    m_test = torch.from_numpy(splits == "id_test")
    from deployment_reliability.combiner import LogisticRegressionCombiner

    phi, correct = cache["phi"], cache["correct"]
    combiner = LogisticRegressionCombiner().fit(phi[m_fit], correct[m_fit].float())
    return {
        "name": display_name,
        "msp_cal": phi[m_cal][:, 0], "correct_cal": correct[m_cal].bool(),
        "msp_test": phi[m_test][:, 0], "correct_test": correct[m_test].bool(),
        "combiner_scores_test": combiner.score(phi[m_test]),
        "question_id_test": np.array(cache["question_id"])[m_test.numpy()],
    }


def main() -> None:
    tau_hi, tau_lo = cost_sensitive_thresholds(
        c_execute_incorrect=C_EXECUTE_INCORRECT, c_verify_correct=C_VERIFY_CORRECT,
        c_verify_incorrect=C_VERIFY_INCORRECT, c_hitl=C_HITL,
    )
    print(f"cost regime: c_E={C_EXECUTE_INCORRECT} c_V={C_VERIFY_CORRECT} c_V'={C_VERIFY_INCORRECT} c_H={C_HITL}")
    print(f"thresholds: tau_lo={tau_lo:.4f}  tau_hi={tau_hi:.4f}")
    print()

    header = (f"{'Config':<32}{'Score':<16}{'Coverage':>9}{'Verify':>8}{'HITL':>6}"
              f"{'ExecErr':>9}{'MeanCost':>10}{'95% CI':>18}")
    print(header)
    print("-" * len(header))

    for filename, name in CACHE_FILES:
        cfg = load_full_config(filename, name)
        if cfg is None:
            continue

        correct_test = cfg["correct_test"].numpy()
        qid = cfg["question_id_test"]
        msp_cal_np = cfg["msp_cal"].numpy()
        correct_cal_np = cfg["correct_cal"].numpy()

        temp = TemperatureScaling().fit(cfg["msp_cal"], cfg["correct_cal"].float())
        msp_temp_test = temp.transform(cfg["msp_test"]).numpy()

        platt = PlattScaling().fit(cfg["msp_cal"], cfg["correct_cal"].float())
        msp_platt_test = platt.transform(cfg["msp_test"]).numpy()

        msp_raw_test = cfg["msp_test"].numpy()
        combiner_test = cfg["combiner_scores_test"].numpy()

        rows = [
            ("MSP (raw)", msp_raw_test),
            ("MSP (temp-scaled)", msp_temp_test),
            ("MSP (Platt-scaled)", msp_platt_test),
            ("Combiner", combiner_test),
        ]

        print(f"--- {name} ---  (fit temp/Platt on threshold_cal: n={len(correct_cal_np)}, "
              f"cal accuracy={correct_cal_np.mean():.3f})")
        summaries = {}
        for row_name, scores in rows:
            summ = route_summary(row_name, scores, correct_test, tau_hi, tau_lo, qid)
            summaries[row_name] = summ
            lo, hi = summ["cost_ci"]
            print(f"{name:<32}{row_name:<16}{summ['coverage']:>9.2%}{summ['n_verify']:>8}{summ['n_hitl']:>6}"
                  f"{summ['exec_error_rate']:>9.2%}{summ['mean_cost']:>10.3f}   [{lo:.3f}, {hi:.3f}]")

        exec_costs = np.where(correct_test, 0.0, C_EXECUTE_INCORRECT)
        verify_costs = np.where(correct_test, C_VERIFY_CORRECT, C_VERIFY_INCORRECT)
        exec_lo, exec_hi = cluster_bootstrap_mean_cost_ci(qid, exec_costs)
        verify_lo, verify_hi = cluster_bootstrap_mean_cost_ci(qid, verify_costs)
        print(f"{name:<32}{'Always-Execute':<16}{'100.00%':>9}{0:>8}{0:>6}"
              f"{(~correct_test).mean():>9.2%}{always_execute_cost(correct_test):>10.3f}   [{exec_lo:.3f}, {exec_hi:.3f}]")
        print(f"{name:<32}{'Always-Verify':<16}{'0.00%':>9}{len(correct_test):>8}{0:>6}"
              f"{'n/a':>9}{always_verify_cost(correct_test):>10.3f}   [{verify_lo:.3f}, {verify_hi:.3f}]")

        for row_name in ("MSP (temp-scaled)", "MSP (Platt-scaled)"):
            costs_row = summaries[row_name]["costs"]
            d_v, lo_v, hi_v = cluster_bootstrap_paired_cost_diff_ci(qid, costs_row, verify_costs)
            d_c, lo_c, hi_c = cluster_bootstrap_paired_cost_diff_ci(qid, costs_row, summaries["Combiner"]["costs"])
            tag_v = "(excludes 0)" if lo_v > 0 or hi_v < 0 else "(includes 0, n.s.)"
            tag_c = "(excludes 0)" if lo_c > 0 or hi_c < 0 else "(includes 0, n.s.)"
            print(f"    {row_name} - Always-Verify: {d_v:+.3f}  95% CI [{lo_v:+.3f}, {hi_v:+.3f}]  {tag_v}")
            print(f"    {row_name} - Combiner:      {d_c:+.3f}  95% CI [{lo_c:+.3f}, {hi_c:+.3f}]  {tag_c}")
        print()


if __name__ == "__main__":
    main()
