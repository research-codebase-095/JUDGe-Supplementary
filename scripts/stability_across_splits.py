"""Answers a reviewer question about judge2026.tex: since every reported
number comes from ONE fit/calibrate/test split (seed=0), and the LLM-backbone
result was shown to be sensitive to a fixable modeling choice
(standardization, paper_diagnostics.py / the paper's Discussion), how much
does the combiner-vs-MSP AUROC gap itself move if the split is redrawn?

Pools each cache's phi/correct across ALL of its original combiner_fit /
threshold_cal / id_test rows (the original split labels are discarded here,
not reused) and redraws five fresh three_way_split(seed=0..4) partitions,
refitting the combiner from scratch each time and evaluating unrefit on that
draw's id_test. Reports mean +/- std of (combiner AUROC - MSP AUROC) across
the five draws, for both the raw linear combiner and the standardized one
diagnosed in the paper as closing most of the LLM gap.

No new model inference - reuses cached phi/correct/logits already collected
for the paper's own Table 1 and Table 2.

Usage:
    python scripts/stability_across_splits.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from deployment_reliability.combiner import LogisticRegressionCombiner  # noqa: E402
from deployment_reliability.router import auroc  # noqa: E402
from deployment_reliability.splits import three_way_split  # noqa: E402

DATA_DIR = os.path.join(REPO_ROOT, "data")
SEEDS = (0, 1, 2, 3, 4)


def load_pool(cache_filename: str) -> tuple[torch.Tensor, torch.Tensor] | None:
    path = os.path.join(DATA_DIR, cache_filename)
    if not os.path.exists(path):
        return None
    cache = torch.load(path)
    return cache["phi"], cache["correct"]


def evaluate_one_seed(phi: torch.Tensor, correct: torch.Tensor, seed: int) -> tuple[float, float]:
    """Returns (linear_combiner_auroc - msp_auroc, standardized_combiner_auroc - msp_auroc) for this seed's draw."""
    n = len(correct)
    fit_idx, _cal_idx, test_idx = three_way_split(n, seed=seed)
    phi_fit, correct_fit = phi[fit_idx], correct[fit_idx].float()
    phi_test, correct_test = phi[test_idx], correct[test_idx].bool()

    msp_test = phi_test[:, 0]
    msp_auroc = auroc(msp_test[correct_test], msp_test[~correct_test])

    lin = LogisticRegressionCombiner().fit(phi_fit, correct_fit)
    lin_scores = lin.score(phi_test)
    lin_auroc = auroc(lin_scores[correct_test], lin_scores[~correct_test])

    mean = phi_fit.mean(dim=0)
    std = phi_fit.std(dim=0).clamp_min(1e-8)
    phi_fit_std = (phi_fit - mean) / std
    phi_test_std = (phi_test - mean) / std
    std_comb = LogisticRegressionCombiner().fit(phi_fit_std, correct_fit)
    std_scores = std_comb.score(phi_test_std)
    std_auroc = auroc(std_scores[correct_test], std_scores[~correct_test])

    return lin_auroc - msp_auroc, std_auroc - msp_auroc


def main() -> None:
    configs = [
        ("llm_feature_cache_gpt2.pt", "GPT-2 (LLM)"),
        ("llm_feature_cache_pythia160m.pt", "Pythia-160m (LLM)"),
        ("llm_feature_cache_qwen05binstruct.pt", "Qwen2.5-0.5B-Instr. (LLM)"),
        ("judge_feature_cache_mtbench.pt", "Qwen2.5-0.5B (judge)"),
        ("judge_feature_cache_mtbench_1p5b.pt", "Qwen2.5-1.5B (judge)"),
        ("judge_feature_cache_mtbench_smollm2_360m.pt", "SmolLM2-360M (judge)"),
    ]

    header = f"{'Config':<28}{'n':>8}{'lin.comb-MSP mean':>19}{'std':>8}{'std.comb-MSP mean':>19}{'std':>8}"
    print(header)
    print("-" * len(header))
    for filename, name in configs:
        loaded = load_pool(filename)
        if loaded is None:
            print(f"[skip] {name}: {filename} not found")
            continue
        phi, correct = loaded
        lin_diffs, std_diffs = [], []
        for seed in SEEDS:
            lin_d, std_d = evaluate_one_seed(phi, correct, seed)
            lin_diffs.append(lin_d)
            std_diffs.append(std_d)
        lin_diffs, std_diffs = np.array(lin_diffs), np.array(std_diffs)
        print(f"{name:<28}{len(correct):>8}{lin_diffs.mean():>19.4f}{lin_diffs.std():>8.4f}"
              f"{std_diffs.mean():>19.4f}{std_diffs.std():>8.4f}")
        print(f"    per-seed (linear):      " + ", ".join(f"{d:+.4f}" for d in lin_diffs))
        print(f"    per-seed (standardized):" + ", ".join(f"{d:+.4f}" for d in std_diffs))


if __name__ == "__main__":
    main()
