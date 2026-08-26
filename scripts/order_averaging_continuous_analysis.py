"""Table 5's continuous order-averaging result (0.5B judge, id_test):
scripts/collect_judge_swap_consistency_continuous.py collects phi_orig and
phi_swapped (both full 5-feature vectors, including MSP) for every one of the
1,284 MT-Bench pairs, but that collection script only ever persists the
tensors -- it does not itself compute the two-pass order-averaged accuracy
Table 5 reports. This script closes that gap: it is the analysis half of
that collection, reading data/judge_swap_consistency_cache_continuous.pt only
(no new model inference).

Method: each pass's MSP score is the model's confidence in ITS OWN predicted
winner, not P(model_a) directly, so it must first be re-expressed as
P(model_a) using that pass's predicted_winner label (P(model_a) = MSP if the
pass predicted model_a, else 1 - MSP). Averaging the two passes' P(model_a)
estimates and thresholding at 0.5 gives the two-pass order-averaged verdict;
this is the genuine continuous-score counterpart to
order_averaging_correction.py's binary-only variants, made possible because
this cache (unlike judge_swap_consistency_cache.pt) also kept phi_swapped.

Usage:
    python scripts/order_averaging_continuous_analysis.py
"""

from __future__ import annotations

import os

import numpy as np
import torch
from scipy.stats import binomtest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data")

N_BOOTSTRAP = 1000
SEED = 0


def p_model_a(msp: np.ndarray, predicted_winner: np.ndarray) -> np.ndarray:
    """Re-express a pass's own-prediction MSP as P(model_a)."""
    return np.where(predicted_winner == "model_a", msp, 1.0 - msp)


def cluster_bootstrap_acc_ci(correct: np.ndarray, question_id: np.ndarray, n_bootstrap: int = N_BOOTSTRAP, seed: int = SEED):
    unique_q = np.unique(question_id)
    q_to_idx = {q: np.where(question_id == q)[0] for q in unique_q}
    rng = np.random.default_rng(seed)
    draws = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        drawn = rng.choice(unique_q, size=len(unique_q), replace=True)
        idx = np.concatenate([q_to_idx[q] for q in drawn])
        draws[i] = correct[idx].mean()
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return float(lo), float(hi)


def main() -> None:
    d = torch.load(os.path.join(DATA_DIR, "judge_swap_consistency_cache_continuous.pt"))
    splits = np.array(d["splits"])
    m_test = splits == "id_test"
    print(f"n id_test = {int(m_test.sum())}")

    phi_orig = d["phi_orig"][m_test]
    phi_swapped = d["phi_swapped"][m_test]
    human = np.array(d["human_winner"])[m_test]
    question_id = np.array(d["question_id"])[m_test]
    pred_orig = np.array(d["predicted_winner_orig"])[m_test]
    pred_swapped = np.array(d["predicted_winner_swapped"])[m_test]

    p_a_orig = p_model_a(phi_orig[:, 0].numpy(), pred_orig)
    p_a_swapped = p_model_a(phi_swapped[:, 0].numpy(), pred_swapped)
    p_a_avg = (p_a_orig + p_a_swapped) / 2.0
    pred_avg = np.where(p_a_avg >= 0.5, "model_a", "model_b")

    correct_orig = (pred_orig == human)
    correct_avg = (pred_avg == human)

    acc_orig = correct_orig.mean()
    acc_avg = correct_avg.mean()
    ci_lo, ci_hi = cluster_bootstrap_acc_ci(correct_avg.astype(np.float64), question_id)

    # McNemar (paired): discordant pairs where original-order and order-averaged disagree on correctness.
    b = int((correct_orig & ~correct_avg).sum())
    c = int((~correct_orig & correct_avg).sum())
    mcnemar_p = binomtest(min(b, c), b + c, 0.5).pvalue if (b + c) > 0 else float("nan")

    print(f"single-order (unswapped) MSP accuracy:        {acc_orig*100:.1f}%")
    print(f"two-pass order-averaged MSP accuracy:          {acc_avg*100:.1f}%  "
          f"(cluster-bootstrap 95% CI [{ci_lo*100:.1f}%, {ci_hi*100:.1f}%])")
    print(f"McNemar (b={b}, c={c}): p={mcnemar_p:.2e}")
    print(f"delta: {(acc_avg-acc_orig)*100:+.1f} points")


if __name__ == "__main__":
    main()
