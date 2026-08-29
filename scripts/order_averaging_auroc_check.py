"""Reviewer follow-up: does the Table 5 / Appendix A.6 two-pass order-averaging
intervention (0.5B judge) improve AUROC (correctness discrimination), not just
accuracy at threshold 0.5?

The Appendix A.6 "Order-averaging (0.5B judge)" paragraph reports only the
accuracy improvement (50.9% -> 64.0%) plus a McNemar test on the discordant
pairs. This script adds the AUROC side of that question.

Reads data/judge_swap_consistency_cache_continuous.pt only -- no new model
inference. Reuses order_averaging_continuous_analysis.py's p_model_a() logic
(not modified, just re-derived inline the same way) and imports
_rank_based_auroc / delong_test from deployment_reliability.significance
rather than reimplementing AUROC or a significance test.

Definitions (matching the paper's existing selective-judging convention of
using confidence to discriminate correct vs incorrect predictions):
  - single-order confidence  = phi_orig[:,0]           (MSP of predicted winner)
  - single-order label       = correct_orig             (pred_orig == human)
  - two-pass confidence      = max(p_a_avg, 1-p_a_avg)   (confidence of the averaged rule)
  - two-pass label           = correct_avg               (pred_avg == human)

Because the two rules produce DIFFERENT per-example correctness labels
(predictions differ), this is NOT a valid DeLong pairing (DeLong requires the
SAME binary outcome for both scores). So: report each AUROC's own cluster
bootstrap CI (resampling question_id, matching this paper's existing
convention in order_averaging_continuous_analysis.py's accuracy CI and
paper_diagnostics.py's cluster_bootstrap_auroc_diff), plus a paired
cluster-bootstrap on the DIFFERENCE (AUROC_avg - AUROC_orig), which handles
the differing-label issue correctly since it never assumes a shared label
vector -- each bootstrap draw recomputes both AUROCs (each against its own
rule's labels) on the same resampled question clusters.

For reference, the script also reports a supplementary (technically valid)
DeLong test that grades both confidence scores against the ORIGINAL rule's
correctness label -- a well-posed but distinct question from the main
each-rule-against-its-own-labels comparison above.

Usage:
    python scripts/order_averaging_auroc_check.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data")
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from deployment_reliability.significance import _rank_based_auroc, delong_test  # noqa: E402

N_BOOTSTRAP = 2000
SEED = 0


def p_model_a(msp: np.ndarray, predicted_winner: np.ndarray) -> np.ndarray:
    return np.where(predicted_winner == "model_a", msp, 1.0 - msp)


def main() -> None:
    d = torch.load(os.path.join(DATA_DIR, "judge_swap_consistency_cache_continuous.pt"), weights_only=False)
    splits = np.array(d["splits"])
    m_test = splits == "id_test"
    n = int(m_test.sum())
    print(f"n id_test = {n}")

    phi_orig = d["phi_orig"][m_test]
    human = np.array(d["human_winner"])[m_test]
    question_id = np.array(d["question_id"])[m_test]
    pred_orig = np.array(d["predicted_winner_orig"])[m_test]
    phi_swapped = d["phi_swapped"][m_test]
    pred_swapped = np.array(d["predicted_winner_swapped"])[m_test]

    p_a_orig = p_model_a(phi_orig[:, 0].numpy(), pred_orig)
    p_a_swapped = p_model_a(phi_swapped[:, 0].numpy(), pred_swapped)
    p_a_avg = (p_a_orig + p_a_swapped) / 2.0
    pred_avg = np.where(p_a_avg >= 0.5, "model_a", "model_b")

    correct_orig = (pred_orig == human)
    correct_avg = (pred_avg == human)

    conf_orig = phi_orig[:, 0].numpy()  # MSP of the predicted winner, single-order
    conf_avg = np.maximum(p_a_avg, 1.0 - p_a_avg)  # confidence of the averaged rule

    def auroc_of(conf: np.ndarray, correct: np.ndarray) -> float:
        pos, neg = conf[correct], conf[~correct]
        return _rank_based_auroc(pos, neg)

    auroc_orig = auroc_of(conf_orig, correct_orig)
    auroc_avg = auroc_of(conf_avg, correct_avg)

    print(f"single-order accuracy:      {correct_orig.mean()*100:.1f}%")
    print(f"two-pass avg accuracy:      {correct_avg.mean()*100:.1f}%")
    print(f"single-order MSP AUROC (confidence discriminating correct_orig):  {auroc_orig:.4f}")
    print(f"two-pass averaged-confidence AUROC (discriminating correct_avg): {auroc_avg:.4f}")
    print(f"raw delta (avg - orig): {auroc_avg - auroc_orig:+.4f}")

    # Cluster bootstrap: per-AUROC CI, plus paired difference distribution.
    unique_q = np.unique(question_id)
    q_to_idx = {q: np.where(question_id == q)[0] for q in unique_q}
    rng = np.random.default_rng(SEED)
    draws_orig = np.empty(N_BOOTSTRAP)
    draws_avg = np.empty(N_BOOTSTRAP)
    for i in range(N_BOOTSTRAP):
        drawn = rng.choice(unique_q, size=len(unique_q), replace=True)
        idx = np.concatenate([q_to_idx[q] for q in drawn])
        c_o, c_a = correct_orig[idx], correct_avg[idx]
        co_orig, co_avg = conf_orig[idx], conf_avg[idx]
        if c_o.sum() < 2 or (~c_o).sum() < 2:
            draws_orig[i] = np.nan
        else:
            draws_orig[i] = _rank_based_auroc(co_orig[c_o], co_orig[~c_o])
        if c_a.sum() < 2 or (~c_a).sum() < 2:
            draws_avg[i] = np.nan
        else:
            draws_avg[i] = _rank_based_auroc(co_avg[c_a], co_avg[~c_a])

    diff = draws_avg - draws_orig
    valid = ~np.isnan(diff)
    ci_lo_o, ci_hi_o = np.nanquantile(draws_orig, [0.025, 0.975])
    ci_lo_a, ci_hi_a = np.nanquantile(draws_avg, [0.025, 0.975])
    ci_lo_d, ci_hi_d = np.nanquantile(diff[valid], [0.025, 0.975])
    frac_pos = float((diff[valid] > 0).mean())

    print(f"single-order AUROC 95% CI (cluster bootstrap, n={N_BOOTSTRAP}): [{ci_lo_o:.4f}, {ci_hi_o:.4f}]")
    print(f"two-pass avg AUROC 95% CI (cluster bootstrap, n={N_BOOTSTRAP}):  [{ci_lo_a:.4f}, {ci_hi_a:.4f}]")
    print(f"paired diff (avg-orig) 95% CI: [{ci_lo_d:.4f}, {ci_hi_d:.4f}]   frac(diff>0)={frac_pos:.3f}")

    # For reference/context, also report what a (technically-mismatched)
    # DeLong test would say if one insisted on using correct_orig as the
    # shared label for both scores anyway (i.e., "does the averaged
    # confidence score, still graded against the ORIGINAL rule's correctness,
    # discriminate better than the original MSP?") -- a well-posed DeLong
    # question in its own right, distinct from the "each rule graded against
    # its own predictions" comparison above.
    dl = delong_test(torch.from_numpy(correct_orig), torch.from_numpy(conf_avg), torch.from_numpy(conf_orig))
    auroc_avg_vs_correct_orig = _rank_based_auroc(conf_avg[correct_orig], conf_avg[~correct_orig])
    print()
    print("[supplementary framing] grading BOTH scores against correct_orig (valid DeLong pairing):")
    print(f"  AUROC(conf_avg | correct_orig)  = {auroc_avg_vs_correct_orig:.4f}")
    print(f"  AUROC(conf_orig | correct_orig) = {auroc_orig:.4f}")
    print(f"  DeLong z={dl.z:.3f} p={dl.p_value:.3e}")


if __name__ == "__main__":
    main()
