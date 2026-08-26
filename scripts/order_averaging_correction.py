"""Order-averaging as a corrective for position bias at the 0.5B judge
configuration, using ONLY data already cached from
scripts/collect_judge_swap_consistency.py's prior run - no new model
inference.

IMPORTANT LIMITATION, discovered by inspecting the cache (not assumed): that
prior run computed full verdict-token logits for BOTH the original-order and
swapped-order passes (see collect_judge_swap_consistency.py's main loop,
`logits_orig` and `logits_swap`), but only featurized and saved `phi_orig`
(the original-order 5-feature vector) to data/judge_swap_consistency_cache.pt.
The swapped-order raw logits/MSP score were never saved - only the resulting
BINARY verdict (`predicted_winner_swapped`) survived. So a genuine
score-averaging order-correction (averaging two continuous MSP-style
probabilities) is NOT reconstructable from this cache without rerunning
inference, which is out of scope here. This script therefore evaluates the
only order-averaging variant actually computable from cached data: a
binary-verdict agreement rule, plus two explicit "always trust order X"
degenerate baselines and a symmetric coin-flip expectation, so the reader can
see exactly what a majority/agreement-based order-average can and cannot
achieve when the underlying two passes give only binary output.

Usage:
    python scripts/order_averaging_correction.py
"""

from __future__ import annotations

import os

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data")


def main() -> None:
    d = torch.load(os.path.join(DATA_DIR, "judge_swap_consistency_cache.pt"))
    n = len(d["human_winner"])
    splits = np.array(d["splits"])
    human = np.array(d["human_winner"])
    pred_orig = np.array(d["predicted_winner_orig"])
    pred_swap = np.array(d["predicted_winner_swapped"])
    swap_consistent = np.array(d["swap_consistent"]) > 0.5

    print(f"n={n} total pairs; swap-consistent (pred_orig == pred_swap): "
          f"{swap_consistent.sum()}/{n} = {swap_consistent.mean()*100:.2f}%")
    print(f"(i.e. the two passes DISAGREE on {n - swap_consistent.sum()}/{n} = "
          f"{(1-swap_consistent.mean())*100:.2f}% of pairs -- this is the population an "
          f"order-averaging tie-break rule actually has to resolve)")
    print()

    # Agreement-rule order-average: use the shared verdict when both passes
    # agree; when they disagree, no principled score-based tie-break exists
    # from this cache (see module docstring), so report three explicit
    # alternatives rather than silently picking one.
    correct_orig = (pred_orig == human)
    correct_swap = (pred_swap == human)

    def report(name: str, correct: np.ndarray, mask: np.ndarray | None = None):
        if mask is None:
            mask = np.ones(n, dtype=bool)
        acc = correct[mask].mean()
        acc_id_test = correct[mask & (splits == "id_test")].mean() if (mask & (splits == "id_test")).any() else float("nan")
        print(f"{name}: full-pool acc={acc:.4f} (n={mask.sum()}), id_test-only acc={acc_id_test:.4f} "
              f"(n={(mask & (splits=='id_test')).sum()})")

    print("=== Baselines already in the paper, recomputed here for direct comparison ===")
    report("Original single-order (unswapped) accuracy", correct_orig)
    always_a_baseline_acc = (human == "model_a").mean()
    print(f"Always-position-following (always guess model_a) baseline: {always_a_baseline_acc:.4f} "
          f"(this is the human label's own model_a base rate)")
    print()

    print("=== Order-averaging variants computable from this cache ===")
    print("Variant 1: on agreement (1.1% of pairs), use the shared verdict; "
          "on disagreement (98.9%), fall back to the ORIGINAL-order verdict (i.e. order-averaging degenerates to pred_orig almost everywhere):")
    report("  -> equals original single-order accuracy by construction on 98.9% of pairs", correct_orig)

    print("Variant 2: on agreement, use shared verdict; on disagreement, fall back to the SWAPPED-order verdict:")
    variant2_correct = np.where(swap_consistent, correct_orig, correct_swap)
    report("  Variant 2 (fallback=swapped)", variant2_correct)

    print("Variant 3: on agreement, use shared verdict; on disagreement (no score to break the tie), "
          "treat as an explicit 50/50 coin flip -- expected accuracy computed in closed form, not simulated:")
    # On agreement, accuracy = correct_orig (== correct_swap) restricted to swap_consistent.
    # On disagreement, a symmetric coin flip between "guess original response_a" and
    # "guess original response_b" gets the human's actual pick with probability exactly 0.5,
    # regardless of the human label's own base rate, since the coin is symmetric between the
    # two identities and is independent of both passes' (both-wrong-in-a-biased-way) predictions.
    agree_acc_contribution = correct_orig[swap_consistent].sum()
    disagree_n = (~swap_consistent).sum()
    expected_coinflip_correct = 0.5 * disagree_n
    expected_total_acc = (agree_acc_contribution + expected_coinflip_correct) / n
    print(f"  Variant 3 (agreement + symmetric coin-flip on disagreement): expected accuracy = {expected_total_acc:.4f} "
          f"(closed-form: ({agree_acc_contribution:.0f} agreement-correct + {expected_coinflip_correct:.1f} expected coin-flip-correct) / {n})")
    print()

    print("=== Interpretation ===")
    print(f"Because swap-consistency is only {swap_consistent.mean()*100:.2f}%, an agreement-based order-average has almost")
    print("no population to average over: on 98.9% of pairs the two passes disagree and there is no raw score in this")
    print("cache to break the tie with, so the 'order-averaged' result is either (a) identical to the original single-order")
    print("accuracy (if you fall back to the original order), (b) identical to the swapped-order accuracy (if you fall back")
    print("the other way), or (c) a coin-flip near 50% (if you refuse to guess). None of these constitutes a real recovery")
    print("of signal beyond the single-order judge's own near-constant-output baseline -- order-averaging over BINARY")
    print("verdicts from a judge that ignores content in both orders just averages two near-random signals, exactly as")
    print("anticipated as a real possibility rather than assumed to fail. A genuine order-averaging correction would need")
    print("the swapped-pass MSP score (continuous), which was computed during the original run but not persisted to this")
    print("cache, and would require rerunning inference to obtain -- out of scope for this existing-data-only check.")


if __name__ == "__main__":
    main()
