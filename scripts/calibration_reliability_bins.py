"""Reviewer follow-up (GAP 2/3 deepening): judge2026.tex currently reports
only summary ECE/Brier numbers for MSP vs. the combiner (Appendix,
"Swap-consistency and calibration detail"). This script computes the
underlying 10-bin reliability-diagram data those summaries are computed from,
to check whether a single scalar (max bin gap) captures the calibration
story or whether a genuinely non-monotonic pattern would need a figure.

Uses the EXACT same bin_edges convention as scripts/paper_diagnostics.py's
ece() (torch.linspace(0, 1, n_bins+1), half-open bins except the last), and
recomputes ECE/Brier with that same logic as a sanity check against the
paper's already-reported summary numbers (ECE 0.35/0.32->0.02/0.05, Brier
0.37/0.34->0.24/0.23 at 0.5B/1.5B, MSP->combiner).

No new model inference - reuses cached phi/correct/combiner via
judge_characterization.load_all_judge_configs().

Usage:
    python scripts/calibration_reliability_bins.py
"""

from __future__ import annotations

import os
import sys

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from judge_characterization import load_all_judge_configs  # noqa: E402
from paper_diagnostics import brier_score, ece  # noqa: E402

N_BINS = 10
MIN_COUNT = 5
GAP_THRESHOLD = 0.1


def reliability_bins(scores: torch.Tensor, correct: torch.Tensor, n_bins: int = N_BINS) -> list[dict]:
    y = correct.float()
    bin_edges = torch.linspace(0, 1, n_bins + 1)
    bins = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (scores >= lo) & (scores < hi) if i < n_bins - 1 else (scores >= lo) & (scores <= hi)
        count = int(mask.sum().item())
        if count == 0:
            continue
        bins.append({
            "lo": float(lo), "hi": float(hi), "count": count,
            "mean_pred": float(scores[mask].mean().item()),
            "emp_acc": float(y[mask].mean().item()),
        })
    return bins


def compact_summary(bins: list[dict], min_count: int = MIN_COUNT, gap_threshold: float = GAP_THRESHOLD) -> dict:
    usable = [b for b in bins if b["count"] >= min_count]
    if not usable:
        return {"max_gap": float("nan"), "n_exceeding": 0, "n_bins_usable": 0}
    gaps = [abs(b["mean_pred"] - b["emp_acc"]) for b in usable]
    return {
        "max_gap": max(gaps),
        "n_exceeding": sum(1 for g in gaps if g > gap_threshold),
        "n_bins_usable": len(usable),
    }


def main() -> None:
    configs = [c for c in load_all_judge_configs() if "Qwen" in c["name"]]
    for cfg in configs:
        print(f"\n{'=' * 90}\n{cfg['name']}\n{'=' * 90}")
        for score_name, scores in [("MSP", cfg["phi"][:, 0]), ("combiner", cfg["combiner"].score(cfg["phi"]))]:
            recomputed_ece = ece(scores, cfg["correct"])
            recomputed_brier = brier_score(scores, cfg["correct"])
            bins = reliability_bins(scores, cfg["correct"])
            summary = compact_summary(bins)
            print(f"\n  {score_name}: recomputed ECE={recomputed_ece:.4f}  Brier={recomputed_brier:.4f}")
            print(f"  {'bin':<14}{'mean_pred':>11}{'emp_acc':>10}{'gap':>8}{'count':>8}")
            for b in bins:
                gap = abs(b["mean_pred"] - b["emp_acc"])
                flag = " *" if b["count"] >= MIN_COUNT and gap > GAP_THRESHOLD else ""
                print(f"  [{b['lo']:.1f},{b['hi']:.1f}]{'':<3}{b['mean_pred']:>11.3f}{b['emp_acc']:>10.3f}"
                      f"{gap:>8.3f}{b['count']:>8}{flag}")
            print(f"  compact summary (bins with count>={MIN_COUNT}): "
                  f"max_gap={summary['max_gap']:.3f}, n_exceeding_{GAP_THRESHOLD}={summary['n_exceeding']}, "
                  f"n_bins_usable={summary['n_bins_usable']}")


if __name__ == "__main__":
    main()
