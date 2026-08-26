"""Reviewer follow-up: judge2026.tex's split-stability check (five reseeds,
scripts/stability_across_splits.py) covers the three LLM backbones and both
Qwen judge configurations, but explicitly NOT ResNet-50 -- vision's +0.008
combiner-vs-best-feature win has rested on the single seed-0 split reported
throughout the paper. Since this is now the paper's only clearly positive
result, a sign-flip here would matter more than a sign-flip on a null; this
was flagged as an open asymmetry (Appendix, "Split stability and power") and
not previously closed.

Vision's split structure differs from the LLM/judge configs (Table 1's
dagger/double-dagger footnote, scripts/best_feature_selected_on_cal.py):
combiner_fit(1500)/threshold_cal(300) come from a SEPARATE small calibration
sample (logit_cache_resnet50.pt), while the actual reported id_test is the
disjoint, fixed 50,000-image full-scale validation set
(logit_cache_imagenet1k_resnet50.pt) -- it is not itself resplit here, since
it is not part of the same pool the fit/cal sample is drawn from (unlike the
LLM/judge configs, where fit/cal/test all come from one reshuffled pool).

This script redraws the small 1800-image fit/cal sample five times (seeds
0-4), refits the combiner and reselects the best single feature (via AUROC on
that draw's threshold_cal, scored on the fixed 50k id_test) each time, and
reports the combiner-vs-best-feature AUROC gap across reseeds -- the same
statistic Table 1 reports for vision, now with the same reseed treatment
given to every other row.

No new model inference - reuses the same cached logits as Table 1's vision
row and scripts/best_feature_selected_on_cal.py.

Usage:
    python scripts/vision_seed_stability.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from deployment_reliability.combiner import LogisticRegressionCombiner  # noqa: E402
from deployment_reliability.features import DEFAULT_FEATURE_DIRECTIONS, DEFAULT_FEATURE_NAMES, featurize  # noqa: E402
from deployment_reliability.router import auroc  # noqa: E402

DATA_DIR = os.path.join(REPO_ROOT, "data")
SEEDS = (0, 1, 2, 3, 4)


def orient(phi: torch.Tensor) -> torch.Tensor:
    return phi * DEFAULT_FEATURE_DIRECTIONS


def feature_aurocs(phi_oriented: torch.Tensor, correct: torch.Tensor) -> dict[str, float]:
    correct_bool = correct.to(dtype=torch.bool)
    return {
        name: auroc(phi_oriented[:, i][correct_bool], phi_oriented[:, i][~correct_bool])
        for i, name in enumerate(DEFAULT_FEATURE_NAMES)
    }


def load_small_pool():
    """The 1800-image fit/cal sample, pooled (original combiner_fit/threshold_cal
    labels discarded -- we redraw our own 1500/300 split each seed, exactly as
    stability_across_splits.py does for the other six rows). logit_cache_resnet50.pt
    also carries this cache's own id_test (1500) and OOD rows (imagenet_a,
    imagenet_o) for unrelated OOD-detection uses elsewhere in this repo -- those
    MUST be excluded here, or the pool is contaminated with data this paper's
    vision task was never meant to include."""
    small = torch.load(os.path.join(DATA_DIR, "logit_cache_resnet50.pt"))
    s = np.array(small["splits"])
    mask = np.isin(s, ["combiner_fit", "threshold_cal"])
    logits, labels = small["logits"][mask], small["labels"][mask]
    phi = featurize(logits)
    correct = (logits.argmax(dim=-1) == labels).float()
    return phi, correct


def load_full_test():
    full = torch.load(os.path.join(DATA_DIR, "logit_cache_imagenet1k_resnet50.pt"))
    phi_test = featurize(full["logits"])
    correct_test = (full["logits"].argmax(dim=-1) == full["labels"]).bool()
    return phi_test, correct_test


def evaluate_one_seed(phi_pool, correct_pool, phi_test, correct_test, seed: int) -> tuple[str, float, float, float]:
    n = len(correct_pool)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_cal = 300
    cal_idx, fit_idx = perm[:n_cal], perm[n_cal:]

    phi_fit, correct_fit = phi_pool[fit_idx], correct_pool[fit_idx]
    phi_cal, correct_cal = phi_pool[cal_idx], correct_pool[cal_idx]

    cal_aurocs = feature_aurocs(orient(phi_cal), correct_cal)
    best_name = max(cal_aurocs, key=cal_aurocs.get)
    best_idx = DEFAULT_FEATURE_NAMES.index(best_name)

    test_oriented = orient(phi_test)
    best_col = test_oriented[:, best_idx]
    best_auroc = auroc(best_col[correct_test], best_col[~correct_test])

    combiner = LogisticRegressionCombiner().fit(phi_fit, correct_fit)
    comb_scores = combiner.score(phi_test)
    comb_auroc = auroc(comb_scores[correct_test], comb_scores[~correct_test])

    return best_name, best_auroc, comb_auroc, comb_auroc - best_auroc


def main() -> None:
    phi_pool, correct_pool = load_small_pool()
    phi_test, correct_test = load_full_test()

    print(f"{'seed':<6}{'best feature':<16}{'best AUROC':>12}{'combiner AUROC':>16}{'gap':>10}")
    gaps = []
    for seed in SEEDS:
        best_name, best_auroc, comb_auroc, gap = evaluate_one_seed(phi_pool, correct_pool, phi_test, correct_test, seed)
        gaps.append(gap)
        print(f"{seed:<6}{best_name:<16}{best_auroc:>12.4f}{comb_auroc:>16.4f}{gap:>+10.4f}")
    gaps = np.array(gaps)
    print()
    print(f"mean gap = {gaps.mean():+.4f}, std = {gaps.std():.4f}, min = {gaps.min():+.4f}, max = {gaps.max():+.4f}")
    print(f"sign flips across seeds: {'YES' if (gaps.min() < 0 < gaps.max()) else 'no'}")


if __name__ == "__main__":
    main()
