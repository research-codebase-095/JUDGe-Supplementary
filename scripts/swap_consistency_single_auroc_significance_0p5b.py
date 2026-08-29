"""Closes a reproducibility gap: judge2026.tex (Section 5.1 and Table 6) reports
"AUROC 0.503, p=0.30" for whether the 0.5B judge's swap-consistency alone
predicts per-example correctness -- the AUROC's point estimate and its
cluster-bootstrap CI (scripts/judge_characterization.py section E(iii)) were
already reproducible, but no script computed the accompanying p-value.

This is a single-sample AUROC significance test (H0: AUROC == 0.5), not a
two-score comparison, so significance.py's delong_test (which requires two
different scores on the same examples) does not directly apply. This script
instead applies DeLong's variance formula to ONE score by reusing the same
_structural_components() primitive delong_test() itself uses (Sun & Xu 2014
fast-DeLong reformulation, already verified against a naive O(n_pos*n_neg)
reference in tests/test_significance.py): Var(theta) = S10[0,0]/n_pos +
S01[0,0]/n_neg (the single-score diagonal terms of the same covariance
DeLong's paired test assembles from both scores), then z = (theta-0.5)/SE,
a standard two-sided normal-approximation test.

No new model inference -- reuses the cached data/judge_swap_consistency_cache.pt
(0.5B judge, full 1,284-pair pool, matching the exact scope
judge_characterization.py's section E(iii) already uses for this AUROC's
point estimate and CI) and only imports _structural_components(), never
reimplementing its ranking/tie logic.

Usage:
    python scripts/swap_consistency_single_auroc_significance_0p5b.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch
from scipy.stats import norm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from deployment_reliability.significance import _structural_components  # noqa: E402

DATA_DIR = os.path.join(REPO_ROOT, "data")


def single_auroc_significance(correct: np.ndarray, score: np.ndarray) -> tuple[float, float, float, float]:
    """AUROC point estimate plus a DeLong-variance z-test against the H0:
    AUROC == 0.5 null, for one score (not a two-score comparison). Returns
    (auroc, se, z, p_value)."""
    pos, neg = score[correct], score[~correct]
    n_pos, n_neg = len(pos), len(neg)
    v10, v01, theta = _structural_components(pos, neg)
    s10 = np.var(v10, ddof=1)
    s01 = np.var(v01, ddof=1)
    se = float((s10 / n_pos + s01 / n_neg) ** 0.5)
    z = (theta - 0.5) / se
    p_value = float(2.0 * norm.sf(abs(z)))
    return theta, se, z, p_value


def main() -> None:
    cache = torch.load(os.path.join(DATA_DIR, "judge_swap_consistency_cache.pt"))
    correct = cache["correct_orig"].numpy().astype(bool)
    swap_consistent = cache["swap_consistent"].numpy().astype(float)

    theta, se, z, p_value = single_auroc_significance(correct, swap_consistent)
    print("=== 0.5B swap-consistency-vs-correctness AUROC, single-sample DeLong significance test ===")
    print(f"n={len(correct)} (full pool, matching judge_characterization.py section E(iii)'s scope)")
    print(f"AUROC={theta:.4f}  DeLong SE={se:.4f}  z={z:.4f}  p_value={p_value:.4f}")
    print("(judge_characterization.py section E(iii) already reports this same AUROC's point estimate "
          "and cluster-bootstrap CI; this script adds the accompanying significance test.)")


if __name__ == "__main__":
    main()
