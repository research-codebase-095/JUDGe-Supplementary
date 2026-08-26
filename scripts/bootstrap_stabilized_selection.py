"""Bootstrap-stabilized feature-direction verification and best-single-feature
selection, replacing the single-split versions used previously.

Two things changed relative to the prior protocol (best_single_feature() in
judge_characterization.py before this script existed):

1. Source data: direction verification and best-feature selection now use the
   POOLED combiner_fit+threshold_cal data, not threshold_cal alone. Pooling
   gives n comparable to id_test itself (e.g. SmolLM2: 200 pooled vs 200
   id_test, vs. just 40 for threshold_cal alone) while staying fully disjoint
   from id_test -- this fixes both the small-n instability AND the
   id_test-verifies-its-own-direction circularity in one step, rather than
   bootstrapping the (still tiny) threshold_cal split alone. A direct
   comparison (see scripts/direction_split_robustness_check.py's original
   threshold_cal-alone check) showed bootstrapping threshold_cal alone can
   give a confident-looking majority that is itself sample-noise (e.g.
   SmolLM2's MSP: 84% of threshold_cal-alone bootstrap resamples say
   "reversed," driven by that split's own n=40 point estimate of 0.359 AUROC
   -- which conflicts with id_test's own n=200 direct estimate of 0.513, and
   the smaller split has no basis for being trusted over the larger one).
   Pooling removes this problem structurally instead of papering over it.
2. Both direction verification and best-feature selection are now
   bootstrap-majority votes over B=1000 resamples of the pooled data (seed
   0), not single point estimates -- reported stability fractions below.

Usage:
    python scripts/bootstrap_stabilized_selection.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from deployment_reliability.features import (  # noqa: E402
    DEFAULT_FEATURE_DIRECTIONS,
    DEFAULT_FEATURE_NAMES,
    verify_feature_directions,
)
from deployment_reliability.router import auroc  # noqa: E402

DATA_DIR = os.path.join(REPO_ROOT, "data")
B = 1000
SEED = 0

CONFIGS = [
    ("Qwen2.5-0.5B-Instruct (judge)", "judge_feature_cache_mtbench.pt"),
    ("Qwen2.5-1.5B-Instruct (judge)", "judge_feature_cache_mtbench_1p5b.pt"),
    ("SmolLM2-360M-Instruct (judge)", "judge_feature_cache_mtbench_smollm2_360m.pt"),
]


def load_pool(path: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    cache = torch.load(os.path.join(DATA_DIR, path))
    splits = np.array(cache["splits"])
    m_fit = splits == "combiner_fit"
    m_cal = splits == "threshold_cal"
    m_test = splits == "id_test"
    phi_pool = torch.cat([cache["phi"][m_fit], cache["phi"][m_cal]])
    correct_pool = torch.cat([cache["correct"][m_fit], cache["correct"][m_cal]])
    return phi_pool, correct_pool, cache["phi"][m_test], cache["correct"][m_test]


def bootstrap_stabilize(phi_pool: torch.Tensor, correct_pool: torch.Tensor, b: int = B, seed: int = SEED) -> dict:
    n = phi_pool.shape[0]
    rng = np.random.default_rng(seed)
    sign_counts = {f: 0 for f in DEFAULT_FEATURE_NAMES}
    best_counts = {f: 0 for f in DEFAULT_FEATURE_NAMES}
    valid = 0
    for _ in range(b):
        idx = rng.integers(0, n, size=n)
        phi_b, correct_b = phi_pool[idx], correct_pool[idx]
        if correct_b.all() or (~correct_b).all():
            continue
        valid += 1
        d = verify_feature_directions(phi_b, correct_b)
        signs = {f: (1.0 if d[f] else -1.0) for f in DEFAULT_FEATURE_NAMES}
        for f in DEFAULT_FEATURE_NAMES:
            if signs[f] > 0:
                sign_counts[f] += 1
        oriented = phi_b * DEFAULT_FEATURE_DIRECTIONS * torch.tensor([signs[f] for f in DEFAULT_FEATURE_NAMES])
        cb = correct_b.bool()
        aurocs = {f: auroc(oriented[:, i][cb], oriented[:, i][~cb]) for i, f in enumerate(DEFAULT_FEATURE_NAMES)}
        best_counts[max(aurocs, key=aurocs.get)] += 1
    stabilized_sign = {f: (1.0 if sign_counts[f] >= valid / 2 else -1.0) for f in DEFAULT_FEATURE_NAMES}
    stabilized_best = max(best_counts, key=best_counts.get)
    return {
        "n_pool": n, "valid": valid,
        "sign_frac": {f: sign_counts[f] / valid for f in DEFAULT_FEATURE_NAMES},
        "best_frac": {f: best_counts[f] / valid for f in DEFAULT_FEATURE_NAMES},
        "stabilized_sign": stabilized_sign, "stabilized_best": stabilized_best,
    }


def main() -> None:
    for name, path in CONFIGS:
        phi_pool, correct_pool, phi_test, correct_test = load_pool(path)
        result = bootstrap_stabilize(phi_pool, correct_pool)
        print(f"=== {name} (pooled fit+cal n={result['n_pool']}, valid resamples={result['valid']}) ===")
        for f in DEFAULT_FEATURE_NAMES:
            print(f"  {f:<20} positive-sign frac: {result['sign_frac'][f]:.3f}   "
                  f"selected-as-best frac: {result['best_frac'][f]:.3f}   "
                  f"stabilized sign: {result['stabilized_sign'][f]:+.0f}")
        print(f"  STABILIZED BEST FEATURE: {result['stabilized_best']}")

        # Score the stabilized best feature on id_test, using id_test's own
        # verify_feature_directions for the id_test-side orientation (id_test
        # is large enough -- 200-642 -- that this is not the same small-n
        # problem; only the SELECTION source was replaced, not the disjoint
        # selection-then-score protocol itself).
        d_test = verify_feature_directions(phi_test, correct_test)
        best_name = result["stabilized_best"]
        idx = DEFAULT_FEATURE_NAMES.index(best_name)
        sign_test = 1.0 if d_test[best_name] else -1.0
        best_col = phi_test[:, idx] * DEFAULT_FEATURE_DIRECTIONS[idx].item() * sign_test
        correct_test_bool = correct_test.bool()
        best_auroc = auroc(best_col[correct_test_bool], best_col[~correct_test_bool])
        print(f"  {best_name} AUROC on id_test (stabilized selection, id_test's own orientation): {best_auroc:.4f}")
        print()


if __name__ == "__main__":
    main()
