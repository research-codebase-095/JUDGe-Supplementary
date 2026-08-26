"""Reviewer follow-up: judge2026.tex's Full Limitations item (iv) states that
best-feature selection noise is a real, unaddressed source of understated
uncertainty, but never quantifies it. best_single_feature() in
judge_characterization.py already resamples the pooled combiner_fit+
threshold_cal data 1000 times to pick a stabilized best feature (majority
vote across resamples), but Table 1's reported combiner-vs-best-feature
AUROC gap and its cluster-bootstrap CI both treat that MAJORITY CHOICE as
fixed -- id_test is then resampled (cluster_bootstrap_auroc_diff) holding
the selected feature constant, which captures id_test sampling noise but not
selection noise.

This script instead holds id_test FIXED (no resampling there -- that
uncertainty is already reported) and resamples the pooled cal data the
selection step depends on, tracking, for EACH resample, which feature would
have been chosen and what the resulting combiner-vs-that-feature gap on
id_test would have been. The resulting distribution of gaps is
"selection-aware": it reflects the fact that a different cal-pool draw could
plausibly have selected a different feature entirely, something the paper's
existing point estimate and CI do not.

No new model inference -- reuses cached phi/correct tensors and the exact
same pooled_calibration/verify_feature_directions machinery
bootstrap_stabilized_direction_and_best_feature already uses, just records
the per-draw selected feature instead of only the aggregate majority.

Usage:
    python scripts/selection_aware_bootstrap.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from deployment_reliability.features import DEFAULT_FEATURE_DIRECTIONS, DEFAULT_FEATURE_NAMES, verify_feature_directions  # noqa: E402
from deployment_reliability.router import auroc  # noqa: E402

from judge_characterization import load_judge_config, pooled_calibration  # noqa: E402

SEED = 0
N_BOOTSTRAP = 1000


def selection_aware_gaps(config: dict, n_bootstrap: int = N_BOOTSTRAP, seed: int = SEED) -> dict:
    phi_pool, correct_pool = pooled_calibration(config)
    n = phi_pool.shape[0]
    rng = np.random.default_rng(seed)

    comb_scores = config["combiner"].score(config["phi"])
    correct_test_bool = config["correct"].bool()
    comb_auroc = auroc(comb_scores[correct_test_bool], comb_scores[~correct_test_bool])

    gaps = []
    chosen = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        phi_b, correct_b = phi_pool[idx], correct_pool[idx]
        if correct_b.all() or (~correct_b).all():
            continue
        d_b = verify_feature_directions(phi_b, correct_b)
        signs_b = torch.tensor([1.0 if d_b[f] else -1.0 for f in DEFAULT_FEATURE_NAMES])
        oriented_b = phi_b * DEFAULT_FEATURE_DIRECTIONS * signs_b
        cb = correct_b.bool()
        aurocs_b = {f: auroc(oriented_b[:, i][cb], oriented_b[:, i][~cb]) for i, f in enumerate(DEFAULT_FEATURE_NAMES)}
        best_name = max(aurocs_b, key=aurocs_b.get)
        chosen.append(best_name)

        # score THIS draw's selected feature on the fixed id_test, using
        # id_test's own verified direction (same final step best_single_feature() uses)
        d_test = verify_feature_directions(config["phi"], config["correct"])
        sign_test = 1.0 if d_test[best_name] else -1.0
        idx_feat = DEFAULT_FEATURE_NAMES.index(best_name)
        best_col_test = config["phi"][:, idx_feat] * DEFAULT_FEATURE_DIRECTIONS[idx_feat].item() * sign_test
        best_auroc_test = auroc(best_col_test[correct_test_bool], best_col_test[~correct_test_bool])
        gaps.append(comb_auroc - best_auroc_test)

    gaps = np.array(gaps)
    chosen = np.array(chosen)
    unique, counts = np.unique(chosen, return_counts=True)
    selection_dist = dict(zip(unique.tolist(), (counts / len(chosen)).tolist()))
    ci_lo, ci_hi = np.percentile(gaps, [2.5, 97.5])
    return {
        "n_draws": len(gaps), "comb_auroc": comb_auroc,
        "mean_gap": float(gaps.mean()), "std_gap": float(gaps.std()),
        "ci95": (float(ci_lo), float(ci_hi)),
        "selection_dist": selection_dist,
        "frac_matching_headline": None,  # filled by caller, who knows the headline choice
    }


def main() -> None:
    configs = [
        load_judge_config("judge_feature_cache_mtbench.pt", "Qwen2.5-0.5B-Instruct (judge)"),
        load_judge_config("judge_feature_cache_mtbench_1p5b.pt", "Qwen2.5-1.5B-Instruct (judge)"),
    ]
    headline_best = {"Qwen2.5-0.5B-Instruct (judge)": "msp", "Qwen2.5-1.5B-Instruct (judge)": "normalized_entropy"}

    for cfg in configs:
        if cfg is None:
            continue
        result = selection_aware_gaps(cfg)
        headline = headline_best[cfg["name"]]
        frac_headline = result["selection_dist"].get(headline, 0.0)
        print(f"=== {cfg['name']} ===")
        print(f"  n valid draws: {result['n_draws']}")
        print(f"  combiner AUROC (fixed, id_test): {result['comb_auroc']:.4f}")
        print(f"  selection distribution across {result['n_draws']} cal-pool resamples:")
        for feat, frac in sorted(result["selection_dist"].items(), key=lambda kv: -kv[1]):
            print(f"    {feat:<20} {frac*100:5.1f}%")
        print(f"  headline choice '{headline}' selected in {frac_headline*100:.1f}% of resamples")
        print(f"  selection-aware combiner-vs-selected-feature gap: mean={result['mean_gap']:+.4f}, "
              f"std={result['std_gap']:.4f}, 95% percentile CI=[{result['ci95'][0]:+.4f}, {result['ci95'][1]:+.4f}]")
        print()


if __name__ == "__main__":
    main()
