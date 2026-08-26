"""Reviewer robustness check: judge2026.tex computes Phi over the verdict
token's FULL next-token vocabulary (~150k tokens for Qwen, ~49k for SmolLM2),
not restricted to the two candidate tokens "A"/"B" themselves. A reviewer
asked whether this dilutes entropy/energy/margin with formatting/continuation
uncertainty unrelated to the verdict, and whether that could be part of why
the judge combiner shows a null result and near-perfect MSP/margin/entropy
collinearity.

scripts/collect_judge_verdicts_2way.py (already run; see
data/judge_feature_cache_mtbench*_2way.pt) reruns all three judge configs
with the identical protocol (same dataset, prompts, splits, seed) except
Phi is computed over ONLY the two-element vector [logit_A, logit_B] instead
of the full vocabulary. This script compares that 2-way Phi against the
full-vocabulary Phi already used throughout the paper, using the exact same
downstream statistics (best-single-feature selected on threshold_cal, scored
on id_test; combiner AUROC; pairwise rank-discordance vs MSP, per-config
empirically-verified orientation) so the two are directly comparable.

No new model inference - reuses the already-cached 2-way and full-vocab
tensors.

Usage:
    python scripts/compare_2way_vs_fullvocab.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from deployment_reliability.combiner import LogisticRegressionCombiner  # noqa: E402
from deployment_reliability.features import (  # noqa: E402
    DEFAULT_FEATURE_DIRECTIONS,
    DEFAULT_FEATURE_NAMES,
    verify_feature_directions,
)
from deployment_reliability.router import auroc  # noqa: E402
from deployment_reliability.significance import delong_test  # noqa: E402

from judge_characterization import (  # noqa: E402
    bootstrap_stabilized_direction_and_best_feature,
    load_judge_config,
    oriented_pooled,
)

DATA_DIR = os.path.join(REPO_ROOT, "data")

CONFIGS = [
    ("Qwen2.5-0.5B-Instruct (judge)", "judge_feature_cache_mtbench.pt", "judge_feature_cache_mtbench_2way.pt"),
    ("Qwen2.5-1.5B-Instruct (judge)", "judge_feature_cache_mtbench_1p5b.pt", "judge_feature_cache_mtbench_1p5b_2way.pt"),
    ("SmolLM2-360M-Instruct (judge)", "judge_feature_cache_mtbench_smollm2_360m.pt", "judge_feature_cache_mtbench_smollm2_360m_2way.pt"),
]


def pairwise_discordance(a: np.ndarray, b: np.ndarray, n_sample: int = 200_000, seed: int = 0) -> float:
    n = len(a)
    rng = np.random.default_rng(seed)
    if n * (n - 1) // 2 <= n_sample:
        ii, jj = np.triu_indices(n, k=1)
    else:
        ii = rng.integers(0, n, size=n_sample)
        jj = rng.integers(0, n, size=n_sample)
        mask = ii != jj
        ii, jj = ii[mask], jj[mask]
    da, db = a[ii] - a[jj], b[ii] - b[jj]
    return float((((da > 0) & (db < 0)) | ((da < 0) & (db > 0))).mean())


def best_single_feature_on_cal(config: dict):
    """Selects the best feature via the bootstrap-stabilized majority vote
    over pooled combiner_fit+threshold_cal (judge_characterization.py's
    bootstrap_stabilized_direction_and_best_feature), then scores that SAME
    feature on id_test using id_test's own correct orientation - matches
    judge2026.tex's disjoint-selection protocol (Table 1), not a single draw
    on threshold_cal alone."""
    result = bootstrap_stabilized_direction_and_best_feature(config)
    best_name = result["stabilized_best"]
    phi_test, correct_test = config["phi"], config["correct"]
    d_test = verify_feature_directions(phi_test, correct_test)
    sign_test = 1.0 if d_test[best_name] else -1.0
    idx = DEFAULT_FEATURE_NAMES.index(best_name)
    col_test = phi_test[:, idx] * DEFAULT_FEATURE_DIRECTIONS[idx].item() * sign_test
    correct_test_bool = correct_test.bool()
    best_auroc = auroc(col_test[correct_test_bool], col_test[~correct_test_bool])
    return best_name, best_auroc


def analyze(name: str, config: dict) -> dict:
    phi_test, correct_test = config["phi"], config["correct"]
    msp_test = phi_test[:, 0]
    correct_test_bool = correct_test.bool()
    msp_auroc = auroc(msp_test[correct_test_bool], msp_test[~correct_test_bool])

    best_name, best_auroc = best_single_feature_on_cal(config)

    combiner = LogisticRegressionCombiner().fit(config["phi_fit"], config["correct_fit"].float())
    comb_scores = combiner.score(phi_test)
    comb_auroc = auroc(comb_scores[correct_test_bool], comb_scores[~correct_test_bool])

    dl = delong_test(correct_test, comb_scores, msp_test)

    # Combiner-vs-best-feature DeLong test (Table 1's actual headline
    # comparison, not combiner-vs-MSP -- best feature differs from MSP at
    # some configs, e.g. SmolLM2's logit_l2_norm).
    idx_best = DEFAULT_FEATURE_NAMES.index(best_name)
    d_test = verify_feature_directions(phi_test, correct_test)
    sign_best = 1.0 if d_test[best_name] else -1.0
    best_scores = phi_test[:, idx_best] * DEFAULT_FEATURE_DIRECTIONS[idx_best].item() * sign_best
    dl_best = delong_test(correct_test, comb_scores, best_scores)

    phi_test_c = oriented_pooled(config)
    msp_c = phi_test_c[:, 0].numpy()
    discordance = {}
    for feat in ("logit_margin", "normalized_entropy", "energy_score", "logit_l2_norm"):
        idx = DEFAULT_FEATURE_NAMES.index(feat)
        col = phi_test_c[:, idx].numpy()
        discordance[feat] = pairwise_discordance(msp_c, col) * 100

    return {
        "name": name, "msp_auroc": msp_auroc, "best_name": best_name, "best_auroc": best_auroc,
        "comb_auroc": comb_auroc, "delong_p": dl.p_value, "delong_p_best": dl_best.p_value,
        "discordance": discordance,
    }


def load_and_analyze(cache_path: str, name: str) -> dict:
    config = load_judge_config(cache_path, name)
    return analyze(name, config)


def main() -> None:
    header = f"{'Config':<32}{'Variant':<12}{'MSP':>8}{'Best feat.':>14}{'Combiner':>10}{'p (winner)':>12}"
    print(header)
    print("-" * len(header))
    for name, full_path, twoway_path in CONFIGS:
        r_full = load_and_analyze(full_path, name)
        r_2way = load_and_analyze(twoway_path, name)
        for tag, r in (("full-vocab", r_full), ("2-way (A/B)", r_2way)):
            print(f"{name:<32}{tag:<12}{r['msp_auroc']:>8.4f}{r['best_auroc']:>10.4f} ({r['best_name']:<10}){r['comb_auroc']:>10.4f}{r['delong_p_best']:>12.4g} (best-feat p; vs-MSP p={r['delong_p']:.4g})")
        print("  discordance vs MSP (full-vocab / 2-way):")
        for feat in ("logit_margin", "normalized_entropy", "energy_score", "logit_l2_norm"):
            print(f"    {feat:<20}{r_full['discordance'][feat]:>8.2f}%   /   {r_2way['discordance'][feat]:>8.2f}%")
        print()


if __name__ == "__main__":
    main()
