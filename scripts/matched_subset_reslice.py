"""GAP 1 follow-up: SmolLM2-360M's judge configuration is reported separately
from the two Qwen configs because it uses a non-random, 4x-smaller MT-Bench
subset (first 400 of 1284 rows, 23 of 80 questions) -- any direct comparison
against Qwen (which sees the full question set) confounds model family with
topic coverage.

This script removes that confound directly: it re-slices Qwen2.5-0.5B and
Qwen2.5-1.5B's already-cached judge data down to EXACTLY the same first-400-
rows / 23-question subset SmolLM2 uses, re-derives a matched
combiner_fit/threshold_cal/id_test split via three_way_split(400, seed=0) --
verified below to reproduce SmolLM2's own cached split assignment bit-for-bit
-- and recomputes MSP AUROC, best-single-feature AUROC (same
bootstrap-stabilized, disjoint-pool selection protocol as Table 1), and
combiner AUROC on this matched id_test, for a clean apples-to-apples
comparison against SmolLM2's own numbers.

No new model inference: reuses only already-cached tensors
(data/judge_feature_cache_mtbench*.pt) and the paper's existing helper
functions (judge_characterization.best_single_feature,
power_analysis's MDE formula), unmodified.

Usage:
    python scripts/matched_subset_reslice.py
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
from deployment_reliability.router import auroc  # noqa: E402
from deployment_reliability.significance import delong_test  # noqa: E402
from deployment_reliability.splits import three_way_split  # noqa: E402

from judge_characterization import best_single_feature, load_judge_config  # noqa: E402

DATA_DIR = os.path.join(REPO_ROOT, "data")
SMOLLM2_PATH = os.path.join(DATA_DIR, "judge_feature_cache_mtbench_smollm2_360m.pt")
QWEN_CONFIGS = [
    ("judge_feature_cache_mtbench.pt", "Qwen2.5-0.5B-Instruct"),
    ("judge_feature_cache_mtbench_1p5b.pt", "Qwen2.5-1.5B-Instruct"),
]
MDE_MULTIPLIER = 1.96 + 0.84  # same as power_analysis.py: z_{alpha/2=0.025} + z_{power=0.80}


def verify_split_alignment(smollm2_cache: dict) -> None:
    """Hard assertion, not an assumption: three_way_split(400, seed=0) must
    reproduce SmolLM2's own cached split assignment element-for-element."""
    combiner_idx, cal_idx, test_idx = three_way_split(400, seed=0)
    reconstructed = np.empty(400, dtype=object)
    reconstructed[combiner_idx.numpy()] = "combiner_fit"
    reconstructed[cal_idx.numpy()] = "threshold_cal"
    reconstructed[test_idx.numpy()] = "id_test"
    cached_splits = np.array(smollm2_cache["splits"])
    assert len(cached_splits) == 400, f"expected SmolLM2 cache n=400, got {len(cached_splits)}"
    n_mismatch = (reconstructed != cached_splits).sum()
    assert n_mismatch == 0, f"three_way_split(400, seed=0) mismatches SmolLM2's cached split at {n_mismatch}/400 rows"
    print(f"[verified] three_way_split(400, seed=0) reproduces SmolLM2's cached split exactly (0/400 mismatches)")
    print(f"[verified] SmolLM2 split sizes: combiner_fit={len(combiner_idx)}, threshold_cal={len(cal_idx)}, id_test={len(test_idx)}")


def verify_question_id_alignment(qwen_cache: dict, smollm2_cache: dict, name: str) -> None:
    qwen_qid_400 = np.array(qwen_cache["question_id"])[:400]
    smollm2_qid = np.array(smollm2_cache["question_id"])
    assert qwen_qid_400.tolist() == smollm2_qid.tolist(), (
        f"{name}: first-400-rows question_id does not match SmolLM2's cached question_id"
    )
    n_unique = len(np.unique(qwen_qid_400))
    print(f"[verified] {name}: first 400 rows' question_id exactly matches SmolLM2's cache "
          f"({n_unique} unique questions)")


def build_matched_config(qwen_cache_path: str) -> dict:
    cache = torch.load(qwen_cache_path)
    phi400, correct400 = cache["phi"][:400], cache["correct"][:400]
    combiner_idx, cal_idx, test_idx = three_way_split(400, seed=0)
    combiner = LogisticRegressionCombiner().fit(phi400[combiner_idx], correct400[combiner_idx].float())
    return {
        "phi_fit": phi400[combiner_idx], "correct_fit": correct400[combiner_idx].bool(),
        "phi_cal": phi400[cal_idx], "correct_cal": correct400[cal_idx].bool(),
        "phi": phi400[test_idx], "correct": correct400[test_idx], "combiner": combiner,
    }


def evaluate(config: dict, name: str) -> None:
    correct_bool = config["correct"].bool()
    msp_scores = config["phi"][:, 0]
    msp_auroc = auroc(msp_scores[correct_bool], msp_scores[~correct_bool])

    best_name, best_scores = best_single_feature(config)
    best_auroc = auroc(best_scores[correct_bool], best_scores[~correct_bool])

    comb_scores = config["combiner"].score(config["phi"])
    comb_auroc = auroc(comb_scores[correct_bool], comb_scores[~correct_bool])

    dl = delong_test(config["correct"], comb_scores, best_scores)
    se = abs(dl.auc_diff / dl.z) if dl.z != 0 else float("nan")
    mde = MDE_MULTIPLIER * se

    print(f"{name}")
    print(f"    n_id_test={len(config['correct'])}  MSP AUROC={msp_auroc:.4f}  "
          f"best feature={best_name} (AUROC={best_auroc:.4f})  combiner AUROC={comb_auroc:.4f}")
    print(f"    combiner-vs-best-feature: diff={dl.auc_diff:+.4f}  naive DeLong p={dl.p_value:.4g}  "
          f"implied SE={se:.4f}  MDE(80% power)={mde:.4f}")
    print()


def main() -> None:
    smollm2_cache = torch.load(SMOLLM2_PATH)
    verify_split_alignment(smollm2_cache)
    print()

    for cache_filename, name in QWEN_CONFIGS:
        qwen_cache = torch.load(os.path.join(DATA_DIR, cache_filename))
        verify_question_id_alignment(qwen_cache, smollm2_cache, name)
    print()

    print("=" * 100)
    print("MATCHED-SUBSET RESLICE: Qwen configs restricted to SmolLM2's exact 23-question/400-row subset")
    print("=" * 100)
    for cache_filename, name in QWEN_CONFIGS:
        matched = build_matched_config(os.path.join(DATA_DIR, cache_filename))
        evaluate(matched, f"{name} (matched subset, n=400 pool)")

    print("=" * 100)
    print("FOR COMPARISON: SmolLM2's own numbers on this exact subset (re-derived via load_judge_config)")
    print("=" * 100)
    smollm2_config = load_judge_config("judge_feature_cache_mtbench_smollm2_360m.pt", "SmolLM2-360M-Instruct")
    evaluate(smollm2_config, "SmolLM2-360M-Instruct (its own native cache)")

    print("=" * 100)
    print("FOR REFERENCE: Qwen configs' FULL (unrestricted, n=1284 pool) numbers, as already reported in Table 1")
    print("=" * 100)
    for cache_filename, name in QWEN_CONFIGS:
        full_config = load_judge_config(cache_filename, name)
        evaluate(full_config, f"{name} (full pool, n=1284, Table 1's existing numbers)")


if __name__ == "__main__":
    main()
