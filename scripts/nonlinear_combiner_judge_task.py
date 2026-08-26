"""Reviewer follow-up: judge2026.tex's central "vectorization does not help
on the judge task" claim is established only against the LINEAR combiner
G_w. A nonlinear (gradient-boosted-tree) combiner already exists in this
repo's own code (paper_diagnostics.py's section10 / judge_characterization.py's
gbt_scores()) and is already reported for the four non-judge backbones, but
was never run and reported for the judge task itself. A linear combiner's
failure to beat the best single feature does not establish that no combiner
could (a linear model cannot represent interaction/XOR-like effects even if
they exist), so this is a real gap: does a strictly more expressive combiner
find anything the linear one misses on the judge task specifically?

This script computes GBT-vs-MSP and GBT-vs-best-feature AUROC for every judge
configuration this paper reports, using the exact same disjoint-selection
protocol (best_single_feature), significance tests (delong_test, naive;
cluster_bootstrap_auroc_diff, cluster-robust by question_id), and calibration
metrics (ece, brier_score) already used throughout this paper -- no shortcuts,
same rigor standard.

No new model inference: gbt_scores() only fits an
sklearn.ensemble.HistGradientBoostingClassifier on already-cached Phi/correct
tensors (data/judge_feature_cache_mtbench*.pt); this is a different combiner
ARCHITECTURE applied to existing features, not new data collection.

Coverage: Qwen2.5-0.5B (full pool), Qwen2.5-1.5B (full pool), SmolLM2-360M
(native n=200 cache), and the matched-subset comparison already established
in scripts/matched_subset_reslice.py (Qwen configs restricted to SmolLM2's
identical 23-question/400-row subset via three_way_split(400, seed=0)) --
does the judge-task nonlinear signal at 0.5B hold up on this smaller,
topic-matched subset, or is it an artifact of the larger full-pool n=642?

Usage:
    python scripts/nonlinear_combiner_judge_task.py
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

from judge_characterization import best_single_feature, gbt_scores, load_judge_config  # noqa: E402
from paper_diagnostics import brier_score, cluster_bootstrap_auroc_diff, ece  # noqa: E402

DATA_DIR = os.path.join(REPO_ROOT, "data")
MDE_MULTIPLIER = 1.96 + 0.84  # z_{alpha/2=0.025} + z_{power=0.80}, same as power_analysis.py

FULL_POOL_CONFIGS = [
    ("judge_feature_cache_mtbench.pt", "Qwen2.5-0.5B-Instruct (judge, full pool)"),
    ("judge_feature_cache_mtbench_1p5b.pt", "Qwen2.5-1.5B-Instruct (judge, full pool)"),
]
SMOLLM2_PATH = "judge_feature_cache_mtbench_smollm2_360m.pt"


def build_matched_config_with_qid(qwen_cache_path: str) -> dict:
    """Same construction as matched_subset_reslice.py's build_matched_config,
    plus a question_id array (needed here for cluster-robust bootstrapping,
    which that script didn't need since it only ran naive DeLong)."""
    cache = torch.load(qwen_cache_path)
    phi400, correct400 = cache["phi"][:400], cache["correct"][:400]
    qid400 = np.array(cache["question_id"])[:400]
    combiner_idx, cal_idx, test_idx = three_way_split(400, seed=0)
    combiner = LogisticRegressionCombiner().fit(phi400[combiner_idx], correct400[combiner_idx].float())
    return {
        "phi_fit": phi400[combiner_idx], "correct_fit": correct400[combiner_idx].bool(),
        "phi_cal": phi400[cal_idx], "correct_cal": correct400[cal_idx].bool(),
        "phi": phi400[test_idx], "correct": correct400[test_idx], "combiner": combiner,
        "question_id": qid400[test_idx.numpy()],
    }


def evaluate(config: dict, name: str) -> None:
    correct_bool = config["correct"].bool()
    correct_np = config["correct"].numpy().astype(bool)
    qid = config["question_id"]

    msp_scores = config["phi"][:, 0]
    msp_auroc = auroc(msp_scores[correct_bool], msp_scores[~correct_bool])

    best_name, best_scores = best_single_feature(config)
    best_auroc = auroc(best_scores[correct_bool], best_scores[~correct_bool])

    lin_scores = config["combiner"].score(config["phi"])
    lin_auroc = auroc(lin_scores[correct_bool], lin_scores[~correct_bool])

    gbt = gbt_scores(config)
    gbt_auroc = auroc(gbt[correct_bool], gbt[~correct_bool])

    print(f"\n{name}  (n_id_test={len(config['correct'])})")
    print(f"    MSP AUROC={msp_auroc:.4f}  best feature={best_name} (AUROC={best_auroc:.4f})  "
          f"linear combiner AUROC={lin_auroc:.4f}  GBT AUROC={gbt_auroc:.4f}")

    for label, other_name, other in [("MSP", "MSP", msp_scores), (f"best-feature({best_name})", "best", best_scores)]:
        dl = delong_test(config["correct"], gbt, other)
        se = abs(dl.auc_diff / dl.z) if dl.z != 0 else float("nan")
        mde = MDE_MULTIPLIER * se
        point, ci_lo, ci_hi, frac_pos = cluster_bootstrap_auroc_diff(
            qid, correct_np, gbt.numpy(), other.numpy()
        )
        sig_naive = "yes" if dl.p_value < 0.05 else "no"
        sig_cluster = "yes" if (ci_lo > 0 or ci_hi < 0) else "no"
        print(f"    GBT vs {label}: diff={dl.auc_diff:+.4f}  naive DeLong p={dl.p_value:.4g} (sig={sig_naive})  "
              f"implied SE={se:.4f}  MDE(80%)={mde:.4f}")
        print(f"        cluster-bootstrap: point={point:+.4f}  95% CI=[{ci_lo:+.4f},{ci_hi:+.4f}] (sig={sig_cluster})  "
              f"frac_draws_favoring_GBT={frac_pos:.3f}")

    gbt_ece = ece(gbt, config["correct"])
    gbt_brier = brier_score(gbt, config["correct"])
    lin_ece = ece(lin_scores.clamp(0, 1), config["correct"])
    lin_brier = brier_score(lin_scores, config["correct"])
    print(f"    Calibration -- GBT: ECE={gbt_ece:.4f} Brier={gbt_brier:.4f}   "
          f"linear combiner: ECE={lin_ece:.4f} Brier={lin_brier:.4f}")


def main() -> None:
    print("=" * 110)
    print("NONLINEAR (GBT) COMBINER ON THE JUDGE TASK -- full coverage across all configs")
    print("=" * 110)

    print("\n" + "#" * 110)
    print("# 1-2. FULL-POOL Qwen configs (id_test n=642), as reported in Table 1")
    print("#" * 110)
    for cache_filename, name in FULL_POOL_CONFIGS:
        cfg = load_judge_config(cache_filename, name)
        evaluate(cfg, name)

    print("\n" + "#" * 110)
    print("# 3. SmolLM2-360M native cache (id_test n=200)")
    print("#" * 110)
    smollm2_cfg = load_judge_config(SMOLLM2_PATH, "SmolLM2-360M-Instruct (judge, native n=200)")
    evaluate(smollm2_cfg, "SmolLM2-360M-Instruct (judge, native n=200)")

    print("\n" + "#" * 110)
    print("# 4. MATCHED-SUBSET: Qwen restricted to SmolLM2's identical 23-question/400-row subset")
    print("#" * 110)
    for cache_filename, name in FULL_POOL_CONFIGS:
        matched = build_matched_config_with_qid(os.path.join(DATA_DIR, cache_filename))
        evaluate(matched, f"{name.replace('full pool', 'matched subset, n=400 pool')}")


if __name__ == "__main__":
    main()
