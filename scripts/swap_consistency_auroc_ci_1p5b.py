"""Fills a gap noted in scripts/judge_characterization.py section E: the
question-id cluster-bootstrap 95% CI for the standalone swap-consistency-vs-
correctness AUROC was only ever computed for the 0.5B judge (point 0.503,
CI [0.4967,0.5113]). This script applies the exact same cluster_bootstrap_auroc
function to the already-cached 1.5B swap-consistency data
(data/judge_swap_consistency_cache_1p5b.pt), reusing that cache's
correct_orig/swap_consistent/question_id fields directly. No new model
inference; reuses the paper's existing cluster-bootstrap methodology.

Usage:
    python scripts/swap_consistency_auroc_ci_1p5b.py
"""
import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from judge_characterization import cluster_bootstrap_auroc  # noqa: E402

DATA_DIR = os.path.join(REPO_ROOT, "data")


def main() -> None:
    cache = torch.load(os.path.join(DATA_DIR, "judge_swap_consistency_cache_1p5b.pt"), weights_only=False)
    correct = cache["correct_orig"].numpy().astype(bool)
    swap_consistent = cache["swap_consistent"].numpy().astype(float)
    qid = np.array(cache["question_id"])

    point, lo, hi = cluster_bootstrap_auroc(qid, correct, swap_consistent)
    print("=== 1.5B swap-consistency-vs-correctness AUROC, cluster-bootstrap 95% CI ===")
    print(f"point={point:.4f}  95% CI=[{lo:.4f}, {hi:.4f}]  (paper's reported point estimate: 0.583)")


if __name__ == "__main__":
    main()
