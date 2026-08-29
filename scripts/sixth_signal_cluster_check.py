"""Adds the missing analysis half for judge2026.tex's Appendix A.6 "Sixth-signal
cluster-aware check": the question-id cluster-bootstrap CI and cluster-SE
z-test on the AUROC gap between the 5-feature (Phi) and 6-feature (Phi +
swap-consistency) 1.5B combiners.

collect_judge_swap_consistency_1p5b.py already fits both combiners and reports
the naive row-level DeLong test (p=2.8e-7) on this same gap; that test assumes
id_test rows are independent, which they are not (rows share question_id).
This script re-fits both combiners identically from the already-cached
data/judge_swap_consistency_cache_1p5b.pt (no new model inference) and adds
the question-id cluster-bootstrap check, reusing the exact
cluster_bootstrap_auroc_diff machinery paper_diagnostics.py already uses for
the combiner-vs-MSP comparisons (Table 1, Appendix A.6 section 13).

Usage:
    python scripts/sixth_signal_cluster_check.py
"""

import os
import sys

import numpy as np
import torch
from scipy.stats import norm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from deployment_reliability.combiner import LogisticRegressionCombiner  # noqa: E402
from deployment_reliability.significance import delong_test  # noqa: E402

from paper_diagnostics import cluster_bootstrap_auroc_diff  # noqa: E402

DATA_DIR = os.path.join(REPO_ROOT, "data")
SEED = 0
N_BOOTSTRAP = 1000  # matches every other question-id cluster-bootstrap CI in this paper (Appendix A.6, "Bootstrap counts")


def main() -> None:
    cache = torch.load(os.path.join(DATA_DIR, "judge_swap_consistency_cache_1p5b.pt"), weights_only=False)

    phi5 = cache["phi_orig"]
    swap_consistent = cache["swap_consistent"].float()
    phi6 = torch.cat([phi5, swap_consistent.unsqueeze(-1)], dim=-1)
    correct = cache["correct_orig"].bool()
    splits = cache["splits"]
    qid = np.array(cache["question_id"])

    fit_idx = torch.tensor([i for i, s in enumerate(splits) if s == "combiner_fit"], dtype=torch.long)
    test_idx = torch.tensor([i for i, s in enumerate(splits) if s == "id_test"], dtype=torch.long)

    combiner5 = LogisticRegressionCombiner().fit(phi5[fit_idx], correct[fit_idx].float())
    combiner6 = LogisticRegressionCombiner().fit(phi6[fit_idx], correct[fit_idx].float())

    scores5_test = combiner5.score(phi5[test_idx])
    scores6_test = combiner6.score(phi6[test_idx])
    correct_test = correct[test_idx]
    qid_test = qid[test_idx.numpy()]

    delong = delong_test(correct_test, scores6_test, scores5_test)
    naive_se = delong.auc_diff / delong.z

    point, ci_lo, ci_hi, frac_pos = cluster_bootstrap_auroc_diff(
        qid_test, correct_test.numpy(), scores6_test.numpy(), scores5_test.numpy(),
        n_bootstrap=N_BOOTSTRAP, seed=SEED,
    )

    # Empirical cluster SE, recovered from the bootstrap CI's spread rather than
    # assumed: under a normal approximation a 95% CI spans ~3.9198 SE, the same
    # approximation the naive-DeLong SE above already relies on.
    cluster_se = (ci_hi - ci_lo) / (2 * 1.959963985)
    se_inflation = cluster_se / naive_se
    z_cluster = point / cluster_se
    p_cluster = 2 * (1 - norm.cdf(abs(z_cluster)))

    print("=== Sixth-signal cluster-aware check (1.5B judge, id_test) ===")
    print(f"5-feature combiner AUROC:   {delong.auc_b:.4f}")
    print(f"6-feature combiner AUROC:   {delong.auc_a:.4f}")
    print(f"AUROC gap (6-minus-5):      {point:.4f}")
    print(f"naive row-level DeLong:     z={delong.z:.4f}  p={delong.p_value:.4g}  (assumes row independence; rows share question_id)")
    print(f"question-id cluster CI:     [{ci_lo:.4f}, {ci_hi:.4f}]  (n_bootstrap={N_BOOTSTRAP}, seed={SEED})")
    print(f"cluster SE / naive DeLong SE: {se_inflation:.4f}x")
    print(f"cluster-SE z-test:          z={z_cluster:.4f}  p={p_cluster:.4g}")


if __name__ == "__main__":
    main()
