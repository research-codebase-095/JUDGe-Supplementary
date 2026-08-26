"""Reviewer item 13 on judge2026.tex: the paper's combiner_fit/threshold_cal/
id_test split (three_way_split, seed=0, ratios 0.4/0.1/0.5) partitions the
1,284 filtered MT-Bench rows at the ROW level. Since those rows span only a
few dozen unique questions, the same question can appear in all three splits
at once. The paper already cluster-bootstraps by question_id at evaluation
time (paper_diagnostics.cluster_bootstrap_auroc_diff), which addresses
INFERENCE-variance correlation (rows sharing a question aren't independent
draws for computing a confidence interval) -- but that is a distinct concern
from whether the combiner GENERALIZES to genuinely unseen questions, since a
row-level split lets the combiner be fit and evaluated on rows drawn from the
identical set of questions it trained on.

This script redoes the split at the QUESTION level instead: every row
belonging to a given question_id is assigned to exactly one of
combiner_fit/threshold_cal/id_test, so no question ever appears in two
splits. Reuses already-cached phi/correct/question_id tensors only -- no new
model inference, just a different partition of data already on disk.

Reuses best_single_feature() from judge_characterization.py unmodified: it
accepts any dict shaped like load_judge_config()'s output
(phi_fit/correct_fit/phi_cal/correct_cal/phi/correct), so a question-level
split can be fed through the identical bootstrap-stabilized selection
protocol Table 1 uses, without reimplementing it.

Usage:
    python scripts/question_level_resplit.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from deployment_reliability.significance import delong_test  # noqa: E402
from deployment_reliability.combiner import LogisticRegressionCombiner  # noqa: E402
from deployment_reliability.router import auroc  # noqa: E402

from judge_characterization import best_single_feature  # noqa: E402

DATA_DIR = os.path.join(REPO_ROOT, "data")
SEED = 0
RATIOS = (0.4, 0.1, 0.5)

CONFIGS = [
    ("judge_feature_cache_mtbench.pt", "Qwen2.5-0.5B-Instruct (judge)"),
    ("judge_feature_cache_mtbench_1p5b.pt", "Qwen2.5-1.5B-Instruct (judge)"),
]


def question_level_split(question_id: np.ndarray, ratios=RATIOS, seed: int = SEED):
    """Splits UNIQUE question_ids (not rows) into three disjoint sets at
    roughly `ratios` proportions of question COUNT, then assigns every row
    to whichever split its question_id landed in. Row-count ratios will not
    exactly match `ratios` (questions vary in row count), which is expected
    and reported explicitly rather than forced."""
    unique_qs = np.unique(question_id)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(unique_qs))
    n_fit_q = int(round(len(unique_qs) * ratios[0]))
    n_cal_q = int(round(len(unique_qs) * ratios[1]))
    fit_qs = set(unique_qs[perm[:n_fit_q]])
    cal_qs = set(unique_qs[perm[n_fit_q:n_fit_q + n_cal_q]])
    test_qs = set(unique_qs[perm[n_fit_q + n_cal_q:]])
    assert fit_qs.isdisjoint(cal_qs) and fit_qs.isdisjoint(test_qs) and cal_qs.isdisjoint(test_qs)

    m_fit = np.isin(question_id, list(fit_qs))
    m_cal = np.isin(question_id, list(cal_qs))
    m_test = np.isin(question_id, list(test_qs))
    assert (m_fit | m_cal | m_test).all() and not (m_fit & m_cal).any() and not (m_fit & m_test).any() and not (m_cal & m_test).any()
    return m_fit, m_cal, m_test, len(fit_qs), len(cal_qs), len(test_qs), len(unique_qs)


def build_config(cache_filename: str, name: str) -> dict:
    cache = torch.load(os.path.join(DATA_DIR, cache_filename))
    phi, correct = cache["phi"], cache["correct"]
    question_id = np.array(cache["question_id"])
    m_fit, m_cal, m_test, n_fit_q, n_cal_q, n_test_q, n_q_total = question_level_split(question_id)

    combiner = LogisticRegressionCombiner().fit(phi[m_fit], correct[m_fit].float())
    config = {
        "name": name,
        "phi_fit": phi[m_fit], "correct_fit": correct[m_fit].bool(),
        "phi_cal": phi[m_cal], "correct_cal": correct[m_cal].bool(),
        "phi": phi[m_test], "correct": correct[m_test], "combiner": combiner,
    }
    meta = {
        "n_q_total": n_q_total, "n_fit_q": n_fit_q, "n_cal_q": n_cal_q, "n_test_q": n_test_q,
        "n_fit_rows": int(m_fit.sum()), "n_cal_rows": int(m_cal.sum()), "n_test_rows": int(m_test.sum()),
    }
    return config, meta


def evaluate(config: dict) -> dict:
    correct_bool = config["correct"].bool()
    msp = config["phi"][:, 0]
    msp_auroc = auroc(msp[correct_bool], msp[~correct_bool])

    best_name, best_scores = best_single_feature(config)
    best_auroc = auroc(best_scores[correct_bool], best_scores[~correct_bool])

    comb_scores = config["combiner"].score(config["phi"])
    comb_auroc = auroc(comb_scores[correct_bool], comb_scores[~correct_bool])

    dl = delong_test(config["correct"], comb_scores, best_scores)
    return {
        "msp_auroc": msp_auroc, "best_name": best_name, "best_auroc": best_auroc,
        "comb_auroc": comb_auroc, "diff": dl.auc_diff, "p": dl.p_value,
    }


EXISTING_ROW_LEVEL = {
    "Qwen2.5-0.5B-Instruct (judge)": dict(msp=0.5522, best="msp", best_auroc=0.5522, comb=0.5734, p=0.1004),
    "Qwen2.5-1.5B-Instruct (judge)": dict(msp=0.6846, best="entropy", best_auroc=0.6850, comb=0.6730, p=0.05824),
}


def main() -> None:
    for cache_filename, name in CONFIGS:
        config, meta = build_config(cache_filename, name)
        result = evaluate(config)

        print(f"\n{'='*100}\n{name} -- QUESTION-LEVEL resplit (seed={SEED})\n{'='*100}")
        print(f"  unique questions total: {meta['n_q_total']}  "
              f"(fit={meta['n_fit_q']}, cal={meta['n_cal_q']}, test={meta['n_test_q']})")
        print(f"  resulting ROW counts:   fit={meta['n_fit_rows']}, cal={meta['n_cal_rows']}, test={meta['n_test_rows']}  "
              f"(target ratios {RATIOS}, exact match not expected -- questions vary in row count)")
        print(f"  MSP AUROC        = {result['msp_auroc']:.4f}")
        print(f"  best feature     = {result['best_name']} (AUROC {result['best_auroc']:.4f})")
        print(f"  combiner AUROC   = {result['comb_auroc']:.4f}")
        print(f"  combiner - best  = {result['diff']:+.4f}  (naive DeLong p = {result['p']:.4g})")

        ref = EXISTING_ROW_LEVEL[name]
        print(f"\n  FOR COMPARISON -- existing ROW-LEVEL split (Table 1, already in the paper):")
        print(f"    MSP={ref['msp']:.4f}  best={ref['best']} ({ref['best_auroc']:.4f})  "
              f"combiner={ref['comb']:.4f}  p={ref['p']:.4g}")


if __name__ == "__main__":
    main()
