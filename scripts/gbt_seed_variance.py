"""Reviewer ask (#21): the paper's GBT-vs-MSP numbers for the three LLM
backbones (judge_characterization.py's gbt_scores(), used identically here)
were each reported from a single random_state=0 fit, with no hyperparameters
stated beyond max_iter and no seed variance - inconsistent with
stability_across_splits.py's 5-seed reporting used elsewhere in this paper.

Hyperparameters actually in effect (sklearn 1.9.0 HistGradientBoostingClassifier
defaults for everything not explicitly passed): loss=log_loss,
learning_rate=0.1, max_iter=100 (explicit, matches default), max_leaf_nodes=31,
max_depth=None, min_samples_leaf=20, l2_regularization=0.0, max_bins=255,
early_stopping='auto' (False below 10,000 samples - all three combiner_fit
splits here exceed that, so early stopping IS active, using a held-out
validation_fraction=0.1 slice of combiner_fit and n_iter_no_change=10,
tol=1e-7), scoring='loss'. Only random_state varies across seeds below.

Re-fits GBT with random_state=0..4 on each LLM backbone's combiner_fit split,
evaluates unrefit on id_test, reports mean+/-std AUROC and mean+/-std of the
GBT-vs-MSP AUROC gap across the five seeds. No new model inference - reuses
cached phi/correct.

Usage:
    python scripts/gbt_seed_variance.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from deployment_reliability.router import auroc  # noqa: E402

DATA_DIR = os.path.join(REPO_ROOT, "data")
SEEDS = (0, 1, 2, 3, 4)


def gbt_scores(phi_fit: torch.Tensor, correct_fit: torch.Tensor, phi_test: torch.Tensor, seed: int) -> torch.Tensor:
    from sklearn.ensemble import HistGradientBoostingClassifier

    gbt = HistGradientBoostingClassifier(random_state=seed, max_iter=100)
    gbt.fit(phi_fit.numpy(), correct_fit.numpy().astype(int))
    return torch.from_numpy(gbt.predict_proba(phi_test.numpy())[:, 1])


def main() -> None:
    configs = [
        ("GPT-2 (LLM)", "llm_feature_cache_gpt2.pt"),
        ("Pythia-160m (LLM)", "llm_feature_cache_pythia160m.pt"),
        ("Qwen2.5-0.5B-Instr. (LLM)", "llm_feature_cache_qwen05binstruct.pt"),
    ]

    header = f"{'Backbone':<28}{'MSP AUROC':>11}{'GBT mean':>11}{'GBT std':>9}{'gap mean':>10}{'gap std':>9}"
    print(header)
    print("-" * len(header))
    for name, filename in configs:
        cache = torch.load(os.path.join(DATA_DIR, filename))
        phi, correct, splits = cache["phi"], cache["correct"], np.array(cache["splits"])
        m_fit = torch.from_numpy(splits == "combiner_fit")
        m_test = torch.from_numpy(splits == "id_test")
        phi_fit, correct_fit = phi[m_fit], correct[m_fit]
        phi_test, correct_test = phi[m_test], correct[m_test].bool()

        msp_test = phi_test[:, 0]
        msp_auroc = auroc(msp_test[correct_test], msp_test[~correct_test])

        gbt_aurocs, gaps = [], []
        for seed in SEEDS:
            scores = gbt_scores(phi_fit, correct_fit, phi_test, seed)
            a = auroc(scores[correct_test], scores[~correct_test])
            gbt_aurocs.append(a)
            gaps.append(a - msp_auroc)
        gbt_aurocs, gaps = np.array(gbt_aurocs), np.array(gaps)
        print(f"{name:<28}{msp_auroc:>11.4f}{gbt_aurocs.mean():>11.4f}{gbt_aurocs.std():>9.4f}"
              f"{gaps.mean():>10.4f}{gaps.std():>9.4f}")
        print(f"    per-seed GBT AUROC: " + ", ".join(f"{a:.4f}" for a in gbt_aurocs))


if __name__ == "__main__":
    main()
