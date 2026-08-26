"""Reviewer check: judge2026.tex diagnoses the LLM combiner's loss to MSP as
an unstandardized-collinear-features-under-a-linear-fit artifact, "largely
resolved by standardization or a nonlinear combiner" - but this was checked
at exactly one L2 penalty (lambda=0.01, LogisticRegressionCombiner's
default), never varied. A reviewer correctly asked whether the failure is a
property of that one arbitrary hyperparameter rather than of the
unstandardized-feature-scale problem itself: if a stronger/weaker penalty
alone (no standardization) recovers MSP-level AUROC, "fixed by
standardization" would be an overclaim.

This script refits the linear combiner at a range of L2 values, no
standardization, on the three LLM backbones' existing combiner_fit/id_test
split, and reports AUROC vs MSP at each. No new model inference - reuses
cached phi/correct tensors.

Usage:
    python scripts/combiner_regularization_sweep.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from deployment_reliability.combiner import LogisticRegressionCombiner  # noqa: E402
from deployment_reliability.router import auroc  # noqa: E402

DATA_DIR = os.path.join(REPO_ROOT, "data")
L2_VALUES = (0.0, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0)


def main() -> None:
    configs = [
        ("GPT-2 (LLM)", "llm_feature_cache_gpt2.pt"),
        ("Pythia-160m (LLM)", "llm_feature_cache_pythia160m.pt"),
        ("Qwen2.5-0.5B-Instruct (LLM)", "llm_feature_cache_qwen05binstruct.pt"),
    ]

    for name, filename in configs:
        cache = torch.load(os.path.join(DATA_DIR, filename))
        splits = np.array(cache["splits"])
        m_fit, m_test = splits == "combiner_fit", splits == "id_test"
        phi_fit, correct_fit = cache["phi"][m_fit], cache["correct"][m_fit].float()
        phi_test, correct_test = cache["phi"][m_test], cache["correct"][m_test]
        msp_test = phi_test[:, 0]
        msp_auroc = auroc(msp_test[correct_test.bool()], msp_test[~correct_test.bool()])

        print(f"=== {name}  (MSP AUROC={msp_auroc:.4f}) ===")
        for l2 in L2_VALUES:
            combiner = LogisticRegressionCombiner(l2=l2).fit(phi_fit, correct_fit)
            scores = combiner.score(phi_test)
            a = auroc(scores[correct_test.bool()], scores[~correct_test.bool()])
            print(f"  l2={l2:<8} combiner AUROC={a:.4f}  (vs MSP: {a - msp_auroc:+.4f})")
        print()


if __name__ == "__main__":
    main()
