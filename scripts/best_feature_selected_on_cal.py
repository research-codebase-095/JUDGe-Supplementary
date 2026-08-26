"""Fixes a test-set-selection bug: paper_ablation_and_effect_size.py's
single_feature_aurocs() and judge_characterization.py's best_single_feature()
both pick the "best" single feature by its OWN AUROC on id_test, then report
that same AUROC on id_test in Table 1 / the ablation table - selecting and
reporting on the same held-out split. This is invalid regardless of which
direction it happens to bias (here it biases AGAINST the combiner, since a
test-set-selected single feature is an over-optimistic baseline).

Fix: select the winning feature using AUROC computed on the threshold_cal
split (never used for fitting or for the reported number itself), then report
THAT feature's AUROC on id_test as the number that goes in the paper.

Vision (ResNet-50) special case: logit_cache_resnet50.pt carries its own
combiner_fit(1500)/threshold_cal(300)/id_test(1500) split, distinct from the
separate logit_cache_imagenet1k_resnet50.pt full-scale cache (50,000 rows,
single split label "id_test_full_scale", no cal band of its own) that the
paper actually reports Table 1's n=50,000 numbers from. There is no
threshold_cal subset of the 50k full-scale set. Feature selection is done on
logit_cache_resnet50.pt's own threshold_cal (n=300) - the same small cache
already used to fit the combiner reported on the full-scale set - and the
winning feature is then evaluated on the full-scale 50k id_test as everywhere
else in the paper.

Usage:
    python scripts/best_feature_selected_on_cal.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from deployment_reliability.features import DEFAULT_FEATURE_DIRECTIONS, DEFAULT_FEATURE_NAMES, featurize  # noqa: E402
from deployment_reliability.router import auroc  # noqa: E402

DATA_DIR = os.path.join(REPO_ROOT, "data")


def orient(phi: torch.Tensor) -> torch.Tensor:
    return phi * DEFAULT_FEATURE_DIRECTIONS


def feature_aurocs(phi_oriented: torch.Tensor, correct: torch.Tensor) -> dict[str, float]:
    correct_bool = correct.to(dtype=torch.bool)
    return {
        name: auroc(phi_oriented[:, i][correct_bool], phi_oriented[:, i][~correct_bool])
        for i, name in enumerate(DEFAULT_FEATURE_NAMES)
    }


def pick_best_on_cal_report_on_test(
    phi_cal: torch.Tensor, correct_cal: torch.Tensor, phi_test: torch.Tensor, correct_test: torch.Tensor
) -> tuple[str, float, float]:
    """Returns (winning_feature_name, its_auroc_on_cal, its_auroc_on_test)."""
    cal_aurocs = feature_aurocs(orient(phi_cal), correct_cal)
    best_name = max(cal_aurocs, key=cal_aurocs.get)
    test_aurocs = feature_aurocs(orient(phi_test), correct_test)
    return best_name, cal_aurocs[best_name], test_aurocs[best_name]


def old_best_on_test(phi_test: torch.Tensor, correct_test: torch.Tensor) -> tuple[str, float]:
    """Reproduces the ORIGINAL (buggy) selection: pick by AUROC on id_test itself."""
    test_aurocs = feature_aurocs(orient(phi_test), correct_test)
    best_name = max(test_aurocs, key=test_aurocs.get)
    return best_name, test_aurocs[best_name]


def load_llm(tag: str) -> dict:
    cache = torch.load(os.path.join(DATA_DIR, f"llm_feature_cache_{tag}.pt"))
    phi, correct, splits = cache["phi"], cache["correct"], np.array(cache["splits"])
    m_cal = torch.from_numpy(splits == "threshold_cal")
    m_test = torch.from_numpy(splits == "id_test")
    return {"phi_cal": phi[m_cal], "correct_cal": correct[m_cal], "phi_test": phi[m_test], "correct_test": correct[m_test]}


def load_judge(cache_filename: str) -> dict:
    cache = torch.load(os.path.join(DATA_DIR, cache_filename))
    phi, correct, splits = cache["phi"], cache["correct"], np.array(cache["splits"])
    m_cal = torch.from_numpy(splits == "threshold_cal")
    m_test = torch.from_numpy(splits == "id_test")
    return {"phi_cal": phi[m_cal], "correct_cal": correct[m_cal], "phi_test": phi[m_test], "correct_test": correct[m_test]}


def load_vision() -> dict:
    small = torch.load(os.path.join(DATA_DIR, "logit_cache_resnet50.pt"))
    s = np.array(small["splits"])
    m_cal = torch.from_numpy(s == "threshold_cal")
    phi_cal = featurize(small["logits"][m_cal])
    correct_cal = small["logits"][m_cal].argmax(dim=-1) == small["labels"][m_cal]

    full = torch.load(os.path.join(DATA_DIR, "logit_cache_imagenet1k_resnet50.pt"))
    phi_test = featurize(full["logits"])
    correct_test = full["logits"].argmax(dim=-1) == full["labels"]
    return {"phi_cal": phi_cal, "correct_cal": correct_cal, "phi_test": phi_test, "correct_test": correct_test}


def main() -> None:
    configs = [
        ("ResNet-50 (vision)", load_vision()),
        ("GPT-2 (LLM)", load_llm("gpt2")),
        ("Pythia-160m (LLM)", load_llm("pythia160m")),
        ("Qwen2.5-0.5B-Instr. (LLM)", load_llm("qwen05binstruct")),
        ("Qwen2.5-0.5B-Instr. (judge)", load_judge("judge_feature_cache_mtbench.pt")),
        ("Qwen2.5-1.5B-Instr. (judge)", load_judge("judge_feature_cache_mtbench_1p5b.pt")),
        ("SmolLM2-360M-Instr. (judge)", load_judge("judge_feature_cache_mtbench_smollm2_360m.pt")),
    ]

    header = f"{'Backbone':<28}{'OLD (selected on test)':<28}{'NEW (selected on cal)':<28}{'changed?':<10}"
    print(header)
    print("-" * len(header))
    for name, d in configs:
        old_name, old_auroc = old_best_on_test(d["phi_test"], d["correct_test"])
        new_name, cal_auroc, new_auroc = pick_best_on_cal_report_on_test(
            d["phi_cal"], d["correct_cal"], d["phi_test"], d["correct_test"]
        )
        changed = "YES" if new_name != old_name else "no"
        print(f"{name:<28}{f'{old_name} ({old_auroc:.4f})':<28}{f'{new_name} ({new_auroc:.4f}, cal={cal_auroc:.4f})':<28}{changed:<10}")


if __name__ == "__main__":
    main()
