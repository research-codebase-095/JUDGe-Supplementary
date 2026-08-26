"""Checks whether the deterministic, evenly-spaced (systematic/stride-based)
window selection used throughout scripts/collect_llm_free_running.py,
check_windowed_correctness.py, and check_self_consistency.py introduces a
detectable bias relative to genuine random sampling - the methodology
question DESIGN.md 14.15 answers.

Systematic sampling is a classical survey-sampling technique (Cochran,
"Sampling Techniques", 3rd ed., 1977, Ch. 8): statistically equivalent to
simple random sampling UNLESS the population has a trend or periodicity
that lines up with the sampling interval. The standard check for this,
used here, is REPLICATED SYSTEMATIC SAMPLING: draw several independent
systematic samples at the same interval but different starting offsets,
and see whether the resulting statistic is sensitive to which offset was
used. This script runs that check (Test 1) plus a direct stride-vs-random
comparison at the AUROC level (Test 2), on real GPT-2/WikiText-2
teacher-forced data - cheap since it needs only one forward pass per
window, not autoregressive generation.

Real result (DESIGN.md 14.15): AUROC across all `stride` independent
offsets spans only 0.0042 (far inside a single offset's own ~0.013-wide
95% CI) and the project's actual stride is statistically indistinguishable
from genuine random sampling (0.8199 vs 0.8181, heavily overlapping CIs) -
no detectable periodicity bias.

Usage: python scripts/check_stride_sampling_bias.py
"""

import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from deployment_reliability.features import featurize  # noqa: E402
from deployment_reliability.llm_backbone import load_frozen_gpt2  # noqa: E402
from deployment_reliability.significance import bootstrap_auroc_ci  # noqa: E402

DATA_DIR = os.path.join(REPO_ROOT, "data")
PROMPT_LEN = 50
GEN_LEN = 50
WINDOW_SIZE = PROMPT_LEN + GEN_LEN
N_WINDOWS = 400


def main() -> None:
    text_path = os.path.join(DATA_DIR, "wikitext2_valid.txt")
    assert os.path.exists(text_path), (
        f"{text_path} not found - run scripts/collect_llm_free_running.py once first (it downloads this file)"
    )
    with open(text_path, encoding="utf-8") as f:
        text = f.read()
    print("loading frozen gpt2...")
    model, tokenizer = load_frozen_gpt2("gpt2")

    full_ids = torch.tensor(tokenizer(text)["input_ids"])
    total_windows_available = len(full_ids) // WINDOW_SIZE
    n_windows = min(N_WINDOWS, total_windows_available)
    stride = max(1, total_windows_available // n_windows)
    print(f"total_windows_available={total_windows_available}  n_windows={n_windows}  stride={stride}")

    @torch.inference_mode()
    def teacher_forced(starts: list[int]):
        correct_list, msp_list = [], []
        for start in starts:
            window = full_ids[start : start + WINDOW_SIZE]
            outputs = model(input_ids=window.unsqueeze(0))
            logits = outputs.logits.squeeze(0)
            pred_logits = logits[PROMPT_LEN - 1 : WINDOW_SIZE - 1]
            target = window[PROMPT_LEN:]
            correct = pred_logits.argmax(dim=-1) == target
            phi = featurize(pred_logits, normalize_l2=True)
            correct_list.append(correct)
            msp_list.append(phi[:, 0])
        return torch.cat(correct_list), torch.cat(msp_list)

    def auroc_for_starts(starts: list[int], label: str):
        correct, msp = teacher_forced(starts)
        acc = correct.float().mean().item()
        pos, neg = msp[correct], msp[~correct]
        r = bootstrap_auroc_ci(pos, neg, n_bootstrap=2000, seed=0)
        print(f"{label:35s} n={len(correct):6d}  accuracy={acc:.4f}  MSP AUROC={r.auroc:.4f}  CI=[{r.ci_lo:.4f}, {r.ci_hi:.4f}]")
        return r.auroc, r.ci_lo, r.ci_hi

    print("\n=== Test 1: replicated systematic sampling - AUROC across all independent stride offsets ===")
    print("(Cochran 1977 Ch.8's standard check: if periodicity mattered, different offsets would diverge)")
    offset_results = []
    for offset in range(stride):
        starts = [
            (offset + i * stride) * WINDOW_SIZE
            for i in range(n_windows)
            if (offset + i * stride) < total_windows_available
        ]
        auroc, lo, hi = auroc_for_starts(starts, f"stride offset={offset}")
        offset_results.append(auroc)

    print(
        f"\nAUROC range across the {stride} independent stride offsets: "
        f"[{min(offset_results):.4f}, {max(offset_results):.4f}]  spread={max(offset_results) - min(offset_results):.4f}"
    )

    print("\n=== Test 2: the project's actual offset=0 stride vs genuine random sampling ===")
    stride_starts = [i * stride * WINDOW_SIZE for i in range(n_windows)]
    auroc_stride, lo_s, hi_s = auroc_for_starts(stride_starts, "project's actual stride (offset=0)")

    rng = np.random.default_rng(0)
    random_block_indices = rng.choice(total_windows_available, size=n_windows, replace=False)
    random_starts = [int(b) * WINDOW_SIZE for b in random_block_indices]
    auroc_random, lo_r, hi_r = auroc_for_starts(random_starts, "genuine random sampling")

    overlap = len(set(stride_starts) & set(random_starts))
    print(f"\nWindow-start overlap between stride and random (out of {n_windows}): {overlap}")
    print(f"(expected under independence: ~{n_windows * n_windows / total_windows_available:.1f})")

    ci_overlap = not (hi_s < lo_r or hi_r < lo_s)
    print(f"\nDo the two AUROC 95% CIs overlap? {'YES - not distinguishable' if ci_overlap else 'NO - real difference'}")


if __name__ == "__main__":
    main()
