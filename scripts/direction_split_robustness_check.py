"""One-off investigation (not wired into the paper): does Table 3's use of
id_test's OWN labels to pick each feature's empirically-verified direction,
then reporting discordance computed on those same oriented id_test values,
matter numerically? Recomputes each feature's direction on the disjoint
threshold_cal split instead, applies that sign to id_test's raw values, and
compares discordance against MSP under both conventions. ResNet-50 has no
threshold_cal split in its cache (a separate 1,800-image calibration sample
is used instead, per Table 1's dagger footnote) so it is out of scope here.

Usage:
    python scripts/direction_split_robustness_check.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from deployment_reliability.features import (  # noqa: E402
    DEFAULT_FEATURE_DIRECTIONS,
    DEFAULT_FEATURE_NAMES,
    verify_feature_directions,
)
from deployment_reliability.router import auroc  # noqa: E402

DATA_DIR = os.path.join(REPO_ROOT, "data")
N_SAMPLE_PAIRS = 200_000
SEED = 0

CONFIGS = [
    ("Qwen2.5-0.5B-Instruct (judge)", "judge_feature_cache_mtbench.pt"),
    ("Qwen2.5-1.5B-Instruct (judge)", "judge_feature_cache_mtbench_1p5b.pt"),
    ("SmolLM2-360M-Instruct (judge)", "judge_feature_cache_mtbench_smollm2_360m.pt"),
]


def pairwise_discordance(a: np.ndarray, b: np.ndarray) -> float:
    n = len(a)
    rng = np.random.default_rng(SEED)
    if n * (n - 1) // 2 <= N_SAMPLE_PAIRS:
        ii, jj = np.triu_indices(n, k=1)
    else:
        ii = rng.integers(0, n, size=N_SAMPLE_PAIRS)
        jj = rng.integers(0, n, size=N_SAMPLE_PAIRS)
        mask = ii != jj
        ii, jj = ii[mask], jj[mask]
    da, db = a[ii] - a[jj], b[ii] - b[jj]
    return float((((da > 0) & (db < 0)) | ((da < 0) & (db > 0))).mean())


def signs_from(phi: torch.Tensor, correct: torch.Tensor) -> dict:
    d = verify_feature_directions(phi, correct)
    return {n: (1.0 if d[n] else -1.0) for n in DEFAULT_FEATURE_NAMES}


def auroc_per_feature(phi_oriented: torch.Tensor, correct: torch.Tensor) -> dict:
    correct_bool = correct.bool()
    out = {}
    for i, name in enumerate(DEFAULT_FEATURE_NAMES):
        col = phi_oriented[:, i]
        out[name] = auroc(col[correct_bool], col[~correct_bool])
    return out


def main() -> None:
    for name, path in CONFIGS:
        cache = torch.load(os.path.join(DATA_DIR, path))
        splits = np.array(cache["splits"])
        m_cal = splits == "threshold_cal"
        m_test = splits == "id_test"

        phi_cal, correct_cal = cache["phi"][m_cal], cache["correct"][m_cal]
        phi_test, correct_test = cache["phi"][m_test], cache["correct"][m_test]
        n_cal, n_test = phi_cal.shape[0], phi_test.shape[0]

        signs_test = signs_from(phi_test, correct_test)   # method (a): current paper
        signs_cal = signs_from(phi_cal, correct_cal)       # method (b): disjoint

        oriented_test_a = phi_test * DEFAULT_FEATURE_DIRECTIONS * torch.tensor(
            [signs_test[n] for n in DEFAULT_FEATURE_NAMES]
        )
        oriented_test_b = phi_test * DEFAULT_FEATURE_DIRECTIONS * torch.tensor(
            [signs_cal[n] for n in DEFAULT_FEATURE_NAMES]
        )

        auroc_test = auroc_per_feature(phi_test * DEFAULT_FEATURE_DIRECTIONS, correct_test)  # base AUROC (global sign) for reference
        auroc_a = auroc_per_feature(oriented_test_a, correct_test)
        auroc_b = auroc_per_feature(oriented_test_b, correct_test)

        print(f"=== {name}  (threshold_cal n={n_cal}, id_test n={n_test}) ===")
        print(f"  {'feature':<20}{'sign(a:id_test)':>16}{'sign(b:cal)':>13}{'flip?':>7}"
              f"{'AUROC id_test (a-sign)':>26}{'AUROC id_test (b-sign)':>26}")
        for fname in DEFAULT_FEATURE_NAMES:
            sa, sb = signs_test[fname], signs_cal[fname]
            flip = "YES" if sa != sb else "no"
            print(f"  {fname:<20}{sa:>16.0f}{sb:>13.0f}{flip:>7}{auroc_a[fname]:>26.4f}{auroc_b[fname]:>26.4f}")

        msp_a = oriented_test_a[:, 0].numpy()
        msp_b = oriented_test_b[:, 0].numpy()
        print(f"\n  Discordance vs MSP, id_test pairs, method (a) current-paper sign vs method (b) cal-disjoint sign:")
        for i, fname in enumerate(DEFAULT_FEATURE_NAMES[1:], start=1):
            col_a = oriented_test_a[:, i].numpy()
            col_b = oriented_test_b[:, i].numpy()
            disc_a = pairwise_discordance(msp_a, col_a) * 100
            disc_b = pairwise_discordance(msp_b, col_b) * 100
            print(f"    {fname:<20} (a) {disc_a:6.2f}%   (b) {disc_b:6.2f}%   delta {disc_b - disc_a:+6.2f} pts")
        print()


if __name__ == "__main__":
    main()
