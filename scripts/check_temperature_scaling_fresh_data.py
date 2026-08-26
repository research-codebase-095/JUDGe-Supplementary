"""Checks whether the temperature-scaling null result (DESIGN.md 15.2,
"Calibration provided no benefit here") generalizes to genuinely NEW real
data, not just resamples of the same 300-point threshold_cal split
(notebooks/16_calibration_sample_size.ipynb already ruled out simple
fitting instability via bootstrap-resampling that same pool - a different,
narrower question).

Imagenette's own images are sampled from ImageNet-1k's TRAINING split; the
real, official ImageNet-1k VALIDATION split (data/logit_cache_imagenet1k_
resnet50.pt, the full-scale cache from scripts/evaluate_full_scale.py) is
a completely different set of image files for the same classes, never
touched by anything in this project's fitting steps. Filtering that cache
to Imagenette's 10 classes gives real, disjoint, genuinely new data.

Protocol: fit the combiner ONCE on the original combiner_fit split
(unchanged, per DESIGN.md 10.5). Reproduce the original baseline (fit
temperature scaling on threshold_cal, evaluate ECE on the held-out
id_test) to confirm this script's methodology matches DESIGN.md 15.2
exactly. Then repeat the identical fit-on-cal/evaluate-on-held-out-test
structure on the fresh data, split into two disjoint halves so both the
fitting and evaluation happen on points nothing in this project has ever
touched before.

Usage: python scripts/check_temperature_scaling_fresh_data.py
Requires data/logit_cache_resnet50.pt and data/logit_cache_imagenet1k_resnet50.pt
(run scripts/collect_logits.py resnet50 and scripts/collect_logits_imagenet1k.py
resnet50 first).
"""

import json
import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from deployment_reliability.calibration import TemperatureScaling  # noqa: E402
from deployment_reliability.combiner import LogisticRegressionCombiner  # noqa: E402
from deployment_reliability.features import featurize  # noqa: E402

DATA_DIR = os.path.join(REPO_ROOT, "data")

IMAGENETTE_SYNSETS = [
    "n01440764", "n02102040", "n02979186", "n03000684", "n03028079",
    "n03394916", "n03417042", "n03425413", "n03445777", "n03888257",
]


def ece(scores: torch.Tensor, correct: torch.Tensor, n_bins: int = 10) -> float:
    bins = torch.linspace(0, 1, n_bins + 1)
    total = 0.0
    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        # Half-open [lo, hi) for every bin except the last, which is closed
        # on the right ([lo, hi]) so a score of exactly 1.0 still lands in a
        # bin instead of being silently dropped from both the numerator and
        # the implicit weight denominator (same fix as evaluate_full_scale.py).
        m = (scores >= lo) & (scores <= hi) if i == n_bins - 1 else (scores >= lo) & (scores < hi)
        if m.sum() == 0:
            continue
        conf = scores[m].mean()
        acc = correct[m].float().mean()
        total += m.float().mean() * (conf - acc).abs()
    return float(total.item())


def main() -> None:
    small_path = os.path.join(DATA_DIR, "logit_cache_resnet50.pt")
    full_path = os.path.join(DATA_DIR, "logit_cache_imagenet1k_resnet50.pt")
    assert os.path.exists(small_path), f"{small_path} not found - run scripts/collect_logits.py resnet50 first"
    assert os.path.exists(full_path), f"{full_path} not found - run scripts/collect_logits_imagenet1k.py resnet50 first"

    with open(os.path.join(REPO_ROOT, "notebooks", "assets", "imagenet_class_index.json")) as f:
        class_index = json.load(f)
    syn_to_idx = {v[0]: int(k) for k, v in class_index.items()}
    imagenette_idxs = torch.tensor([syn_to_idx[s] for s in IMAGENETTE_SYNSETS])

    small = torch.load(small_path)
    splits = np.array(small["splits"])
    m_fit = torch.from_numpy(splits == "combiner_fit")
    m_cal_orig = torch.from_numpy(splits == "threshold_cal")
    m_test_orig = torch.from_numpy(splits == "id_test")

    logits_small, labels_small = small["logits"], small["labels"]
    correct_small = logits_small.argmax(dim=-1) == labels_small
    phi_small = featurize(logits_small)

    combiner = LogisticRegressionCombiner().fit(phi_small[m_fit], correct_small[m_fit].float())

    # --- Baseline: fit on threshold_cal, evaluate ECE on held-out id_test ---
    S_cal_orig = combiner.score(phi_small[m_cal_orig])
    correct_cal_orig = correct_small[m_cal_orig]
    temp_orig = TemperatureScaling().fit(S_cal_orig, correct_cal_orig.float())

    S_test_orig = combiner.score(phi_small[m_test_orig])
    correct_test_orig = correct_small[m_test_orig]
    ece_before_orig = ece(S_test_orig, correct_test_orig)
    ece_after_orig = ece(temp_orig.transform(S_test_orig), correct_test_orig)
    print(
        f"Baseline (fit threshold_cal n={int(m_cal_orig.sum())}, eval id_test n={int(m_test_orig.sum())}): "
        f"ECE before={ece_before_orig:.4f}  after={ece_after_orig:.4f}  T={temp_orig.log_t.exp().item():.4f}"
    )

    # --- Fresh, real, disjoint data: official ImageNet-1k val split, same classes ---
    full = torch.load(full_path)
    labels_full = full["labels"]
    mask_fresh = torch.isin(labels_full, imagenette_idxs)
    logits_fresh_all = full["logits"][mask_fresh]
    labels_fresh_all = labels_full[mask_fresh]
    n_fresh_total = len(logits_fresh_all)
    print(f"\nFresh, disjoint, same-class real data available: n={n_fresh_total}")

    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(n_fresh_total, generator=g)
    n_fresh_cal = n_fresh_total // 2
    cal_idx, test_idx = perm[:n_fresh_cal], perm[n_fresh_cal:]

    logits_fresh_cal, labels_fresh_cal = logits_fresh_all[cal_idx], labels_fresh_all[cal_idx]
    logits_fresh_test, labels_fresh_test = logits_fresh_all[test_idx], labels_fresh_all[test_idx]
    correct_fresh_cal = logits_fresh_cal.argmax(dim=-1) == labels_fresh_cal
    correct_fresh_test = logits_fresh_test.argmax(dim=-1) == labels_fresh_test
    phi_fresh_cal = featurize(logits_fresh_cal)
    phi_fresh_test = featurize(logits_fresh_test)

    S_fresh_cal = combiner.score(phi_fresh_cal)
    temp_fresh = TemperatureScaling().fit(S_fresh_cal, correct_fresh_cal.float())

    S_fresh_test = combiner.score(phi_fresh_test)
    ece_before_fresh = ece(S_fresh_test, correct_fresh_test)
    ece_after_fresh = ece(temp_fresh.transform(S_fresh_test), correct_fresh_test)
    print(
        f"Fresh data (fit fresh-cal n={len(cal_idx)}, eval fresh-test n={len(test_idx)}): "
        f"ECE before={ece_before_fresh:.4f}  after={ece_after_fresh:.4f}  T={temp_fresh.log_t.exp().item():.4f}"
    )

    ece_after_fresh_by_orig_temp = ece(temp_orig.transform(S_fresh_test), correct_fresh_test)
    print(
        f"Fresh-test, scaled by the ORIGINAL (threshold_cal-fit) temperature "
        f"T={temp_orig.log_t.exp().item():.4f}: ECE={ece_after_fresh_by_orig_temp:.4f}"
    )

    rng = np.random.default_rng(0)
    n_test = len(S_fresh_test)
    deltas = np.empty(2000)
    for i in range(2000):
        idx = rng.integers(0, n_test, size=n_test)
        s_b, c_b = S_fresh_test[idx], correct_fresh_test[idx]
        deltas[i] = ece(temp_fresh.transform(s_b), c_b) - ece(s_b, c_b)
    ci_lo, ci_hi = np.quantile(deltas, [0.025, 0.975])
    print(
        f"\nBootstrap 95% CI on (ECE_after - ECE_before) for fresh data, n_bootstrap=2000: "
        f"point={ece_after_fresh - ece_before_fresh:.4f}, CI=[{ci_lo:.4f}, {ci_hi:.4f}]"
    )

    acc_fresh = torch.cat([correct_fresh_cal, correct_fresh_test]).float().mean().item()
    acc_orig_test = correct_test_orig.float().mean().item()
    print(f"\nAccuracy sanity check: fresh-data accuracy={acc_fresh:.4f}  original id_test accuracy={acc_orig_test:.4f}")


if __name__ == "__main__":
    main()
