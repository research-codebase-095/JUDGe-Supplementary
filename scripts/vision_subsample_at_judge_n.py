"""Reviewer control: judge2026.tex's vision "win" (combiner beats margin by
+0.008 AUROC, DeLong p=3e-58, Table 1) is measured at n=50,000. The judge
task's null results are measured at n=642, where the paper's own MDE
analysis (power_analysis.py) shows only gaps of ~0.018-0.036 AUROC are
detectable at 80% power -- an effect the size of vision's +0.008 would be
invisible there. The open question this script answers: does the
already-established vision effect (already-fit combiner from Table 1's own
protocol, margin as the already-established best single feature) remain
statistically detectable once evaluated at n=642, or does it disappear into
noise the same way the judge task's real (or possibly equally-sized,
never-detected) effect does?

Two clearly separated analyses:

Analysis 1 (primary, detectability): reuse Table 1's own fitted combiner
(fit once on the small 1500-image combiner_fit pool, logit_cache_resnet50.pt,
exactly as scripts/best_feature_selected_on_cal.py and Table 1 do) and
margin as the already-established best feature (no reselection per draw).
Subsample the 50,000-image id_test pool (logit_cache_imagenet1k_resnet50.pt)
down to n=642 without replacement, 1000 times (seeds 0..999), and run a
paired DeLong test (combiner vs. margin) on each subsample. This isolates
"does an already-known-good effect survive DETECTION at judge-task n" from
any question about whether n=642 is enough to FIT a good combiner in the
first place.

Analysis 2 (secondary, fit-and-test at judge scale): repeat, but also refit
the combiner from scratch on a freshly-subsampled pool sized to match the
judge task's own absolute split sizes (combiner_fit=514, threshold_cal=128,
id_test=642, drawn from the same 50k pool each time, non-overlapping),
reselecting the best feature via cal-set AUROC each draw (same protocol as
vision_seed_stability.py). This tests whether FITTING at judge-scale data
also degrades the effect, not just detecting an already-good fit.

No new model inference -- reuses the same cached logits as Table 1's vision
row and scripts/best_feature_selected_on_cal.py / vision_seed_stability.py.

Usage:
    python scripts/vision_subsample_at_judge_n.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from deployment_reliability.combiner import LogisticRegressionCombiner  # noqa: E402
from deployment_reliability.features import DEFAULT_FEATURE_DIRECTIONS, DEFAULT_FEATURE_NAMES, featurize  # noqa: E402
from deployment_reliability.router import auroc  # noqa: E402
from deployment_reliability.significance import delong_test  # noqa: E402

DATA_DIR = os.path.join(REPO_ROOT, "data")
N_DRAWS = 1000
JUDGE_N_TEST = 642
JUDGE_N_FIT = 514
JUDGE_N_CAL = 128


def orient(phi: torch.Tensor) -> torch.Tensor:
    return phi * DEFAULT_FEATURE_DIRECTIONS


def feature_aurocs(phi_oriented: torch.Tensor, correct: torch.Tensor) -> dict[str, float]:
    correct_bool = correct.to(dtype=torch.bool)
    return {
        name: auroc(phi_oriented[:, i][correct_bool], phi_oriented[:, i][~correct_bool])
        for i, name in enumerate(DEFAULT_FEATURE_NAMES)
    }


def load_established_combiner_and_full_test():
    """Table 1's own protocol: combiner fit once on the 1500-image
    combiner_fit pool; margin is the already-established best feature
    (selected on threshold_cal by best_feature_selected_on_cal.py, matching
    Table 1's printed row)."""
    small = torch.load(os.path.join(DATA_DIR, "logit_cache_resnet50.pt"))
    s = np.array(small["splits"])
    m_fit = s == "combiner_fit"
    phi_fit = featurize(small["logits"][m_fit])
    correct_fit = (small["logits"][m_fit].argmax(dim=-1) == small["labels"][m_fit]).float()
    combiner = LogisticRegressionCombiner().fit(phi_fit, correct_fit)

    full = torch.load(os.path.join(DATA_DIR, "logit_cache_imagenet1k_resnet50.pt"))
    phi_test_full = featurize(full["logits"])
    correct_test_full = (full["logits"].argmax(dim=-1) == full["labels"]).bool()
    return combiner, phi_test_full, correct_test_full


def load_50k_pool_for_refit():
    """The 50k full-scale pool, for Analysis 2's fresh subsampled fit/cal/test
    draws. Distinct from the small 1800-image combiner_fit/threshold_cal
    cache used by Table 1 itself -- here we draw ALL of fit/cal/test from the
    same 50k pool each iteration, matching the judge task's own single-pool
    split design (unlike vision's actual Table-1 protocol, which uses a
    separate small calibration sample; this is a deliberate departure to
    mimic "if vision had judge-task-sized data available end-to-end")."""
    full = torch.load(os.path.join(DATA_DIR, "logit_cache_imagenet1k_resnet50.pt"))
    phi = featurize(full["logits"])
    correct = (full["logits"].argmax(dim=-1) == full["labels"]).bool()
    return phi, correct


def analysis1_detectability(combiner, phi_test_full, correct_test_full):
    margin_idx = DEFAULT_FEATURE_NAMES.index("logit_margin")
    phi_test_oriented = orient(phi_test_full)
    margin_col_full = phi_test_oriented[:, margin_idx]
    comb_scores_full = combiner.score(phi_test_full)

    n_full = len(correct_test_full)
    p_values = np.empty(N_DRAWS)
    gaps = np.empty(N_DRAWS)
    for i, seed in enumerate(range(N_DRAWS)):
        rng = np.random.default_rng(seed)
        idx = rng.choice(n_full, size=JUDGE_N_TEST, replace=False)
        correct_sub = correct_test_full[idx]
        margin_sub = margin_col_full[idx]
        comb_sub = comb_scores_full[idx]
        # DeLong requires both classes present; ImageNet top-1 correctness at
        # n=642 should always have both, but guard anyway rather than crash.
        if correct_sub.all() or (~correct_sub).all():
            p_values[i] = np.nan
            gaps[i] = np.nan
            continue
        dl = delong_test(correct_sub, comb_sub, margin_sub)
        p_values[i] = dl.p_value
        gaps[i] = dl.auc_diff

    valid = ~np.isnan(p_values)
    p_values, gaps = p_values[valid], gaps[valid]
    frac_sig = (p_values < 0.05).mean()
    print(f"Analysis 1 -- detectability of Table 1's ALREADY-FIT vision combiner-vs-margin effect at n={JUDGE_N_TEST}")
    print(f"  draws: {len(p_values)} (of {N_DRAWS} requested, seeds 0..{N_DRAWS-1})")
    print(f"  observed AUROC gap:  mean={gaps.mean():+.4f}  std={gaps.std():.4f}  min={gaps.min():+.4f}  max={gaps.max():+.4f}")
    print(f"  DeLong p-value:      mean={p_values.mean():.4f}  median={np.median(p_values):.4f}")
    print(f"  fraction of draws with p<0.05: {frac_sig:.3f}  ({int(frac_sig*len(p_values))}/{len(p_values)})")
    print()


def analysis2_refit_at_judge_scale(phi_pool, correct_pool):
    n_pool = len(correct_pool)
    n_needed = JUDGE_N_FIT + JUDGE_N_CAL + JUDGE_N_TEST
    p_values = np.empty(N_DRAWS)
    gaps = np.empty(N_DRAWS)
    best_names = []
    for i, seed in enumerate(range(N_DRAWS)):
        rng = np.random.default_rng(seed + 10_000)  # distinct seed stream from Analysis 1
        idx = rng.choice(n_pool, size=n_needed, replace=False)
        fit_idx = idx[:JUDGE_N_FIT]
        cal_idx = idx[JUDGE_N_FIT : JUDGE_N_FIT + JUDGE_N_CAL]
        test_idx = idx[JUDGE_N_FIT + JUDGE_N_CAL :]

        phi_fit, correct_fit = phi_pool[fit_idx], correct_pool[fit_idx].float()
        phi_cal, correct_cal = phi_pool[cal_idx], correct_pool[cal_idx]
        phi_test, correct_test = phi_pool[test_idx], correct_pool[test_idx]

        cal_aurocs = feature_aurocs(orient(phi_cal), correct_cal)
        best_name = max(cal_aurocs, key=cal_aurocs.get)
        best_names.append(best_name)
        best_idx = DEFAULT_FEATURE_NAMES.index(best_name)
        best_col_test = orient(phi_test)[:, best_idx]

        combiner = LogisticRegressionCombiner().fit(phi_fit, correct_fit)
        comb_scores_test = combiner.score(phi_test)

        if correct_test.all() or (~correct_test).all():
            p_values[i] = np.nan
            gaps[i] = np.nan
            continue
        dl = delong_test(correct_test, comb_scores_test, best_col_test)
        p_values[i] = dl.p_value
        gaps[i] = dl.auc_diff

    valid = ~np.isnan(p_values)
    p_values, gaps = p_values[valid], gaps[valid]
    frac_sig = (p_values < 0.05).mean()
    from collections import Counter

    print(f"Analysis 2 -- REFIT combiner + reselect best feature at judge-scale n (fit={JUDGE_N_FIT}, cal={JUDGE_N_CAL}, test={JUDGE_N_TEST})")
    print(f"  draws: {len(p_values)} (of {N_DRAWS} requested)")
    print(f"  best-feature selection across draws: {dict(Counter(best_names))}")
    print(f"  observed AUROC gap:  mean={gaps.mean():+.4f}  std={gaps.std():.4f}  min={gaps.min():+.4f}  max={gaps.max():+.4f}")
    print(f"  DeLong p-value:      mean={p_values.mean():.4f}  median={np.median(p_values):.4f}")
    print(f"  fraction of draws with p<0.05: {frac_sig:.3f}  ({int(frac_sig*len(p_values))}/{len(p_values)})")
    print()


def main() -> None:
    combiner, phi_test_full, correct_test_full = load_established_combiner_and_full_test()
    analysis1_detectability(combiner, phi_test_full, correct_test_full)

    phi_pool, correct_pool = load_50k_pool_for_refit()
    analysis2_refit_at_judge_scale(phi_pool, correct_pool)

    print("For comparison -- judge task's own actual observed results (Table 1, already in the paper):")
    print("  Qwen 0.5B (judge): combiner-vs-best-feature diff=+0.0211, naive DeLong p=0.100")
    print("  Qwen 1.5B (judge): combiner-vs-best-feature diff=-0.0121, naive DeLong p=0.058")


if __name__ == "__main__":
    main()
