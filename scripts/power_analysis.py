"""Answers a reviewer question about judge2026.tex: at n=642 (Qwen judges) and
n=200 (SmolLM2 subset), what AUROC gap between the combiner and the best
single feature could this sample size actually detect?

Uses the DeLong test's own asymptotic variance (delong_test already computes
this to produce its z-statistic) to back out a standard error for the
observed AUROC difference, then reports the minimum detectable effect (MDE)
at alpha=0.05, 80% power, via the standard two-sided normal-approximation
rule MDE = (z_{1-alpha/2} + z_{power}) * SE = 2.80 * SE - no new model
inference, reuses the same disjoint threshold_cal-then-id_test best-feature
selection and cached data as Table 1 (best_single_feature() in
judge_characterization.py was fixed to this protocol; it previously selected
in-sample on id_test itself, which agreed with Table 1 at 0.5B/SmolLM2 but
picked normalized_entropy over MSP at 1.5B on a near-tie, silently shifting
this script's 1.5B MDE from 0.017 to 0.018).

Usage:
    python scripts/power_analysis.py
"""

from __future__ import annotations

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from deployment_reliability.significance import delong_test  # noqa: E402

from judge_characterization import best_single_feature, load_all_judge_configs  # noqa: E402

Z_ALPHA_OVER_2 = 1.96  # two-sided, alpha=0.05
Z_POWER_80 = 0.84  # 80% power
MDE_MULTIPLIER = Z_ALPHA_OVER_2 + Z_POWER_80  # ~2.80


def main() -> None:
    configs = load_all_judge_configs()
    header = f"{'Config':<32}{'n':>6}{'obs. diff':>11}{'DeLong z':>10}{'implied SE':>12}{'MDE (80% power)':>18}"
    print(header)
    print("-" * len(header))
    for cfg in configs:
        correct = cfg["correct"]
        comb_scores = cfg["combiner"].score(cfg["phi"])
        _best_name, best_scores = best_single_feature(cfg)
        dl = delong_test(correct, comb_scores, best_scores)
        se = abs(dl.auc_diff / dl.z) if dl.z != 0 else float("nan")
        mde = MDE_MULTIPLIER * se
        print(f"{cfg['name']:<32}{len(correct):>6}{dl.auc_diff:>+11.4f}{dl.z:>10.3f}{se:>12.4f}{mde:>18.4f}")
    print()
    print("MDE = smallest true AUROC gap this n would detect 80% of the time at alpha=0.05,")
    print("under this comparison's observed variance - not itself a claim about the true gap.")


if __name__ == "__main__":
    main()
