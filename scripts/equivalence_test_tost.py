"""Reviewer question (Gap 6 follow-up): judge2026.tex already calls the
combiner-vs-best-feature judge-task result a "diagnosed null, not a
demonstrated equivalence" -- a failed-to-reject-null p-value alone cannot
distinguish "no effect" from "underpowered test." This script makes that
distinction precise via a TOST (two one-sided tests, Schuirmann 1987)
equivalence test: does the observed combiner-vs-best-feature AUROC gap fall,
with 95% confidence (equivalently, a 90% CI), entirely inside a pre-specified
equivalence margin around zero?

Two margins are tested, side by side, neither picked in isolation:
  (1) a literature-convention margin, 0.02 AUROC, independent of this
      study's own noise;
  (2) each config's own minimum-detectable-effect (MDE = 2.80*SE, the exact
      formula and derivation scripts/power_analysis.py already uses) as a
      sensitivity margin -- explicitly a looser, somewhat circular choice
      (derived from the same variance the test itself uses), reported for
      transparency, not as the primary criterion.

No new model inference - reuses the same cached data, best_single_feature()
selection, and delong_test() as Table 1 and scripts/power_analysis.py.

Usage:
    python scripts/equivalence_test_tost.py
"""

from __future__ import annotations

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from scipy.stats import norm  # noqa: E402

from deployment_reliability.significance import delong_test  # noqa: E402

from judge_characterization import best_single_feature, load_all_judge_configs  # noqa: E402

MDE_MULTIPLIER = 1.96 + 0.84  # matches power_analysis.py's 2.80 exactly
CONVENTION_MARGIN = 0.02
Z_90 = norm.ppf(0.95)  # 1.645, for a 90% two-sided CI == TOST at alpha=0.05


def tost(diff: float, se: float, margin: float, alpha: float = 0.05) -> dict:
    """Two one-sided tests for |true diff| < margin. Equivalence at level
    alpha iff BOTH one-sided tests reject, i.e. p_tost = max(p_lower, p_upper) < alpha.
    Equivalent framing: the (1 - 2*alpha)*100% CI of diff is a subset of
    [-margin, +margin] -- for alpha=0.05 this is the 90% CI."""
    z_lower = (diff + margin) / se
    z_upper = (margin - diff) / se
    p_lower = norm.sf(z_lower)
    p_upper = norm.sf(z_upper)
    p_tost = max(p_lower, p_upper)
    ci_lo, ci_hi = diff - Z_90 * se, diff + Z_90 * se
    equivalent = ci_lo >= -margin and ci_hi <= margin
    return {
        "p_tost": p_tost, "p_lower": p_lower, "p_upper": p_upper,
        "ci90": (ci_lo, ci_hi), "equivalent": equivalent,
    }


def main() -> None:
    configs = load_all_judge_configs()
    for cfg in configs:
        if "SmolLM2" in cfg["name"]:
            continue  # SmolLM2 reported separately in the paper; keep this script scoped to the two Qwen configs
        correct = cfg["correct"]
        comb_scores = cfg["combiner"].score(cfg["phi"])
        best_name, best_scores = best_single_feature(cfg)
        dl = delong_test(correct, comb_scores, best_scores)
        se = abs(dl.auc_diff / dl.z) if dl.z != 0 else float("nan")
        mde = MDE_MULTIPLIER * se

        print(f"=== {cfg['name']} (n={len(correct)}, best feature = {best_name}) ===")
        print(f"  observed combiner-minus-best diff = {dl.auc_diff:+.4f}, DeLong-implied SE = {se:.4f}, own MDE = {mde:.4f}")
        for margin, label in [(CONVENTION_MARGIN, "0.02 convention"), (mde, "own-MDE sensitivity")]:
            r = tost(dl.auc_diff, se, margin)
            lo, hi = r["ci90"]
            verdict = "EQUIVALENT (established)" if r["equivalent"] else "NOT equivalent (not established)"
            print(f"  margin={margin:.4f} ({label}): 90% CI [{lo:+.4f}, {hi:+.4f}], TOST p={r['p_tost']:.4f} -> {verdict}")
        print()

    print("TOST equivalence is a distinct claim from significance: 'not significant' only means")
    print("a nonzero difference wasn't detected; 'equivalent' means the difference was shown small")
    print("enough to not matter, within the stated margin. Reporting both here shows precisely")
    print("whether this null is a genuine double-underpowering (fails both tests) or a real equivalence.")


if __name__ == "__main__":
    main()
