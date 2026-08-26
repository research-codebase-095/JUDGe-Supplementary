"""Reviewer request (Gap-1 follow-up): a concrete, per-config failure-mode
table -- high-confidence-wrong, low-confidence-correct, high signal
disagreement, and combiner-vs-best-single-feature disagreement -- rather than
these rates scattered across prose. No new experiment: everything here is
computed from the same cached tensors (data/judge_feature_cache_mtbench*.pt)
and the same combiner_fit/threshold_cal/id_test splits already used
throughout the paper.

Methodology, made explicit so it doesn't get conflated with other tables:
- High-confidence-wrong / low-confidence-correct: EXACTLY Rule 1 / Rule 2 from
  scripts/judge_characterization.py -- MSP's 75th/25th percentile computed on
  the FULL id_test distribution (not on the correct/incorrect subset itself,
  which would be tautologically ~25% by construction), then checked against
  the incorrect/correct subsets respectively. Recomputed here only to attach
  it to the same table as the two new rows below, not because the original
  numbers were in doubt (an earlier draft of this script mistakenly
  conditioned the percentile on the subset itself and produced a tautological
  ~25% at every config; fixed to match judge_characterization.py exactly).
- High signal disagreement: a NEW per-example rule, deliberately NOT the same
  as Table 3's pairwise rank-discordance (which is a fraction of example
  PAIRS, not a per-example flag) and NOT the same as judge_characterization.py's
  old Rule 3 (z-score joint-extremity), which this project already flagged as
  methodologically inconsistent with the rest of the paper. Instead: an
  example counts as "high disagreement" if MSP (oriented) sits at or above its
  own 75th percentile on id_test while energy_score (oriented) sits at or
  below its own 25th percentile, or vice versa -- i.e. the two signals are in
  opposite extreme tails, unconditional on correctness (this is about internal
  uncertainty/disagreement, not accuracy). This mirrors Rule 1/Rule 2's own
  percentile convention rather than inventing a third, incompatible metric.
  Energy is used (not L2 norm) because Table 3 shows energy has comparable or
  larger MSP-discordance at all three configs (62/42/68%) and is the feature
  Section 6.1 already singles out alongside L2 norm as the driver.
- Combiner vs. best single feature: recomputed against each config's OWN best
  feature selected on threshold_cal (matching Table 1's own selection
  protocol) -- MSP at 0.5B and 1.5B, logit_l2_norm at SmolLM2 -- not uniformly
  against MSP, since the reviewer's request was literally "combiner disagrees
  with the best single feature," and that feature differs at SmolLM2. Sign
  compared against each score's own median on id_test (same convention as the
  combiner-vs-MSP flip rate already in the paper).

Usage:
    python scripts/failure_mode_table.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from deployment_reliability.combiner import LogisticRegressionCombiner  # noqa: E402
from deployment_reliability.features import (  # noqa: E402
    DEFAULT_FEATURE_DIRECTIONS,
    DEFAULT_FEATURE_NAMES,
    verify_feature_directions,
)
from deployment_reliability.router import auroc  # noqa: E402

from judge_characterization import (  # noqa: E402
    bootstrap_stabilized_direction_and_best_feature,
    load_judge_config,
    oriented_pooled,
)

DATA_DIR = os.path.join(REPO_ROOT, "data")

CONFIGS = [
    ("Qwen2.5-0.5B-Instruct (judge)", "judge_feature_cache_mtbench.pt"),
    ("Qwen2.5-1.5B-Instruct (judge)", "judge_feature_cache_mtbench_1p5b.pt"),
    ("SmolLM2-360M-Instruct (judge)", "judge_feature_cache_mtbench_smollm2_360m.pt"),
]


def analyze(name: str, path: str) -> dict:
    config = load_judge_config(path, name)
    phi_fit, correct_fit = config["phi_fit"], config["correct_fit"]
    phi_test, correct_test = config["phi"], config["correct"]
    correct_test_bool = correct_test.bool()
    n = phi_test.shape[0]

    # Orientation for Rule 1/2/3 (msp, energy): pooled bootstrap-stabilized
    # direction (scripts/judge_characterization.py), same protocol as Table 3
    # -- not id_test's own verify_feature_directions, which is circular
    # against id_test-computed rates (scripts/direction_split_robustness_check.py).
    phi_test_c = oriented_pooled(config)
    msp = phi_test_c[:, DEFAULT_FEATURE_NAMES.index("msp")].numpy() if "msp" in DEFAULT_FEATURE_NAMES else phi_test_c[:, 0].numpy()
    energy = phi_test_c[:, DEFAULT_FEATURE_NAMES.index("energy_score")].numpy()

    # High-confidence-wrong / low-confidence-correct: EXACT Rule 1 / Rule 2 from
    # judge_characterization.py -- percentile thresholds computed on the FULL
    # id_test MSP distribution (not on the correct/incorrect subset itself,
    # which would be tautological), then checked against the conditioned
    # subsets.
    correct_np_bool = correct_test_bool.numpy()
    q75_all, q25_all = np.percentile(msp, 75), np.percentile(msp, 25)
    n_incorrect, n_correct = (~correct_np_bool).sum(), correct_np_bool.sum()
    high_conf_wrong = ((~correct_np_bool) & (msp >= q75_all)).sum() / n_incorrect
    low_conf_correct = (correct_np_bool & (msp <= q25_all)).sum() / n_correct

    # High signal disagreement (new, unconditional on correctness)
    p75_msp, p25_msp = np.percentile(msp, 75), np.percentile(msp, 25)
    p75_energy, p25_energy = np.percentile(energy, 75), np.percentile(energy, 25)
    disagree = ((msp >= p75_msp) & (energy <= p25_energy)) | ((msp <= p25_msp) & (energy >= p75_energy))
    disagree_rate = disagree.mean()

    # Combiner vs. best single feature: bootstrap-stabilized selection over
    # pooled combiner_fit+threshold_cal (matching Table 1's protocol exactly
    # -- scripts/judge_characterization.py's best_single_feature()), not a
    # single draw on threshold_cal alone.
    best_name = bootstrap_stabilized_direction_and_best_feature(config)["stabilized_best"]
    d_test = verify_feature_directions(phi_test, correct_test)
    sign_test = 1.0 if d_test[best_name] else -1.0
    idx = DEFAULT_FEATURE_NAMES.index(best_name)
    best_col = (phi_test[:, idx] * DEFAULT_FEATURE_DIRECTIONS[idx].item() * sign_test).numpy()

    combiner = LogisticRegressionCombiner().fit(phi_fit, correct_fit.float())
    comb_scores = combiner.score(phi_test).numpy()

    comb_sign = np.sign(comb_scores - np.median(comb_scores))
    best_sign = np.sign(best_col - np.median(best_col))
    flip_rate = (comb_sign != best_sign).mean()

    return {
        "name": name, "n": n, "best_name": best_name,
        "high_conf_wrong": high_conf_wrong * 100,
        "low_conf_correct": low_conf_correct * 100,
        "disagree_rate": disagree_rate * 100,
        "flip_rate": flip_rate * 100,
    }


def main() -> None:
    print(f"{'Config':<32}{'n':>6}{'HighConf+Wrong':>16}{'LowConf+Correct':>18}{'HighDisagree':>14}{'Combiner!=Best':>16}  BestFeat")
    for name, path in CONFIGS:
        r = analyze(name, path)
        print(f"{r['name']:<32}{r['n']:>6}{r['high_conf_wrong']:>15.1f}%{r['low_conf_correct']:>17.1f}%{r['disagree_rate']:>13.1f}%{r['flip_rate']:>15.1f}%  {r['best_name']}")


if __name__ == "__main__":
    main()
