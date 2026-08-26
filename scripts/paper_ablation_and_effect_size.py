"""Answers three reviewer questions about judge2026.tex's Table 1, from this
project's EXISTING cached data - no new inference, mirrors
make_paper_figures.py's loading protocol exactly (same caches, same
combiner-fit split, same id_test evaluation split).

1. Which single feature drives the ImageNet win / the LLM loss? (leave-in
   ablation: AUROC of each of the 5 features alone, not just MSP vs combiner)
2. Are there OOD baselines beyond MSP the combiner should be compared to?
   (energy-alone and entropy-alone AUROC, since both are named in Related
   Work as prior OOD-detection scores)
3. What is the actual effect size, with a confidence interval, behind the
   Table 1 AUROC gaps - not just the DeLong p-value? Paired bootstrap CI on
   the AUROC DIFFERENCE (combiner - MSP), which DeLong's z/p alone doesn't
   give directly.

Single-feature AUROC is computed with each feature ORIENTED by
features.FEATURE_DIRECTIONS first (entropy and logit_l2_norm are "lower =
more confident" in raw form - see features.py's own documented sign
convention) so every column in the printed table is on the same "higher =
more reliable" scale as MSP and the combiner score.

Usage: python scripts/paper_ablation_and_effect_size.py
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
from deployment_reliability.significance import _rank_based_auroc, delong_test  # noqa: E402

DATA_DIR = os.path.join(REPO_ROOT, "data")

N_BOOTSTRAP = 2000
SEED = 0


def load_vision_full_scale(backbone: str = "resnet50") -> dict:
    small_cache = torch.load(os.path.join(DATA_DIR, f"logit_cache_{backbone}.pt"))
    splits = np.array(small_cache["splits"])
    m_fit = torch.from_numpy(splits == "combiner_fit")
    phi_fit = featurize(small_cache["logits"][m_fit])
    correct_fit = (small_cache["logits"][m_fit].argmax(dim=-1) == small_cache["labels"][m_fit]).float()
    combiner = LogisticRegressionCombiner().fit(phi_fit, correct_fit)

    full_cache = torch.load(os.path.join(DATA_DIR, f"logit_cache_imagenet1k_{backbone}.pt"))
    logits, labels = full_cache["logits"], full_cache["labels"]
    correct = logits.argmax(dim=-1) == labels
    phi = featurize(logits)
    return {"name": "ResNet-50", "phi": phi, "correct": correct, "combiner_scores": combiner.score(phi)}


def load_llm(tag: str, display_name: str) -> dict:
    cache = torch.load(os.path.join(DATA_DIR, f"llm_feature_cache_{tag}.pt"))
    phi, correct, splits = cache["phi"], cache["correct"], np.array(cache["splits"])
    m_fit = torch.from_numpy(splits == "combiner_fit")
    m_test = torch.from_numpy(splits == "id_test")
    combiner = LogisticRegressionCombiner().fit(phi[m_fit], correct[m_fit].float())
    phi_test, correct_test = phi[m_test], correct[m_test]
    return {"name": display_name, "phi": phi_test, "correct": correct_test, "combiner_scores": combiner.score(phi_test)}


def paired_bootstrap_auroc_diff(
    correct: torch.Tensor, score_a: torch.Tensor, score_b: torch.Tensor, n_bootstrap: int = N_BOOTSTRAP, seed: int = SEED
) -> tuple[float, float, float]:
    """Paired bootstrap CI for AUROC(score_a) - AUROC(score_b), same paired
    examples resampled jointly for both scores each draw (unlike
    bootstrap_auroc_ci's independent pos/neg resampling, which is correct for
    a single AUROC's own CI but not for the CI of a paired DIFFERENCE, since
    it would throw away the positive correlation between score_a and score_b
    on the same examples - the same reason delong_test exists instead of two
    independent bootstrap_auroc_ci calls). Complementary to delong_test's
    asymptotic normal-theory z/p: this is a finite-sample interval on the
    actual effect size, not just a significance flag.
    """
    correct_np = correct.detach().cpu().numpy().astype(bool)
    a_np = score_a.detach().cpu().to(dtype=torch.float64).numpy()
    b_np = score_b.detach().cpu().to(dtype=torch.float64).numpy()
    pos_idx_all = np.where(correct_np)[0]
    neg_idx_all = np.where(~correct_np)[0]
    n_pos, n_neg = len(pos_idx_all), len(neg_idx_all)

    point = _rank_based_auroc(a_np[pos_idx_all], a_np[neg_idx_all]) - _rank_based_auroc(b_np[pos_idx_all], b_np[neg_idx_all])

    rng = np.random.default_rng(seed)
    draws = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        pos_draw = rng.choice(pos_idx_all, size=n_pos, replace=True)
        neg_draw = rng.choice(neg_idx_all, size=n_neg, replace=True)
        auc_a = _rank_based_auroc(a_np[pos_draw], a_np[neg_draw])
        auc_b = _rank_based_auroc(b_np[pos_draw], b_np[neg_draw])
        draws[i] = auc_a - auc_b

    ci_lo, ci_hi = np.quantile(draws, [0.025, 0.975])
    return point, float(ci_lo), float(ci_hi)


def orient(phi: torch.Tensor) -> torch.Tensor:
    """Apply FEATURE_DIRECTIONS so every column reads 'higher = more reliable' -
    entropy and logit_l2_norm are flipped, matching features.py's documented
    convention (see features.py's FEATURE_DIRECTIONS docstring)."""
    return phi * DEFAULT_FEATURE_DIRECTIONS


def single_feature_aurocs(phi_oriented: torch.Tensor, correct: torch.Tensor) -> dict[str, float]:
    correct_bool = correct.to(dtype=torch.bool)
    out = {}
    for i, name in enumerate(DEFAULT_FEATURE_NAMES):
        col = phi_oriented[:, i]
        out[name] = auroc(col[correct_bool], col[~correct_bool])
    return out


def main() -> None:
    backbones = [
        load_vision_full_scale("resnet50"),
        load_llm("gpt2", "GPT-2"),
        load_llm("pythia160m", "Pythia-160m"),
        load_llm("qwen05binstruct", "Qwen2.5-0.5B-Instruct"),
    ]

    print("=== 1-2: single-feature AUROC (each of the 5 features alone, oriented higher=better) ===")
    header = f"{'Backbone':<24}" + "".join(f"{n:>13}" for n in DEFAULT_FEATURE_NAMES) + f"{'combiner':>13}"
    print(header)
    print("-" * len(header))
    for b in backbones:
        oriented = orient(b["phi"])
        sf = single_feature_aurocs(oriented, b["correct"])
        combiner_auroc = auroc(b["combiner_scores"][b["correct"]], b["combiner_scores"][~b["correct"]])
        row = f"{b['name']:<24}" + "".join(f"{sf[n]:>13.4f}" for n in DEFAULT_FEATURE_NAMES) + f"{combiner_auroc:>13.4f}"
        print(row)
        b["single_feature_aurocs"] = sf
        b["combiner_auroc"] = combiner_auroc

    print()
    print("=== 3: paired bootstrap CI on AUROC(combiner) - AUROC(MSP), vs. DeLong z/p ===")
    header2 = f"{'Backbone':<24}{'diff':>10}{'95% CI':>22}{'DeLong p':>14}{'winner':>10}"
    print(header2)
    print("-" * len(header2))
    for b in backbones:
        msp_scores = b["phi"][:, 0]  # column 0 = msp per DEFAULT_FEATURE_NAMES
        diff, ci_lo, ci_hi = paired_bootstrap_auroc_diff(b["correct"], b["combiner_scores"], msp_scores)
        dl = delong_test(b["correct"], b["combiner_scores"], msp_scores)
        winner = "Combiner" if dl.auc_diff > 0 else "MSP"
        print(f"{b['name']:<24}{diff:>10.4f}   [{ci_lo:>7.4f}, {ci_hi:>7.4f}]{dl.p_value:>14.2e}{winner:>10}")


if __name__ == "__main__":
    main()
