"""Answers a large batch of reviewer-raised diagnostic questions about
judge2026.tex's Table 1/2 results, entirely from data already cached in
data/*.pt - no new model inference. Extends paper_ablation_and_effect_size.py
(kept separate rather than merged into it, so that script's existing,
already-cited-in-the-paper numbers stay untouched and independently
reproducible).

Sections, each printed with a clear header:
  1. Best-single-feature-vs-combiner DeLong test, all 5 backbones (the
     corrected headline comparison - MSP is not always the best baseline).
  2. Fitted combiner weights (w, b) per backbone.
  3. Standardized-feature refit vs. unstandardized, per backbone.
  4. Train (combiner_fit) vs. test (id_test) AUROC per backbone.
  5. Feature correlation matrix per backbone (largest off-diagonal values).
  6. Link-function variant: logit(MSP) in place of raw MSP.
  7. Brier score + ECE, MSP vs. combiner, all 5 backbones.
  8. Selective risk at 50/70/90% coverage, MSP vs. combiner, all 5 backbones.
  9. Expected cost under one concrete cost model satisfying the CORRECTED
     ordering (c_V <= c_H <= c_V' < c_E, c_V < c_V'), ResNet-50 and GPT-2.
  10. Nonlinear (gradient-boosted-tree) combiner AUROC per backbone.
  11. Real ImageNet counterexample pair mined from cached full-scale logits.
  12. Chunk-level cluster bootstrap CI/p for the 3 LLM backbones (tokens
      within a 1024-token chunk are not i.i.d.; DeLong assumes they are).
  13. question_id-level cluster bootstrap CI/p for the judge backbone.

Usage: python scripts/paper_diagnostics.py
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
from deployment_reliability.router import auroc, risk_coverage_curve  # noqa: E402
from deployment_reliability.significance import _rank_based_auroc, delong_test  # noqa: E402

DATA_DIR = os.path.join(REPO_ROOT, "data")
CHUNK_SIZE = 1024  # matches scripts/collect_llm_logits.py
SEED = 0


# ---------------------------------------------------------------------------
# Loading (mirrors make_paper_figures.py / paper_ablation_and_effect_size.py)
# ---------------------------------------------------------------------------

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
    return {
        "name": "ResNet-50", "phi_fit": phi_fit, "correct_fit": correct_fit.bool(),
        "phi": phi, "correct": correct, "combiner": combiner, "logits": logits, "raw_logits_full": logits,
    }


def load_llm(tag: str, display_name: str) -> dict:
    cache = torch.load(os.path.join(DATA_DIR, f"llm_feature_cache_{tag}.pt"))
    phi, correct, splits = cache["phi"], cache["correct"], np.array(cache["splits"])
    m_fit = torch.from_numpy(splits == "combiner_fit")
    m_test = torch.from_numpy(splits == "id_test")
    combiner = LogisticRegressionCombiner().fit(phi[m_fit], correct[m_fit].float())
    return {
        "name": display_name, "phi_fit": phi[m_fit], "correct_fit": correct[m_fit].bool(),
        "phi": phi[m_test], "correct": correct[m_test], "combiner": combiner,
        "full_phi": phi, "full_correct": correct, "test_mask": m_test,
    }


def load_judge() -> dict:
    cache = torch.load(os.path.join(DATA_DIR, "judge_feature_cache_mtbench.pt"))
    phi, correct, splits = cache["phi"], cache["correct"], np.array(cache["splits"])
    m_fit = torch.from_numpy(splits == "combiner_fit")
    m_test = torch.from_numpy(splits == "id_test")
    combiner = LogisticRegressionCombiner().fit(phi[m_fit], correct[m_fit].float())
    return {
        "name": "Qwen2.5-0.5B-Instruct (judge)", "phi_fit": phi[m_fit], "correct_fit": correct[m_fit].bool(),
        "phi": phi[m_test], "correct": correct[m_test], "combiner": combiner,
        "question_id": np.array(cache["question_id"])[m_test.numpy()],
    }


def load_all() -> list[dict]:
    return [
        load_vision_full_scale("resnet50"),
        load_llm("gpt2", "GPT-2"),
        load_llm("pythia160m", "Pythia-160m"),
        load_llm("qwen05binstruct", "Qwen2.5-0.5B-Instruct"),
        load_judge(),
    ]


def orient(phi: torch.Tensor) -> torch.Tensor:
    return phi * DEFAULT_FEATURE_DIRECTIONS


# ---------------------------------------------------------------------------
# 1. Best-single-feature baseline
# ---------------------------------------------------------------------------

def section1(backbones: list[dict]) -> None:
    print("=" * 100)
    print("1. BEST-SINGLE-FEATURE vs COMBINER (corrected headline comparison)")
    print("=" * 100)
    header = f"{'Backbone':<28}{'MSP':>8}{'best feat':>12}{'combiner':>10}{'comb-best':>11}{'p (DeLong)':>13}{'winner':>10}"
    print(header)
    print("-" * len(header))
    for b in backbones:
        oriented = orient(b["phi"])
        correct_bool = b["correct"].bool()
        feat_aurocs = {}
        for i, name in enumerate(DEFAULT_FEATURE_NAMES):
            col = oriented[:, i]
            feat_aurocs[name] = auroc(col[correct_bool], col[~correct_bool])
        best_name = max(feat_aurocs, key=feat_aurocs.get)
        best_idx = DEFAULT_FEATURE_NAMES.index(best_name)
        best_scores = oriented[:, best_idx]
        combiner_scores = b["combiner"].score(b["phi"])
        msp_auroc = feat_aurocs["msp"]
        combiner_auroc = auroc(combiner_scores[correct_bool], combiner_scores[~correct_bool])
        dl = delong_test(b["correct"], combiner_scores, best_scores)
        winner = "combiner" if dl.p_value < 0.05 and dl.auc_diff > 0 else ("best-feat" if dl.p_value < 0.05 else "n.s.")
        print(f"{b['name']:<28}{msp_auroc:>8.4f}{feat_aurocs[best_name]:>12.4f}({best_name:<9}){combiner_auroc:>10.4f}"
              f"{combiner_auroc - feat_aurocs[best_name]:>11.4f}{dl.p_value:>13.2e}{winner:>10}")
    print()


# ---------------------------------------------------------------------------
# 2. Fitted weights
# ---------------------------------------------------------------------------

def section2(backbones: list[dict]) -> None:
    print("=" * 100)
    print("2. FITTED COMBINER WEIGHTS (w, b)")
    print("=" * 100)
    header = f"{'Backbone':<28}" + "".join(f"{n:>13}" for n in DEFAULT_FEATURE_NAMES) + f"{'bias':>10}"
    print(header)
    print("-" * len(header))
    for b in backbones:
        w = b["combiner"].weight.detach().numpy()
        bias = b["combiner"].bias.detach().item()
        print(f"{b['name']:<28}" + "".join(f"{wi:>13.4f}" for wi in w) + f"{bias:>10.4f}")
    print()


# ---------------------------------------------------------------------------
# 3. Standardization test
# ---------------------------------------------------------------------------

def section3(backbones: list[dict]) -> None:
    print("=" * 100)
    print("3. STANDARDIZED-FEATURE REFIT vs UNSTANDARDIZED")
    print("=" * 100)
    header = f"{'Backbone':<28}{'unstd. AUROC':>14}{'std. AUROC':>14}{'delta':>10}"
    print(header)
    print("-" * len(header))
    for b in backbones:
        mean = b["phi_fit"].mean(dim=0)
        std = b["phi_fit"].std(dim=0).clamp_min(1e-8)
        phi_fit_std = (b["phi_fit"] - mean) / std
        phi_test_std = (b["phi"] - mean) / std
        combiner_std = LogisticRegressionCombiner().fit(phi_fit_std, b["correct_fit"].float())
        scores_std = combiner_std.score(phi_test_std)
        scores_unstd = b["combiner"].score(b["phi"])
        correct_bool = b["correct"].bool()
        auroc_std = auroc(scores_std[correct_bool], scores_std[~correct_bool])
        auroc_unstd = auroc(scores_unstd[correct_bool], scores_unstd[~correct_bool])
        print(f"{b['name']:<28}{auroc_unstd:>14.4f}{auroc_std:>14.4f}{auroc_std - auroc_unstd:>10.4f}")
    print()


# ---------------------------------------------------------------------------
# 4. Train vs test AUROC
# ---------------------------------------------------------------------------

def section4(backbones: list[dict]) -> None:
    print("=" * 100)
    print("4. TRAIN (combiner_fit) vs TEST (id_test) AUROC")
    print("=" * 100)
    header = f"{'Backbone':<28}{'MSP train':>11}{'comb train':>12}{'MSP test':>11}{'comb test':>11}"
    print(header)
    print("-" * len(header))
    for b in backbones:
        fit_bool = b["correct_fit"].bool()
        msp_fit = b["phi_fit"][:, 0]
        msp_train_auroc = auroc(msp_fit[fit_bool], msp_fit[~fit_bool])
        comb_fit_scores = b["combiner"].score(b["phi_fit"])
        comb_train_auroc = auroc(comb_fit_scores[fit_bool], comb_fit_scores[~fit_bool])

        test_bool = b["correct"].bool()
        msp_test = b["phi"][:, 0]
        msp_test_auroc = auroc(msp_test[test_bool], msp_test[~test_bool])
        comb_test_scores = b["combiner"].score(b["phi"])
        comb_test_auroc = auroc(comb_test_scores[test_bool], comb_test_scores[~test_bool])
        print(f"{b['name']:<28}{msp_train_auroc:>11.4f}{comb_train_auroc:>12.4f}{msp_test_auroc:>11.4f}{comb_test_auroc:>11.4f}")
    print()


# ---------------------------------------------------------------------------
# 5. Feature correlation
# ---------------------------------------------------------------------------

def section5(backbones: list[dict]) -> None:
    print("=" * 100)
    print("5. FEATURE CORRELATION MATRIX (largest |off-diagonal| per backbone)")
    print("=" * 100)
    for b in backbones:
        phi_np = b["phi"].numpy()
        corr = np.corrcoef(phi_np, rowvar=False)
        n = len(DEFAULT_FEATURE_NAMES)
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append((abs(corr[i, j]), DEFAULT_FEATURE_NAMES[i], DEFAULT_FEATURE_NAMES[j], corr[i, j]))
        pairs.sort(reverse=True)
        top3 = ", ".join(f"{a}~{c}: r={d:+.2f}" for _, a, c, d in pairs[:3])
        print(f"{b['name']:<28} top correlations: {top3}")
    print()


# ---------------------------------------------------------------------------
# 6. Link-function variant
# ---------------------------------------------------------------------------

def section6(backbones: list[dict]) -> None:
    print("=" * 100)
    print("6. LINK-FUNCTION VARIANT: logit(MSP) replacing raw MSP")
    print("=" * 100)
    header = f"{'Backbone':<28}{'raw-MSP AUROC':>15}{'logit-MSP AUROC':>17}{'delta':>10}"
    print(header)
    print("-" * len(header))
    eps = 1e-6
    for b in backbones:
        phi_fit_variant = b["phi_fit"].clone()
        p = phi_fit_variant[:, 0].clamp(eps, 1 - eps)
        phi_fit_variant[:, 0] = torch.log(p / (1 - p))
        phi_test_variant = b["phi"].clone()
        p_test = phi_test_variant[:, 0].clamp(eps, 1 - eps)
        phi_test_variant[:, 0] = torch.log(p_test / (1 - p_test))

        combiner_variant = LogisticRegressionCombiner().fit(phi_fit_variant, b["correct_fit"].float())
        scores_variant = combiner_variant.score(phi_test_variant)
        scores_orig = b["combiner"].score(b["phi"])
        correct_bool = b["correct"].bool()
        auroc_variant = auroc(scores_variant[correct_bool], scores_variant[~correct_bool])
        auroc_orig = auroc(scores_orig[correct_bool], scores_orig[~correct_bool])
        print(f"{b['name']:<28}{auroc_orig:>15.4f}{auroc_variant:>17.4f}{auroc_variant - auroc_orig:>10.4f}")
    print()


# ---------------------------------------------------------------------------
# 7. Proper scoring: Brier + ECE
# ---------------------------------------------------------------------------

def brier_score(scores: torch.Tensor, correct: torch.Tensor) -> float:
    y = correct.float()
    return float(((scores - y) ** 2).mean().item())


def ece(scores: torch.Tensor, correct: torch.Tensor, n_bins: int = 10) -> float:
    y = correct.float()
    bin_edges = torch.linspace(0, 1, n_bins + 1)
    total = 0.0
    n = len(scores)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (scores >= lo) & (scores < hi) if i < n_bins - 1 else (scores >= lo) & (scores <= hi)
        if mask.sum() == 0:
            continue
        bin_conf = scores[mask].mean().item()
        bin_acc = y[mask].mean().item()
        total += (mask.sum().item() / n) * abs(bin_conf - bin_acc)
    return total


def section7(backbones: list[dict]) -> None:
    print("=" * 100)
    print("7. PROPER SCORING: Brier score + ECE, MSP vs combiner")
    print("=" * 100)
    header = f"{'Backbone':<28}{'MSP Brier':>11}{'comb Brier':>12}{'MSP ECE':>10}{'comb ECE':>10}"
    print(header)
    print("-" * len(header))
    for b in backbones:
        msp = b["phi"][:, 0]
        comb = b["combiner"].score(b["phi"])
        correct = b["correct"]
        print(f"{b['name']:<28}{brier_score(msp, correct):>11.4f}{brier_score(comb, correct):>12.4f}"
              f"{ece(msp, correct):>10.4f}{ece(comb, correct):>10.4f}")
    print()


# ---------------------------------------------------------------------------
# 8. Selective risk at fixed coverage
# ---------------------------------------------------------------------------

def risk_at_coverage(scores: torch.Tensor, correct: torch.Tensor, coverage_target: float) -> float:
    coverage, risk = risk_coverage_curve(scores, correct)
    idx = int(round(coverage_target * len(scores))) - 1
    idx = max(0, min(idx, len(scores) - 1))
    return float(risk[idx].item())


def section8(backbones: list[dict]) -> None:
    print("=" * 100)
    print("8. SELECTIVE RISK AT FIXED COVERAGE (50/70/90%), MSP vs combiner")
    print("=" * 100)
    header = f"{'Backbone':<28}{'MSP@50':>9}{'comb@50':>9}{'MSP@70':>9}{'comb@70':>9}{'MSP@90':>9}{'comb@90':>9}"
    print(header)
    print("-" * len(header))
    for b in backbones:
        msp = b["phi"][:, 0]
        comb = b["combiner"].score(b["phi"])
        correct = b["correct"]
        row = f"{b['name']:<28}"
        for cov in (0.5, 0.7, 0.9):
            row += f"{risk_at_coverage(msp, correct, cov):>9.4f}{risk_at_coverage(comb, correct, cov):>9.4f}"
        print(row)
    print()


# ---------------------------------------------------------------------------
# 9. Expected cost under one concrete cost model (corrected ordering)
# ---------------------------------------------------------------------------

def expected_cost(scores: torch.Tensor, correct: torch.Tensor, tau_lo: float, tau_hi: float,
                   c_e: float, c_v: float, c_vp: float, c_h: float) -> float:
    total = 0.0
    n = len(scores)
    for s, y in zip(scores.tolist(), correct.tolist()):
        if s < tau_lo:
            total += c_h
        elif s < tau_hi:
            total += c_v if y else c_vp
        else:
            total += 0.0 if y else c_e
    return total / n


def section9(backbones: list[dict]) -> None:
    print("=" * 100)
    print("9. EXPECTED COST under one concrete cost model (c_V=0.3 <= c_H=0.5 <= c_V'=1.0 < c_E=5.0)")
    print("=" * 100)
    c_v, c_h, c_vp, c_e = 0.3, 0.5, 1.0, 5.0
    tau_lo = (c_h - c_vp) / (c_v - c_vp)
    tau_hi = (c_e - c_vp) / (c_e + c_v - c_vp)
    print(f"tau_lo={tau_lo:.4f}  tau_hi={tau_hi:.4f}")
    header = f"{'Backbone':<28}{'MSP E[cost]':>13}{'combiner E[cost]':>18}"
    print(header)
    print("-" * len(header))
    for b in backbones[:2]:  # ResNet-50 and GPT-2, per plan
        msp = b["phi"][:, 0]
        comb = b["combiner"].score(b["phi"])
        correct = b["correct"]
        cost_msp = expected_cost(msp, correct, tau_lo, tau_hi, c_e, c_v, c_vp, c_h)
        cost_comb = expected_cost(comb, correct, tau_lo, tau_hi, c_e, c_v, c_vp, c_h)
        print(f"{b['name']:<28}{cost_msp:>13.4f}{cost_comb:>18.4f}")
    print()


# ---------------------------------------------------------------------------
# 10. Nonlinear combiner
# ---------------------------------------------------------------------------

def section10(backbones: list[dict]) -> None:
    print("=" * 100)
    print("10. NONLINEAR (gradient-boosted-tree) COMBINER vs LINEAR, and vs MSP (DeLong)")
    print("=" * 100)
    from sklearn.ensemble import HistGradientBoostingClassifier
    header = f"{'Backbone':<28}{'linear AUROC':>14}{'GBT AUROC':>12}{'delta':>10}{'MSP AUROC':>12}{'GBT-MSP':>10}{'DeLong p':>13}{'sig?':>6}"
    print(header)
    print("-" * len(header))
    for b in backbones:
        gbt = HistGradientBoostingClassifier(random_state=0, max_iter=100)
        gbt.fit(b["phi_fit"].numpy(), b["correct_fit"].numpy().astype(int))
        gbt_scores = torch.from_numpy(gbt.predict_proba(b["phi"].numpy())[:, 1])
        correct_bool = b["correct"].bool()
        gbt_auroc = auroc(gbt_scores[correct_bool], gbt_scores[~correct_bool])
        lin_scores = b["combiner"].score(b["phi"])
        lin_auroc = auroc(lin_scores[correct_bool], lin_scores[~correct_bool])
        msp_scores = b["phi"][:, 0]
        msp_auroc = auroc(msp_scores[correct_bool], msp_scores[~correct_bool])
        dl = delong_test(b["correct"], gbt_scores, msp_scores)
        sig = "yes" if dl.p_value < 0.05 else "NO"
        print(f"{b['name']:<28}{lin_auroc:>14.4f}{gbt_auroc:>12.4f}{gbt_auroc - lin_auroc:>10.4f}"
              f"{msp_auroc:>12.4f}{dl.auc_diff:>10.4f}{dl.p_value:>13.4g}{sig:>6}")
    print()


# ---------------------------------------------------------------------------
# 11. Real ImageNet counterexample
# ---------------------------------------------------------------------------

def section11(backbones: list[dict]) -> None:
    print("=" * 100)
    print("11. REAL IMAGENET COUNTEREXAMPLE (MSP vs -||z||_2 rank-disagreement)")
    print("=" * 100)
    resnet = backbones[0]
    logits = resnet["raw_logits_full"]
    from deployment_reliability.features import logit_l2_norm, msp as msp_fn
    msp_vals = msp_fn(logits)
    l2_vals = logit_l2_norm(logits)
    v_vals = -l2_vals  # oriented so higher = more reliable, per the paper's convention

    # Find a pair with large disagreement: x1 with high MSP but high L2 (bad V),
    # x2 with lower MSP but low L2 (good V) - search real data instead of hand-picking.
    n = len(msp_vals)
    rng = np.random.default_rng(SEED)
    best_pair, best_gap = None, -1.0
    candidates = rng.choice(n, size=min(20000, n), replace=False)
    msp_np, v_np = msp_vals.numpy(), v_vals.numpy()
    # Sort candidates by MSP descending to find high-MSP/bad-V vs lower-MSP/good-V pairs efficiently
    order = candidates[np.argsort(-msp_np[candidates])]
    top_msp = order[:200]
    bottom_v = candidates[np.argsort(v_np[candidates])][:200]  # worst V (most negative, i.e. largest L2)
    for i in top_msp[:50]:
        for j in candidates[np.argsort(-v_np[candidates])][:50]:  # best V among candidates
            if msp_np[i] > msp_np[j] and v_np[i] < v_np[j]:
                gap = (msp_np[i] - msp_np[j]) + (v_np[j] - v_np[i])
                if gap > best_gap:
                    best_gap = gap
                    best_pair = (int(i), int(j))
    if best_pair:
        i, j = best_pair
        print(f"x1 (idx {i}): MSP={msp_np[i]:.4f}  V=-||z||_2={v_np[i]:.4f}  ||z||_2={l2_vals[i].item():.4f}  correct={bool(resnet['correct'][i])}")
        print(f"x2 (idx {j}): MSP={msp_np[j]:.4f}  V=-||z||_2={v_np[j]:.4f}  ||z||_2={l2_vals[j].item():.4f}  correct={bool(resnet['correct'][j])}")
        print(f"U(x1)={msp_np[i]:.4f} > U(x2)={msp_np[j]:.4f}   but   V(x1)={v_np[i]:.4f} < V(x2)={v_np[j]:.4f}  -- confirmed real rank-disagreement")
    else:
        print("No qualifying pair found in sampled candidates - widen search if this prints.")
    print()


# ---------------------------------------------------------------------------
# 12/13. Cluster bootstrap
# ---------------------------------------------------------------------------

def cluster_bootstrap_auroc_diff(cluster_ids: np.ndarray, correct: np.ndarray, score_a: np.ndarray,
                                  score_b: np.ndarray, n_bootstrap: int = 1000, seed: int = SEED) -> tuple[float, float, float, float]:
    """Resamples whole clusters with replacement, recomputes AUROC(a)-AUROC(b)
    within each resampled dataset, returns (point, ci_lo, ci_hi, frac_positive)."""
    unique_clusters = np.unique(cluster_ids)
    rng = np.random.default_rng(seed)

    def compute_diff(idx: np.ndarray) -> float:
        c, a, bb = correct[idx], score_a[idx], score_b[idx]
        pos, neg = c.astype(bool), ~c.astype(bool)
        if pos.sum() < 2 or neg.sum() < 2:
            return np.nan
        auc_a = _rank_based_auroc(a[pos], a[neg])
        auc_b = _rank_based_auroc(bb[pos], bb[neg])
        return auc_a - auc_b

    point = compute_diff(np.arange(len(correct)))
    cluster_to_indices = {c: np.where(cluster_ids == c)[0] for c in unique_clusters}
    draws = []
    for _ in range(n_bootstrap):
        drawn_clusters = rng.choice(unique_clusters, size=len(unique_clusters), replace=True)
        idx = np.concatenate([cluster_to_indices[c] for c in drawn_clusters])
        d = compute_diff(idx)
        if not np.isnan(d):
            draws.append(d)
    draws = np.array(draws)
    ci_lo, ci_hi = np.quantile(draws, [0.025, 0.975])
    frac_positive = float((draws > 0).mean())
    return point, float(ci_lo), float(ci_hi), frac_positive


def section12(backbones: list[dict]) -> None:
    print("=" * 100)
    print("12. CHUNK-LEVEL (1024-token) CLUSTER BOOTSTRAP, combiner vs MSP AUROC diff, LLM backbones")
    print("=" * 100)
    header = f"{'Backbone':<28}{'point diff':>11}{'95% CI':>22}{'frac(diff>0)':>13}"
    print(header)
    print("-" * len(header))
    for b in backbones[1:4]:  # GPT-2, Pythia-160m, Qwen (index 0=vision, 4=judge)
        full_correct = b["full_correct"].numpy()
        full_phi = b["full_phi"]
        n_full = len(full_correct)
        chunk_ids = np.arange(n_full) // CHUNK_SIZE  # position within the FULL cache before test-split filtering

        test_mask = b["test_mask"].numpy()
        test_idx = np.where(test_mask)[0]
        cluster_ids_test = chunk_ids[test_idx]
        correct_test = full_correct[test_idx]
        msp_test = full_phi[test_idx, 0].numpy()
        combiner_scores_test = b["combiner"].score(full_phi[test_idx]).numpy()

        point, ci_lo, ci_hi, frac_pos = cluster_bootstrap_auroc_diff(
            cluster_ids_test, correct_test, combiner_scores_test, msp_test
        )
        print(f"{b['name']:<28}{point:>11.4f}   [{ci_lo:>7.4f}, {ci_hi:>7.4f}]{frac_pos:>13.3f}")
    print()


def section13(backbones: list[dict]) -> None:
    print("=" * 100)
    print("13. question_id-LEVEL CLUSTER BOOTSTRAP, combiner vs MSP AUROC diff, judge backbone")
    print("=" * 100)
    judge = backbones[4]
    correct = judge["correct"].numpy()
    msp = judge["phi"][:, 0].numpy()
    combiner_scores = judge["combiner"].score(judge["phi"]).numpy()
    qid = judge["question_id"]
    point, ci_lo, ci_hi, frac_pos = cluster_bootstrap_auroc_diff(qid, correct, combiner_scores, msp)
    print(f"{'Qwen judge':<28}{point:>11.4f}   [{ci_lo:>7.4f}, {ci_hi:>7.4f}]{frac_pos:>13.3f}")
    print()


def main() -> None:
    backbones = load_all()
    section1(backbones)
    section2(backbones)
    section3(backbones)
    section4(backbones)
    section5(backbones)
    section6(backbones)
    section7(backbones)
    section8(backbones)
    section9(backbones)
    section10(backbones)
    section11(backbones)
    section12(backbones)
    section13(backbones)


if __name__ == "__main__":
    main()
