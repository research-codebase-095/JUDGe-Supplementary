"""GAP 1 (judge-specific empirical characterization) analyses A-E, run entirely
over already-cached judge data (data/judge_feature_cache_mtbench*.pt,
data/judge_swap_consistency_cache.pt) - no new model inference in this script
itself (the third judge-model run, collect_judge_verdicts_smollm2.py, is a
separate script producing the cache this one reads once it exists).

Reuses infrastructure already built and used in the paper rather than
reimplementing it: LogisticRegressionCombiner, delong_test, _rank_based_auroc,
DEFAULT_FEATURE_NAMES/DIRECTIONS from src/deployment_reliability, and
brier_score/ece/cluster_bootstrap_auroc_diff from scripts/paper_diagnostics.py
(imported directly, not copy-pasted).

Sections:
  A. Cluster-bootstrap (question_id-level) AUROC point + 95% CI for every
     single feature and every combiner variant (linear/standardized/GBT),
     per judge config.
  B. Cross-config comparison: per-feature AUROC, ranking, combiner AUROC,
     Brier/ECE, side by side.
  C. Feature-behavior: mean/std split by correctness, point-biserial r with
     correctness, feature correlation matrix, per judge config.
  D. Failure-case rates under four PRE-SPECIFIED rules (defined in this
     docstring and in section D's header, not tuned after looking at data):
       1. high-confidence disagreement: MSP >= id_test's own 75th percentile,
          among INCORRECT examples (rate = count / n_incorrect).
       2. low-confidence agreement: MSP <= id_test's own 25th percentile,
          among CORRECT examples (rate = count / n_correct).
       3. signal disagreement: standardized MSP and standardized entropy
          (z-scored on combiner_fit mean/std) have opposite sign, each with
          |z| > 0.5 (rate = count / n_id_test).
       4. combiner-vs-best-feature flip: sign(combiner_score - median) !=
          sign(best_feature_score - median), medians taken over id_test
          itself (rate = count / n_id_test). "Best feature" is selected on
          threshold_cal and scored on id_test (best_single_feature() below),
          the same disjoint protocol as Table 1 -- not in-sample selection on
          id_test, which this function used to do and which picked a
          different feature than Table 1 at Qwen2.5-1.5B-judge (entropy vs.
          MSP, a near-tie) even though no rounded rate in this table changed.
  E. Swap/order-effect analysis on judge_swap_consistency_cache.pt (0.5B
     only. Swap-consistency itself is also collected and analyzed for 1.5B
     elsewhere in this paper - a headline finding, AUROC 0.583, sixth-signal
     boost 0.673->0.728 - but this specific signal-disagreement breakdown has
     only been run at 0.5B here; it was never run for SmolLM2 at all):
       (i) swap-consistency rate by MSP confidence decile,
       (ii) swap-consistency rate under the same signal-disagreement rule
            as D.3, compared across disagreement/no-disagreement groups,
       (iii) cluster-bootstrap 95% CI on the swap-consistency-vs-correctness
             AUROC (point estimate 0.503 already in the paper).

Usage:
    python scripts/judge_characterization.py
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
from deployment_reliability.significance import _rank_based_auroc, delong_test  # noqa: E402

from paper_diagnostics import brier_score, cluster_bootstrap_auroc_diff, ece  # noqa: E402

DATA_DIR = os.path.join(REPO_ROOT, "data")
SEED = 0


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

MIN_EXAMPLES = 100  # below this a cache is a leftover smoke-test artifact, not a real run


def load_judge_config(cache_filename: str, display_name: str) -> dict | None:
    path = os.path.join(DATA_DIR, cache_filename)
    if not os.path.exists(path):
        print(f"[skip] {display_name}: {cache_filename} not found yet")
        return None
    cache = torch.load(path)
    if len(cache["correct"]) < MIN_EXAMPLES:
        print(f"[skip] {display_name}: {cache_filename} has only {len(cache['correct'])} examples "
              f"(< {MIN_EXAMPLES}) -- looks like a leftover smoke-test cache, not a completed run")
        return None
    phi, correct, splits = cache["phi"], cache["correct"], np.array(cache["splits"])
    m_fit = torch.from_numpy(splits == "combiner_fit")
    m_cal = torch.from_numpy(splits == "threshold_cal")
    m_test = torch.from_numpy(splits == "id_test")
    combiner = LogisticRegressionCombiner().fit(phi[m_fit], correct[m_fit].float())
    return {
        "name": display_name,
        "phi_fit": phi[m_fit], "correct_fit": correct[m_fit].bool(),
        "phi_cal": phi[m_cal], "correct_cal": correct[m_cal].bool(),
        "phi": phi[m_test], "correct": correct[m_test], "combiner": combiner,
        "question_id": np.array(cache["question_id"])[m_test.numpy()],
    }


def load_all_judge_configs() -> list[dict]:
    configs = [
        load_judge_config("judge_feature_cache_mtbench.pt", "Qwen2.5-0.5B-Instruct (judge)"),
        load_judge_config("judge_feature_cache_mtbench_1p5b.pt", "Qwen2.5-1.5B-Instruct (judge)"),
        load_judge_config("judge_feature_cache_mtbench_smollm2.pt", "SmolLM2-1.7B-Instruct (judge)"),
        load_judge_config("judge_feature_cache_mtbench_smollm2_360m.pt", "SmolLM2-360M-Instruct (judge, n=400 subset)"),
    ]
    return [c for c in configs if c is not None]


def orient(phi: torch.Tensor) -> torch.Tensor:
    return phi * DEFAULT_FEATURE_DIRECTIONS


def standardized_combiner(config: dict) -> tuple[LogisticRegressionCombiner, torch.Tensor, torch.Tensor]:
    mean = config["phi_fit"].mean(dim=0)
    std = config["phi_fit"].std(dim=0).clamp_min(1e-8)
    phi_fit_std = (config["phi_fit"] - mean) / std
    phi_test_std = (config["phi"] - mean) / std
    combiner_std = LogisticRegressionCombiner().fit(phi_fit_std, config["correct_fit"].float())
    return combiner_std, phi_fit_std, phi_test_std


def gbt_scores(config: dict) -> torch.Tensor:
    from sklearn.ensemble import HistGradientBoostingClassifier
    gbt = HistGradientBoostingClassifier(random_state=0, max_iter=100)
    gbt.fit(config["phi_fit"].numpy(), config["correct_fit"].numpy().astype(int))
    return torch.from_numpy(gbt.predict_proba(config["phi"].numpy())[:, 1])


def pooled_calibration(config: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """combiner_fit + threshold_cal, concatenated: the full disjoint (from
    id_test) pool available for feature-direction verification and
    best-feature selection. Used instead of threshold_cal alone because
    threshold_cal alone is tiny (n=128 for Qwen configs, n=40 for SmolLM2) and
    demonstrably unstable at that size (scripts/direction_split_robustness_check.py:
    SmolLM2's own MSP scores 0.359 AUROC on threshold_cal alone vs. 0.513 on
    id_test -- a swing driven by threshold_cal's small n, not new information).
    Pooling combiner_fit in gives n comparable to id_test itself (642/642/200)
    while staying fully disjoint from it -- fixing both the small-n problem and
    the id_test-verifies-its-own-direction circularity in the same step.
    combiner_fit is safe to reuse here because verify_feature_directions and
    best-feature selection are not the same operation as fitting Gw's own
    weights (Gw learns its weights regardless of any assumed sign convention;
    see features.py's verify_feature_directions docstring), so pooling it in
    does not leak id_test information or bias Gw's own fit."""
    phi_pool = torch.cat([config["phi_fit"], config["phi_cal"]])
    correct_pool = torch.cat([config["correct_fit"], config["correct_cal"]])
    return phi_pool, correct_pool


def bootstrap_stabilized_direction_and_best_feature(
    config: dict, n_bootstrap: int = 1000, seed: int = SEED
) -> dict:
    """Bootstrap-majority direction (per feature) and best-feature identity,
    resampled from the pooled combiner_fit+threshold_cal data (see
    pooled_calibration). Replaces a single point estimate on threshold_cal
    alone with a majority vote over 1000 resamples, reported alongside the
    resulting stability fractions so the choice is auditable rather than
    asserted."""
    phi_pool, correct_pool = pooled_calibration(config)
    n = phi_pool.shape[0]
    rng = np.random.default_rng(seed)
    sign_counts = {f: 0 for f in DEFAULT_FEATURE_NAMES}
    best_counts = {f: 0 for f in DEFAULT_FEATURE_NAMES}
    valid = 0
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        phi_b, correct_b = phi_pool[idx], correct_pool[idx]
        if correct_b.all() or (~correct_b).all():
            continue
        valid += 1
        d = verify_feature_directions(phi_b, correct_b)
        signs = {f: (1.0 if d[f] else -1.0) for f in DEFAULT_FEATURE_NAMES}
        for f in DEFAULT_FEATURE_NAMES:
            if signs[f] > 0:
                sign_counts[f] += 1
        oriented = phi_b * DEFAULT_FEATURE_DIRECTIONS * torch.tensor([signs[f] for f in DEFAULT_FEATURE_NAMES])
        cb = correct_b.bool()
        aurocs = {f: auroc(oriented[:, i][cb], oriented[:, i][~cb]) for i, f in enumerate(DEFAULT_FEATURE_NAMES)}
        best_counts[max(aurocs, key=aurocs.get)] += 1
    stabilized_sign = {f: (1.0 if sign_counts[f] >= valid / 2 else -1.0) for f in DEFAULT_FEATURE_NAMES}
    stabilized_best = max(best_counts, key=best_counts.get)
    return {
        "n_pool": n, "valid": valid,
        "sign_frac": {f: sign_counts[f] / valid for f in DEFAULT_FEATURE_NAMES},
        "best_frac": {f: best_counts[f] / valid for f in DEFAULT_FEATURE_NAMES},
        "stabilized_sign": stabilized_sign, "stabilized_best": stabilized_best,
    }


def oriented_pooled(config: dict) -> torch.Tensor:
    """id_test's phi, oriented by the pooled bootstrap-stabilized per-config
    direction (not id_test's own verify_feature_directions call, which is
    circular against id_test-computed discordance/AUROC figures -- see
    pooled_calibration). This is the orientation Table 3 (disagreement),
    Table 2 is NOT this -- Table 2 deliberately keeps the fixed global
    convention throughout, unaffected by this function, precisely so genuine
    reversals stay visible as sub-chance entries there."""
    result = bootstrap_stabilized_direction_and_best_feature(config)
    signs = torch.tensor([result["stabilized_sign"][n] for n in DEFAULT_FEATURE_NAMES])
    return config["phi"] * DEFAULT_FEATURE_DIRECTIONS * signs


def best_single_feature(config: dict) -> tuple[str, torch.Tensor]:
    """Selects the best feature via a bootstrap-majority vote over the pooled
    combiner_fit+threshold_cal data (see bootstrap_stabilized_direction_and_best_feature),
    then returns that SAME feature's id_test scores, oriented by id_test's own
    verified direction (id_test is large enough at every judge config,
    200-642, that this final orientation step is not the small-n problem the
    selection-source fix above addresses) -- the disjoint selection/reporting
    protocol Table 1 and Section 4 use throughout. This replaced two earlier
    versions: in-sample selection on id_test itself (disagreed with Table 1 at
    a near-tie, entropy vs. MSP at Qwen2.5-1.5B-judge), and single-draw
    selection on threshold_cal alone (unstable at n=40/128 -- see
    pooled_calibration's docstring)."""
    result = bootstrap_stabilized_direction_and_best_feature(config)
    best_name = result["stabilized_best"]

    d_test = verify_feature_directions(config["phi"], config["correct"])
    sign_test = 1.0 if d_test[best_name] else -1.0
    idx = DEFAULT_FEATURE_NAMES.index(best_name)
    best_col = config["phi"][:, idx] * DEFAULT_FEATURE_DIRECTIONS[idx].item() * sign_test
    return best_name, best_col


# ---------------------------------------------------------------------------
# Cluster bootstrap for a single score (point + CI), generalizing
# paper_diagnostics.cluster_bootstrap_auroc_diff (diff of two scores) to one.
# ---------------------------------------------------------------------------

def cluster_bootstrap_auroc(cluster_ids: np.ndarray, correct: np.ndarray, scores: np.ndarray,
                             n_bootstrap: int = 1000, seed: int = SEED) -> tuple[float, float, float]:
    """Resamples whole question_id clusters with replacement (MT-Bench pairs
    sharing a question_id are not independent - the same justification the
    paper already uses for the combiner-vs-MSP diff CI). Returns
    (point, ci_lo, ci_hi)."""
    unique_clusters = np.unique(cluster_ids)
    rng = np.random.default_rng(seed)

    def compute(idx: np.ndarray) -> float:
        c, s = correct[idx], scores[idx]
        pos, neg = c.astype(bool), ~c.astype(bool)
        if pos.sum() < 2 or neg.sum() < 2:
            return np.nan
        return _rank_based_auroc(s[pos], s[neg])

    point = compute(np.arange(len(correct)))
    cluster_to_indices = {c: np.where(cluster_ids == c)[0] for c in unique_clusters}
    draws = []
    for _ in range(n_bootstrap):
        drawn_clusters = rng.choice(unique_clusters, size=len(unique_clusters), replace=True)
        idx = np.concatenate([cluster_to_indices[c] for c in drawn_clusters])
        d = compute(idx)
        if not np.isnan(d):
            draws.append(d)
    ci_lo, ci_hi = np.quantile(draws, [0.025, 0.975])
    return point, float(ci_lo), float(ci_hi)


# ---------------------------------------------------------------------------
# A. Cluster-bootstrap CIs, every feature + combiner variant, per config
# ---------------------------------------------------------------------------

def section_a(configs: list[dict]) -> None:
    print("=" * 100)
    print("A. CLUSTER-BOOTSTRAP (question_id-level) AUROC POINT + 95% CI, per judge config")
    print("=" * 100)
    header = f"{'Config':<30}{'Quantity':<16}{'AUROC':>8}{'95% CI':>20}"
    print(header)
    print("-" * len(header))
    for cfg in configs:
        qid = cfg["question_id"]
        correct = cfg["correct"].numpy()
        oriented = orient(cfg["phi"]).numpy()
        for i, name in enumerate(DEFAULT_FEATURE_NAMES):
            point, lo, hi = cluster_bootstrap_auroc(qid, correct, oriented[:, i])
            print(f"{cfg['name']:<30}{name:<16}{point:>8.4f}   [{lo:>7.4f}, {hi:>7.4f}]")

        lin_scores = cfg["combiner"].score(cfg["phi"]).numpy()
        point, lo, hi = cluster_bootstrap_auroc(qid, correct, lin_scores)
        print(f"{cfg['name']:<30}{'linear comb.':<16}{point:>8.4f}   [{lo:>7.4f}, {hi:>7.4f}]")

        std_combiner, _, phi_test_std = standardized_combiner(cfg)
        std_scores = std_combiner.score(phi_test_std).numpy()
        point, lo, hi = cluster_bootstrap_auroc(qid, correct, std_scores)
        print(f"{cfg['name']:<30}{'std. comb.':<16}{point:>8.4f}   [{lo:>7.4f}, {hi:>7.4f}]")

        gbt = gbt_scores(cfg).numpy()
        point, lo, hi = cluster_bootstrap_auroc(qid, correct, gbt)
        print(f"{cfg['name']:<30}{'GBT comb.':<16}{point:>8.4f}   [{lo:>7.4f}, {hi:>7.4f}]")

        best_name, best_scores = best_single_feature(cfg)
        point_diff, lo_diff, hi_diff, frac_pos = cluster_bootstrap_auroc_diff(
            qid, correct, lin_scores, best_scores.numpy()
        )
        print(f"{cfg['name']:<30}{'comb-bestfeat diff':<16}{point_diff:>8.4f}   [{lo_diff:>7.4f}, {hi_diff:>7.4f}]"
              f"  (best={best_name}, frac(diff>0)={frac_pos:.3f})")
        print()
    print()


# ---------------------------------------------------------------------------
# B. Cross-config comparison
# ---------------------------------------------------------------------------

def section_b(configs: list[dict]) -> None:
    print("=" * 100)
    print("B. CROSS-CONFIG COMPARISON: per-feature AUROC, ranking, combiner AUROC, Brier/ECE")
    print("=" * 100)
    header = f"{'Config':<30}" + "".join(f"{n:>10}" for n in DEFAULT_FEATURE_NAMES) + f"{'combiner':>10}{'MSP Brier':>11}{'comb Brier':>12}{'MSP ECE':>9}{'comb ECE':>9}"
    print(header)
    print("-" * len(header))
    for cfg in configs:
        oriented = orient(cfg["phi"])
        correct_bool = cfg["correct"].bool()
        aurocs = []
        for i in range(len(DEFAULT_FEATURE_NAMES)):
            col = oriented[:, i]
            aurocs.append(auroc(col[correct_bool], col[~correct_bool]))
        comb_scores = cfg["combiner"].score(cfg["phi"])
        comb_auroc = auroc(comb_scores[correct_bool], comb_scores[~correct_bool])
        msp = cfg["phi"][:, 0]
        row = f"{cfg['name']:<30}" + "".join(f"{a:>10.4f}" for a in aurocs) + f"{comb_auroc:>10.4f}"
        row += f"{brier_score(msp, cfg['correct']):>11.4f}{brier_score(comb_scores, cfg['correct']):>12.4f}"
        row += f"{ece(msp, cfg['correct']):>9.4f}{ece(comb_scores, cfg['correct']):>9.4f}"
        print(row)

        ranking = sorted(zip(DEFAULT_FEATURE_NAMES, aurocs), key=lambda t: -t[1])
        print(f"    feature ranking (best to worst): " + " > ".join(f"{n}({a:.3f})" for n, a in ranking))
    print()


# ---------------------------------------------------------------------------
# C. Feature-behavior analysis
# ---------------------------------------------------------------------------

def point_biserial_r(feature_vals: np.ndarray, correct: np.ndarray) -> float:
    return float(np.corrcoef(feature_vals, correct.astype(float))[0, 1])


def section_c(configs: list[dict]) -> None:
    print("=" * 100)
    print("C. FEATURE-BEHAVIOR: mean/std by correctness, point-biserial r, correlation matrix")
    print("=" * 100)
    for cfg in configs:
        print(f"--- {cfg['name']} ---")
        oriented = orient(cfg["phi"]).numpy()
        correct = cfg["correct"].numpy()
        header = f"{'Feature':<12}{'mean|correct':>13}{'std|correct':>12}{'mean|wrong':>12}{'std|wrong':>11}{'r(biserial)':>13}"
        print(header)
        print("-" * len(header))
        for i, name in enumerate(DEFAULT_FEATURE_NAMES):
            col = oriented[:, i]
            m_c, s_c = col[correct].mean(), col[correct].std()
            m_w, s_w = col[~correct].mean(), col[~correct].std()
            r = point_biserial_r(col, correct)
            print(f"{name:<12}{m_c:>13.4f}{s_c:>12.4f}{m_w:>12.4f}{s_w:>11.4f}{r:>13.4f}")

        phi_np = cfg["phi"].numpy()
        corr = np.corrcoef(phi_np, rowvar=False)
        n = len(DEFAULT_FEATURE_NAMES)
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append((abs(corr[i, j]), DEFAULT_FEATURE_NAMES[i], DEFAULT_FEATURE_NAMES[j], corr[i, j]))
        pairs.sort(reverse=True)
        top3 = ", ".join(f"{a}~{c}: r={d:+.2f}" for _, a, c, d in pairs[:3])
        print(f"top correlations: {top3}")
        print()
    print()


# ---------------------------------------------------------------------------
# D. Failure-case analysis (rules pre-specified in module docstring)
# ---------------------------------------------------------------------------

def section_d(configs: list[dict]) -> None:
    print("=" * 100)
    print("D. FAILURE-CASE RATES (rules defined in this script's module docstring, fixed a priori)")
    print("=" * 100)
    header = f"{'Config':<30}{'n_id_test':>10}{'1.high-conf wrong':>19}{'2.low-conf right':>18}{'3.signal disagree':>19}{'4.comb-vs-best flip':>21}"
    print(header)
    print("-" * len(header))
    for cfg in configs:
        msp = cfg["phi"][:, 0].numpy()
        correct = cfg["correct"].numpy()
        n = len(correct)

        q75, q25 = np.percentile(msp, 75), np.percentile(msp, 25)
        n_incorrect = (~correct).sum()
        n_correct = correct.sum()
        rule1_count = int(((~correct) & (msp >= q75)).sum())
        rule1_rate = rule1_count / n_incorrect if n_incorrect > 0 else float("nan")
        rule2_count = int((correct & (msp <= q25)).sum())
        rule2_rate = rule2_count / n_correct if n_correct > 0 else float("nan")

        mean = cfg["phi_fit"].mean(dim=0).numpy()
        std = cfg["phi_fit"].std(dim=0).clamp_min(1e-8).numpy()
        phi_test_std = (cfg["phi"].numpy() - mean) / std
        z_msp = phi_test_std[:, 0]
        z_entropy = phi_test_std[:, DEFAULT_FEATURE_NAMES.index("normalized_entropy")] * DEFAULT_FEATURE_DIRECTIONS[
            DEFAULT_FEATURE_NAMES.index("normalized_entropy")
        ].item()
        z_msp_oriented = z_msp * DEFAULT_FEATURE_DIRECTIONS[0].item()
        rule3_mask = (np.sign(z_msp_oriented) != np.sign(z_entropy)) & (np.abs(z_msp_oriented) > 0.5) & (np.abs(z_entropy) > 0.5)
        rule3_count = int(rule3_mask.sum())
        rule3_rate = rule3_count / n

        comb_scores = cfg["combiner"].score(cfg["phi"]).numpy()
        best_name, best_scores_t = best_single_feature(cfg)
        best_scores = best_scores_t.numpy()
        comb_med, best_med = np.median(comb_scores), np.median(best_scores)
        rule4_mask = np.sign(comb_scores - comb_med) != np.sign(best_scores - best_med)
        rule4_count = int(rule4_mask.sum())
        rule4_rate = rule4_count / n

        print(f"{cfg['name']:<30}{n:>10}"
              f"{f'{rule1_count}/{n_incorrect} ({rule1_rate:.2%})':>19}"
              f"{f'{rule2_count}/{n_correct} ({rule2_rate:.2%})':>18}"
              f"{f'{rule3_count}/{n} ({rule3_rate:.2%})':>19}"
              f"{f'{rule4_count}/{n} ({rule4_rate:.2%}, best={best_name})':>21}")
    print()


# ---------------------------------------------------------------------------
# E. Swap/order-effect analysis (0.5B only, existing swap-consistency cache)
# ---------------------------------------------------------------------------

def section_e() -> None:
    print("=" * 100)
    print("E. SWAP/ORDER-EFFECT ANALYSIS (Qwen2.5-0.5B-Instruct judge, existing swap-consistency cache)")
    print("=" * 100)
    path = os.path.join(DATA_DIR, "judge_swap_consistency_cache.pt")
    if not os.path.exists(path):
        print("[skip] judge_swap_consistency_cache.pt not found")
        return
    cache = torch.load(path)
    phi = cache["phi_orig"]
    correct = cache["correct_orig"].numpy()
    swap_consistent = cache["swap_consistent"].numpy()
    qid = np.array(cache["question_id"])
    n = len(correct)

    msp = phi[:, 0].numpy()
    print("(i) swap-consistency rate by MSP confidence decile")
    deciles = np.percentile(msp, np.arange(0, 101, 10))
    header = f"{'decile':<10}{'MSP range':<22}{'n':>6}{'swap-consistency rate':>24}"
    print(header)
    print("-" * len(header))
    for d in range(10):
        lo, hi = deciles[d], deciles[d + 1]
        mask = (msp >= lo) & (msp <= hi) if d == 9 else (msp >= lo) & (msp < hi)
        rate = swap_consistent[mask].mean() if mask.sum() > 0 else float("nan")
        print(f"{d + 1:<10}{f'[{lo:.4f}, {hi:.4f}]':<22}{int(mask.sum()):>6}{rate:>24.4f}")
    print()

    print("(ii) swap-consistency rate under the signal-disagreement rule (same rule as D.3)")
    mean = phi.mean(dim=0).numpy()
    std = phi.std(dim=0).clamp_min(1e-8).numpy()
    phi_std = (phi.numpy() - mean) / std
    z_msp = phi_std[:, 0] * DEFAULT_FEATURE_DIRECTIONS[0].item()
    entropy_idx = DEFAULT_FEATURE_NAMES.index("normalized_entropy")
    z_entropy = phi_std[:, entropy_idx] * DEFAULT_FEATURE_DIRECTIONS[entropy_idx].item()
    disagree_mask = (np.sign(z_msp) != np.sign(z_entropy)) & (np.abs(z_msp) > 0.5) & (np.abs(z_entropy) > 0.5)
    rate_disagree = swap_consistent[disagree_mask].mean() if disagree_mask.sum() > 0 else float("nan")
    rate_agree = swap_consistent[~disagree_mask].mean()
    print(f"signal-disagreement examples: n={int(disagree_mask.sum())}  swap-consistency rate={rate_disagree:.4f}")
    print(f"no-disagreement examples:     n={int((~disagree_mask).sum())}  swap-consistency rate={rate_agree:.4f}")

    # cluster-bootstrap CI on the difference in swap-consistency rate between groups
    rng = np.random.default_rng(SEED)
    unique_q = np.unique(qid)
    q_to_idx = {q: np.where(qid == q)[0] for q in unique_q}
    diffs = []
    for _ in range(1000):
        drawn = rng.choice(unique_q, size=len(unique_q), replace=True)
        idx = np.concatenate([q_to_idx[q] for q in drawn])
        dm = disagree_mask[idx]
        sc = swap_consistent[idx]
        if dm.sum() < 2 or (~dm).sum() < 2:
            continue
        diffs.append(sc[dm].mean() - sc[~dm].mean())
    if diffs:
        lo, hi = np.quantile(diffs, [0.025, 0.975])
        print(f"cluster-bootstrap 95% CI on (disagree rate - agree rate): [{lo:.4f}, {hi:.4f}]")
    print()

    print("(iii) cluster-bootstrap 95% CI, swap-consistency-vs-correctness AUROC (point estimate already in paper: 0.503)")
    point, lo, hi = cluster_bootstrap_auroc(qid, correct, swap_consistent)
    print(f"point={point:.4f}  95% CI=[{lo:.4f}, {hi:.4f}]")
    print()
    print("NOTE: this specific signal-disagreement-vs-swap-consistency breakdown was only run for the 0.5B judge "
          "here. Swap-consistency itself IS collected and analyzed for 1.5B elsewhere in this paper (a headline "
          "finding: AUROC 0.583, sixth-signal boost 0.673->0.728). It was never collected for SmolLM2, so whether "
          "order-sensitivity interacts with judge model/size beyond the two tested Qwen sizes remains untested.")
    print()


def main() -> None:
    configs = load_all_judge_configs()
    print(f"loaded {len(configs)} judge config(s): {[c['name'] for c in configs]}")
    print()
    section_a(configs)
    section_b(configs)
    section_c(configs)
    section_d(configs)
    section_e()


if __name__ == "__main__":
    main()
