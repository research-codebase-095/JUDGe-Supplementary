"""Statistical significance for AUROC comparisons: bootstrap CIs and the
DeLong (1988) paired test, per the external-critique response documented in
STUDY_PLAN.md/DESIGN.md.

Every AUROC/AURC number this project reports elsewhere (router.py's `auroc`,
the disagreement matrix, the OOD suite, the LLM extension) is a point
estimate with no measure of sampling noise attached. This module adds two,
complementary tools for that:

- `bootstrap_auroc_ci`: a generic nonparametric bootstrap CI for a single
  AUROC (resample the positive/negative groups independently, recompute AUROC
  per draw, report the percentile CI). Applicable to ANY two-group comparison
  (correct vs. incorrect, id vs. OOD, one backbone vs. another) - valid for
  both paired and unpaired situations, but on its own says nothing about
  whether two AUROCs differ from EACH OTHER (two overlapping-looking CIs can
  still hide a real paired difference, and non-overlapping CIs are a
  conservative, not exact, test).

- `delong_test`: the exact test for the one situation bootstrap CIs handle
  poorly - two DIFFERENT scores (e.g. combiner S vs. raw MSP) evaluated
  against the SAME binary outcome on the SAME samples (e.g. the same
  `id_test` correctness labels). This is a paired/correlated comparison
  (both scores see the same hard/easy examples), and treating it as if it
  were two independent samples - e.g. via non-overlapping bootstrap CIs -
  would be conservative in the wrong direction (understating significance)
  because it throws away the positive correlation between the two scores.

**A real, caught-before-it-shipped performance bug, logged here rather than
silently fixed.** The first version of this module computed AUROC via the
same O(n_pos * n_neg) dense pairwise-comparison matrix `router.auroc` uses
(`diff = pos.unsqueeze(1) - neg.unsqueeze(0)`) - exact, and genuinely "cheap
enough" (router.py's own docstring) at the ~1,500-2,000-sample scale that
function was written for. Run inside a 2,000-3,000-draw bootstrap loop at
this project's actual largest real sample sizes (the LLM extension's
`id_test`, ~130,000 tokens; the full-scale ImageNet-1k OOD comparison,
n=50,000), that same formula requires allocating a dense matrix with tens of
billions of entries *per draw* - it was caught failing with a literal
`RuntimeError: ... not enough memory: you tried to allocate 16085602400
bytes` when the equivalent full-dataset call was hit elsewhere in this
project's own test suite (`tests/test_llm_extension.py`,
`tests/test_llm_extension_pythia.py`), and would have made the bootstrap loop
computationally infeasible (trillions of pairwise comparisons) even with
unlimited memory. Both `_rank_based_auroc` (below) and `delong_test`'s
structural components are therefore implemented via the O(n log n)
rank/midrank formulation - respectively the standard Mann-Whitney-U-via-ranks
identity, and DeLong's own comparison's efficient reformulation (Sun & Xu,
2014, "Fast Implementation of DeLong's Algorithm..." - the paper's whole
point was avoiding exactly the O(n^2) construction a naive reading of the
1988 paper's covariance matrix suggests). Both are proven algebraically
equivalent to the exact pairwise/psi definitions below, and that equivalence
is checked numerically (not just claimed) in `tests/test_significance.py`
against a small, independent, deliberately-naive O(n_pos*n_neg) reference
implementation before either is trusted on real data.

Two different scores over DIFFERENT samples (e.g. combiner AUROC on
ResNet-50's `id_test` vs. combiner AUROC on ViT-B/16's `id_test`, which have
different sizes and are not the same underlying predictions) are NOT a
DeLong scenario - DeLong requires literally the same paired sample. Use
`bootstrap_auroc_ci` on each independently and compare CIs for those cases.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import torch
from scipy.stats import norm as _norm_dist
from scipy.stats import rankdata as _rankdata


def _rank_based_auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """AUROC via the Mann-Whitney U-statistic/rank-sum identity - O(n log n)
    (dominated by one `argsort`-based rank computation), exactly equivalent to
    `router.auroc`'s O(n_pos*n_neg) pairwise-comparison definition (including
    its tie convention: ties count as half a win, matched here by
    `rankdata`'s `method="average"`, the standard "midrank" tie-breaking
    rule). Equivalence proof: for combined ranks `R` of `pos` within
    `concatenate(pos, neg)` (average/midrank for ties), the Mann-Whitney
    U-statistic `U = sum(R) - n_pos*(n_pos+1)/2` counts exactly
    `#{(i,j): pos_i > neg_j} + 0.5*#{(i,j): pos_i == neg_j}` - the same
    quantity `router.auroc` computes directly - and `AUROC = U/(n_pos*n_neg)`
    is the standard, textbook Mann-Whitney-AUC equivalence. Verified
    numerically against `router.auroc` on random data (including deliberately
    tie-heavy cases) in `tests/test_significance.py` before being used for
    anything real, rather than trusted from the derivation alone.
    """
    n_pos, n_neg = len(pos), len(neg)
    combined = np.concatenate([pos, neg])
    ranks = _rankdata(combined, method="average")  # 1-indexed, ties averaged (midranks)
    rank_sum_pos = ranks[:n_pos].sum()
    u_statistic = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return float(u_statistic / (n_pos * n_neg))


class BootstrapAurocResult(NamedTuple):
    """Point estimate plus a percentile bootstrap CI (`ci_lo`, `ci_hi`) at
    the requested confidence level, and the full array of per-draw AUROCs
    (`draws`) in case a caller wants to inspect the distribution directly
    (e.g. to check it isn't badly skewed before trusting the percentile CI).
    """

    auroc: float
    ci_lo: float
    ci_hi: float
    confidence: float
    n_bootstrap: int
    draws: np.ndarray


def bootstrap_auroc_ci(
    pos_scores: torch.Tensor,
    neg_scores: torch.Tensor,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: int | None = 0,
) -> BootstrapAurocResult:
    """Nonparametric percentile bootstrap CI for AUROC(pos_scores, neg_scores)
    (numerically equivalent to `router.auroc`, computed via the O(n log n)
    `_rank_based_auroc` so this stays tractable at this project's largest real
    sample sizes - see the module docstring for why the naive O(n_pos*n_neg)
    approach this started with was not just slow but actually infeasible
    there).

    Resamples `pos_scores` and `neg_scores` INDEPENDENTLY with replacement
    (each of size len(pos_scores)/len(neg_scores) respectively) on every
    draw, recomputes AUROC, and reports the `confidence`-level percentile
    interval of the resulting distribution - the standard two-sample
    bootstrap-CI-for-AUC procedure. Independent resampling of the two groups
    is correct here regardless of whether `pos_scores`/`neg_scores` came from
    a paired or unpaired original design, because AUROC itself is a function
    of the (unordered) positive and negative score SETS, not of any pairing
    between them.

    `seed=0` by default (not None) so a call with no arguments is
    reproducible, matching this project's general preference for explicit,
    documented seeds (`splits.py`, `notebooks/16`) over silently
    nondeterministic Monte Carlo - pass `seed=None` for a fresh draw each call.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        raise ValueError("pos_scores and neg_scores must both be non-empty")

    pos_np = pos_scores.detach().cpu().to(dtype=torch.float64).numpy()
    neg_np = neg_scores.detach().cpu().to(dtype=torch.float64).numpy()

    point = _rank_based_auroc(pos_np, neg_np)
    rng = np.random.default_rng(seed)
    n_pos, n_neg = len(pos_np), len(neg_np)
    draws = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        pos_idx = rng.integers(0, n_pos, size=n_pos)
        neg_idx = rng.integers(0, n_neg, size=n_neg)
        draws[i] = _rank_based_auroc(pos_np[pos_idx], neg_np[neg_idx])

    alpha = 1.0 - confidence
    ci_lo, ci_hi = np.quantile(draws, [alpha / 2.0, 1.0 - alpha / 2.0])
    return BootstrapAurocResult(
        auroc=point, ci_lo=float(ci_lo), ci_hi=float(ci_hi), confidence=confidence, n_bootstrap=n_bootstrap, draws=draws
    )


def _structural_components(pos: np.ndarray, neg: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """One score's DeLong structural components (Sun & Xu, 2014, "fast
    DeLong"), computed via midranks in O(n log n) - NOT the O(n_pos*n_neg)
    dense psi-matrix a literal reading of DeLong's 1988 covariance formula
    might suggest (see the module docstring for why that naive approach was
    tried first, caught, and replaced).

    Derivation (self-contained, not just cited): let `psi(x,y) = 1` if
    `x>y`, `0.5` if `x==y`, `0` if `x<y` (the Mann-Whitney placement kernel).
    `V10_i = mean_j psi(pos_i, neg_j)`, `V01_j = mean_i psi(pos_i, neg_j)`,
    and `theta = mean(V10) = mean(V01)` is the AUC. Let `Z = concatenate(pos,
    neg)`, and `TZ`/`TX`/`TY` be midranks (average-rank-for-ties, `1`-indexed)
    of `Z`/`pos`/`neg` computed WITHIN THEIR OWN array. For positive `i`, if
    `a_Y = #{neg_j < pos_i}` and `b_Y = #{neg_j == pos_i}`, then `TZ_i - TX_i
    = a_Y + b_Y/2 = sum_j psi(pos_i, neg_j)` - the two `pos`-internal tie/order
    terms common to both midranks cancel exactly, leaving only pos_i's
    relationship to the `neg` array, which is exactly what `V10_i` needs. So
    `V10_i = (TZ_i - TX_i) / n_neg`. The symmetric argument for negatives
    (using `m - (...)` since `psi(x,y) = 1 - psi(y,x)` when `x != y`, `0.5`
    when equal) gives `V01_j = 1 - (TZ_{n_pos+j} - TY_j) / n_pos`. Both
    identities, and `mean(V10) == mean(V01) == theta`, are checked numerically
    against the direct O(n_pos*n_neg) psi-matrix definition on random data
    (including ties) in `tests/test_significance.py` before being trusted.
    """
    n_pos, n_neg = len(pos), len(neg)
    combined = np.concatenate([pos, neg])
    tz = _rankdata(combined, method="average")
    tx = _rankdata(pos, method="average")
    ty = _rankdata(neg, method="average")

    v10 = (tz[:n_pos] - tx) / n_neg
    v01 = 1.0 - (tz[n_pos:] - ty) / n_pos
    theta = float(v10.mean())
    return v10, v01, theta


class DelongResult(NamedTuple):
    auc_a: float
    auc_b: float
    auc_diff: float  # auc_a - auc_b
    z: float
    p_value: float


def delong_test(correct: torch.Tensor, score_a: torch.Tensor, score_b: torch.Tensor) -> DelongResult:
    """DeLong (1988) paired test for AUC_a == AUC_b, computed via the O(n log
    n) structural-components/Mann-Whitney-U reformulation (Sun & Xu, 2014),
    which is mathematically identical to DeLong's original U-statistic-based
    covariance derivation - see `_structural_components`'s docstring for the
    self-contained equivalence derivation, and the module docstring for why
    computing it this way (rather than the O(n_pos*n_neg) dense construction a
    literal reading of the 1988 paper's covariance matrix might suggest)
    matters at this project's real sample sizes.

    `correct` is the SAME binary outcome for both scores (e.g. "was this
    id_test prediction correct"), and `score_a`/`score_b` are two different
    scores computed on the SAME underlying examples (e.g. combiner S vs. raw
    MSP) - this is what makes the two AUCs correlated rather than
    independent, and what a plain two-sample test would get wrong.

    Method: for each score, split into per-positive placements V10 (mean
    Mann-Whitney placement of that positive above all negatives) and
    per-negative placements V01 (symmetric). Their means recover the AUC
    (`theta_a`/`theta_b`); their sample covariance across scores A and B -
    taken separately over the positive set (`S10`) and the negative set
    (`S01`), matching DeLong's derivation that these are the two independent
    sources of variance - gives the (co)variance of (theta_a, theta_b) once
    scaled by 1/n_pos and 1/n_neg respectively. `Var(theta_a - theta_b) =
    S[0,0] + S[1,1] - 2*S[0,1]` follows directly. `z = diff / sqrt(Var)` is
    then asymptotically standard normal under H0: AUC_a == AUC_b, giving a
    two-sided p-value from the normal survival function.

    Degenerate case: if `score_a` and `score_b` are identical (or the
    variance estimate is exactly 0 for any other reason, e.g. n_pos==1 or
    n_neg==1), returns z=0.0, p_value=1.0 rather than dividing by zero -
    "no evidence of a difference" is the correct answer when there is no
    variance to detect one, not a NaN.
    """
    correct = correct.to(dtype=torch.bool)
    if correct.all() or not correct.any():
        raise ValueError("delong_test needs both correct and incorrect examples in `correct`")
    if not (len(correct) == len(score_a) == len(score_b)):
        raise ValueError("correct, score_a, and score_b must all have the same length (paired samples)")

    correct_np = correct.detach().cpu().numpy()
    a_np = score_a.detach().cpu().to(dtype=torch.float64).numpy()
    b_np = score_b.detach().cpu().to(dtype=torch.float64).numpy()

    pos_a, neg_a = a_np[correct_np], a_np[~correct_np]
    pos_b, neg_b = b_np[correct_np], b_np[~correct_np]
    n_pos, n_neg = int(correct_np.sum()), int((~correct_np).sum())
    if n_pos < 2 or n_neg < 2:
        raise ValueError(f"DeLong's variance estimate needs >=2 positives and >=2 negatives, got {n_pos}/{n_neg}")

    v10_a, v01_a, theta_a = _structural_components(pos_a, neg_a)
    v10_b, v01_b, theta_b = _structural_components(pos_b, neg_b)

    v10 = np.stack([v10_a, v10_b], axis=1)  # (n_pos, 2)
    v01 = np.stack([v01_a, v01_b], axis=1)  # (n_neg, 2)
    s10 = np.cov(v10, rowvar=False, ddof=1)  # (2, 2), DeLong's positive-set variance component
    s01 = np.cov(v01, rowvar=False, ddof=1)  # (2, 2), negative-set component
    s = s10 / n_pos + s01 / n_neg

    var_diff = float(s[0, 0] + s[1, 1] - 2.0 * s[0, 1])
    diff = theta_a - theta_b
    if var_diff <= 0.0:
        return DelongResult(auc_a=theta_a, auc_b=theta_b, auc_diff=diff, z=0.0, p_value=1.0)

    z = diff / (var_diff**0.5)
    p_value = float(2.0 * _norm_dist.sf(abs(z)))
    return DelongResult(auc_a=theta_a, auc_b=theta_b, auc_diff=diff, z=float(z), p_value=p_value)
