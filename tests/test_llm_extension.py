import os

import numpy as np
import torch

from deployment_reliability.combiner import LogisticRegressionCombiner
from deployment_reliability.features import (
    DEFAULT_FEATURE_NAMES,
    FEATURE_DIRECTIONS,
    aggregate_sequence_features,
    sequence_correctness,
    verify_feature_directions,
)
from deployment_reliability.router import auroc, risk_coverage_curve, threshold_for_target_risk
from deployment_reliability.splits import three_way_split

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LLM_CACHE = os.path.join(REPO_ROOT, "data", "llm_feature_cache_gpt2.pt")
CHUNK_SIZE = 1024  # must match scripts/collect_llm_logits.py's context window (shared by GPT-2 and Pythia-160m - see collect_llm_logits.py's module docstring)
TOKENS_PER_CHUNK = CHUNK_SIZE - 1  # logits[:-1] drops the last position per chunk
N_CHUNKS = 252  # full WikiText-2 validation split at CHUNK_SIZE=1024, GPT-2 tokenizer specifically


def _build_sequence_windows(phi, correct, window_size, method="mean", threshold=1.0):
    # Windows never cross a chunk boundary: consecutive chunks ARE adjacent
    # in the original text, but each was a separate forward pass with no
    # cross-chunk attention, so treating a boundary-spanning window as a
    # real "sequence" would assume model continuity that never happened.
    # n_chunks is derived from len(phi) rather than a hardcoded constant, so
    # this same helper works unchanged for any cache (tests/test_llm_extension_pythia.py
    # reuses it directly) regardless of that model's own tokenizer producing
    # a different total token count than GPT-2's.
    n_chunks = len(phi) // TOKENS_PER_CHUNK
    windows_phi, windows_correct = [], []
    for c in range(n_chunks):
        start = c * TOKENS_PER_CHUNK
        chunk_phi = phi[start : start + TOKENS_PER_CHUNK]
        chunk_correct = correct[start : start + TOKENS_PER_CHUNK]
        n_windows = TOKENS_PER_CHUNK // window_size
        for w in range(n_windows):
            s = w * window_size
            window_phi = chunk_phi[s : s + window_size]
            window_correct = chunk_correct[s : s + window_size]
            windows_phi.append(aggregate_sequence_features(window_phi, method=method))
            windows_correct.append(sequence_correctness(window_correct, threshold=threshold))
    return torch.stack(windows_phi), torch.stack(windows_correct)


def test_real_gpt2_wikitext2_core_hypothesis_transfers_to_a_new_domain():
    # DESIGN.md 0's research hypothesis - logit geometry predicts prediction
    # reliability - stated and validated only for vision classifiers (§15)
    # until now. This is the first test of it on a structurally different
    # domain: autoregressive next-token prediction, real GPT-2 (124M) on the
    # real, complete WikiText-2 validation corpus (257,796 token-level
    # predictions, DESIGN.md 14.5/notebooks 14), with correctness defined at
    # the token level (y = 1[argmax(token_logits) == true_next_token]),
    # deliberately sidestepping the still-open sequence-aggregation question
    # (DESIGN.md 14.3/§7) rather than trying to resolve it in the same pass.
    assert os.path.exists(LLM_CACHE), "run scripts/collect_llm_logits.py first"
    cache = torch.load(LLM_CACHE)
    phi, correct, splits = cache["phi"], cache["correct"], cache["splits"]
    splits_arr = np.array(splits)

    def mask(name):
        return torch.from_numpy(splits_arr == name)

    m_fit, m_test = mask("combiner_fit"), mask("id_test")
    assert int(m_fit.sum()) > 0 and int(m_test.sum()) > 0

    combiner = LogisticRegressionCombiner().fit(phi[m_fit], correct[m_fit].float())
    S = combiner.score(phi)

    # Falsification criterion from DESIGN.md 0/STUDY_PLAN.md §2: the
    # hypothesis is falsified if logit-derived features cannot separate
    # correct from incorrect predictions better than chance, and cannot
    # improve the risk-coverage trade-off over a no-reject baseline. Checked
    # directly, not assumed.
    a = auroc(S[m_test][correct[m_test]], S[m_test][~correct[m_test]])
    assert a > 0.6, f"correctness AUROC {a:.4f} should be clearly better than chance (0.5)"

    unconditional_acc = correct[m_test].float().mean().item()
    coverage, risk = risk_coverage_curve(S[m_test], correct[m_test])
    idx_80 = (coverage >= 0.8).nonzero()[0, 0]
    acc_at_80 = 1 - risk[idx_80].item()
    assert acc_at_80 > unconditional_acc, "selective accuracy at 80% coverage should beat the no-reject baseline"


def test_real_gpt2_wikitext2_all_feature_directions_confirmed():
    # Same direction-audit discipline already applied to all three vision
    # backbones (tests/test_features.py) - reused here rather than assumed
    # to transfer, given logit_l2_norm's direction was found backwards on
    # vision data. Checked directly: all five confirm correctly oriented on
    # this domain too.
    assert os.path.exists(LLM_CACHE), "run scripts/collect_llm_logits.py first"
    cache = torch.load(LLM_CACHE)
    phi, correct, splits = cache["phi"], cache["correct"], cache["splits"]
    splits_arr = np.array(splits)
    m_fit = torch.from_numpy(splits_arr == "combiner_fit")

    result = verify_feature_directions(phi[m_fit], correct[m_fit])
    failed = [name for name, ok in result.items() if not ok]
    assert not failed, f"FEATURE_DIRECTIONS assumption contradicted by real GPT-2/WikiText-2 data for {failed}"
    assert set(result) == set(DEFAULT_FEATURE_NAMES)


def test_real_gpt2_wikitext2_margin_dominates_the_fitted_combiner():
    # Consistent with DESIGN.md 15.7's finding across all three vision
    # backbones (logit_margin dominates the L2-regularized fit) - checked
    # here as an observation about this new domain, not assumed to
    # automatically transfer.
    assert os.path.exists(LLM_CACHE), "run scripts/collect_llm_logits.py first"
    cache = torch.load(LLM_CACHE)
    phi, correct, splits = cache["phi"], cache["correct"], cache["splits"]
    splits_arr = np.array(splits)
    m_fit = torch.from_numpy(splits_arr == "combiner_fit")

    combiner = LogisticRegressionCombiner().fit(phi[m_fit], correct[m_fit].float())
    margin_idx = DEFAULT_FEATURE_NAMES.index("logit_margin")
    weights = combiner.weight.detach().abs()
    assert weights[margin_idx] == weights.max(), "expected logit_margin to carry the largest fitted weight magnitude"


def test_real_gpt2_wikitext2_sequence_level_mean_aggregation_beats_min_for_most_features():
    # DESIGN.md 14.3 originally suggested min ("weakest link") as the
    # natural sequence-aggregation rule for msp/margin - checked directly
    # on real 5-token windows rather than assumed, and found backwards:
    # mean aggregation gives higher single-feature correctness AUROC than
    # min for msp, logit_margin, normalized_entropy, and energy_score.
    # logit_l2_norm is the one consistent exception (min beats mean for
    # it) - not asserted here as "wrong," just recorded as the real,
    # mixed finding DESIGN.md 14.6 documents.
    assert os.path.exists(LLM_CACHE), "run scripts/collect_llm_logits.py first"
    cache = torch.load(LLM_CACHE)
    phi, correct = cache["phi"], cache["correct"]

    window_size = 5
    phi_mean, seq_correct = _build_sequence_windows(phi, correct, window_size, method="mean")
    phi_min, seq_correct_min = _build_sequence_windows(phi, correct, window_size, method="min")
    assert torch.equal(seq_correct, seq_correct_min)  # correctness labels don't depend on the aggregation method

    mean_wins, min_wins = 0, 0
    for i, name in enumerate(DEFAULT_FEATURE_NAMES):
        direction = FEATURE_DIRECTIONS[name]
        s_mean = phi_mean[:, i] * direction
        s_min = phi_min[:, i] * direction
        a_mean = auroc(s_mean[seq_correct], s_mean[~seq_correct])
        a_min = auroc(s_min[seq_correct], s_min[~seq_correct])
        if a_mean > a_min:
            mean_wins += 1
        else:
            min_wins += 1
    assert mean_wins >= 4, f"expected mean aggregation to win for at least 4/5 features, mean_wins={mean_wins}"


def test_real_gpt2_wikitext2_sequence_level_combiner_beats_chance_and_no_reject_baseline():
    # The actual payoff of sequence-level aggregation: fit a combiner on
    # mean-aggregated 5-token windows and confirm the same falsification
    # criterion used for token-level correctness (DESIGN.md 0/STUDY_PLAN.md
    # section 2) also holds at the sequence level.
    assert os.path.exists(LLM_CACHE), "run scripts/collect_llm_logits.py first"
    cache = torch.load(LLM_CACHE)
    phi, correct = cache["phi"], cache["correct"]

    phi_seq, seq_correct = _build_sequence_windows(phi, correct, window_size=5, method="mean")
    n = len(seq_correct)
    fit_idx, cal_idx, test_idx = three_way_split(n, seed=0)
    combiner = LogisticRegressionCombiner().fit(phi_seq[fit_idx], seq_correct[fit_idx].float())
    S = combiner.score(phi_seq)

    a = auroc(S[test_idx][seq_correct[test_idx]], S[test_idx][~seq_correct[test_idx]])
    assert a > 0.7, f"sequence-level correctness AUROC {a:.4f} should be clearly better than chance"

    unconditional_acc = seq_correct[test_idx].float().mean().item()
    coverage, risk = risk_coverage_curve(S[test_idx], seq_correct[test_idx])
    idx_80 = (coverage >= 0.8).nonzero()[0, 0]
    acc_at_80 = 1 - risk[idx_80].item()
    assert acc_at_80 > unconditional_acc, "selective accuracy at 80% coverage should beat the no-reject baseline"


def test_real_gpt2_wikitext2_sequence_level_absolute_selective_accuracy_is_near_its_ceiling():
    # DESIGN.md 14.6's key diagnosis: the strict all-correct target's low
    # absolute selective accuracy (1.57% -> 3.00% at k=5) looks like a
    # failure next to the token-level result, but is actually close to the
    # best ANY classifier could achieve given how rare "all 5 tokens
    # correct" is. At coverage c with true positive rate p < c, the best
    # possible selective accuracy is p/c (a perfect oracle ranking puts
    # every positive in the selected set). This test locks in that the
    # fitted combiner captures the large majority of that ceiling, not
    # just "does okay."
    assert os.path.exists(LLM_CACHE), "run scripts/collect_llm_logits.py first"
    cache = torch.load(LLM_CACHE)
    phi, correct = cache["phi"], cache["correct"]

    phi_seq, seq_correct = _build_sequence_windows(phi, correct, window_size=5, method="mean", threshold=1.0)
    n = len(seq_correct)
    fit_idx, cal_idx, test_idx = three_way_split(n, seed=0)
    combiner = LogisticRegressionCombiner().fit(phi_seq[fit_idx], seq_correct[fit_idx].float())
    S = combiner.score(phi_seq)

    base_rate = seq_correct[test_idx].float().mean().item()
    coverage, risk = risk_coverage_curve(S[test_idx], seq_correct[test_idx])
    idx_50 = (coverage >= 0.5).nonzero()[0, 0]
    observed = 1 - risk[idx_50].item()
    ceiling = min(1.0, base_rate / 0.5)
    assert observed / ceiling > 0.85, (
        f"expected the fitted combiner to capture most of the achievable ceiling at 50% coverage, "
        f"got observed={observed:.4f} ceiling={ceiling:.4f} ratio={observed/ceiling:.3f}"
    )


def test_real_gpt2_wikitext2_strict_target_starves_target_risk_calibration():
    # The distinct, second real problem with the strict all-correct target
    # (DESIGN.md 14.7/14.8): it's not just that absolute selective accuracy
    # looks weak, it's that the calibration split doesn't contain enough
    # positive windows to fit router.threshold_for_target_risk reliably.
    # Locked in as a regression test for the FAILURE MODE the fractional
    # threshold (next test) is meant to fix - if this ever stops failing,
    # the fractional-threshold justification in DESIGN.md 14.8 needs
    # revisiting, not silently left as-is.
    assert os.path.exists(LLM_CACHE), "run scripts/collect_llm_logits.py first"
    cache = torch.load(LLM_CACHE)
    phi, correct = cache["phi"], cache["correct"]

    phi_seq, seq_correct = _build_sequence_windows(phi, correct, window_size=5, method="mean", threshold=1.0)
    n = len(seq_correct)
    fit_idx, cal_idx, test_idx = three_way_split(n, seed=0)
    combiner = LogisticRegressionCombiner().fit(phi_seq[fit_idx], seq_correct[fit_idx].float())
    S = combiner.score(phi_seq)

    n_pos_cal = int(seq_correct[cal_idx].sum())
    assert n_pos_cal < 100, f"expected few calibration-split positives at the strict target, got {n_pos_cal}"

    target_risk = 0.10
    tau = threshold_for_target_risk(S[cal_idx], seq_correct[cal_idx], target_risk=target_risk)
    accepted = S[test_idx] >= tau
    actual_risk = (1 - seq_correct[test_idx][accepted].float().mean().item()) if accepted.any() else float("nan")
    assert actual_risk > target_risk * 1.5, (
        f"expected the strict target's threshold calibration to overshoot its target risk "
        f"(too few cal positives to fit reliably); got actual_risk={actual_risk:.4f} vs target={target_risk}"
    )


def test_real_gpt2_wikitext2_fractional_threshold_fixes_calibration_and_keeps_discrimination():
    # DESIGN.md 14.8: threshold=0.8 (tolerate one wrong token in five) is
    # the minimal relaxation from strict correctness that gives
    # threshold_for_target_risk enough calibration-split positives to work
    # properly, while keeping discrimination quality reasonably close to
    # the strict target's. Both properties checked directly.
    assert os.path.exists(LLM_CACHE), "run scripts/collect_llm_logits.py first"
    cache = torch.load(LLM_CACHE)
    phi, correct = cache["phi"], cache["correct"]

    phi_seq, seq_correct = _build_sequence_windows(phi, correct, window_size=5, method="mean", threshold=0.8)
    n = len(seq_correct)
    fit_idx, cal_idx, test_idx = three_way_split(n, seed=0)
    combiner = LogisticRegressionCombiner().fit(phi_seq[fit_idx], seq_correct[fit_idx].float())
    S = combiner.score(phi_seq)

    n_pos_cal = int(seq_correct[cal_idx].sum())
    assert n_pos_cal > 400, f"expected substantially more calibration-split positives than the strict target, got {n_pos_cal}"

    a = auroc(S[test_idx][seq_correct[test_idx]], S[test_idx][~seq_correct[test_idx]])
    assert a > 0.75, f"expected discrimination quality reasonably close to the strict target's, got AUROC={a:.4f}"

    target_risk = 0.10
    tau = threshold_for_target_risk(S[cal_idx], seq_correct[cal_idx], target_risk=target_risk)
    accepted = S[test_idx] >= tau
    actual_risk = (1 - seq_correct[test_idx][accepted].float().mean().item()) if accepted.any() else float("nan")
    assert actual_risk <= target_risk * 1.5, (
        f"expected the fractional target's threshold calibration to roughly honor its target risk, "
        f"got actual_risk={actual_risk:.4f} vs target={target_risk}"
    )
