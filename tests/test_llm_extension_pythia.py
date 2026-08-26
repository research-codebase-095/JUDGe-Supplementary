import os

import numpy as np
import torch

from deployment_reliability.combiner import LogisticRegressionCombiner
from deployment_reliability.features import (
    DEFAULT_FEATURE_NAMES,
    FEATURE_DIRECTIONS,
    verify_feature_directions,
)
from deployment_reliability.router import auroc, risk_coverage_curve, threshold_for_target_risk
from deployment_reliability.splits import three_way_split
from test_llm_extension import _build_sequence_windows

# STUDY_PLAN.md 3.6 item 1b: a second, architecturally distinct LLM
# backbone (EleutherAI/pythia-160m, GPT-NeoX family, trained on the Pile -
# not GPT-2/WebText) run through the identical pipeline
# (scripts/collect_llm_logits.py --model EleutherAI/pythia-160m --auto), to
# check whether test_llm_extension.py's token/sequence-level discrimination-
# quality findings are a property of the design or an accident of GPT-2
# specifically. These tests mirror that file's real-data tests directly
# (same structure, same thresholds where the real numbers support them),
# reusing its _build_sequence_windows helper rather than duplicating it -
# that helper derives n_chunks from len(phi) rather than a hardcoded
# constant specifically so it works unchanged for a differently-tokenized
# model's cache.

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LLM_CACHE = os.path.join(REPO_ROOT, "data", "llm_feature_cache_pythia160m.pt")


def test_real_pythia160m_wikitext2_core_hypothesis_transfers_to_a_second_llm():
    # Real numbers found: token-level accuracy 0.4124 (GPT-2: 0.4108),
    # correctness AUROC 0.8134 (GPT-2: 0.83) - the discrimination-quality
    # finding replicates closely on a structurally different model family.
    assert os.path.exists(LLM_CACHE), "run scripts/collect_llm_logits.py --model EleutherAI/pythia-160m --auto first"
    cache = torch.load(LLM_CACHE)
    phi, correct, splits = cache["phi"], cache["correct"], cache["splits"]
    assert phi.dtype == torch.float32, "regression check for the float16-checkpoint overflow bug llm_backbone.py fixed"
    splits_arr = np.array(splits)

    def mask(name):
        return torch.from_numpy(splits_arr == name)

    m_fit, m_test = mask("combiner_fit"), mask("id_test")
    assert int(m_fit.sum()) > 0 and int(m_test.sum()) > 0

    combiner = LogisticRegressionCombiner().fit(phi[m_fit], correct[m_fit].float())
    S = combiner.score(phi)

    a = auroc(S[m_test][correct[m_test]], S[m_test][~correct[m_test]])
    assert a > 0.75, f"expected token-level AUROC clearly in the same range as GPT-2's 0.83, got {a:.4f}"

    unconditional_acc = correct[m_test].float().mean().item()
    coverage, risk = risk_coverage_curve(S[m_test], correct[m_test])
    idx_80 = (coverage >= 0.8).nonzero()[0, 0]
    acc_at_80 = 1 - risk[idx_80].item()
    assert acc_at_80 > unconditional_acc, "selective accuracy at 80% coverage should beat the no-reject baseline"


def test_real_pythia160m_wikitext2_four_of_five_feature_directions_confirmed_energy_score_is_the_exception():
    # A real, checked DIFFERENCE from GPT-2, not smoothed over: on GPT-2, all
    # five FEATURE_DIRECTIONS assumptions confirmed (test_llm_extension.py's
    # test_real_gpt2_wikitext2_all_feature_directions_confirmed). On
    # Pythia-160m, energy_score's assumed direction (+1, higher=more
    # confident) is CONTRADICTED - checked directly: mean energy_score is
    # slightly LOWER for correct predictions (835.0) than incorrect ones
    # (836.8) on the combiner_fit split, the opposite of the assumed sign.
    # The other four (msp, logit_margin, normalized_entropy, logit_l2_norm)
    # confirm correctly oriented, same as GPT-2. This is reported as a real,
    # backbone-specific finding - the same discipline already applied to
    # logit_l2_norm's direction on vision backbones (DESIGN.md 20.4) - not
    # silently special-cased away.
    assert os.path.exists(LLM_CACHE), "run scripts/collect_llm_logits.py --model EleutherAI/pythia-160m --auto first"
    cache = torch.load(LLM_CACHE)
    phi, correct, splits = cache["phi"], cache["correct"], cache["splits"]
    splits_arr = np.array(splits)
    m_fit = torch.from_numpy(splits_arr == "combiner_fit")

    result = verify_feature_directions(phi[m_fit], correct[m_fit])
    assert set(result) == set(DEFAULT_FEATURE_NAMES)
    failed = [name for name, ok in result.items() if not ok]
    assert failed == ["energy_score"], (
        f"expected energy_score to be the one contradicted direction on Pythia-160m data, got failed={failed} - "
        f"if this changes, STUDY_PLAN.md 3.6/DESIGN.md's write-up of this finding needs updating"
    )


def test_real_pythia160m_wikitext2_margin_dominates_the_fitted_combiner():
    # Same as GPT-2 (test_real_gpt2_wikitext2_margin_dominates_the_fitted_combiner)
    # and all three vision backbones (DESIGN.md 15.7) - checked here as an
    # observation about this second LLM, not assumed to automatically transfer.
    assert os.path.exists(LLM_CACHE), "run scripts/collect_llm_logits.py --model EleutherAI/pythia-160m --auto first"
    cache = torch.load(LLM_CACHE)
    phi, correct, splits = cache["phi"], cache["correct"], cache["splits"]
    splits_arr = np.array(splits)
    m_fit = torch.from_numpy(splits_arr == "combiner_fit")

    combiner = LogisticRegressionCombiner().fit(phi[m_fit], correct[m_fit].float())
    margin_idx = DEFAULT_FEATURE_NAMES.index("logit_margin")
    weights = combiner.weight.detach().abs()
    assert weights[margin_idx] == weights.max(), "expected logit_margin to carry the largest fitted weight magnitude"


def test_real_pythia160m_wikitext2_sequence_level_mean_aggregation_beats_min_for_most_features():
    # Same pattern as GPT-2 (mean_wins>=4/5), checked directly rather than
    # assumed to transfer: real numbers found mean_wins=4, min_wins=1, with
    # logit_l2_norm again the one consistent exception where min beats mean -
    # an exact replication of DESIGN.md 14.6's finding on a second model.
    assert os.path.exists(LLM_CACHE), "run scripts/collect_llm_logits.py --model EleutherAI/pythia-160m --auto first"
    cache = torch.load(LLM_CACHE)
    phi, correct = cache["phi"], cache["correct"]

    window_size = 5
    phi_mean, seq_correct = _build_sequence_windows(phi, correct, window_size, method="mean")
    phi_min, seq_correct_min = _build_sequence_windows(phi, correct, window_size, method="min")
    assert torch.equal(seq_correct, seq_correct_min)

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


def test_real_pythia160m_wikitext2_sequence_level_combiner_beats_chance_and_no_reject_baseline():
    # Same protocol as GPT-2's sequence-level test; real number found:
    # strict-target (k=5, threshold=1.0) sequence-level AUROC 0.8946 (GPT-2: 0.91).
    assert os.path.exists(LLM_CACHE), "run scripts/collect_llm_logits.py --model EleutherAI/pythia-160m --auto first"
    cache = torch.load(LLM_CACHE)
    phi, correct = cache["phi"], cache["correct"]

    phi_seq, seq_correct = _build_sequence_windows(phi, correct, window_size=5, method="mean")
    n = len(seq_correct)
    fit_idx, cal_idx, test_idx = three_way_split(n, seed=0)
    combiner = LogisticRegressionCombiner().fit(phi_seq[fit_idx], seq_correct[fit_idx].float())
    S = combiner.score(phi_seq)

    a = auroc(S[test_idx][seq_correct[test_idx]], S[test_idx][~seq_correct[test_idx]])
    assert a > 0.8, f"sequence-level correctness AUROC {a:.4f} should be clearly better than chance"

    unconditional_acc = seq_correct[test_idx].float().mean().item()
    coverage, risk = risk_coverage_curve(S[test_idx], seq_correct[test_idx])
    idx_80 = (coverage >= 0.8).nonzero()[0, 0]
    acc_at_80 = 1 - risk[idx_80].item()
    assert acc_at_80 > unconditional_acc, "selective accuracy at 80% coverage should beat the no-reject baseline"


def test_real_pythia160m_wikitext2_fractional_threshold_replicates_the_gpt2_fix():
    # DESIGN.md 14.8's fractional-correctness fix (threshold=0.8), re-checked
    # on this second model rather than assumed to transfer: real numbers
    # found n_pos_cal=532 (GPT-2: 544, both well above the >400 bar that
    # distinguishes this from the strict target's calibration-starved
    # failure mode) and AUROC 0.8297 (GPT-2: 0.83) - both properties hold.
    assert os.path.exists(LLM_CACHE), "run scripts/collect_llm_logits.py --model EleutherAI/pythia-160m --auto first"
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
