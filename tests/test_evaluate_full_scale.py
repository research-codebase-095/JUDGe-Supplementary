"""Unit tests for scripts/evaluate_full_scale.py's own metric helpers
(ece/selective_accuracy_at_coverages) - these produce §24's headline
full-scale numbers directly, so they're worth a tight, hand-checked
regression test even though the script itself lives outside the tested
`deployment_reliability` package."""

import os
import sys

import torch

SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from evaluate_full_scale import bucket_report, ece, fit_pipeline, score_dataset  # noqa: E402

REPO_ROOT = os.path.dirname(SCRIPTS_DIR)
DATA_DIR = os.path.join(REPO_ROOT, "data")
SMALL_CACHE = os.path.join(DATA_DIR, "logit_cache_resnet50.pt")
FULL_CACHE = os.path.join(DATA_DIR, "logit_cache_imagenet1k_resnet50.pt")


def test_ece_matches_hand_computed_two_bin_example():
    # Bin 0 = [0, 0.5): scores 0.1, 0.2, 0.3 -> conf=0.2, acc=2/3, weight=0.5
    # Bin 1 = [0.5, 1.0]: scores 0.6, 0.7, 0.9 -> conf=0.7333.., acc=2/3, weight=0.5
    scores = torch.tensor([0.1, 0.2, 0.3, 0.6, 0.7, 0.9])
    correct = torch.tensor([True, False, True, True, False, True])
    expected = 0.5 * abs(0.2 - 2 / 3) + 0.5 * abs((0.6 + 0.7 + 0.9) / 3 - 2 / 3)
    assert abs(ece(scores, correct, n_bins=2) - expected) < 1e-6


def test_ece_includes_a_score_of_exactly_one_in_the_last_bin():
    # A perfectly-confident score of 1.0 must land in the last bin
    # ([0.9, 1.0] must be right-closed), not be silently dropped from both
    # the numerator and the implicit weight denominator - it previously fell
    # through every bin's strict `scores < hi` check, since even the last
    # bin's `hi` is exactly 1.0.
    #
    # correct is deliberately chosen (False for the score==1.0 point, True
    # for the other) so that dropping the score==1.0 point - the pre-fix
    # behavior - gives a different, hand-computable wrong answer (0.35, from
    # only the score=0.3 point counting) rather than coincidentally matching
    # the correct one:
    #   bin[0.3,0.4): conf=0.3, acc=1.0 (correct=True), weight=0.5 -> 0.35
    #   bin[0.9,1.0]: conf=1.0, acc=0.0 (correct=False), weight=0.5 -> 0.50
    #   correct total = 0.35 + 0.50 = 0.85
    scores = torch.tensor([1.0, 0.3])
    correct = torch.tensor([False, True])
    assert abs(ece(scores, correct, n_bins=10) - 0.85) < 1e-6


def test_ece_perfect_calibration_is_zero():
    # Each bin's mean confidence exactly equals its mean accuracy (0.0/1.0)
    # by construction, including the boundary score of 1.0 in the last bin.
    scores = torch.tensor([0.0, 0.0, 1.0, 1.0])
    correct = torch.tensor([False, False, True, True])
    assert abs(ece(scores, correct, n_bins=10) - 0.0) < 1e-9


def test_real_resnet50_bucket_risk_is_monotonic_execute_lt_verify_lt_hitl():
    # The entire 3-way routing design (DESIGN.md 10) implicitly depends on
    # empirical risk being monotonic across bands - Execute < Verify < HITL
    # is what makes "route to HITL when least trustworthy" a coherent policy
    # at all, rather than an arbitrary three-way split with no ordering
    # guarantee. This was never explicitly checked anywhere in this project
    # before (found during a numerical audit) - checked here directly, on
    # real full-scale data, for both the naive default threshold and the
    # Bonferroni-corrected alternative (DESIGN.md 23.6), rather than assumed
    # from the design's intent.
    assert os.path.exists(SMALL_CACHE), "run scripts/collect_logits.py resnet50 first"
    assert os.path.exists(FULL_CACHE), "run scripts/collect_logits_imagenet1k.py resnet50 first"

    pipeline = fit_pipeline(SMALL_CACHE)
    full_cache = torch.load(FULL_CACHE)
    _, correct, s, _ = score_dataset(pipeline, full_cache["logits"], full_cache["labels"])

    for tau_hi_key, tau_lo_key in [("tau_hi", "tau_lo"), ("tau_hi_bonferroni", "tau_lo_bonferroni")]:
        buckets = bucket_report(s, correct, pipeline[tau_hi_key], pipeline[tau_lo_key])
        execute_risk = buckets["Execute"]["empirical_risk"]
        verify_risk = buckets["Verify"]["empirical_risk"]
        hitl_risk = buckets["HITL"]["empirical_risk"]
        assert execute_risk < verify_risk < hitl_risk, (
            f"[{tau_hi_key}] expected monotonic risk Execute < Verify < HITL, "
            f"got {execute_risk:.4f} / {verify_risk:.4f} / {hitl_risk:.4f}"
        )
