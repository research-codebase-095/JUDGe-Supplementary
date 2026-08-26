import os

import numpy as np
import torch

from deployment_reliability.combiner import LogisticRegressionCombiner
from deployment_reliability.features import featurize
from deployment_reliability.normalization import ReferenceNormalizer
from deployment_reliability.pipeline import (
    ReliabilityPipeline,
    anomaly_augmented_policy,
    joint_cost_sensitive_policy,
    scalar_threshold_policy,
)
from deployment_reliability.reliability import estimate_reliability_state
from deployment_reliability.router import EXECUTE, HITL, VERIFY, cost_sensitive_joint_policy, cost_sensitive_thresholds, route, route_joint

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESNET50_CACHE = os.path.join(REPO_ROOT, "data", "logit_cache_resnet50.pt")


def _toy_combiner():
    torch.manual_seed(0)
    n = 300
    phi = torch.randn(n, 5)
    y = (phi[:, 0] + 0.1 * torch.randn(n) > 0.0).float()
    return LogisticRegressionCombiner().fit(phi, y)


def test_pipeline_estimate_matches_estimate_reliability_state_directly():
    combiner = _toy_combiner()
    logits = torch.randn(8, 1000)
    pipeline = ReliabilityPipeline(combiner)
    state_a = pipeline.estimate(logits)
    state_b = estimate_reliability_state(logits, combiner)
    assert torch.equal(state_a.correctness, state_b.correctness)
    assert torch.equal(state_a.uncertainty, state_b.uncertainty)


def test_pipeline_decide_applies_an_arbitrary_policy_to_the_state():
    combiner = _toy_combiner()
    logits = torch.randn(5, 1000)
    pipeline = ReliabilityPipeline(combiner)

    calls = []

    def recording_policy(state):
        calls.append(state)
        return ["Execute"] * len(state.correctness)

    result = pipeline.decide(logits, recording_policy)
    assert result == ["Execute"] * 5
    assert len(calls) == 1
    assert calls[0].correctness.shape == (5,)


def test_scalar_threshold_policy_reproduces_router_route_exactly():
    # The point of this policy: the old scalar Psi_theta pipeline is a
    # SPECIAL CASE of a Policy over r(x), not a separate system.
    combiner = _toy_combiner()
    torch.manual_seed(1)
    logits = torch.randn(20, 1000) * 4
    pipeline = ReliabilityPipeline(combiner)
    state = pipeline.estimate(logits)

    policy = scalar_threshold_policy(tau_hi=0.7, tau_lo=0.3, field="correctness")
    via_policy = policy(state)
    via_router = route(state.correctness, tau_hi=0.7, tau_lo=0.3)
    assert via_policy == via_router


def test_anomaly_augmented_policy_downgrades_execute_when_anomaly_is_high():
    combiner = _toy_combiner()
    torch.manual_seed(2)
    reference_phi = torch.randn(500, 5)
    normalizer = ReferenceNormalizer().fit(reference_phi)
    logits = torch.randn(6, 1000)
    pipeline = ReliabilityPipeline(combiner, reference_normalizer=normalizer)
    state = pipeline.estimate(logits)

    # Force a controlled scenario: everything would Execute on correctness
    # alone, but one entry has an anomaly score well past the threshold.
    state = state._replace(
        correctness=torch.full((6,), 0.95),
        distribution_shift=torch.tensor([0.1, 0.1, 5.0, 0.1, 0.1, 0.1]),
    )
    policy = anomaly_augmented_policy(tau_hi=0.7, tau_lo=0.3, anomaly_threshold=2.0, field="correctness")
    decisions = policy(state)
    assert decisions == [EXECUTE, EXECUTE, VERIFY, EXECUTE, EXECUTE, EXECUTE]


def test_anomaly_augmented_policy_never_downgrades_verify_or_hitl():
    combiner = _toy_combiner()
    state_correctness = torch.tensor([0.95, 0.5, 0.05])  # Execute, Verify, HITL under tau_hi=0.7/tau_lo=0.3
    state = estimate_reliability_state(torch.randn(3, 1000), combiner)
    state = state._replace(correctness=state_correctness, distribution_shift=torch.full((3,), 100.0))
    policy = anomaly_augmented_policy(tau_hi=0.7, tau_lo=0.3, anomaly_threshold=1.0, field="correctness")
    decisions = policy(state)
    # Only the Execute entry can be downgraded (to Verify); Verify/HITL entries
    # are untouched regardless of how anomalous they look.
    assert decisions == [VERIFY, VERIFY, HITL]


def test_anomaly_augmented_policy_is_a_no_op_without_a_reference_normalizer():
    # distribution_shift comes back as NaN with no reference_normalizer
    # (reliability.py's documented behavior); every NaN > threshold
    # comparison is False, so this must degrade to scalar_threshold_policy
    # silently rather than erroring or downgrading everything.
    combiner = _toy_combiner()
    torch.manual_seed(3)
    logits = torch.randn(10, 1000) * 4
    pipeline = ReliabilityPipeline(combiner)  # no reference_normalizer
    state = pipeline.estimate(logits)
    assert torch.isnan(state.distribution_shift).all()

    scalar_decisions = scalar_threshold_policy(tau_hi=0.7, tau_lo=0.3, field="correctness")(state)
    anomaly_decisions = anomaly_augmented_policy(tau_hi=0.7, tau_lo=0.3, anomaly_threshold=2.0, field="correctness")(
        state
    )
    assert scalar_decisions == anomaly_decisions


def test_real_resnet50_anomaly_augmented_policy_improves_imagenet_a_catch_rate():
    # Reproduces PROGRESS_REPORT.md 3.7's finding as tested library code
    # instead of one-off notebook glue: downgrading Execute->Verify when
    # distribution_shift is high should catch strictly more imagenet_a
    # errors than the plain scalar policy, without touching id_test's
    # Execute-band error rate.
    assert os.path.exists(RESNET50_CACHE), "run scripts/collect_logits.py resnet50 first"
    cache = torch.load(RESNET50_CACHE)
    logits, labels = cache["logits"], cache["labels"]
    splits_arr = np.array(cache["splits"])

    def mask(name):
        return torch.from_numpy(splits_arr == name)

    m_fit, m_test, m_a = mask("combiner_fit"), mask("id_test"), mask("imagenet_a")
    correct = logits.argmax(dim=-1) == labels

    phi = featurize(logits)
    combiner = LogisticRegressionCombiner().fit(phi[m_fit], correct[m_fit].float())
    normalizer = ReferenceNormalizer().fit(phi[m_fit])
    pipeline = ReliabilityPipeline(combiner, reference_normalizer=normalizer)

    tau_hi, tau_lo = 0.7, 0.3
    scalar_policy = scalar_threshold_policy(tau_hi, tau_lo, field="correctness")
    augmented_policy = anomaly_augmented_policy(
        tau_hi, tau_lo, anomaly_threshold=2.0, field="correctness", anomaly_field="distribution_shift"
    )

    scalar_decisions_a = pipeline.decide(logits[m_a], scalar_policy)
    augmented_decisions_a = pipeline.decide(logits[m_a], augmented_policy)

    def catch_rate(decisions, correct_mask):
        errors = ~correct_mask
        caught = sum(
            1 for d, err in zip(decisions, errors.tolist()) if err and d != EXECUTE
        )
        return caught / max(1, int(errors.sum()))

    scalar_catch = catch_rate(scalar_decisions_a, correct[m_a])
    augmented_catch = catch_rate(augmented_decisions_a, correct[m_a])
    assert augmented_catch >= scalar_catch, "anomaly-augmented policy should never catch fewer imagenet_a errors"

    # And it should not change anything on id_test's Execute-band error rate
    # by more than a negligible amount (it's meant to be a targeted rescue,
    # not a general-purpose tightening of the threshold).
    scalar_decisions_id = pipeline.decide(logits[m_test], scalar_policy)
    augmented_decisions_id = pipeline.decide(logits[m_test], augmented_policy)

    def execute_error_rate(decisions, correct_mask):
        idx = [i for i, d in enumerate(decisions) if d == EXECUTE]
        if not idx:
            return 0.0
        errs = (~correct_mask[idx]).float().mean().item()
        return errs

    scalar_err = execute_error_rate(scalar_decisions_id, correct[m_test])
    augmented_err = execute_error_rate(augmented_decisions_id, correct[m_test])
    assert augmented_err <= scalar_err + 0.01


def test_joint_cost_sensitive_policy_reduces_to_scalar_when_anomaly_cost_is_zero():
    combiner = _toy_combiner()
    torch.manual_seed(4)
    reference_phi = torch.randn(500, 5)
    normalizer = ReferenceNormalizer().fit(reference_phi)
    logits = torch.randn(15, 1000) * 4
    pipeline = ReliabilityPipeline(combiner, reference_normalizer=normalizer)

    cost_kwargs = dict(c_execute_incorrect=10.0, c_verify_correct=1.0, c_verify_incorrect=2.0, c_hitl=3.0)
    p_hi, p_lo = cost_sensitive_thresholds(**cost_kwargs)
    zero_cost_policy = cost_sensitive_joint_policy(**cost_kwargs, anomaly_cost_execute=0.0)
    joint_decisions = pipeline.decide(
        logits, joint_cost_sensitive_policy(zero_cost_policy, field="correctness", anomaly_field="distribution_shift")
    )
    scalar_decisions = pipeline.decide(logits, scalar_threshold_policy(p_hi, p_lo, field="correctness"))
    assert joint_decisions == scalar_decisions


def test_joint_cost_sensitive_policy_raises_on_nan_anomaly_field_instead_of_silently_misrouting():
    # Regression test for a real bug found during development: route_joint's
    # argmin degenerates to always selecting HITL when the anomaly field is
    # NaN (0*NaN=NaN under IEEE754, so even actions with a zero q-weight get
    # a NaN cost) - a silent, dangerous failure mode, not the graceful
    # fallback anomaly_augmented_policy has. Must raise instead.
    combiner = _toy_combiner()
    logits = torch.randn(5, 1000)
    pipeline = ReliabilityPipeline(combiner)  # no reference_normalizer -> distribution_shift is all-NaN
    cost_policy = cost_sensitive_joint_policy(
        c_execute_incorrect=10.0, c_verify_correct=1.0, c_verify_incorrect=2.0, c_hitl=3.0, anomaly_cost_execute=5.0
    )
    policy = joint_cost_sensitive_policy(cost_policy, field="correctness", anomaly_field="distribution_shift")
    try:
        pipeline.decide(logits, policy)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_real_resnet50_joint_policy_matches_or_beats_fixed_anomaly_policy_with_no_more_friction():
    # The full no-peeking validation (DESIGN.md's joint-policy section):
    # calibrate anomaly_cost_execute purely from combiner_fit/id_test (a 5%
    # friction budget on id_test's own Execute band, mirroring the same
    # precedent DESIGN.md 15.5 used for anomaly_augmented_policy's fixed
    # threshold) - imagenet_a is touched only for the final, evaluation-only
    # comparison below.
    assert os.path.exists(RESNET50_CACHE), "run scripts/collect_logits.py resnet50 first"
    cache = torch.load(RESNET50_CACHE)
    logits, labels = cache["logits"], cache["labels"]
    splits_arr = np.array(cache["splits"])

    def mask(name):
        return torch.from_numpy(splits_arr == name)

    m_fit, m_test, m_a = mask("combiner_fit"), mask("id_test"), mask("imagenet_a")
    correct = logits.argmax(dim=-1) == labels

    phi = featurize(logits)
    combiner = LogisticRegressionCombiner().fit(phi[m_fit], correct[m_fit].float())
    normalizer = ReferenceNormalizer().fit(phi[m_fit])
    pipeline = ReliabilityPipeline(combiner, reference_normalizer=normalizer)

    cost_kwargs = dict(c_execute_incorrect=10.0, c_verify_correct=1.0, c_verify_incorrect=2.0, c_hitl=3.0)
    p_hi, p_lo = cost_sensitive_thresholds(**cost_kwargs)

    # --- ID-only calibration (never touches imagenet_a) ---
    state_id = pipeline.estimate(logits[m_test])
    p_id, q_id = state_id.correctness, state_id.distribution_shift
    execute_mask_id = p_id > p_hi

    def friction_at(anomaly_cost_execute):
        candidate = cost_sensitive_joint_policy(**cost_kwargs, anomaly_cost_execute=anomaly_cost_execute)
        decisions = route_joint(p_id, q_id, candidate)
        flips = sum(1 for i, d in enumerate(decisions) if execute_mask_id[i] and d != EXECUTE)
        return flips / max(1, int(execute_mask_id.sum()))

    lo, hi, target = 0.0, 1000.0, 0.05
    for _ in range(40):
        mid = (lo + hi) / 2
        if friction_at(mid) < target:
            lo = mid
        else:
            hi = mid
    anomaly_cost_execute = hi

    # --- Build the three policies to compare ---
    scalar_policy = scalar_threshold_policy(p_hi, p_lo, field="correctness")
    fixed_threshold = torch.quantile(q_id, 0.95).item()  # same style as DESIGN.md 15.5's precedent
    augmented_policy = anomaly_augmented_policy(
        p_hi, p_lo, anomaly_threshold=fixed_threshold, field="correctness", anomaly_field="distribution_shift"
    )
    joint_policy = joint_cost_sensitive_policy(
        cost_sensitive_joint_policy(**cost_kwargs, anomaly_cost_execute=anomaly_cost_execute),
        field="correctness",
        anomaly_field="distribution_shift",
    )

    # --- Evaluation-only on imagenet_a (never used for fitting/calibration above) ---
    def catch_rate(decisions, correct_mask):
        errors = ~correct_mask
        caught = sum(1 for d, err in zip(decisions, errors.tolist()) if err and d != EXECUTE)
        return caught / max(1, int(errors.sum()))

    def execute_rate(decisions):
        return sum(1 for d in decisions if d == EXECUTE) / len(decisions)

    scalar_catch_a = catch_rate(pipeline.decide(logits[m_a], scalar_policy), correct[m_a])
    augmented_catch_a = catch_rate(pipeline.decide(logits[m_a], augmented_policy), correct[m_a])
    joint_catch_a = catch_rate(pipeline.decide(logits[m_a], joint_policy), correct[m_a])

    assert augmented_catch_a >= scalar_catch_a
    assert joint_catch_a >= augmented_catch_a - 0.005, "joint policy should be at least competitive with the fixed rule's catch-rate gain"

    scalar_exec_rate_id = execute_rate(pipeline.decide(logits[m_test], scalar_policy))
    augmented_exec_rate_id = execute_rate(pipeline.decide(logits[m_test], augmented_policy))
    joint_exec_rate_id = execute_rate(pipeline.decide(logits[m_test], joint_policy))

    augmented_friction = scalar_exec_rate_id - augmented_exec_rate_id
    joint_friction = scalar_exec_rate_id - joint_exec_rate_id
    assert joint_friction <= augmented_friction + 0.01, (
        "a 'more principled' joint policy costing MORE id_test friction for a comparable "
        "imagenet_a gain would not be an improvement worth keeping"
    )
