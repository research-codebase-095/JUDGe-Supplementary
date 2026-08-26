"""The Reliability Pipeline: composes Evidence -> ReliabilityState -> Decision
as one object, so a caller drives the whole chain through a single interface
instead of wiring reliability.py's `estimate_reliability_state` and router.py's
`route` together by hand every time.

This exists in direct response to a specific criticism: an earlier version of
this package treated `ReliabilityState` (reliability.py) as a side diagnostic
sitting next to the "real" scalar pipeline (`combiner.py` -> `router.py`),
rather than as what decisions actually get made from. `ReliabilityPipeline`
makes the vector the thing a decision is a function *of* - `decide()` takes a
`Policy: ReliabilityState -> decision`, so any routing rule that wants to look
at more than one field of the state can be expressed and tested directly,
instead of being one-off code glued onto a scalar in a notebook.

`scalar_threshold_policy` shows the old scalar pipeline is a special case of
this, not a different thing (`tests/test_pipeline.py` checks it reproduces
`router.route`'s output exactly). `anomaly_augmented_policy` turns
PROGRESS_REPORT.md 3.7's real, notebook-only finding - downgrade Execute to
Verify when an anomaly signal is high, regardless of how confident the
correctness score itself is - into reusable, tested code that could only be
expressed once there was more than one field to look at.
"""

from __future__ import annotations

from typing import Callable

import torch

from .reliability import ReliabilityState, estimate_reliability_state
from .router import EXECUTE, VERIFY, AffineCostPolicy, route, route_joint

Policy = Callable[[ReliabilityState], list[str]]


class ReliabilityPipeline:
    """Evidence -> ReliabilityState -> Decision, as one composed object.

    `combiner` must already be fitted (§8); `reference_normalizer`/
    `calibrator` are optional and passed straight through to
    `estimate_reliability_state` (reliability.py) - see its docstring for
    what happens to `distribution_shift`/`familiarity`/`calibration_confidence`
    when they're omitted. `mahalanobis_scorer` (STUDY_PLAN.md 3.6 item 2) is
    likewise optional and passed straight through; when given, `estimate()`
    also requires a `features` argument (see there) to actually compute
    `feature_space_trust` - passing `mahalanobis_scorer` here alone, with no
    `features` at call time, still yields NaN for that field, the same
    honest-unknown default as omitting it entirely.

    `mc_dropout_model`/`mc_dropout_preprocess`/`mc_dropout_passes`
    (STUDY_PLAN.md 3.6 item 5) are also optional, all default `None`/`10`:
    when `mc_dropout_model` is given (a ViT-B/16 loaded via
    `backbone.load_frozen_vit_b16_with_dropout` - see `epistemic.py`'s scope
    notes for why this is ViT-B/16-only) AND `estimate()`/`decide()` are
    called with `images` (the raw PIL images, NOT derivable from `logits`),
    `epistemic_uncertainty` is computed via `epistemic.mc_dropout_predict`
    with an explicit extra cost of `mc_dropout_passes` forward passes.
    Without both, `epistemic_uncertainty` comes back NaN-filled, same
    honest-unknown default as every other optional field here.
    """

    def __init__(
        self,
        combiner,
        reference_normalizer=None,
        calibrator=None,
        mahalanobis_scorer=None,
        mc_dropout_model=None,
        mc_dropout_preprocess=None,
        mc_dropout_passes: int = 10,
    ) -> None:
        self.combiner = combiner
        self.reference_normalizer = reference_normalizer
        self.calibrator = calibrator
        self.mahalanobis_scorer = mahalanobis_scorer
        self.mc_dropout_model = mc_dropout_model
        self.mc_dropout_preprocess = mc_dropout_preprocess
        self.mc_dropout_passes = mc_dropout_passes

    def estimate(
        self, logits: torch.Tensor, features: torch.Tensor | None = None, images=None
    ) -> ReliabilityState:
        epistemic_uncertainty = None
        if images is not None and self.mc_dropout_model is not None:
            from .epistemic import mc_dropout_predict  # local import: keeps epistemic.py's torch.nn.MultiheadAttention-poking optional for callers who never use MC-dropout

            _, epistemic_uncertainty = mc_dropout_predict(
                self.mc_dropout_model, self.mc_dropout_preprocess, images, passes=self.mc_dropout_passes
            )
        return estimate_reliability_state(
            logits, self.combiner, self.reference_normalizer, self.calibrator,
            mahalanobis_scorer=self.mahalanobis_scorer, features=features,
            epistemic_uncertainty=epistemic_uncertainty,
        )

    def decide(
        self, logits: torch.Tensor, policy: Policy, features: torch.Tensor | None = None, images=None
    ) -> list[str]:
        """Compute r(x) then apply `policy` to it - the whole Evidence-to-Decision
        chain through one call, with the policy free to look at any/all of
        r(x)'s fields rather than being handed a pre-collapsed scalar.
        `features`/`images` are optional and only matter if this pipeline was
        built with a `mahalanobis_scorer`/`mc_dropout_model` respectively (see
        `estimate()`)."""
        state = self.estimate(logits, features=features, images=images)
        return policy(state)


def scalar_threshold_policy(tau_hi: float, tau_lo: float, field: str = "calibration_confidence") -> Policy:
    """The old Psi_theta pipeline (Sec4.1), expressed as a Policy: reads exactly
    one field of the state and thresholds it via router.route, unchanged.
    Exists to make explicit that the scalar pipeline is one particular,
    simple choice of policy over r(x), not a separate system this one
    replaces."""

    def policy(state: ReliabilityState) -> list[str]:
        return route(getattr(state, field), tau_hi=tau_hi, tau_lo=tau_lo)

    return policy


def anomaly_augmented_policy(
    tau_hi: float,
    tau_lo: float,
    anomaly_threshold: float,
    field: str = "calibration_confidence",
    anomaly_field: str = "distribution_shift",
) -> Policy:
    """Downgrades an Execute decision to Verify whenever `anomaly_field`
    exceeds `anomaly_threshold`, regardless of how confident `field` itself
    is - PROGRESS_REPORT.md 3.7's finding, as a reusable policy instead of
    notebook-only glue code. If `anomaly_field` is NaN for every input (no
    `reference_normalizer` was given to `estimate_reliability_state` -
    reliability.py's documented behavior), every comparison against
    `anomaly_threshold` is false and this policy is silently identical to
    `scalar_threshold_policy` - a deliberate fail-safe default (no anomaly
    data means no anomaly-based downgrades), not a bug, but worth knowing
    about if decisions look unexpectedly unaffected.
    """

    def policy(state: ReliabilityState) -> list[str]:
        base = route(getattr(state, field), tau_hi=tau_hi, tau_lo=tau_lo)
        anomaly = getattr(state, anomaly_field)
        return [
            VERIFY if (decision == EXECUTE and a > anomaly_threshold) else decision
            for decision, a in zip(base, anomaly.tolist())
        ]

    return policy


def joint_cost_sensitive_policy(
    cost_policy: AffineCostPolicy,
    field: str = "calibration_confidence",
    anomaly_field: str = "distribution_shift",
) -> Policy:
    """Wraps router.route_joint over (state[field], state[anomaly_field]) as a
    Policy - the recommended, more principled replacement for
    anomaly_augmented_policy's fixed hand-picked threshold (kept in this
    module unchanged, since it's working, tested code, not something to
    delete). `cost_policy` comes from router.cost_sensitive_joint_policy();
    see its docstring for the important scope caveat (argmin-optimal for a
    stated cost model, NOT unqualified "Bayes-optimal" - `anomaly_field` is
    not a probability).

    UNLIKE anomaly_augmented_policy, this does NOT fail safe if
    `anomaly_field` is NaN (no `reference_normalizer` was given to
    estimate_reliability_state): under IEEE754, 0*NaN=NaN, so even an
    action whose q-weight is exactly 0 (HITL and, by default, Verify) gets
    a NaN cost once q is NaN, and torch.argmin over an all-NaN row
    degenerates to always picking index 0 (HITL) - checked directly, not
    assumed; every decision would silently become HITL, not a graceful
    fallback to the p-only rule. Raises ValueError instead of allowing that
    silent failure mode; use scalar_threshold_policy or
    anomaly_augmented_policy (whose `a > threshold` comparison is safely
    False for NaN) when no reference_normalizer is available.
    """

    def policy(state: ReliabilityState) -> list[str]:
        p = getattr(state, field)
        q = getattr(state, anomaly_field)
        if torch.isnan(q).any():
            raise ValueError(
                f"'{anomaly_field}' contains NaN (no reference_normalizer was given to "
                "estimate_reliability_state) - route_joint's argmin degenerates to always "
                "selecting HITL under NaN, not a safe fallback. Use scalar_threshold_policy "
                "or anomaly_augmented_policy instead when no reference_normalizer is available."
            )
        return route_joint(p, q, cost_policy)

    return policy
