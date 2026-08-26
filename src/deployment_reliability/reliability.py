"""Multi-dimensional reliability state r(x), as an alternative to collapsing
everything into the single scalar S that combiner.py/router.py operate on.

Motivation, grounded in this project's own results rather than the framing
alone: DESIGN.md sections 15.3-15.6 found that on real cached ResNet-50/
ViT-B16/ConvNeXt-Tiny logits, correctness (what the combiner is trained to
predict), the ReferenceNormalizer anomaly score (normalization.py), and
OOD-ness (ImageNet-O AUROC) diverge from one another - each catches failure
modes the others miss (ImageNet-A defeats correctness-style signals while
leaving anomaly-style signals mostly intact; ImageNet-O is the reverse). A
single scalar S cannot represent three different latent variables well at
once; this module keeps them separate rather than forcing a premature
combination.

This does not replace the scalar S / threshold-router pipeline
(combiner.py, router.py) - it's an additional, coarser-grained diagnostic
view for callers who want to see *why* something looks unreliable, not just
whether. Nothing here has its own routing rule; `router.py`'s cost-sensitive
thresholds still operate on a scalar, by design (DESIGN.md 10.3.1's proof is
for a scalar score) - reducing r(x) back to a scalar for routing, if wanted,
is left to the caller.
"""

from __future__ import annotations

from typing import NamedTuple

import torch

from .features import DEFAULT_FEATURE_NAMES, featurize, normalized_entropy
from .normalization import ReferenceNormalizer


class ReliabilityState(NamedTuple):
    """r(x): seven independent reliability signals, each shape (...,).

    Deliberately excludes two dimensions from the "reliability vector"
    framing this module is based on - prediction stability and explanation
    consistency - since both need infrastructure this repo doesn't have and
    hasn't validated: stability needs a temporal sequence of predictions for
    the *same* underlying entity (see temporal.py, itself unvalidated on
    real video/robot data), and explanation consistency needs an explanation
    method (e.g. saliency, LIME) this project has never implemented or
    tested. Adding untested fields here would misrepresent them as being on
    the same empirical footing as the fields below, which are computed from
    real cached logits and checked in tests/test_reliability.py.

    `feature_space_trust` (STUDY_PLAN.md 3.6 item 2) and
    `epistemic_uncertainty` (item 5) are the two fields with a default
    (`None`), added strictly after the original five: computing either needs
    something `estimate_reliability_state` doesn't always have -
    `feature_space_trust` needs penultimate-layer FEATURES, not just logits;
    `epistemic_uncertainty` needs T extra stochastic forward passes over the
    ORIGINAL images (`epistemic.mc_dropout_predict`, ViT-B/16-only), not
    derivable from logits at all. The defaults keep every existing keyword-
    construction call site across the codebase - `pipeline.py`,
    `tests/test_reliability.py`, `tests/test_pipeline.py` - working
    unchanged; both are the only fields not required at construction.
    """

    correctness: torch.Tensor  # P(argmax(z) == true label) from a fitted combiner (combiner.py)
    uncertainty: torch.Tensor  # normalized entropy - spread of belief across classes (features.py)
    distribution_shift: torch.Tensor  # |z-score| of energy_score vs. a reference set (normalization.py); NaN if no reference
    familiarity: torch.Tensor  # |z-score| of logit L2 norm vs. a reference set (normalization.py); NaN if no reference
    calibration_confidence: torch.Tensor  # correctness after post-hoc calibration; equals correctness if uncalibrated
    feature_space_trust: torch.Tensor | None = None  # mahalanobis.MahalanobisScorer.score(); NaN-filled if no scorer/features given via estimate_reliability_state, None if constructed directly without this field
    epistemic_uncertainty: torch.Tensor | None = None  # epistemic.mc_dropout_predict's predictive_variance, ViT-B/16-only, T-pass opt-in; NaN-filled if not given via estimate_reliability_state, None if constructed directly without this field
    # NOT the same thing as combiner.py's LogisticRegressionCombiner.epistemic_std
    # (DESIGN.md 23) - that measures uncertainty of the COMBINER's OWN fitted
    # (weight, bias), and DESIGN.md 23.5 explicitly decided AGAINST exposing it
    # as a ReliabilityState field (it was found to run backwards - LOWER, not
    # higher, on ImageNet-A/O than id_test). This field is a structurally
    # different signal: MC-dropout predictive variance of the BACKBONE's own
    # output distribution (STUDY_PLAN.md 3.6 item 5), which DESIGN.md 23's
    # decision says nothing about one way or the other - the two are not in
    # tension, but sharing the word "epistemic" invites confusing them.


def estimate_reliability_state(
    logits: torch.Tensor,
    combiner,
    reference_normalizer: ReferenceNormalizer | None = None,
    calibrator=None,
    mahalanobis_scorer=None,
    features: torch.Tensor | None = None,
    epistemic_uncertainty: torch.Tensor | None = None,
) -> ReliabilityState:
    """Compute r(x) from raw logits plus an already-fitted combiner (DESIGN.md 8).

    `combiner` must already be fitted (combiner.py's WeightedLinearCombiner
    or LogisticRegressionCombiner, or anything with a compatible `.score()`).
    `reference_normalizer`/`calibrator` are optional: without a fitted
    ReferenceNormalizer, distribution_shift/familiarity come back as NaN -
    not zero, since zero would silently assert "no shift detected," which is
    a false claim, not an honest "unknown." Without a calibrator,
    calibration_confidence equals the combiner's raw score.

    `mahalanobis_scorer`/`features` (STUDY_PLAN.md 3.6 item 2) are also
    optional and both must be given together for `feature_space_trust` to be
    computed: `mahalanobis_scorer` is an already-fitted `mahalanobis.MahalanobisScorer`,
    `features` are that same backbone's penultimate-layer features for these
    exact inputs (backbone.py's `logits_and_features_for_images`, NOT
    derivable from `logits` alone - unlike every other field here). Without
    both, `feature_space_trust` comes back NaN-filled, the same "honest
    unknown, not silently zero" convention `distribution_shift`/`familiarity`
    already use.

    `epistemic_uncertainty` (STUDY_PLAN.md 3.6 item 5) is likewise
    optional: this function has no way to compute it itself (it would need T
    extra stochastic forward passes over the ORIGINAL images - not logits -
    with a ViT-B/16 loaded via `backbone.load_frozen_vit_b16_with_dropout`;
    see `epistemic.mc_dropout_predict`), so a caller who wants this field
    populated must compute `predictive_variance` there first and pass it
    through here. Without it, `epistemic_uncertainty` comes back NaN-filled,
    the same honest-unknown convention as the other optional fields.
    """
    phi = featurize(logits)
    correctness = combiner.score(phi)
    uncertainty = normalized_entropy(logits)

    if reference_normalizer is not None:
        z = reference_normalizer.transform(phi).abs()
        energy_idx = DEFAULT_FEATURE_NAMES.index("energy_score")
        norm_idx = DEFAULT_FEATURE_NAMES.index("logit_l2_norm")
        distribution_shift = z[..., energy_idx]
        familiarity = z[..., norm_idx]
    else:
        distribution_shift = torch.full_like(correctness, float("nan"))
        familiarity = torch.full_like(correctness, float("nan"))

    calibration_confidence = calibrator.transform(correctness) if calibrator is not None else correctness

    if mahalanobis_scorer is not None and features is not None:
        feature_space_trust = mahalanobis_scorer.score(features)
    else:
        feature_space_trust = torch.full_like(correctness, float("nan"))

    if epistemic_uncertainty is None:
        epistemic_uncertainty = torch.full_like(correctness, float("nan"))

    return ReliabilityState(
        correctness=correctness,
        uncertainty=uncertainty,
        distribution_shift=distribution_shift,
        familiarity=familiarity,
        calibration_confidence=calibration_confidence,
        feature_space_trust=feature_space_trust,
        epistemic_uncertainty=epistemic_uncertainty,
    )
