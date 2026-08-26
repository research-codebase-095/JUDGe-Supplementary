"""Logit-derived feature extraction, per DESIGN.md sections 5-6.

Every function takes raw logits `z` of shape (..., C) and returns a tensor
of shape (...,) - one scalar per prediction, batched over any leading dims.
No function touches anything but the logit vector itself (DESIGN.md 3.1).
"""

import math

import torch

DEFAULT_FEATURE_NAMES = ("msp", "logit_margin", "normalized_entropy", "energy_score", "logit_l2_norm")

# +1: higher = more confident. -1: lower = more confident. Any code that
# combines these features linearly (e.g. WeightedLinearCombiner) must apply
# this before averaging/summing, or a feature like entropy - where LOWER is
# better - silently contributes in the wrong direction (DESIGN.md 8).
#
# logit_l2_norm is -1, not the originally-assumed +1: it was checked
# directly against real cached logits across all three evaluated backbones
# (DESIGN.md's disagreement-matrix section) rather than left as an
# unverified assumption, and on every one of them, incorrect predictions
# have a HIGHER raw L2 norm than correct ones (e.g. ResNet-50: 18.8 vs
# 15.3) - the backbone reacts more strongly, not more confidently, when
# it's about to be wrong. LogisticRegressionCombiner was never affected by
# this (it learns its own sign from labeled data); WeightedLinearCombiner,
# the only user of this fixed direction, was silently working against
# itself on this feature until the fix - confirmed by a measurable AURC
# improvement on real id_test data across all three backbones once
# corrected.
FEATURE_DIRECTIONS = {
    "msp": 1,
    "logit_margin": 1,
    "normalized_entropy": -1,
    "energy_score": 1,
    "logit_l2_norm": -1,
}
DEFAULT_FEATURE_DIRECTIONS = torch.tensor([FEATURE_DIRECTIONS[name] for name in DEFAULT_FEATURE_NAMES], dtype=torch.float32)


def msp(logits: torch.Tensor) -> torch.Tensor:
    """Max softmax probability: p_(1). Higher = more confident."""
    return torch.softmax(logits, dim=-1).amax(dim=-1)


def logit_margin(logits: torch.Tensor) -> torch.Tensor:
    """Top-1 minus top-2 raw logit gap: z_(1) - z_(2). Higher = more confident."""
    top2 = torch.topk(logits, k=2, dim=-1).values
    return top2[..., 0] - top2[..., 1]


def normalized_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Shannon entropy of the softmax distribution, normalized by log(C) to [0, 1].

    Lower = more confident (probability mass concentrated on fewer classes).
    """
    c = logits.shape[-1]
    p = torch.softmax(logits, dim=-1)
    h = -(p * torch.log(p.clamp_min(torch.finfo(p.dtype).tiny))).sum(dim=-1)
    return h / torch.log(torch.tensor(float(c), dtype=logits.dtype, device=logits.device))


def energy_score(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Confidence-aligned free energy: T * logsumexp(z / T). Higher = more confident.

    This is the negation of the raw free energy E(z;T) = -T*logsumexp(z/T) used in
    OOD literature (DESIGN.md 5.3), flipped here so that, like the other four
    features, a higher value means "more trustworthy."
    """
    return temperature * torch.logsumexp(logits / temperature, dim=-1)


def logit_l2_norm(logits: torch.Tensor, per_class: bool = False) -> torch.Tensor:
    """L2 norm of the logit vector: ||z||_2. Higher = stronger backbone reaction.

    Under similar per-logit variance, ||z||_2 grows with sqrt(C), so raw values
    from a 1,000-way CNN head and a 50k-128k-way LLM vocabulary aren't directly
    comparable (DESIGN.md 7, 14.4). Set `per_class=True` to divide by sqrt(C) and
    make the value comparable across differently-sized output spaces - this only
    rescales by a constant for any single fixed C, so it never changes rankings
    within one backbone/domain, only comparability across them.

    "Higher = stronger backbone reaction" is a factual description of the raw
    quantity, not a confidence direction - see FEATURE_DIRECTIONS below, whose
    sign for this feature was checked and corrected against real cached data
    across all three backbones (DESIGN.md's disagreement-matrix section): a
    stronger reaction turns out to mean *less* trustworthy here, not more.
    """
    norm = torch.linalg.vector_norm(logits, ord=2, dim=-1)
    if per_class:
        c = logits.shape[-1]
        norm = norm / math.sqrt(c)
    return norm


def verify_feature_directions(
    phi: torch.Tensor,
    correct: torch.Tensor,
    feature_names: tuple[str, ...] = DEFAULT_FEATURE_NAMES,
    directions: dict[str, int] = FEATURE_DIRECTIONS,
) -> dict[str, bool]:
    """Empirically check each feature's assumed FEATURE_DIRECTIONS sign against
    labeled data, instead of trusting the assumption.

    `logit_l2_norm`'s direction was assumed `+1` for a long time before being
    checked directly and found backwards (DESIGN.md's disagreement-matrix
    section) - that was caught by noticing one suspiciously low AUROC by
    hand, which only happened because a matrix of AUROCs was being built for
    an unrelated reason. This function makes that check something you can run
    on purpose, for any feature set and any backbone's cached data, instead
    of relying on someone happening to notice again.

    For each feature, orients its raw values by the assumed direction and
    checks `mean(oriented | correct) > mean(oriented | incorrect)` - the same
    comparison that surfaced the `logit_l2_norm` bug. Returns `{feature_name:
    True/False}`; `False` means the assumed direction contradicts this data
    and should be investigated (as `logit_l2_norm`'s was), not silently kept.
    """
    directions_tensor = torch.tensor(
        [directions[name] for name in feature_names], dtype=phi.dtype, device=phi.device
    )
    oriented = phi * directions_tensor
    correct = correct.to(dtype=torch.bool, device=phi.device)
    mean_correct = oriented[correct].mean(dim=0)
    mean_incorrect = oriented[~correct].mean(dim=0)
    return {
        name: bool((mean_correct[i] > mean_incorrect[i]).item()) for i, name in enumerate(feature_names)
    }


def aggregate_sequence_features(phi_sequence: torch.Tensor, method: str = "mean") -> torch.Tensor:
    """Aggregate an ordered sequence of per-step phi(z) vectors into one
    sequence-level feature vector - DESIGN.md 14.3's sequence-aggregation
    question, now implemented and empirically checked (DESIGN.md 14.6) rather
    than left as a design intent.

    `phi_sequence` has shape (seq_len, num_features); returns (num_features,).
    Domain-agnostic by construction - nothing here assumes LLM tokens
    specifically, any ordered sequence of phi(z) vectors works (e.g. video
    frames, once/if that domain is validated).

    `method="mean"` (the default) is the empirically better choice for four
    of the five default features on real GPT-2/WikiText-2 data (DESIGN.md
    14.6: msp/margin/entropy/energy_score all show higher sequence-level
    correctness AUROC under mean than under min) - notably including
    msp/margin, where DESIGN.md 14.3 originally suggested min ("weakest
    link") as the natural choice before this was actually checked.
    `logit_l2_norm` is the one consistent exception (min outperforms mean
    for it specifically); this function does not special-case it, since
    resolving that asymmetry with a single clean rule is exactly the kind
    of thing DESIGN.md 14.6 reports as still unresolved rather than papered
    over - a caller who wants a different method for a specific feature can
    call this twice and select columns.
    """
    if method == "mean":
        return phi_sequence.mean(dim=0)
    elif method == "min":
        return phi_sequence.min(dim=0).values
    else:
        raise ValueError(f"unknown aggregation method {method!r}; choose 'mean' or 'min'")


def sequence_correctness(token_correct: torch.Tensor, threshold: float = 1.0) -> torch.Tensor:
    """Sequence-level correctness label: fraction of correct tokens in the
    window >= threshold. `threshold=1.0` (the default) is exactly
    `token_correct.all(dim=0)` - DESIGN.md 14.6's original strict target,
    preserved as the default so this is a strict generalization, not a
    silent behavior change.

    `token_correct` has shape (seq_len,) or (seq_len, ...) for a batch of
    windows stacked along a trailing dim; returns a boolean tensor with the
    seq_len axis reduced away.

    Exists because the strict `threshold=1.0` target turned out to have a
    real, verified cost beyond just looking weak (DESIGN.md 14.6's ceiling
    analysis already showed the SIGNAL wasn't the problem): its base rate is
    so low that a calibration split sized the way this project's existing
    protocol (splits.py, DESIGN.md 10.5) produces doesn't contain enough
    positive examples to fit a reliable threshold_for_target_risk (router.py)
    - checked directly, not assumed: at threshold=1.0, k=5, the calibration
    split had only 74 positive windows out of 25,780, and the resulting
    threshold badly overshot its target risk on held-out data. `threshold=0.8`
    (DESIGN.md 14.8) is the empirically-grounded fix - the minimal relaxation
    that gives calibration enough positive examples to work, chosen for that
    specific, checkable reason rather than for producing better-looking
    numbers.
    """
    if not 0.0 < threshold <= 1.0:
        raise ValueError(f"threshold must be in (0, 1], got {threshold}")
    fraction_correct = token_correct.to(dtype=torch.float32).mean(dim=0)
    return fraction_correct >= threshold


def featurize(logits: torch.Tensor, temperature: float = 1.0, normalize_l2: bool = False) -> torch.Tensor:
    """Stack the recommended minimal feature set (DESIGN.md 6) into phi(z).

    Returns a tensor of shape (..., 5), columns ordered per DEFAULT_FEATURE_NAMES.
    `normalize_l2=True` applies the sqrt(C) correction from `logit_l2_norm` -
    turn it on when comparing/combining phi(z) across backbones with different
    output widths (DESIGN.md 14.4); defaults to False to keep single-domain
    results (e.g. all-ResNet-50) unchanged.
    """
    return torch.stack(
        [
            msp(logits),
            logit_margin(logits),
            normalized_entropy(logits),
            energy_score(logits, temperature=temperature),
            logit_l2_norm(logits, per_class=normalize_l2),
        ],
        dim=-1,
    )
