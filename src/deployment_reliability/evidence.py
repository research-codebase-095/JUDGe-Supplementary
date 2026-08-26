"""Evidence Geometry: named operators over a per-class evidence vector.

This is deliberately a thin renaming/wrapping layer over features.py, not a
new signal source - all five of concentration/separability/ambiguity/
plausibility/magnitude are exactly the features.py functions of the same
underlying quantity, under a name chosen to not presuppose softmax/classifier
semantics (conflict is the one genuinely new sixth operator, see below). The point of the
relabeling is genericity: "concentration of evidence on the top outcome" is
meaningful for classifier logits, LLM token logits, or RL per-action
Q-values alike, whereas "max softmax probability" only means something once
you've already committed to a softmax over classes.

Do not read this module as claiming new empirical power over features.py -
`concentration`/`separability`/`ambiguity`/`plausibility`/`magnitude` each
produce a value identical to the corresponding features.py call on the same
input. `conflict` is the one genuinely new operator (see its docstring).
"""

from __future__ import annotations

import torch

from .features import energy_score, logit_l2_norm, logit_margin, msp, normalized_entropy


class Evidence:
    """Wraps one evidence vector - any real-valued, finite score per discrete
    outcome (classifier logits, LLM token logits, per-action Q-values, ...) -
    and exposes it through named geometric operators instead of raw features.

    Only `concentration`, `separability`, `ambiguity`, `plausibility`,
    `magnitude`, and `conflict` are implemented: all six are computable from
    a single evidence vector with no extra state. `consistency` and
    `stability` are NOT implemented here, since both need information a
    single vector doesn't contain - an ensemble of vectors for the same
    input, or a sequence of vectors over time (see temporal.py for the
    latter) - and are left as explicit NotImplementedError stubs rather than
    silently returning a plausible-looking number for a signal this repo has
    never computed or validated.
    """

    def __init__(self, evidence: torch.Tensor) -> None:
        self.evidence = evidence

    def concentration(self) -> torch.Tensor:
        """How much belief mass sits on the single top outcome. == features.msp."""
        return msp(self.evidence)

    def separability(self) -> torch.Tensor:
        """How far the top outcome is from the runner-up. == features.logit_margin."""
        return logit_margin(self.evidence)

    def ambiguity(self) -> torch.Tensor:
        """How spread belief is across all outcomes. == features.normalized_entropy."""
        return normalized_entropy(self.evidence)

    def plausibility(self, temperature: float = 1.0) -> torch.Tensor:
        """How in-distribution the evidence looks overall, independent of which
        outcome wins. == features.energy_score."""
        return energy_score(self.evidence, temperature=temperature)

    def magnitude(self, per_class: bool = False) -> torch.Tensor:
        """Raw strength of the evidence vector. == features.logit_l2_norm."""
        return logit_l2_norm(self.evidence, per_class=per_class)

    def conflict(self) -> torch.Tensor:
        """How much the runner-up alone, rather than the field of remaining
        outcomes collectively, threatens the top outcome: p_(2) / (1 - p_(1)),
        the runner-up's share of all NON-top probability mass. Bounded in
        (0, 1]: near 1 means almost every unit of "not the top pick" belief
        sits on a single specific rival (a concentrated, two-horse-race
        situation); near 0 means the non-top mass is spread thinly across
        many outcomes (a diffuse, no-clear-second-place situation) even if
        the numeric gap to the runner-up is identical.

        An earlier version of this operator defined conflict as
        p_(2)/(p_(1)+p_(2)) - restricted to the top two classes only. That
        turns out to be exactly sigmoid(-(z_(1) - z_(2)))—a monotonic
        function of `separability`/features.logit_margin alone - so it adds
        a bounded *scale* but zero new *ranking* information; two evidence
        vectors with the same top-1/top-2 gap always got the same old-style
        conflict regardless of how the rest of the distribution looked. This
        version fixes that by normalizing against the *entire* non-top mass
        (1 - p_(1)) instead of just the top two, so it depends on the full
        vector, not only its top two entries - it can rank two vectors with
        an identical margin differently (see
        tests/test_evidence.py's concentrated-vs-diffuse-rival test), which
        is what actually justifies calling this a distinct operator from
        `separability` rather than a rescaled copy of it.
        """
        p = torch.softmax(self.evidence, dim=-1)
        p2 = torch.topk(p, k=2, dim=-1).values[..., 1]
        # Non-top mass as a direct sum of the (small) non-top probabilities,
        # not as "1 - p_(1)" - when p_(1) is very close to 1 (a highly
        # confident vector), 1 - p_(1) is a subtraction of two nearly-equal
        # numbers and loses almost all precision in float32, which can blow
        # the ratio up to nonsense far outside (0, 1]. Masking out the top
        # index and summing what's left avoids that cancellation entirely.
        top1_idx = p.argmax(dim=-1, keepdim=True)
        non_top_mass = p.scatter(-1, top1_idx, 0.0).sum(dim=-1)
        return p2 / non_top_mass.clamp_min(torch.finfo(p.dtype).tiny)

    def consistency(self) -> torch.Tensor:
        raise NotImplementedError(
            "consistency requires multiple evidence vectors for the same input "
            "(e.g. an ensemble, MC-dropout samples, or repeated stochastic "
            "forward passes) - a single Evidence instance only ever holds one "
            "vector. Not implemented because this repo has never run such an "
            "ensemble and has no validated way to test this signal."
        )

    def stability(self) -> torch.Tensor:
        raise NotImplementedError(
            "stability is a temporal property (does the evidence stay reliable "
            "across consecutive frames/steps for the same entity) and needs a "
            "sequence, not a single vector - use "
            "temporal.TemporalReliabilityTracker over a stream of scores "
            "instead. Not implemented directly on Evidence because this repo "
            "has no video/robot/streaming dataset to validate it against."
        )
