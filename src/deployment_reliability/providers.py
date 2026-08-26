"""Domain-agnostic Evidence Provider abstraction.

Every module in this package up to here (features.py, combiner.py, ...)
assumes its input is already a "logit-like" real-valued vector over a
discrete outcome space. This module makes that assumption explicit as an
interface (`EvidenceProvider`), so a caller can plug in a different domain's
native model output and get the same vector shape back, instead of writing
per-domain glue code ad hoc.

`VisionLogitProvider` and `LLMTokenProvider` are implemented and backed by
real, tested data (notebooks 07-10 and notebooks 14 respectively,
tests/test_providers.py) - both are thin adapters over what
backbone.py/llm_backbone.py already produce. The remaining four providers
below are honest interface stubs: this repo has never run an RL agent,
diffusion model, GNN, or time-series forecaster, so claiming a tested
implementation for any of them would fabricate validation that doesn't
exist. Each stub's docstring states exactly what real signal it *would*
wrap and what's missing to implement it for real.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch


class EvidenceProvider(ABC):
    """Maps one domain's native model output to a logit-like evidence vector
    of shape (..., C) that features.py/evidence.py can consume unchanged."""

    @abstractmethod
    def get_evidence(self, raw_output: Any) -> torch.Tensor:
        """Return a real-valued, finite tensor of shape (..., C)."""
        raise NotImplementedError


class VisionLogitProvider(EvidenceProvider):
    """Classifier logits (ResNet-50/ViT-B16/ConvNeXt-Tiny, backbone.py) are
    already exactly the shape this whole package expects, so this is an
    identity adapter. Validated: notebooks 07-10 use this data path directly."""

    def get_evidence(self, raw_output: torch.Tensor) -> torch.Tensor:
        return raw_output


class LLMTokenProvider(EvidenceProvider):
    """Wraps a single decoding step's pre-softmax token logits (shape
    (..., |V|), |V| = vocabulary size, e.g. 50,257 for GPT-2) - an identity
    adapter, exactly like VisionLogitProvider, since llm_backbone.py's
    logits_for_token_chunk already produces exactly this shape. Validated on
    real data: DESIGN.md 14.5, notebooks/14 - GPT-2 (124M) run on real
    WikiText-2 text, per-token correctness (does argmax match the actual
    next token) as the token-level analogue of DESIGN.md 8b's
    y = 1[argmax(z) == true_label].

    The unit of "prediction" here is deliberately the individual token, not
    the full generated sequence - DESIGN.md 14.3/section 7's sequence-level
    aggregation question (min/mean pooling across a generation) is a
    separate, still-open design question this first validation pass
    sidesteps rather than resolves, the same way ResNet-50 was validated
    before ViT-B/16 before ConvNeXt-Tiny rather than all three at once.
    """

    def get_evidence(self, raw_output: torch.Tensor) -> torch.Tensor:
        return raw_output


class RLQValueProvider(EvidenceProvider):
    """Would treat per-action Q-values as the evidence vector (one scalar per
    discrete action, structurally identical to one scalar per class) -
    concentration/separability/ambiguity would then read as "how dominant is
    the greedy action" instead of "how confident is the top class."
    Unimplemented: this repo has no RL environment or trained policy to
    source Q-values from, and the mapping has not been checked against any
    real RL agent's actual Q-value distributions, which (unlike classifier
    logits) are not trained under any softmax-cross-entropy objective - so
    features tuned for logits may behave differently there. An open
    question, not a settled one."""

    def get_evidence(self, raw_output: Any) -> torch.Tensor:
        raise NotImplementedError("no RL environment/policy in this repo to source real Q-values from")


class DiffusionNoiseProvider(EvidenceProvider):
    """Would derive an evidence vector from a denoising step's predicted-noise
    statistics (e.g. per-channel or per-region residual magnitude) -
    conceptually closest to `magnitude`/`plausibility` (evidence.py), since
    diffusion models have no natural discrete-class softmax to source
    `concentration`/`ambiguity`/`conflict` from. Unimplemented and
    unvalidated: this repo has no diffusion model integration, and the
    mapping from continuous noise-prediction statistics to this package's
    discrete-outcome-shaped operators is not even fully designed, let alone
    tested."""

    def get_evidence(self, raw_output: Any) -> torch.Tensor:
        raise NotImplementedError(
            "no diffusion model integration in this repo, and the continuous-noise-to-"
            "discrete-evidence mapping is undesigned, not just untested"
        )


class GNNNodeConfidenceProvider(EvidenceProvider):
    """Would treat a node classifier's per-node, per-class logits as the
    evidence vector - structurally identical to VisionLogitProvider applied
    per-node, so the mapping itself is not the open question here.
    Unimplemented because this repo has no GNN model or graph dataset to
    source real per-node logits from and confirm nothing graph-structural
    (e.g. neighborhood homophily) changes what the features actually
    measure."""

    def get_evidence(self, raw_output: Any) -> torch.Tensor:
        raise NotImplementedError(
            "no GNN model/graph dataset in this repo to source real per-node logits from"
        )


class TimeSeriesResidualProvider(EvidenceProvider):
    """Would derive an evidence vector from forecast residual statistics
    (e.g. per-horizon-step predicted-vs-realized error) rather than a
    per-class score - the least natural fit of the six, since there is no
    discrete outcome space at all. Unimplemented and, unlike the others
    above, not just missing data: the operator mapping (what "concentration"
    or "conflict" would even mean for a scalar forecast residual) is not
    resolved, so implementing this would mean inventing a definition with no
    grounding, not applying an existing one to new data."""

    def get_evidence(self, raw_output: Any) -> torch.Tensor:
        raise NotImplementedError(
            "time-series forecast residuals have no natural discrete-outcome evidence-vector "
            "mapping defined yet - an unresolved design question, not just missing data/model access"
        )
