import pytest
import torch

from deployment_reliability.providers import (
    DiffusionNoiseProvider,
    GNNNodeConfidenceProvider,
    LLMTokenProvider,
    RLQValueProvider,
    TimeSeriesResidualProvider,
    VisionLogitProvider,
)

STUB_PROVIDERS = [
    RLQValueProvider,
    DiffusionNoiseProvider,
    GNNNodeConfidenceProvider,
    TimeSeriesResidualProvider,
]


def test_vision_logit_provider_is_an_identity_adapter():
    logits = torch.randn(4, 1000)
    provider = VisionLogitProvider()
    assert torch.equal(provider.get_evidence(logits), logits)


def test_llm_token_provider_is_an_identity_adapter():
    # Implemented (DESIGN.md 14.5, notebooks/14): unlike the STUB_PROVIDERS
    # below, this one is backed by a real, tested pipeline
    # (llm_backbone.py + real GPT-2/WikiText-2 data) - the class itself is
    # a thin identity adapter, matching VisionLogitProvider's pattern,
    # since llm_backbone.logits_for_token_chunk already returns exactly the
    # shape this package expects.
    token_logits = torch.randn(50, 50257)
    provider = LLMTokenProvider()
    assert torch.equal(provider.get_evidence(token_logits), token_logits)


@pytest.mark.parametrize("provider_cls", STUB_PROVIDERS)
def test_unimplemented_providers_raise_not_implemented_rather_than_fabricate_output(provider_cls):
    provider = provider_cls()
    with pytest.raises(NotImplementedError):
        provider.get_evidence(object())
