import math
import os
import sys

import numpy as np
import torch
from scipy.stats import spearmanr

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from check_cross_model_correctness import (  # noqa: E402
    bootstrap_spearman_ci,
    cross_model_mean_logprob,
    free_running_generate_with_confidence,
    repetition_ratio,
)

# check_cross_model_correctness.py implements DESIGN.md 14.14's named next
# step for a working divergence-robust correctness notion: score a
# generating model's free-running output with a SECOND, independent model
# rather than anything derived from the generating model's own logits. These
# tests follow tests/test_llm_free_running.py's precedent: import the pure
# functions directly (no real model download required) and check them
# against an independently hand-computed expected result.


class _FakeCausalLMOutput:
    def __init__(self, logits, past_key_values):
        self.logits = logits
        self.past_key_values = past_key_values


def test_repetition_ratio_all_distinct_is_zero():
    assert repetition_ratio(torch.tensor([1, 2, 3, 4, 5])) == 0.0


def test_repetition_ratio_all_same_is_one():
    assert repetition_ratio(torch.tensor([7, 7, 7, 7])) == 1.0


def test_repetition_ratio_hand_computed_mixed_case():
    # [3,3,1,1,1,9]: immediate-repeat pairs are (3,3)True (3,1)False (1,1)True (1,1)True (1,9)False -> 3/5
    generated = torch.tensor([3, 3, 1, 1, 1, 9])
    assert math.isclose(repetition_ratio(generated), 3.0 / 5.0, rel_tol=1e-6)


def test_repetition_ratio_single_token_is_zero():
    assert repetition_ratio(torch.tensor([5])) == 0.0


def test_bootstrap_spearman_ci_point_matches_scipy_directly():
    rng = np.random.default_rng(0)
    x = rng.normal(size=40)
    y = 2.0 * x + rng.normal(scale=0.1, size=40)
    point, lo, hi = bootstrap_spearman_ci(x, y, n_bootstrap=500)
    assert point == spearmanr(x, y).correlation
    assert lo <= point <= hi


def test_bootstrap_spearman_ci_perfect_monotonic_relationship_is_one():
    x = np.arange(30, dtype=float)
    y = x**3  # monotonic but nonlinear - Spearman is exactly 1.0 regardless
    point, lo, hi = bootstrap_spearman_ci(x, y, n_bootstrap=200)
    assert point == 1.0
    assert math.isclose(lo, 1.0, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(hi, 1.0, rel_tol=1e-9, abs_tol=1e-9)


class _FakeWordTokenizer:
    """Deterministic word-level fake tokenizer: splits on whitespace, maps
    each word to a fixed id via a simple hash. Word-level (not sub-word)
    splitting guarantees tokenizer(prompt_text)["input_ids"] is always an
    exact PREFIX of tokenizer(prompt_text + continuation_text)["input_ids"]
    - the real function's boundary-alignment assumption, made exact here so
    the test's expected boundary is unambiguous rather than approximate the
    way a real BPE tokenizer's seam token occasionally is.
    """

    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size

    def __call__(self, text):
        words = text.split()
        ids = [sum(ord(c) for c in w) % self.vocab_size for w in words]
        return {"input_ids": ids}


class _FakeConfig:
    def __init__(self, max_position_embeddings):
        self.max_position_embeddings = max_position_embeddings


class _UniformLogitsFakeModel:
    """Always returns all-zero logits -> log_softmax is exactly uniform
    (-log(vocab_size) for every token at every position), so the expected
    mean log-probability of ANY continuation is exactly -log(vocab_size),
    computable by hand with no dependence on which tokens were generated.
    """

    def __init__(self, vocab_size: int, max_position_embeddings: int = 10_000):
        self.vocab_size = vocab_size
        self.config = _FakeConfig(max_position_embeddings)

    def __call__(self, input_ids):
        seq_len = input_ids.shape[1]
        logits = torch.zeros(1, seq_len, self.vocab_size)
        return _FakeCausalLMOutput(logits=logits, past_key_values=None)


def test_cross_model_mean_logprob_matches_hand_computed_uniform_case():
    vocab_size = 11
    tokenizer = _FakeWordTokenizer(vocab_size)
    model = _UniformLogitsFakeModel(vocab_size)

    result = cross_model_mean_logprob(model, tokenizer, "alpha beta gamma", " delta epsilon")

    assert math.isclose(result, -math.log(vocab_size), rel_tol=1e-6)


def test_cross_model_mean_logprob_returns_nan_when_continuation_adds_no_new_tokens():
    vocab_size = 11
    tokenizer = _FakeWordTokenizer(vocab_size)
    model = _UniformLogitsFakeModel(vocab_size)

    # Empty continuation -> full_ids == prompt_ids exactly -> boundary ==
    # len(full_ids), the degenerate case the function must detect and skip
    # rather than compute a spurious mean over zero positions.
    result = cross_model_mean_logprob(model, tokenizer, "alpha beta gamma", "")

    assert math.isnan(result)


def test_cross_model_mean_logprob_truncates_long_prompt_from_the_left_not_the_continuation():
    # Regression test: a full 400-window real run hit an IndexError from
    # GPT-2's 1024-position embedding table when a WikiText-2 window,
    # re-tokenized under a different BPE vocabulary, expanded past 1024
    # tokens. cross_model_mean_logprob must truncate the PROMPT (from the
    # left) rather than crash or silently truncate into the continuation -
    # checked here with a scorer whose context window is tiny (5 tokens)
    # so the truncation path triggers deterministically.
    vocab_size = 11
    tokenizer = _FakeWordTokenizer(vocab_size)
    model = _UniformLogitsFakeModel(vocab_size, max_position_embeddings=5)

    # 8 prompt words + 3 continuation words = 11 tokens total, well past
    # the fake scorer's 5-token limit - the continuation (3 tokens) still
    # fits after truncating the prompt down to 2 tokens.
    prompt_text = "one two three four five six seven eight"
    continuation_text = " nine ten eleven"

    result = cross_model_mean_logprob(model, tokenizer, prompt_text, continuation_text)

    # Uniform logits -> exact expected value regardless of how much of the
    # prompt was kept, as long as the function didn't crash or return nan.
    assert math.isclose(result, -math.log(vocab_size), rel_tol=1e-6)


def test_cross_model_mean_logprob_skips_when_continuation_alone_exceeds_context_window():
    vocab_size = 11
    tokenizer = _FakeWordTokenizer(vocab_size)
    # Context window smaller than the continuation's own token count (3) -
    # no amount of prompt truncation can make this fit without cutting
    # into the continuation itself, so the function must skip (nan), not
    # silently score a truncated continuation.
    model = _UniformLogitsFakeModel(vocab_size, max_position_embeddings=2)

    result = cross_model_mean_logprob(model, tokenizer, "one two three", " four five six")

    assert math.isnan(result)


class _GrowingConfidenceFakeModel:
    """A fake causal LM ignoring its actual input entirely and returning,
    on its t-th call (0-indexed), 2-way logits [log(t+2), 0] - argmax is
    always token 0, and softmax's top-1 probability (msp) is exactly
    (t+2)/(t+3), an exact closed form independently re-derivable in the
    test without calling featurize's own implementation.
    """

    def __init__(self):
        self.call_count = 0

    def __call__(self, input_ids, past_key_values=None, use_cache=True):
        t = self.call_count
        self.call_count += 1
        seq_len = input_ids.shape[1]
        logits = torch.tensor([[math.log(t + 2), 0.0]]).repeat(seq_len, 1).unsqueeze(0)
        return _FakeCausalLMOutput(logits=logits, past_key_values=t)


def test_free_running_generate_with_confidence_matches_hand_computed_msp_sequence():
    torch.manual_seed(0)
    prompt_ids = torch.tensor([0, 1, 0])
    gen_len = 6

    model = _GrowingConfidenceFakeModel()
    generated_ids, mean_msp = free_running_generate_with_confidence(model, prompt_ids, gen_len)

    # argmax is always token 0 for every call (log(t+2) > 0 for all t >= 0)
    assert generated_ids.tolist() == [0] * gen_len

    # Step i's msp comes from call i (t=i): call 0 is the initial prompt
    # forward pass, calls 1..gen_len-1 are the incremental generation steps
    # - independently re-derived here via the model's own documented
    # closed form, not by calling featurize.
    expected_msp = [(t + 2) / (t + 3) for t in range(gen_len)]
    expected_mean = sum(expected_msp) / gen_len

    assert math.isclose(mean_msp, expected_mean, rel_tol=1e-6)
    # gen_len calls happen in the loop, plus 1 initial call = gen_len + 1 total
    assert model.call_count == gen_len + 1
