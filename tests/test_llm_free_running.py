import os
import sys

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from check_windowed_correctness import windowed_correct  # noqa: E402
from collect_llm_free_running import free_running_generate, teacher_forced_logits_same_positions  # noqa: E402

# This file closes a real gap: scripts/collect_llm_free_running.py and
# scripts/check_windowed_correctness.py (DESIGN.md 14.10-14.12's free-running
# generation results) had zero pytest coverage before this - every other
# script with a nontrivial, checkable algorithm (e.g. collect_odin_scores.py,
# see test_odin.py) has at least one direct unit test of its pure logic, not
# just a "run it once and record the printed number" validation. These tests
# follow that same precedent: import the pure functions directly (no model
# download required) and check them against an independently hand-computed
# expected result, not against the functions' own output re-run.


class _FakeCausalLMOutput:
    def __init__(self, logits, past_key_values):
        self.logits = logits
        self.past_key_values = past_key_values


class _HistoryConditionedFakeModel:
    """A fake causal LM whose next-token argmax is a deterministic function of
    the FULL conditioning history it has actually been fed (prompt + every
    previously generated token), and nothing else - no access to any
    "real_continuation" is possible because free_running_generate's signature
    never passes one in. `past_key_values` is repurposed here to carry the
    growing history list itself (opaque to the caller, exactly as a real KV
    cache object is) so this fake can compute "sum of history mod vocab_size"
    as a stand-in for "the model's actual next-token prediction," letting the
    test independently re-derive the expected generated sequence by the same
    recursion in plain Python and compare.
    """

    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.call_input_lengths = []

    def __call__(self, input_ids, past_key_values=None, use_cache=False):
        tokens = input_ids[0].tolist()
        self.call_input_lengths.append(len(tokens))
        history = list(tokens) if past_key_values is None else list(past_key_values) + list(tokens)
        next_token = sum(history) % self.vocab_size
        logits = torch.full((1, len(tokens), self.vocab_size), -100.0)
        logits[0, -1, next_token] = 100.0
        return _FakeCausalLMOutput(logits=logits, past_key_values=history)


def test_free_running_generate_conditions_only_on_prompt_plus_self_generated_tokens():
    # Independently re-derive the expected generated sequence via the exact
    # same "sum-of-history mod vocab" recursion the fake model implements,
    # but in a plain Python loop with no KV-cache machinery at all - then
    # check free_running_generate's actual output matches it exactly. If the
    # KV-cache indexing in free_running_generate ever accidentally reused
    # stale logits, skipped a step, or (structurally impossible here, since
    # the function has no such parameter, but checked anyway via the
    # recursion match) conditioned on something other than prompt+generated,
    # this would diverge.
    torch.manual_seed(0)
    vocab_size = 37
    prompt_ids = torch.tensor([3, 11, 5, 29, 2])
    gen_len = 12

    model = _HistoryConditionedFakeModel(vocab_size)
    generated_ids, logits_per_step = free_running_generate(model, prompt_ids, gen_len)

    # Independent reference simulation, deliberately not calling any of this
    # project's own code.
    history = prompt_ids.tolist()
    expected = []
    for _ in range(gen_len):
        next_token = sum(history) % vocab_size
        expected.append(next_token)
        history.append(next_token)

    assert generated_ids.tolist() == expected
    assert logits_per_step.shape == (gen_len, vocab_size)
    # Step i's returned logits must be the ones whose argmax produced
    # generated_ids[i] - the function's own documented contract.
    assert logits_per_step.argmax(dim=-1).tolist() == expected

    # KV-cache mechanics: the first call sees the whole prompt; every
    # subsequent call feeds exactly one new token (the incremental,
    # cache-using path), never the whole growing sequence again.
    assert model.call_input_lengths[0] == len(prompt_ids)
    assert model.call_input_lengths[1:] == [1] * gen_len


def test_teacher_forced_logits_same_positions_slices_the_correct_corpus_positions():
    # DESIGN.md 14.10's "direct, same-positions teacher-forced baseline"
    # depends entirely on this slice lining up with the exact same
    # real_continuation positions free_running_generate is compared against:
    # position (prompt_len-1) predicts the token AT prompt_len (the first
    # real_continuation token), ..., position (window_size-2) predicts the
    # token at (window_size-1) (the last one) - gen_len positions total, not
    # gen_len+1 or gen_len-1 (the off-by-one this test targets specifically).
    prompt_len, gen_len = 4, 6
    window_size = prompt_len + gen_len
    vocab_size = 9

    class _FakeModel:
        def __call__(self, input_ids):
            # logits[0, p, :] argmax = p itself, so the returned slice's
            # argmax sequence directly reveals which positions were sliced.
            seq_len = input_ids.shape[1]
            logits = torch.full((1, seq_len, vocab_size), -100.0)
            for p in range(seq_len):
                logits[0, p, p % vocab_size] = 100.0
            return _FakeCausalLMOutput(logits=logits, past_key_values=None)

    full_window_ids = torch.arange(window_size)
    result = teacher_forced_logits_same_positions(_FakeModel(), full_window_ids, prompt_len, window_size)

    assert result.shape == (gen_len, vocab_size)
    sliced_positions = result.argmax(dim=-1).tolist()
    expected_positions = [(p % vocab_size) for p in range(prompt_len - 1, window_size - 1)]
    assert sliced_positions == expected_positions
    assert len(expected_positions) == gen_len


def test_windowed_correct_matches_hand_computed_membership_within_window():
    # windowed_correct[i] := generated_token[i] in real_continuation[i-w:i+w+1]
    # (DESIGN.md 14.12) - hand-verified on a tiny sequence rather than trusted
    # from a real run alone.
    generated = torch.tensor([9, 1, 2, 3, 4, 9])
    real_continuation = torch.tensor([0, 1, 9, 3, 8, 8])
    # w=2 -> window [max(0,i-2), min(6,i+3)):
    # i=0: [0,3) -> {0,1,9}     ; 9 in it -> True
    # i=1: [0,4) -> {0,1,9,3}   ; 1 in it -> True
    # i=2: [0,5) -> {0,1,9,3,8} ; 2 not in it -> False
    # i=3: [1,6) -> {1,9,3,8,8} ; 3 in it -> True
    # i=4: [2,6) -> {9,3,8,8}   ; 4 not in it -> False
    # i=5: [3,6) -> {3,8,8}     ; 9 not in it -> False
    expected = torch.tensor([True, True, False, True, False, False])

    result = windowed_correct(generated, real_continuation, w=2)
    assert torch.equal(result, expected)


def test_windowed_correct_reduces_to_exact_match_at_w_zero():
    generated = torch.tensor([1, 2, 3, 4])
    real_continuation = torch.tensor([1, 9, 9, 4])
    result = windowed_correct(generated, real_continuation, w=0)
    assert torch.equal(result, generated == real_continuation)


def test_windowed_correct_is_a_superset_of_strict_correctness():
    # Widening the window can only add matches, never remove one that exact
    # positional match already found - a monotonicity property any correct
    # implementation must have.
    torch.manual_seed(0)
    generated = torch.randint(0, 5, (20,))
    real_continuation = torch.randint(0, 5, (20,))
    strict = generated == real_continuation
    windowed = windowed_correct(generated, real_continuation, w=3)
    assert bool((strict & ~windowed).any()) is False
