import torch

from deployment_reliability.llm_backbone import load_frozen_causal_lm, load_frozen_gpt2, logits_for_token_chunk


def test_load_frozen_gpt2_is_frozen_and_in_eval_mode():
    model, tokenizer = load_frozen_gpt2()
    assert model.training is False
    assert all(not p.requires_grad for p in model.parameters())
    assert model.config.vocab_size == 50257
    assert tokenizer is not None


def test_logits_for_token_chunk_shape_and_no_grad():
    model, tokenizer = load_frozen_gpt2()
    token_ids = torch.tensor(tokenizer("The quick brown fox jumps over the lazy dog.")["input_ids"])
    logits = logits_for_token_chunk(model, token_ids)
    assert logits.shape == (len(token_ids), model.config.vocab_size)
    assert not logits.requires_grad
    assert torch.isfinite(logits).all()


def test_logits_for_token_chunk_predicts_a_plausible_next_token():
    # Not a formal correctness test (that's DESIGN.md 14.5/notebooks 14's job
    # on real data at scale) - just a sanity check that a well-known, near-
    # deterministic continuation gets its top prediction right, catching a
    # badly broken model/tokenizer pairing before trusting anything
    # downstream. Verified directly against the real model's actual output
    # before being locked in as a test, not assumed.
    model, tokenizer = load_frozen_gpt2()
    text = "Once upon a"
    token_ids = torch.tensor(tokenizer(text)["input_ids"])
    logits = logits_for_token_chunk(model, token_ids)
    next_token_id = logits[-1].argmax().item()
    predicted = tokenizer.decode([next_token_id])
    assert predicted.strip().lower() == "time", f"expected 'time', got {predicted!r}"


def test_load_frozen_causal_lm_pythia160m_is_frozen_float32_and_in_eval_mode():
    # STUDY_PLAN.md 3.6 item 1b: the architecture-agnostic loader, tested
    # against a real second model (EleutherAI/pythia-160m, GPT-NeoX family,
    # trained on the Pile - not GPT-2/WebText). Explicitly checks dtype ==
    # float32: a real bug was found and fixed here (see llm_backbone.py's
    # docstring) - pythia-160m's published checkpoint is float16, which
    # silently overflowed features.logit_l2_norm to inf on this model's
    # much-larger-magnitude raw logits before load_frozen_causal_lm started
    # forcing torch_dtype=torch.float32 explicitly.
    model, tokenizer = load_frozen_causal_lm("EleutherAI/pythia-160m")
    assert model.training is False
    assert all(not p.requires_grad for p in model.parameters())
    assert next(model.parameters()).dtype == torch.float32
    assert tokenizer is not None


def test_load_frozen_causal_lm_pythia160m_logits_are_finite_and_well_scaled_in_float32():
    model, tokenizer = load_frozen_causal_lm("EleutherAI/pythia-160m")
    token_ids = torch.tensor(tokenizer("The quick brown fox jumps over the lazy dog.")["input_ids"])
    logits = logits_for_token_chunk(model, token_ids)
    assert logits.dtype == torch.float32
    assert torch.isfinite(logits).all()
    # Regression check for the float16-overflow bug this function fixes:
    # squaring and summing these real logits across pythia-160m's ~50k
    # vocabulary must not overflow, which it silently did at float16.
    l2_norm = torch.linalg.vector_norm(logits, dim=-1)
    assert torch.isfinite(l2_norm).all()


def test_load_frozen_causal_lm_is_a_separate_path_from_load_frozen_gpt2():
    # load_frozen_gpt2 must stay completely unaffected by load_frozen_causal_lm's
    # existence - both are separately tested here to lock that in, not just
    # asserted in a docstring.
    gpt2_model, _ = load_frozen_gpt2()
    assert gpt2_model.config.vocab_size == 50257
    assert next(gpt2_model.parameters()).dtype == torch.float32

    auto_gpt2_model, _ = load_frozen_causal_lm("gpt2")
    assert auto_gpt2_model.config.vocab_size == 50257
    assert next(auto_gpt2_model.parameters()).dtype == torch.float32
