"""Frozen LLM backbone loading, per DESIGN.md 14.3/14.4's LLM extension design
intent - now implemented and validated, not just proposed. Mirrors backbone.py's
shape (model, tokenizer) so this stays a genuinely new *domain*, not a
different code path: the rest of the package (features.py/combiner.py/
router.py) never sees this module at all, only the logit tensor it produces,
exactly as DESIGN.md 4/14.1 requires.

This module never trains or modifies a backbone - it exists purely to produce
the per-token logit vectors the rest of the package consumes, the same
constraint backbone.py operates under for vision models.
"""

from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel, GPT2Tokenizer


def load_frozen_causal_lm(model_name: str) -> tuple[Any, Any]:
    """Architecture-agnostic frozen causal-LM loader via AutoModelForCausalLM/
    AutoTokenizer, in eval mode with gradients disabled (DESIGN.md 3.2:
    inference-only) - the generalization of `load_frozen_gpt2` below to any
    model family HuggingFace's Auto* classes resolve, not just GPT-2.

    Added for item 1b of STUDY_PLAN.md 3.6's synthesis: the LLM extension
    (14.5) previously validated exactly one model (GPT-2, 124M), leaving open
    whether its token/sequence-level discrimination-quality finding was a
    property of the design or an accident of that one specific model. This
    function is what lets a second, architecturally distinct model (e.g.
    EleutherAI/pythia-160m, a GPT-NeoX-family model trained on a different
    corpus - the Pile, not GPT-2's WebText) be loaded through the same
    interface `load_frozen_gpt2` already uses, so `logits_for_token_chunk`
    and every downstream consumer (features.py/scripts/collect_llm_logits.py)
    work unchanged regardless of which model was loaded.

    Returns (model, tokenizer) - same shape as `load_frozen_gpt2`.

    Explicitly forces `torch_dtype=torch.float32` - a real bug, found and
    fixed while producing this item's results, not a preemptive guess:
    `AutoModelForCausalLM.from_pretrained` without an explicit `torch_dtype`
    preserves whatever dtype a checkpoint was SAVED in, and
    `EleutherAI/pythia-160m`'s published checkpoint is float16. GPT-2's
    checkpoint happens to be float32, so `load_frozen_gpt2` (which also
    never sets `torch_dtype`) never surfaced this. Pythia-160m's raw logits
    reach magnitude ~800+ (vs. GPT-2's typically two orders of magnitude
    smaller), and `features.logit_l2_norm` squares each logit before
    summing across the ~50k-wide vocabulary - well past float16's ~65504
    max representable value, silently overflowing to `inf` in the cached
    phi(z) before this fix (caught by inspecting the raw cached tensor, not
    by a crash - `torch.save`/`torch.load` round-trip `inf` values without
    complaint; the LBFGS fit in `combiner.py` is what actually raised,
    downstream, with an opaque `RuntimeError: value cannot be converted to
    type c10::Half without overflow`). Forcing float32 here matches every
    other numeric assumption this project's feature/combiner/router code
    already makes.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model, tokenizer


def load_frozen_gpt2(model_name: str = "gpt2") -> tuple[GPT2LMHeadModel, GPT2Tokenizer]:
    """Load pretrained GPT-2 (124M, the smallest standard variant) in eval mode
    with gradients disabled (DESIGN.md 3.2: inference-only) - the LLM-side
    counterpart to backbone.py's load_frozen_resnet50/vit_b16/convnext_tiny.

    Chosen deliberately as the smallest, most standard, fully openly-licensed
    causal LM (no registration/gated access, unlike some newer model families),
    with a real 50,257-token vocabulary - a genuine, non-hypothetical test of
    DESIGN.md 14.4's vocabulary-scaling claims, which previously only had
    ImageNet's ~1,000-class scale as a real reference point.

    Kept as its own thin, unchanged function (rather than a call-site rewrite
    to `load_frozen_causal_lm`) so every existing caller of this exact name
    keeps working identically - `load_frozen_causal_lm` above is the new,
    additive, architecture-agnostic path, not a replacement for this one.

    Returns (model, tokenizer) - the frozen model and its matching tokenizer,
    the LLM-domain parallel to (model, preprocess) from backbone.py.
    """
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model, tokenizer


@torch.inference_mode()
def logits_for_token_chunk(model, token_ids: torch.Tensor) -> torch.Tensor:
    """Run a frozen causal-LM forward pass on one chunk of token ids, returning
    raw per-position logits. `token_ids` is a 1D tensor of length <= the
    model's context window (1024 for GPT-2, 2048 for Pythia-160m, though this
    project chunks both at 1024 for direct comparability - see
    scripts/collect_llm_logits.py); returns a tensor of shape (len, vocab_size).

    The single forward pass DESIGN.md 3.2 requires the whole reliability layer
    to live within - no additional passes, no sampling, no ensembling. This is
    the LLM-domain counterpart to backbone.py's logits_for_images: identical
    contract (raw logits in, nothing else), different modality. Works
    unchanged for any model returned by `load_frozen_gpt2` or
    `load_frozen_causal_lm` - nothing here is GPT-2-specific, only the type
    hint historically was.
    """
    outputs = model(input_ids=token_ids.unsqueeze(0))
    return outputs.logits.squeeze(0)
