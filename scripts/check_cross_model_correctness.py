"""A divergence-robust correctness notion for free-running generation, built
from information genuinely EXTERNAL to the generating model's own forward
pass - the concrete gap DESIGN.md 14.14 named after its self-consistency
candidate failed for a "more fundamental reason" (circularity: any
correctness notion built purely from post-hoc transformations of the same
logit vector a confidence score already reads risks recomputing that same
score in disguise, however elaborate the transformation looks).

Two independent problems, both real, motivated this project's two prior
failed attempts (DESIGN.md 14.12, 14.14):

    1. Free-running generation quickly diverges from any FIXED reference
       text (14.10/14.11), so "does token i match the real corpus" stops
       being a meaningful question after a few tokens - not a correctness
       notion at all past that point, just noise.
    2. Any proxy computed from the SAME model's own logits at those same
       positions (14.14's self-consistency resampling) is at risk of just
       re-deriving the confidence score being evaluated, in a different
       shape - confirmed exactly that way (analytic P(resample==argmax) had
       AUROC statistically indistinguishable from MSP's, r=0.939 Pearson).

This script's design: use a SECOND, independently trained model - different
weights, different architecture family (GPT-NeoX vs. GPT-2's own), different
tokenizer, different training corpus (The Pile vs. WebText) - to score the
generating model's free-running continuation. This sidesteps problem 1 (no
fixed reference text is needed at all; the second model reads whatever text
was actually generated, however far it drifted) and problem 2 (the scoring
model never sees the generating model's logits, only the decoded TEXT its
greedy decoding produced - a completely different computational object, not
a transform of the same forward pass).

Concretely, per window:
    - Model A free-running-generates `gen_len` tokens from a real
      `prompt_len`-token WikiText-2 prompt (identical protocol to
      collect_llm_free_running.py's free_running_generate).
    - A's own mean top-1 softmax probability (MSP) during that generation
      is cached as `a_confidence[w]` - the same confidence signal this
      project's headline results are built on, at the point it is emitted
      (a real, live quantity, not recomputed afterward).
    - A's generated tokens are decoded to text (A's own tokenizer), the
      real prompt text is prepended, and the whole string is RE-TOKENIZED
      with model B's tokenizer (necessary: GPT-2's BPE vocabulary and
      Pythia's GPT-NeoX-20B vocabulary are different token spaces, so B
      cannot consume A's token ids directly). Model B teacher-forces this
      B-tokenized sequence and its mean per-token log-probability on the
      continuation portion is `b_logprob[w]` - an external, independent
      judgment of how plausible A's actual output is, with no access to
      why A produced it.

The correctness question this notion answers: "does A's own live
confidence signal predict whether an independent model finds A's free-
running output plausible" - answerable at any generation length, immune to
reference drift by construction (there IS no fixed reference), and not
circular by construction (B never reads A's logits).

Run BOTH directions (A=GPT-2 generates/B=Pythia-160m scores, and the
reverse) for backbone-generality, matching this project's established
practice (14.11) of not trusting a single-backbone finding.

A known risk, checked explicitly rather than assumed away: greedy
free-running decoding is well documented (Holtzman et al., ICLR 2020, "The
Curious Case of Neural Text Degeneration") to collapse into repetition
loops, and repeated text is trivially predictable by ANY model, including
an independent one - so b_logprob could end up high for degenerate,
low-quality repetition rather than for genuinely good text, which would
make any correlation with a_confidence a repetition-driven confound rather
than the intended signal. This script computes a repetition-ratio
statistic per window and checks its correlation with both a_confidence and
b_logprob before trusting any headline correlation number.

Generality is a separate, later question from whether the notion works at
all (DESIGN.md 14.16 established that it does, on GPT-2/Pythia-160m/
WikiText-2): --model-a/--model-b/--corpus-url let a later run test a third
backbone or a second corpus in isolation, one variable at a time, without
re-deriving the design above.

Usage:
    python scripts/check_cross_model_correctness.py
    python scripts/check_cross_model_correctness.py --n-windows 400
    # third-backbone generality check (BLOOM-560m, genuinely different
    # architecture/tokenizer/training corpus from both GPT-2 and Pythia):
    python scripts/check_cross_model_correctness.py --model-b bigscience/bloom-560m --model-b-tag bloom560m
    # second-corpus generality check (Tiny Shakespeare, structurally
    # different register/era from WikiText-2's formal encyclopedic prose):
    python scripts/check_cross_model_correctness.py --corpus-url https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt --corpus-tag tinyshakespeare
"""

import argparse
import os
import sys
import time
import urllib.request

import numpy as np
import torch
from scipy.stats import spearmanr

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from deployment_reliability.features import featurize  # noqa: E402
from deployment_reliability.llm_backbone import load_frozen_causal_lm, load_frozen_gpt2  # noqa: E402
from deployment_reliability.significance import bootstrap_auroc_ci  # noqa: E402

DATA_DIR = os.path.join(REPO_ROOT, "data")

WIKITEXT2_VALID_URL = (
    "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/valid.txt"
)

PROMPT_LEN = 50
GEN_LEN = 50


def download_corpus(url: str, cache_filename: str) -> str:
    """Generic version of download_wikitext2_valid's caching pattern
    (shared with collect_llm_logits.py/collect_llm_free_running.py for the
    WikiText-2 case specifically), parameterized so a second corpus (e.g.
    Tiny Shakespeare, for the corpus-generality check) can reuse the same
    raw-URL-download-and-cache approach rather than depending on the
    `datasets` library, matching this project's existing convention.
    """
    cache_path = os.path.join(DATA_DIR, cache_filename)
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            return f.read()
    print(f"downloading corpus from {url}...")
    text = urllib.request.urlopen(url, timeout=60).read().decode("utf-8")
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        f.write(text)
    return text


@torch.inference_mode()
def free_running_generate_with_confidence(model, prompt_ids: torch.Tensor, gen_len: int) -> tuple[torch.Tensor, float]:
    """Greedy free-running generation (identical protocol to
    collect_llm_free_running.py's free_running_generate), also returning
    the mean MSP across the gen_len generation steps - A's own live
    confidence signal, read directly off the same logits that produced
    each generated token, not recomputed after the fact.
    """
    outputs = model(input_ids=prompt_ids.unsqueeze(0), use_cache=True)
    past = outputs.past_key_values
    last_logits = outputs.logits[0, -1]
    generated_ids = []
    msp_per_step = []
    next_token = int(last_logits.argmax().item())
    for _ in range(gen_len):
        phi = featurize(last_logits.unsqueeze(0), normalize_l2=True)
        msp_per_step.append(float(phi[0, 0].item()))  # index 0 == msp, DEFAULT_FEATURE_NAMES order
        generated_ids.append(next_token)
        outputs = model(input_ids=torch.tensor([[next_token]]), past_key_values=past, use_cache=True)
        past = outputs.past_key_values
        last_logits = outputs.logits[0, -1]
        next_token = int(last_logits.argmax().item())
    return torch.tensor(generated_ids), float(np.mean(msp_per_step))


@torch.inference_mode()
def cross_model_mean_logprob(scorer_model, scorer_tokenizer, prompt_text: str, continuation_text: str) -> float:
    """Model B's mean per-token log-probability of A's generated
    continuation, computed by re-tokenizing the DECODED TEXT with B's own
    tokenizer (A's and B's token ids live in different, incompatible
    vocabularies - GPT-2 BPE vs. Pythia's GPT-NeoX-20B tokenizer) and
    teacher-forcing B over prompt_text + continuation_text.

    Splitting point: tokenize prompt_text alone and prompt_text +
    continuation_text together, and take the boundary as
    len(prompt_only_ids) - the standard practical approach for
    cross-tokenizer boundary alignment (BPE merges can occasionally shift
    the exact seam token by one; this affects at most one token out of
    gen_len and does not change the mean appreciably).
    """
    prompt_ids = scorer_tokenizer(prompt_text)["input_ids"]
    full_ids = scorer_tokenizer(prompt_text + continuation_text)["input_ids"]
    boundary = len(prompt_ids)
    if boundary >= len(full_ids) - 1:
        return float("nan")  # continuation contributed no new tokens under B's tokenizer - degenerate window, skip

    # A rare real failure mode, hit on a full 400-window run rather than
    # anticipated in advance: re-tokenizing DECODED TEXT under a different
    # BPE vocabulary can expand token count well past the ~prompt_len+
    # gen_len=100 tokens either model's own tokenizer produced, when a
    # WikiText-2 window contains unusual unicode/formatting one tokenizer
    # merges compactly and the other explodes into many byte-level
    # fallback tokens - occasionally past GPT-2's 1024-position limit.
    # Truncate from the LEFT (drop earliest prompt tokens first) so the
    # full continuation - the actual quantity being scored - is preserved
    # whenever physically possible.
    max_len = getattr(scorer_model.config, "max_position_embeddings", None) or getattr(
        scorer_model.config, "n_positions", None
    )
    if max_len is not None and len(full_ids) > max_len:
        trim = len(full_ids) - max_len
        full_ids = full_ids[trim:]
        boundary = max(1, boundary - trim)
        if boundary >= len(full_ids) - 1:
            return float("nan")  # continuation itself exceeds the scorer's context window - skip, not truncate mid-continuation
    full_ids_t = torch.tensor(full_ids)
    logits = scorer_model(input_ids=full_ids_t.unsqueeze(0)).logits.squeeze(0)  # (len, vocab)
    logprobs = torch.log_softmax(logits, dim=-1)
    # position i predicts token i+1; continuation tokens are full_ids[boundary:]
    target_positions = torch.arange(boundary - 1, len(full_ids) - 1)
    targets = full_ids_t[boundary:]
    token_logprobs = logprobs[target_positions, targets]
    return float(token_logprobs.mean().item())


def repetition_ratio(generated_ids: torch.Tensor) -> float:
    """Fraction of generated tokens that are an exact immediate repeat of
    the token before them - a direct, cheap symptom of the greedy
    repetition-loop degeneration Holtzman et al. (2020) document, used
    here purely as a confound check on b_logprob/a_confidence, not as
    part of the correctness notion itself.
    """
    if len(generated_ids) < 2:
        return 0.0
    return float((generated_ids[1:] == generated_ids[:-1]).float().mean().item())


def bootstrap_spearman_ci(x: np.ndarray, y: np.ndarray, n_bootstrap: int = 2000, confidence: float = 0.95, seed: int = 0):
    """Paired percentile bootstrap CI for Spearman correlation - resamples
    window INDICES jointly (x[i], y[i] always resampled together, unlike
    bootstrap_auroc_ci's independent two-group resampling) since x and y
    here are two measurements of the SAME window, not two independent
    populations.
    """
    point = spearmanr(x, y).correlation
    rng = np.random.default_rng(seed)
    n = len(x)
    draws = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        draws[i] = spearmanr(x[idx], y[idx]).correlation
    alpha = 1.0 - confidence
    lo, hi = np.quantile(draws, [alpha / 2.0, 1.0 - alpha / 2.0])
    return point, float(lo), float(hi)


def run_direction(
    gen_model, gen_tokenizer, gen_name: str,
    score_model, score_tokenizer, score_name: str,
    full_ids: torch.Tensor, window_starts: list, prompt_len: int, gen_len: int,
) -> dict:
    a_confidence, b_logprob, rep_ratio = [], [], []
    t0 = time.time()
    for w, start in enumerate(window_starts):
        prompt_ids = full_ids[start : start + prompt_len]
        prompt_text = gen_tokenizer.decode(prompt_ids)

        generated_ids, msp_mean = free_running_generate_with_confidence(gen_model, prompt_ids, gen_len)
        continuation_text = gen_tokenizer.decode(generated_ids)

        b_lp = cross_model_mean_logprob(score_model, score_tokenizer, prompt_text, continuation_text)
        if np.isnan(b_lp):
            continue

        a_confidence.append(msp_mean)
        b_logprob.append(b_lp)
        rep_ratio.append(repetition_ratio(generated_ids))

        if w % 50 == 0:
            elapsed = time.time() - t0
            print(f"  [{gen_name} generates, {score_name} scores] window {w + 1}/{len(window_starts)}  elapsed={elapsed:.0f}s")

    a_confidence = np.array(a_confidence)
    b_logprob = np.array(b_logprob)
    rep_ratio = np.array(rep_ratio)
    n = len(a_confidence)

    corr, corr_lo, corr_hi = bootstrap_spearman_ci(a_confidence, b_logprob)

    median_b = np.median(b_logprob)
    high_mask = b_logprob >= median_b
    pos_scores = torch.tensor(a_confidence[high_mask])
    neg_scores = torch.tensor(a_confidence[~high_mask])
    auroc_result = bootstrap_auroc_ci(pos_scores, neg_scores)

    confound_conf, confound_conf_lo, confound_conf_hi = bootstrap_spearman_ci(rep_ratio, a_confidence)
    confound_b, confound_b_lo, confound_b_hi = bootstrap_spearman_ci(rep_ratio, b_logprob)

    return {
        "gen_name": gen_name,
        "score_name": score_name,
        "n": n,
        "spearman": corr,
        "spearman_ci": (corr_lo, corr_hi),
        "auroc": auroc_result.auroc,
        "auroc_ci": (auroc_result.ci_lo, auroc_result.ci_hi),
        "mean_repetition_ratio": float(rep_ratio.mean()),
        "confound_repetition_vs_confidence": (confound_conf, confound_conf_lo, confound_conf_hi),
        "confound_repetition_vs_blogprob": (confound_b, confound_b_lo, confound_b_hi),
    }


def print_result(r: dict) -> None:
    print(f"\n=== {r['gen_name']} generates, {r['score_name']} scores (n={r['n']} windows) ===")
    print(f"Spearman(A's own MSP confidence, B's cross-model log-prob) = {r['spearman']:.4f}  "
          f"95% CI [{r['spearman_ci'][0]:.4f}, {r['spearman_ci'][1]:.4f}]")
    print(f"AUROC(A's MSP confidence ranks B-rated-high-quality windows above B-rated-low-quality) = "
          f"{r['auroc']:.4f}  95% CI [{r['auroc_ci'][0]:.4f}, {r['auroc_ci'][1]:.4f}]")
    print(f"mean repetition ratio (exact immediate-token repeats): {r['mean_repetition_ratio']:.4f}")
    rc, rc_lo, rc_hi = r["confound_repetition_vs_confidence"]
    rb, rb_lo, rb_hi = r["confound_repetition_vs_blogprob"]
    print(f"confound check: Spearman(repetition ratio, A confidence) = {rc:.4f}  CI [{rc_lo:.4f}, {rc_hi:.4f}]")
    print(f"confound check: Spearman(repetition ratio, B log-prob)   = {rb:.4f}  CI [{rb_lo:.4f}, {rb_hi:.4f}]")


def _load_model(model_name: str):
    # load_frozen_gpt2 is kept as the GPT-2 path specifically (matches
    # every other script in this project that loads GPT-2); anything else
    # goes through the architecture-agnostic AutoModelForCausalLM path,
    # which is exactly what a genuinely different third backbone needs.
    if model_name.lower() == "gpt2":
        return load_frozen_gpt2()
    return load_frozen_causal_lm(model_name)


def main(
    n_windows: int, prompt_len: int, gen_len: int,
    model_a: str, model_a_tag: str, model_b: str, model_b_tag: str,
    corpus_url: str, corpus_tag: str,
) -> None:
    window_size = prompt_len + gen_len
    # Suffixes only appear when a run departs from the original validated
    # default (gpt2/pythia160m/wikitext2, gen_len=50) - matches
    # collect_llm_free_running.py's convention and keeps the already-
    # documented default run's output path stable, while giving later
    # generality-check runs (a third backbone or a second corpus) their
    # own non-colliding file (a real near-miss already happened once at
    # this project's two different --gen-len values before that fix).
    model_suffix = "" if (model_a_tag, model_b_tag) == ("gpt2", "pythia160m") else f"_{model_a_tag}_{model_b_tag}"
    corpus_suffix = "" if corpus_tag == "wikitext2" else f"_{corpus_tag}"
    gen_suffix = "" if gen_len == GEN_LEN else f"_genlen{gen_len}"
    out_path = os.path.join(DATA_DIR, f"cross_model_correctness_results{model_suffix}{corpus_suffix}{gen_suffix}.pt")

    cache_filename = "wikitext2_valid.txt" if corpus_tag == "wikitext2" else f"{corpus_tag}.txt"
    text = download_corpus(corpus_url, cache_filename)
    print(f"loaded corpus ({corpus_tag}): {len(text)} chars")

    print(f"loading frozen {model_a} ({model_a_tag})...")
    model_a_obj, tok_a = _load_model(model_a)
    print(f"loading frozen {model_b} ({model_b_tag})...")
    model_b_obj, tok_b = _load_model(model_b)

    full_ids = torch.tensor(tok_a(text)["input_ids"])
    total_windows_available = len(full_ids) // window_size
    n_windows = min(n_windows, total_windows_available)
    stride = max(1, total_windows_available // n_windows)
    window_starts = [i * stride * window_size for i in range(n_windows)]
    print(f"total tokens: {len(full_ids)}  windows available: {total_windows_available}  using {n_windows} (stride {stride})")

    result_a_b = run_direction(
        model_a_obj, tok_a, model_a_tag, model_b_obj, tok_b, model_b_tag,
        full_ids, window_starts, prompt_len, gen_len,
    )
    result_b_a = run_direction(
        model_b_obj, tok_b, model_b_tag, model_a_obj, tok_a, model_a_tag,
        full_ids, window_starts, prompt_len, gen_len,
    )

    print_result(result_a_b)
    print_result(result_b_a)

    torch.save(
        {f"{model_a_tag}_generates_{model_b_tag}_scores": result_a_b, f"{model_b_tag}_generates_{model_a_tag}_scores": result_b_a},
        out_path,
    )
    print(f"\nsaved results to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-windows", type=int, default=200, help="number of windows to evaluate per direction (default: 200)")
    parser.add_argument("--prompt-len", type=int, default=PROMPT_LEN)
    parser.add_argument("--gen-len", type=int, default=GEN_LEN)
    parser.add_argument("--model-a", default="gpt2", help="HF model name/path for model A (default: gpt2)")
    parser.add_argument("--model-a-tag", default=None, help="tag for model A, used in output filenames (default: derived from --model-a)")
    parser.add_argument("--model-b", default="EleutherAI/pythia-160m", help="HF model name/path for model B (default: EleutherAI/pythia-160m)")
    parser.add_argument("--model-b-tag", default=None, help="tag for model B, used in output filenames (default: derived from --model-b)")
    parser.add_argument("--corpus-url", default=WIKITEXT2_VALID_URL, help="raw-text URL for the corpus (default: WikiText-2 validation split)")
    parser.add_argument("--corpus-tag", default="wikitext2", help="tag for the corpus, used in cache/output filenames (default: wikitext2)")
    args = parser.parse_args()
    model_a_tag = args.model_a_tag or args.model_a.split("/")[-1].replace("-", "")
    model_b_tag = args.model_b_tag or args.model_b.split("/")[-1].replace("-", "")
    main(
        args.n_windows, args.prompt_len, args.gen_len,
        args.model_a, model_a_tag, args.model_b, model_b_tag,
        args.corpus_url, args.corpus_tag,
    )
