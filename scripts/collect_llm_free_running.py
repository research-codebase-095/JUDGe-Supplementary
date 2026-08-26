"""Free-running (non-teacher-forced) generation, the harder regime DESIGN.md
14.5/14.6 and STUDY_PLAN.md 6.5/7 explicitly flag as untested: everything
validated so far conditions each next-token prediction on the REAL previous
tokens from the corpus ("teacher-forced"). Real deployment does not do this -
a deployed LLM conditions on its OWN previously generated tokens. This script
runs the frozen model autoregressively (greedy argmax, no sampling, matching
this project's existing "argmax" correctness convention exactly, DESIGN.md
8b) and caches per-generation-step features + two different correctness
labels, so notebooks/22_llm_free_running.ipynb doesn't need to re-run
inference every time it's opened.

Correctness definition, chosen and justified here (this is a genuinely open
design question - DESIGN.md/STUDY_PLAN.md never resolved it, since nothing
before this script generated free-running text at all):

    For a real WikiText-2 window, take a real PROMPT_LEN-token prefix as the
    prompt and record the GEN_LEN real tokens that actually follow it in the
    corpus ("real_continuation"). Generate GEN_LEN tokens autoregressively -
    each step's argmax conditions on the prompt plus every PREVIOUSLY
    GENERATED token, never on real_continuation. free_running_correct[i] :=
    1[generated_token_i == real_continuation_i] - the exact same "argmax
    matches the real corpus token" rule as DESIGN.md 8b/14.5's
    y = 1[argmax(z) == true_label], unchanged. What changes is only the
    CONTEXT the argmax conditions on (self-generated vs. real), not the
    definition of correctness itself - deliberately not a new formalism.

    Why this, and not a perplexity- or self-consistency-based proxy: this
    project's actual deployment question (DESIGN.md 3.2, 14.5) is "does the
    model's own logit geometry predict whether ITS OUTPUT can be trusted."
    The one available ground truth for "trusted, here" is the real text that
    was actually written at that corpus position - re-using it, under
    self-generated conditioning, is the most direct test of whether
    generation has drifted from that reference, and needs no new
    terminology or invented target. The explicit, disclosed cost: once a
    generation diverges from the real continuation (likely quickly, since
    Wikipedia's specific wording is only one of many reasonable
    continuations), every later position in the same window is judged
    against a real-text target that no longer describes the context the
    model is actually conditioning on - a structural mismatch, not a model
    failure, and not the same thing as "the generation is bad." This is
    disclosed explicitly in DESIGN.md's write-up rather than smoothed over,
    exactly parallel to how token-level accuracy already sits at 41% in the
    teacher-forced case for the same underlying reason (many reasonable
    continuations exist; only one is "correct").

    A second, complementary quantity is cached alongside this one:
    teacher-forced correctness on the IDENTICAL corpus positions (computed
    once per window from a single ordinary forward pass over
    prompt + real_continuation) - not a proxy for free-running correctness,
    but the direct, same-positions comparison baseline DESIGN.md 14.5's
    headline numbers do not otherwise provide.

Usage:
    python scripts/collect_llm_free_running.py                                          # gpt2 (default)
    python scripts/collect_llm_free_running.py --model EleutherAI/pythia-160m --tag pythia160m --auto
"""

import argparse
import os
import sys
import time
import urllib.request

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from deployment_reliability.features import featurize  # noqa: E402
from deployment_reliability.llm_backbone import load_frozen_causal_lm, load_frozen_gpt2  # noqa: E402
from deployment_reliability.splits import three_way_split  # noqa: E402

DATA_DIR = os.path.join(REPO_ROOT, "data")

WIKITEXT2_VALID_URL = (
    "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/valid.txt"
)

PROMPT_LEN = 50
GEN_LEN = 50
WINDOW_SIZE = PROMPT_LEN + GEN_LEN


def download_wikitext2_valid() -> str:
    # Shared cache file with scripts/collect_llm_logits.py - same corpus, same URL.
    cache_path = os.path.join(DATA_DIR, "wikitext2_valid.txt")
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            return f.read()
    print("downloading WikiText-2 validation split...")
    text = urllib.request.urlopen(WIKITEXT2_VALID_URL, timeout=60).read().decode("utf-8")
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        f.write(text)
    return text


@torch.inference_mode()
def free_running_generate(model, prompt_ids: torch.Tensor, gen_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Greedy (argmax, no sampling) autoregressive generation, KV-cached for
    speed. KV-caching is a pure computational optimization for a causal
    model with no dropout in eval mode - it changes nothing about the raw
    logit values returned at each step versus a full un-cached recompute of
    the growing sequence, so this stays within DESIGN.md 3.2's "raw logits,
    nothing else" contract exactly as logits_for_token_chunk does; only the
    single-token incremental forward call differs mechanically from that
    function's whole-chunk call.

    Returns (generated_token_ids, logits_at_each_step), both length gen_len.
    logits_at_each_step[i] is the raw logit vector whose argmax produced
    generated_token_ids[i] by construction - the free-running counterpart of
    "position j's logits predict token j+1" in the teacher-forced case.
    """
    outputs = model(input_ids=prompt_ids.unsqueeze(0), use_cache=True)
    past = outputs.past_key_values
    last_logits = outputs.logits[0, -1]
    generated_ids = []
    logits_per_step = []
    next_token = int(last_logits.argmax().item())
    for _ in range(gen_len):
        logits_per_step.append(last_logits)
        generated_ids.append(next_token)
        outputs = model(input_ids=torch.tensor([[next_token]]), past_key_values=past, use_cache=True)
        past = outputs.past_key_values
        last_logits = outputs.logits[0, -1]
        next_token = int(last_logits.argmax().item())
    return torch.tensor(generated_ids), torch.stack(logits_per_step, dim=0)


@torch.inference_mode()
def teacher_forced_logits_same_positions(
    model, full_window_ids: torch.Tensor, prompt_len: int, window_size: int
) -> torch.Tensor:
    """One ordinary forward pass over prompt+real_continuation; returns the
    gen_len logit vectors that predict exactly the same corpus positions
    free_running_generate's output is compared against - the direct,
    same-positions teacher-forced baseline.
    """
    outputs = model(input_ids=full_window_ids.unsqueeze(0))
    logits = outputs.logits.squeeze(0)  # (window_size, vocab)
    # position (prompt_len-1) predicts token at prompt_len, ..., position
    # (window_size-2) predicts token at window_size-1 - the same gen_len
    # real_continuation positions free-running generation targets.
    return logits[prompt_len - 1 : window_size - 1]


def main(model_name: str, tag: str, use_auto: bool, n_windows: int, prompt_len: int = PROMPT_LEN, gen_len: int = GEN_LEN) -> None:
    window_size = prompt_len + gen_len
    suffix = "" if gen_len == GEN_LEN else f"_genlen{gen_len}"
    out_path = os.path.join(DATA_DIR, f"llm_free_running_cache_{tag}{suffix}.pt")
    text = download_wikitext2_valid()
    print(f"loaded WikiText-2 validation split: {len(text)} chars")

    print(f"loading frozen {model_name}...")
    model, tokenizer = load_frozen_causal_lm(model_name) if use_auto else load_frozen_gpt2(model_name)

    full_ids = torch.tensor(tokenizer(text)["input_ids"])
    total_windows_available = len(full_ids) // window_size
    n_windows = min(n_windows, total_windows_available)
    # Evenly spaced across the whole validation split (not just its first
    # N_windows*window_size tokens) so the sample isn't biased toward
    # whatever WikiText-2 happens to open with.
    stride = max(1, total_windows_available // n_windows)
    window_starts = [i * stride * window_size for i in range(n_windows)]
    print(
        f"total tokens: {len(full_ids)}  windows available (non-overlapping, size {window_size}): "
        f"{total_windows_available}  using {n_windows} (stride {stride})"
    )

    all_fr_phi, all_fr_correct = [], []
    all_tf_phi, all_tf_correct = [], []
    all_window_id, all_offset = [], []

    t0 = time.time()
    for w, start in enumerate(window_starts):
        full_window = full_ids[start : start + window_size]
        prompt_ids = full_window[:prompt_len]
        real_continuation = full_window[prompt_len:]

        generated_ids, fr_logits = free_running_generate(model, prompt_ids, gen_len)
        fr_correct = generated_ids == real_continuation
        fr_phi = featurize(fr_logits, normalize_l2=True)

        tf_logits = teacher_forced_logits_same_positions(model, full_window, prompt_len, window_size)
        tf_correct = tf_logits.argmax(dim=-1) == real_continuation
        tf_phi = featurize(tf_logits, normalize_l2=True)

        all_fr_phi.append(fr_phi)
        all_fr_correct.append(fr_correct)
        all_tf_phi.append(tf_phi)
        all_tf_correct.append(tf_correct)
        all_window_id.append(torch.full((gen_len,), w, dtype=torch.long))
        all_offset.append(torch.arange(gen_len, dtype=torch.long))

        if w % 20 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (w + 1) * (n_windows - w - 1)
            print(f"window {w + 1}/{n_windows}  elapsed={elapsed:.0f}s  eta={eta:.0f}s")

    fr_phi = torch.cat(all_fr_phi, dim=0)
    fr_correct = torch.cat(all_fr_correct, dim=0)
    tf_phi = torch.cat(all_tf_phi, dim=0)
    tf_correct = torch.cat(all_tf_correct, dim=0)
    window_id = torch.cat(all_window_id, dim=0)
    offset = torch.cat(all_offset, dim=0)

    print(f"done in {time.time() - t0:.1f}s. {n_windows} windows x {gen_len} generated tokens = {len(fr_correct)} total")
    print(f"free-running accuracy (vs real continuation): {fr_correct.float().mean().item():.4f}")
    print(f"teacher-forced accuracy (same positions):      {tf_correct.float().mean().item():.4f}")

    # Split assignment at the WINDOW level, not the token level - tokens
    # within one generated window are highly correlated (each conditions on
    # every previous one), so a per-token random split would leak strongly
    # correlated examples across combiner_fit/threshold_cal/id_test, unlike
    # the teacher-forced token-level case (DESIGN.md 10.5) where consecutive
    # tokens are only weakly coupled through the frozen backbone's context
    # window, not through each other's own generated output.
    combiner_idx, cal_idx, test_idx = three_way_split(n_windows, seed=0)
    window_split = [""] * n_windows
    for idx, name in ((combiner_idx, "combiner_fit"), (cal_idx, "threshold_cal"), (test_idx, "id_test")):
        for i in idx.tolist():
            window_split[i] = name
    splits = [window_split[w] for w in window_id.tolist()]

    torch.save(
        {
            "fr_phi": fr_phi,
            "fr_correct": fr_correct,
            "tf_phi": tf_phi,
            "tf_correct": tf_correct,
            "window_id": window_id,
            "offset": offset,
            "splits": splits,
            "prompt_len": prompt_len,
            "gen_len": gen_len,
            "n_windows": n_windows,
        },
        out_path,
    )
    print("saved cache to", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt2", help="HF model name/path (default: gpt2)")
    parser.add_argument(
        "--tag", default=None, help="cache filename tag -> data/llm_free_running_cache_<tag>.pt (default: derived from --model)"
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="load via load_frozen_causal_lm (AutoModelForCausalLM/AutoTokenizer) instead of load_frozen_gpt2",
    )
    parser.add_argument("--n-windows", type=int, default=400, help="number of non-overlapping windows to generate (default: 400)")
    parser.add_argument("--prompt-len", type=int, default=PROMPT_LEN, help=f"prompt length in tokens (default: {PROMPT_LEN})")
    parser.add_argument("--gen-len", type=int, default=GEN_LEN, help=f"number of tokens to generate per window (default: {GEN_LEN})")
    args = parser.parse_args()
    tag = args.tag or args.model.split("/")[-1].replace("-", "")
    main(args.model, tag, args.auto, args.n_windows, prompt_len=args.prompt_len, gen_len=args.gen_len)
