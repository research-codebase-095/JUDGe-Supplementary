"""Tests one candidate divergence-robust correctness notion for free-running
LLM generation (DESIGN.md 14.12), named as open work in DESIGN.md 14.10:
exact-position correctness becomes a poor proxy once free-running generation
has diverged from a fixed real-text reference, since every later position is
then scored against a reference token that no longer describes the model's
actual context.

Candidate: WINDOWED correctness. windowed_correct[i] := generated_token[i] is
present anywhere in real_continuation[i-w : i+w+1] - tolerates a small
positional drift (e.g. one inserted/dropped word) instead of requiring exact
alignment.

Result (real, checked, disclosed as a negative finding, DESIGN.md 14.12):
this candidate does NOT work, for two independent, diagnosed reasons this
script checks directly rather than asserts:

1. A direct paired bootstrap on (windowed AUROC - strict AUROC), using the
   SAME MSP scores under both labelings, shows no established difference -
   the CI includes zero. Two individually-marginal CIs (one crossing 0.5,
   one not) is not evidence the two AUROCs differ from EACH OTHER.
2. The extra tokens windowed correctness credits over strict are dominated
   by chance token-frequency collisions (common words like "the", corpus
   markup fragments), not genuine local realignment with the reference -
   checked against the corpus's own token-frequency distribution, not
   asserted from eyeballing a few examples.

Usage: python scripts/check_windowed_correctness.py
"""

import os
import sys
import time
from collections import Counter

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from deployment_reliability.features import featurize  # noqa: E402
from deployment_reliability.llm_backbone import load_frozen_gpt2  # noqa: E402
from deployment_reliability.significance import _rank_based_auroc  # noqa: E402

DATA_DIR = os.path.join(REPO_ROOT, "data")

PROMPT_LEN = 50
GEN_LEN = 100
WINDOW_SIZE = PROMPT_LEN + GEN_LEN
N_WINDOWS = 30
MATCH_WINDOW = 3


@torch.inference_mode()
def free_running_generate(model, prompt_ids: torch.Tensor, gen_len: int):
    outputs = model(input_ids=prompt_ids.unsqueeze(0), use_cache=True)
    past = outputs.past_key_values
    last_logits = outputs.logits[0, -1]
    generated_ids, logits_per_step = [], []
    next_token = int(last_logits.argmax().item())
    for _ in range(gen_len):
        logits_per_step.append(last_logits)
        generated_ids.append(next_token)
        outputs = model(input_ids=torch.tensor([[next_token]]), past_key_values=past, use_cache=True)
        past = outputs.past_key_values
        last_logits = outputs.logits[0, -1]
        next_token = int(last_logits.argmax().item())
    return torch.tensor(generated_ids), torch.stack(logits_per_step, dim=0)


def windowed_correct(generated_ids: torch.Tensor, real_continuation: torch.Tensor, w: int) -> torch.Tensor:
    n = len(generated_ids)
    out = torch.zeros(n, dtype=torch.bool)
    for i in range(n):
        lo, hi = max(0, i - w), min(n, i + w + 1)
        out[i] = (generated_ids[i] == real_continuation[lo:hi]).any()
    return out


def main() -> None:
    text_path = os.path.join(DATA_DIR, "wikitext2_valid.txt")
    assert os.path.exists(text_path), (
        f"{text_path} not found - run scripts/collect_llm_free_running.py once first (it downloads this file)"
    )
    with open(text_path, encoding="utf-8") as f:
        text = f.read()
    print(f"loaded WikiText-2 validation split: {len(text)} chars")

    print("loading frozen gpt2...")
    model, tokenizer = load_frozen_gpt2("gpt2")

    full_ids = torch.tensor(tokenizer(text)["input_ids"])
    token_counts = Counter(full_ids.tolist())

    total_windows_available = len(full_ids) // WINDOW_SIZE
    n_windows = min(N_WINDOWS, total_windows_available)
    stride = max(1, total_windows_available // n_windows)
    window_starts = [i * stride * WINDOW_SIZE for i in range(n_windows)]
    print(f"using {n_windows} windows (stride {stride})")

    all_strict, all_windowed, all_fr_phi = [], [], []
    windowed_only_tokens = []

    t0 = time.time()
    for w, start in enumerate(window_starts):
        full_window = full_ids[start : start + WINDOW_SIZE]
        prompt_ids = full_window[:PROMPT_LEN]
        real_continuation = full_window[PROMPT_LEN:]

        generated_ids, fr_logits = free_running_generate(model, prompt_ids, GEN_LEN)
        strict = generated_ids == real_continuation
        windowed = windowed_correct(generated_ids, real_continuation, MATCH_WINDOW)
        fr_phi = featurize(fr_logits, normalize_l2=True)

        windowed_only_tokens.extend(generated_ids[windowed & ~strict].tolist())

        all_strict.append(strict)
        all_windowed.append(windowed)
        all_fr_phi.append(fr_phi)

        if w % 10 == 0:
            print(f"window {w + 1}/{n_windows}  elapsed={time.time() - t0:.0f}s")

    strict_correct = torch.cat(all_strict)
    windowed_correct_all = torch.cat(all_windowed)
    fr_phi = torch.cat(all_fr_phi, dim=0)
    msp = fr_phi[:, 0]

    print(f"\ndone in {time.time() - t0:.1f}s")
    print(f"strict accuracy: {strict_correct.float().mean().item():.4f}  windowed accuracy: {windowed_correct_all.float().mean().item():.4f}")
    print(f"n windowed-only positives (windowed=True, strict=False): {len(windowed_only_tokens)}")

    print("\n=== Reason 1: chance-matching confound (are windowed-only matches common tokens?) ===")
    decoded = [tokenizer.decode([t]) for t in windowed_only_tokens]
    ctr = Counter(decoded)
    for tok, cnt in ctr.most_common(10):
        print(f"  {tok!r:15s}  count={cnt:3d}  ({cnt / len(windowed_only_tokens):.1%} of all windowed-only matches)")

    sorted_by_freq = [tok for tok, _ in token_counts.most_common()]
    rank_of = {tok: i for i, tok in enumerate(sorted_by_freq)}
    ranks = [rank_of[t] for t in windowed_only_tokens]
    top20_frac = sum(1 for r in ranks if r < 20) / len(ranks)
    baseline_ranks = [rank_of.get(t, len(sorted_by_freq)) for t in full_ids[:20000].tolist()]
    baseline_top20 = sum(1 for r in baseline_ranks if r < 20) / len(baseline_ranks)
    print(f"fraction of windowed-only matches in corpus top-20 most frequent tokens: {top20_frac:.1%}")
    print(f"(baseline: {baseline_top20:.1%} of all corpus tokens are in the top-20 - compare against this)")

    print("\n=== Reason 2: is the windowed-vs-strict AUROC gap itself real? (paired bootstrap) ===")
    n = len(msp)
    rng = np.random.default_rng(0)
    msp_np = msp.detach().cpu().numpy().astype(np.float64)
    strict_np = strict_correct.numpy()
    windowed_np = windowed_correct_all.numpy()

    auroc_strict = _rank_based_auroc(msp_np[strict_np], msp_np[~strict_np])
    auroc_windowed = _rank_based_auroc(msp_np[windowed_np], msp_np[~windowed_np])
    print(f"point estimates: strict AUROC={auroc_strict:.4f}  windowed AUROC={auroc_windowed:.4f}  diff={auroc_windowed - auroc_strict:.4f}")

    diffs = []
    for _ in range(2000):
        idx = rng.integers(0, n, size=n)
        m, s, wdw = msp_np[idx], strict_np[idx], windowed_np[idx]
        if s.sum() == 0 or (~s).sum() == 0 or wdw.sum() == 0 or (~wdw).sum() == 0:
            continue
        diffs.append(_rank_based_auroc(m[wdw], m[~wdw]) - _rank_based_auroc(m[s], m[~s]))
    diffs = np.array(diffs)
    ci_lo, ci_hi = np.quantile(diffs, [0.025, 0.975])
    print(f"paired bootstrap 95% CI on (windowed - strict) AUROC: [{ci_lo:.4f}, {ci_hi:.4f}]  (n_bootstrap={len(diffs)})")
    print(f"establishes windowed > strict? {'yes' if ci_lo > 0 else 'no - not established'}")


if __name__ == "__main__":
    main()
