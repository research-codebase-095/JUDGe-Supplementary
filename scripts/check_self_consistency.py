"""Tests a second candidate divergence-robust correctness notion for
free-running LLM generation (DESIGN.md 14.14), attempted after the first
candidate (windowed token-match, DESIGN.md 14.12) was ruled out. Named as
open work in DESIGN.md 14.10: exact-position correctness becomes a poor
proxy once free-running generation has diverged from a fixed real-text
reference, since every later position is scored against a reference token
that no longer describes the model's actual context.

Candidate: LOCAL DECISION-STABILITY, not reference-matching at all. Along
the single, deterministic greedy generation trajectory, resample each
step's token (temperature T=0.7) from the SAME logits that produced the
greedy choice, and ask: does the resample reproduce the greedy pick?
locally_stable[i] := resample_token[i] == greedy_token[i]. This never needs
a stale real-text reference and, because it always resamples from the
greedy path's own real, single context (never a second, independently-
diverging trajectory), it should be immune to the divergence problem this
whole exercise is trying to escape.

An EARLIER, flawed version of this candidate compared two independently-
generated trajectories (greedy vs. temperature-sampled, each accumulating
its OWN KV cache) - caught on code review, before any numbers were
reported, not after: once the two paths first disagreed, all later
"agreement" checks were comparing tokens generated from two DIFFERENT
contexts, silently reintroducing the exact divergence problem the
candidate was meant to solve. That design is not reproduced here; this
script implements only the corrected, single-trajectory version.

Result on the corrected design (real, checked, disclosed honestly,
DESIGN.md 14.14): local decision-stability does NOT decay across the
generation window (unlike both reference-matching candidates) and MSP
AUROC for predicting it is very high (~0.94) - but this is NOT a working
correctness notion, for a genuinely different and more fundamental reason
than either prior candidate's failure mode: local-stability's own label is
(near-)DETERMINISTICALLY computable from the same softmax distribution MSP
is read off of. This script's `analytic P(resample==argmax)` - a closed-
form function of the logits alone, no sampling needed - achieves
essentially the same AUROC as MSP does for predicting the empirical
resampling outcome, and correlates with MSP at r~0.94. "MSP predicts local
stability well" is therefore close to tautological (confidence predicting
a near-deterministic function of itself), not evidence MSP carries genuine
information about whether the generated content is trustworthy - the
actual, still-unsolved question.

Usage: python scripts/check_self_consistency.py
"""

import os
import sys
import time
from collections import Counter

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from deployment_reliability.features import featurize  # noqa: E402
from deployment_reliability.llm_backbone import load_frozen_gpt2  # noqa: E402
from deployment_reliability.significance import bootstrap_auroc_ci  # noqa: E402

DATA_DIR = os.path.join(REPO_ROOT, "data")

PROMPT_LEN = 50
GEN_LEN = 100
WINDOW_SIZE = PROMPT_LEN + GEN_LEN
N_WINDOWS = 40
TEMPERATURE = 0.7
LATE_OFFSET_CUTOFF = 5


@torch.inference_mode()
def generate_with_local_stability(model, prompt_ids: torch.Tensor, gen_len: int, temperature: float, seed: int):
    """Single greedy trajectory. At each step, ALSO draws a temperature-
    sampled token from the exact same logits (same context) that produced
    the greedy choice - a genuine "would this specific decision survive
    resampling" check, never conditioned on a second, diverging path.

    Returns (greedy_ids, resample_ids, greedy_logits_per_step).
    """
    torch.manual_seed(seed)
    outputs = model(input_ids=prompt_ids.unsqueeze(0), use_cache=True)
    past = outputs.past_key_values
    last_logits = outputs.logits[0, -1]

    greedy_ids, resample_ids, logits_per_step = [], [], []
    next_token = int(last_logits.argmax().item())

    for _ in range(gen_len):
        logits_per_step.append(last_logits)
        greedy_ids.append(next_token)
        probs = torch.softmax(last_logits / temperature, dim=-1)
        resample_ids.append(int(torch.multinomial(probs, num_samples=1).item()))

        # Advance along the GREEDY path only - the resampled draw never
        # feeds back into the trajectory, so there is exactly one real
        # context at every step, never a second, diverging one.
        outputs = model(input_ids=torch.tensor([[next_token]]), past_key_values=past, use_cache=True)
        past = outputs.past_key_values
        last_logits = outputs.logits[0, -1]
        next_token = int(last_logits.argmax().item())

    return (
        torch.tensor(greedy_ids),
        torch.tensor(resample_ids),
        torch.stack(logits_per_step, dim=0),
    )


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

    all_stable, all_strict_correct, all_logits, all_offset = [], [], [], []
    late_agreement_tokens = []

    t0 = time.time()
    for w, start in enumerate(window_starts):
        full_window = full_ids[start : start + WINDOW_SIZE]
        prompt_ids = full_window[:PROMPT_LEN]
        real_continuation = full_window[PROMPT_LEN:]

        greedy_ids, resample_ids, greedy_logits = generate_with_local_stability(
            model, prompt_ids, GEN_LEN, TEMPERATURE, seed=w
        )
        locally_stable = greedy_ids == resample_ids
        strict_correct = greedy_ids == real_continuation

        late_mask = torch.arange(GEN_LEN) >= LATE_OFFSET_CUTOFF
        late_agreement_tokens.extend(greedy_ids[locally_stable & late_mask].tolist())

        all_stable.append(locally_stable)
        all_strict_correct.append(strict_correct)
        all_logits.append(greedy_logits)
        all_offset.append(torch.arange(GEN_LEN, dtype=torch.long))

        if w % 10 == 0:
            print(f"window {w + 1}/{n_windows}  elapsed={time.time() - t0:.0f}s")

    locally_stable = torch.cat(all_stable)
    strict_correct = torch.cat(all_strict_correct)
    logits_all = torch.cat(all_logits, dim=0)
    offset = torch.cat(all_offset)
    phi = featurize(logits_all, normalize_l2=True)
    msp = phi[:, 0]

    print(f"\ndone in {time.time() - t0:.1f}s")
    print(f"local-stability rate (resample == greedy, same context): {locally_stable.float().mean().item():.4f}")
    print(f"strict reference-match rate: {strict_correct.float().mean().item():.4f}")

    print("\n=== local stability vs strict reference-match, by offset ===")
    for step in [0, 1, 2, 3, 4, 5, 9, 19, 29, 49, 69, 99]:
        mask = offset == step
        if mask.sum() == 0:
            continue
        stab = locally_stable[mask].float().mean().item()
        strict = strict_correct[mask].float().mean().item()
        print(f"  offset={step:3d}  n={int(mask.sum())}  local_stability={stab:.4f}  strict_match={strict:.4f}")

    print("\n=== MSP AUROC discrimination ===")
    for name, labels in [("local stability", locally_stable), ("strict reference-match", strict_correct)]:
        pos, neg = msp[labels], msp[~labels]
        print(f"  {name}: n_pos={len(pos)} n_neg={len(neg)}")
        if len(pos) > 1 and len(neg) > 1:
            r = bootstrap_auroc_ci(pos, neg, n_bootstrap=2000, seed=0)
            print(f"    MSP AUROC = {r.auroc:.4f}  95% CI [{r.ci_lo:.4f}, {r.ci_hi:.4f}]")

    print(f"\n=== chance-matching confound check on LATE (offset>={LATE_OFFSET_CUTOFF}) agreements ===")
    print(f"n late agreements: {len(late_agreement_tokens)}")
    decoded = [tokenizer.decode([t]) for t in late_agreement_tokens]
    ctr = Counter(decoded)
    for tok, cnt in ctr.most_common(10):
        print(f"  {tok!r:15s}  count={cnt:3d}  ({cnt / len(late_agreement_tokens):.1%})")
    sorted_by_freq = [tok for tok, _ in token_counts.most_common()]
    rank_of = {tok: i for i, tok in enumerate(sorted_by_freq)}
    ranks = [rank_of.get(t, len(sorted_by_freq)) for t in late_agreement_tokens]
    top20_frac = sum(1 for r in ranks if r < 20) / len(ranks)
    baseline_ranks = [rank_of.get(t, len(sorted_by_freq)) for t in full_ids[:20000].tolist()]
    baseline_top20 = sum(1 for r in baseline_ranks if r < 20) / len(baseline_ranks)
    print(f"fraction in corpus top-20 most frequent tokens: {top20_frac:.1%}  (baseline: {baseline_top20:.1%})")

    print("\n=== circularity check: is 'local stability' just a restatement of MSP? ===")
    logp = torch.log_softmax(logits_all, dim=-1)
    q = torch.softmax(logp / TEMPERATURE, dim=-1)
    analytic_p = q.max(dim=-1).values  # closed-form P(resample == argmax), no sampling needed
    pos_a, neg_a = analytic_p[locally_stable], analytic_p[~locally_stable]
    r_analytic = bootstrap_auroc_ci(pos_a, neg_a, n_bootstrap=2000, seed=0)
    pos_m, neg_m = msp[locally_stable], msp[~locally_stable]
    r_msp_ls = bootstrap_auroc_ci(pos_m, neg_m, n_bootstrap=2000, seed=0)
    print(f"Analytic P(resample==argmax) AUROC for empirical local_stability: {r_analytic.auroc:.4f}  CI=[{r_analytic.ci_lo:.4f}, {r_analytic.ci_hi:.4f}]")
    print(f"MSP AUROC for empirical local_stability:                         {r_msp_ls.auroc:.4f}  CI=[{r_msp_ls.ci_lo:.4f}, {r_msp_ls.ci_hi:.4f}]")
    corr = torch.corrcoef(torch.stack([msp, analytic_p]))[0, 1].item()
    print(f"Pearson correlation(MSP, analytic P(resample==argmax)): {corr:.4f}")


if __name__ == "__main__":
    main()
