"""Reviewer follow-up: judge2026.tex's order-averaging check (order_averaging_correction.py)
found that a BINARY-verdict order-average at 0.5B is uninformative (swap
orders agree on only 1.1% of pairs, so any tie-break rule collapses to
within 2 points of the position-following baseline), and the paper's Full
Limitations item explicitly says the CONTINUOUS-score version (averaging
both orders' raw MSP before thresholding, not just combining the two binary
verdicts) is not reconstructable from the existing swap-consistency cache,
because the original collect_judge_swap_consistency.py run persisted only
phi_orig (the original-order 5-feature vector) and the resulting binary
verdict for the swapped order, not phi_swapped itself.

This script closes that gap: it is collect_judge_swap_consistency.py with
ONE addition -- featurizing and persisting phi_swapped alongside phi_orig --
so a genuine continuous-score order-average (mean or other combination of
phi_orig's and phi_swapped's MSP, oriented back onto the ORIGINAL response
identities) becomes computable afterward without any further inference.

This does require a fresh 2-pass (original + swapped order) forward-pass run
over all 1,284 MT-Bench pairs at the 0.5B judge -- the raw swapped-order
logits were never saved by the original run, only the final verdict, so
there is no way to get phi_swapped without rerunning inference. Reuses
collect_judge_verdicts.py's build_prompt/load_filtered_dataset/TOKEN_ID_A/B/
verdict_logits/MODEL_NAME exactly as collect_judge_swap_consistency.py does,
so results should exactly reproduce phi_orig/correct_orig/swap_consistent
already in judge_swap_consistency_cache.pt (checked at the end via the same
determinism check that script already runs), while additionally saving
phi_swapped.

Usage:
    python scripts/collect_judge_swap_consistency_continuous.py
"""

from __future__ import annotations

import json
import os
import sys
import time

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from deployment_reliability.features import featurize  # noqa: E402
from deployment_reliability.llm_backbone import load_frozen_causal_lm  # noqa: E402

import collect_judge_verdicts as cjv  # noqa: E402

DATA_DIR = os.path.join(REPO_ROOT, "data")
CHECKPOINT_PATH = os.path.join(DATA_DIR, "_judge_checkpoint_swap_consistency_continuous.pt")
CHECKPOINT_EVERY = 100


def main(limit: int | None = None) -> None:
    out_cache_path = os.path.join(DATA_DIR, "judge_swap_consistency_cache_continuous.pt")
    existing_cache_path = os.path.join(DATA_DIR, "judge_feature_cache_mtbench.pt")

    filtered = cjv.load_filtered_dataset()
    rows = list(filtered)
    if limit is not None:
        rows = rows[:limit]
        print(f"using a fixed-size subset of {len(rows)} examples (--limit {limit})")
    else:
        print(f"using the full filtered set: {len(rows)} examples")

    existing_cache = torch.load(existing_cache_path)
    if limit is None:
        if len(existing_cache["question_id"]) != len(rows):
            raise RuntimeError(
                f"existing cache has {len(existing_cache['question_id'])} rows but this run's "
                f"filtered set has {len(rows)} rows."
            )
        for i, row in enumerate(rows):
            if row["question_id"] != existing_cache["question_id"][i]:
                raise RuntimeError(f"row {i}: question_id mismatch.")
        print("row alignment against judge_feature_cache_mtbench.pt verified: "
              f"{len(rows)} question_ids match exactly, in order")

    all_phi_orig, all_phi_swapped = [], []
    all_correct_orig, all_correct_swapped, all_swap_consistent = [], [], []
    predicted_winner_orig, predicted_winner_swapped = [], []
    human_winner, question_id = [], []
    start_idx = 0
    if os.path.exists(CHECKPOINT_PATH):
        ckpt = torch.load(CHECKPOINT_PATH)
        if ckpt.get("total_rows") == len(rows):
            all_phi_orig = list(ckpt["phi_orig"].unbind(0)) if len(ckpt["correct_orig"]) > 0 else []
            all_phi_swapped = list(ckpt["phi_swapped"].unbind(0)) if len(ckpt["correct_orig"]) > 0 else []
            all_correct_orig = list(ckpt["correct_orig"])
            all_correct_swapped = list(ckpt["correct_swapped"])
            all_swap_consistent = list(ckpt["swap_consistent"])
            predicted_winner_orig = ckpt["predicted_winner_orig"]
            predicted_winner_swapped = ckpt["predicted_winner_swapped"]
            human_winner = ckpt["human_winner"]
            question_id = ckpt["question_id"]
            start_idx = len(all_correct_orig)
            print(f"resuming from checkpoint {CHECKPOINT_PATH}: {start_idx}/{len(rows)} examples already done")
        else:
            print(f"checkpoint at {CHECKPOINT_PATH} was for a different row count "
                  f"({ckpt.get('total_rows')} != {len(rows)}) - ignoring it, starting fresh")

    print(f"loading frozen {cjv.MODEL_NAME}...")
    model, tokenizer = load_frozen_causal_lm(cjv.MODEL_NAME)

    def save_checkpoint() -> None:
        torch.save(
            {
                "phi_orig": torch.stack(all_phi_orig, dim=0) if all_phi_orig else torch.empty(0, 5),
                "phi_swapped": torch.stack(all_phi_swapped, dim=0) if all_phi_swapped else torch.empty(0, 5),
                "correct_orig": all_correct_orig,
                "correct_swapped": all_correct_swapped,
                "swap_consistent": all_swap_consistent,
                "predicted_winner_orig": predicted_winner_orig,
                "predicted_winner_swapped": predicted_winner_swapped,
                "human_winner": human_winner,
                "question_id": question_id,
                "total_rows": len(rows),
            },
            CHECKPOINT_PATH,
        )

    t0 = time.time()
    for i in range(start_idx, len(rows)):
        row = rows[i]
        instruction = row["conversation_a"][0]["content"]
        response_a = row["conversation_a"][1]["content"]
        response_b = row["conversation_b"][1]["content"]

        prompt_orig = cjv.build_prompt(tokenizer, instruction, response_a, response_b)
        logits_orig = cjv.verdict_logits(model, tokenizer, prompt_orig)
        pred_orig = "model_a" if logits_orig[cjv.TOKEN_ID_A].item() > logits_orig[cjv.TOKEN_ID_B].item() else "model_b"

        prompt_swap = cjv.build_prompt(tokenizer, instruction, response_b, response_a)
        logits_swap = cjv.verdict_logits(model, tokenizer, prompt_swap)
        pred_swap = "model_b" if logits_swap[cjv.TOKEN_ID_A].item() > logits_swap[cjv.TOKEN_ID_B].item() else "model_a"

        correct_orig = pred_orig == row["winner"]
        correct_swap = pred_swap == row["winner"]
        swap_consistent = pred_orig == pred_swap

        phi_orig = featurize(logits_orig, normalize_l2=True)
        phi_swapped = featurize(logits_swap, normalize_l2=True)

        all_phi_orig.append(phi_orig)
        all_phi_swapped.append(phi_swapped)
        all_correct_orig.append(correct_orig)
        all_correct_swapped.append(correct_swap)
        all_swap_consistent.append(swap_consistent)
        predicted_winner_orig.append(pred_orig)
        predicted_winner_swapped.append(pred_swap)
        human_winner.append(row["winner"])
        question_id.append(row["question_id"])

        if (i + 1) % 50 == 0 or (i + 1) == len(rows):
            elapsed = time.time() - t0
            running_acc_orig = sum(all_correct_orig) / len(all_correct_orig)
            running_swap_rate = sum(all_swap_consistent) / len(all_swap_consistent)
            print(f"{i + 1}/{len(rows)}  elapsed={elapsed:.0f}s  "
                  f"running orig accuracy={running_acc_orig:.4f}  running swap-consistency rate={running_swap_rate:.4f}")
        if (i + 1) % CHECKPOINT_EVERY == 0:
            save_checkpoint()
            print(f"checkpoint saved at {i + 1}/{len(rows)}")

    phi_orig_t = torch.stack(all_phi_orig, dim=0)
    phi_swapped_t = torch.stack(all_phi_swapped, dim=0)
    correct_orig_t = torch.tensor(all_correct_orig, dtype=torch.bool)
    correct_swap_t = torch.tensor(all_correct_swapped, dtype=torch.bool)
    swap_consistent_t = torch.tensor(all_swap_consistent, dtype=torch.float32)
    n = len(correct_orig_t)

    print(f"done in {time.time() - t0:.1f}s. total examples: {n}")

    if limit is None:
        matches = torch.equal(correct_orig_t, existing_cache["correct"])
        print(f"determinism check vs judge_feature_cache_mtbench.pt's `correct`: "
              f"{'MATCH' if matches else 'MISMATCH'} "
              f"({int((correct_orig_t == existing_cache['correct']).sum().item())}/{n} agree)")

    torch.save(
        {
            "phi_orig": phi_orig_t, "phi_swapped": phi_swapped_t,
            "correct_orig": correct_orig_t, "correct_swapped": correct_swap_t,
            "swap_consistent": swap_consistent_t,
            "splits": existing_cache["splits"] if limit is None else None,
            "predicted_winner_orig": predicted_winner_orig, "predicted_winner_swapped": predicted_winner_swapped,
            "human_winner": human_winner, "question_id": question_id,
        },
        out_cache_path,
    )
    print(f"saved {out_cache_path}")
    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)
        print("run completed - removed intermediate checkpoint", CHECKPOINT_PATH)


if __name__ == "__main__":
    main(limit=None)
