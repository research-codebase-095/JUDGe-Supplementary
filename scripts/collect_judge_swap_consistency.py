"""Swap-consistency LLM-judge reliability experiment, extending
scripts/collect_judge_verdicts.py's real judge run rather than replacing it.

Motivation (reviewer-flagged gap): MT-Bench pairwise judging has known
position bias (Zheng et al. 2023; Wang et al. 2024) - a judge can favor
whichever response happens to land in slot A (or slot B), independent of
actual quality. collect_judge_verdicts.py never controlled for this: it only
ever queries each pair in one fixed order. This script judges every one of
the same 1,284 filtered MT-Bench pairs in BOTH orders (original A/B, and
swapped B/A) and checks whether the verdict - correctly re-mapped back onto
the original response_a/response_b identities - agrees across the swap. That
agreement ("swap-consistency") is a cheap, zero-extra-labels reliability
signal: examples where flipping the slots flips the verdict are exactly the
examples a position-biased judge is least trustworthy on.

Reuses collect_judge_verdicts.py's load_filtered_dataset() (identical
row order/set), build_prompt() (identical prompt template, called twice per
example - once per slot ordering), TOKEN_ID_A/TOKEN_ID_B (identical verdict
extraction), verdict_logits() (identical one-forward-pass logit extraction),
and MODEL_NAME (same frozen Qwen2.5-0.5B-Instruct judge) by importing that
module directly - not re-deriving any of this, so the two runs are directly
comparable rather than a different methodology in each.

Verdict re-mapping for the swapped pass: build_prompt(tokenizer, instruction,
response_b, response_a) presents the ORIGINAL response_b in the "Response A"
slot and the ORIGINAL response_a in the "Response B" slot. So if the model's
swapped-pass logits favor token "A", it picked the original response_b, and
vice versa - the swapped-pass predicted winner (in terms of the ORIGINAL a/b
identities) is the mirror image of the unswapped rule.

Also reuses judge_feature_cache_mtbench.pt's `splits` field directly (loaded
via torch.load, not re-derived) so the 6-feature combiner is fit/evaluated on
the EXACT same combiner_fit/id_test row partition as the original 5-feature
combiner - required for the DeLong paired comparison between them to be valid
(DeLong needs the same underlying examples for both scores).

Checkpointing mirrors collect_judge_verdicts.py's pattern exactly (see that
script's main() docstring note): this run does 2x the forward passes of the
original (two orders per example instead of one), which already needed
checkpointing to survive the environment's background-task kill behavior, so
checkpointing here is not optional.

Usage:
    python scripts/collect_judge_swap_consistency.py
"""

import argparse
import json
import os
import sys
import time

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from deployment_reliability.combiner import LogisticRegressionCombiner  # noqa: E402
from deployment_reliability.features import featurize  # noqa: E402
from deployment_reliability.llm_backbone import load_frozen_causal_lm  # noqa: E402
from deployment_reliability.router import auroc  # noqa: E402
from deployment_reliability.significance import delong_test  # noqa: E402

import collect_judge_verdicts as cjv  # noqa: E402 - reuse build_prompt/load_filtered_dataset/TOKEN_ID_A/B/verdict_logits/MODEL_NAME

DATA_DIR = os.path.join(REPO_ROOT, "data")
CHECKPOINT_PATH = os.path.join(DATA_DIR, "_judge_checkpoint_swap_consistency.pt")
CHECKPOINT_EVERY = 100  # examples between checkpoint saves - matches collect_judge_verdicts.py


def main(limit: int | None) -> None:
    out_cache_path = os.path.join(DATA_DIR, "judge_swap_consistency_cache.pt")
    out_results_path = os.path.join(DATA_DIR, "judge_swap_consistency_results.json")
    existing_cache_path = os.path.join(DATA_DIR, "judge_feature_cache_mtbench.pt")
    existing_results_path = os.path.join(DATA_DIR, "judge_experiment_results.json")

    filtered = cjv.load_filtered_dataset()
    rows = list(filtered)
    if limit is not None:
        rows = rows[:limit]
        print(f"using a fixed-size subset of {len(rows)} examples (--limit {limit})")
    else:
        print(f"using the full filtered set: {len(rows)} examples")

    # Load the ORIGINAL run's cache up front (not just at the end) so we can
    # sanity-check row alignment (same question_id sequence) before spending
    # any compute, and so its `splits` field is available for the 6-feature
    # combiner fit later without re-deriving a new random split.
    existing_cache = torch.load(existing_cache_path)
    with open(existing_results_path, "r", encoding="utf-8") as f:
        existing_results = json.load(f)

    if limit is None:
        if len(existing_cache["question_id"]) != len(rows):
            raise RuntimeError(
                f"existing cache has {len(existing_cache['question_id'])} rows but this run's "
                f"filtered set has {len(rows)} rows - load_filtered_dataset() logic has diverged "
                "between the two scripts; cannot reuse `splits` safely."
            )
        for i, row in enumerate(rows):
            if row["question_id"] != existing_cache["question_id"][i]:
                raise RuntimeError(
                    f"row {i}: question_id mismatch between this run ({row['question_id']}) and "
                    f"the existing cache ({existing_cache['question_id'][i]}) - row order has diverged."
                )
        print("row alignment against judge_feature_cache_mtbench.pt verified: "
              f"{len(rows)} question_ids match exactly, in order")

    # Checkpointing: identical pattern to collect_judge_verdicts.py. Keyed
    # only by how many rows have been processed - `rows` is a deterministic
    # prefix of the same HF dataset filter every time, so resuming is only
    # valid if `--limit` is unchanged between runs of the same checkpoint file.
    all_phi_orig = []
    all_correct_orig, all_correct_swapped, all_swap_consistent = [], [], []
    predicted_winner_orig, predicted_winner_swapped = [], []
    human_winner, question_id = [], []
    start_idx = 0
    if os.path.exists(CHECKPOINT_PATH):
        ckpt = torch.load(CHECKPOINT_PATH)
        if ckpt.get("total_rows") == len(rows):
            all_phi_orig = list(ckpt["phi_orig"].unbind(0)) if len(ckpt["correct_orig"]) > 0 else []
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

        # Pass 1: original order, identical to collect_judge_verdicts.py.
        prompt_orig = cjv.build_prompt(tokenizer, instruction, response_a, response_b)
        logits_orig = cjv.verdict_logits(model, tokenizer, prompt_orig)
        pred_orig = "model_a" if logits_orig[cjv.TOKEN_ID_A].item() > logits_orig[cjv.TOKEN_ID_B].item() else "model_b"

        # Pass 2: swapped order - original response_b is now presented as
        # "Response A", original response_a as "Response B". A verdict of
        # "A" in this pass therefore points at the ORIGINAL response_b, and
        # a verdict of "B" points at the ORIGINAL response_a - the mirror
        # image of pass 1's mapping.
        prompt_swap = cjv.build_prompt(tokenizer, instruction, response_b, response_a)
        logits_swap = cjv.verdict_logits(model, tokenizer, prompt_swap)
        pred_swap = "model_b" if logits_swap[cjv.TOKEN_ID_A].item() > logits_swap[cjv.TOKEN_ID_B].item() else "model_a"

        correct_orig = pred_orig == row["winner"]
        correct_swap = pred_swap == row["winner"]
        swap_consistent = pred_orig == pred_swap  # both already expressed in original a/b identities

        phi_orig = featurize(logits_orig, normalize_l2=True)

        all_phi_orig.append(phi_orig)
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

    phi = torch.stack(all_phi_orig, dim=0)
    correct_orig_t = torch.tensor(all_correct_orig, dtype=torch.bool)
    correct_swap_t = torch.tensor(all_correct_swapped, dtype=torch.bool)
    swap_consistent_t = torch.tensor(all_swap_consistent, dtype=torch.float32)  # 0/1 used directly as a score
    n = len(correct_orig_t)

    print(f"done in {time.time() - t0:.1f}s. total examples: {n}")

    if limit is None:
        # Determinism sanity check: this run's fresh forward passes over the
        # identical original-order prompts should reproduce the ORIGINAL
        # run's correctness labels exactly (eval mode, no dropout, CPU/
        # deterministic ops, identical inputs) - a real check, not assumed.
        matches = torch.equal(correct_orig_t, existing_cache["correct"])
        print(f"determinism check vs judge_feature_cache_mtbench.pt's `correct`: "
              f"{'MATCH' if matches else 'MISMATCH'} "
              f"({int((correct_orig_t == existing_cache['correct']).sum().item())}/{n} agree)")

    swap_consistency_rate = swap_consistent_t.mean().item()
    original_order_accuracy = correct_orig_t.float().mean().item()
    swapped_order_accuracy = correct_swap_t.float().mean().item()

    orig_a_frac = sum(1 for p in predicted_winner_orig if p == "model_a") / n
    swap_a_frac = sum(1 for p in predicted_winner_swapped if p == "model_a") / n
    print(f"predicted_winner_orig: model_a fraction={orig_a_frac:.4f}  model_b fraction={1 - orig_a_frac:.4f}")
    print(f"predicted_winner_swapped: model_a fraction={swap_a_frac:.4f}  model_b fraction={1 - swap_a_frac:.4f}")

    # The actual test: does swap-consistency predict original-order
    # correctness? swap_consistent_t in {0,1} used directly as the score,
    # separated into the two outcome groups router.auroc expects (pos =
    # scores of examples EXPECTED to score higher, i.e. the correct group).
    pos_scores = swap_consistent_t[correct_orig_t]
    neg_scores = swap_consistent_t[~correct_orig_t]
    swap_consistency_auroc = auroc(pos_scores, neg_scores)

    # 6-feature combiner: existing 5 features (original-order logits) + swap
    # consistency as a 6th feature, fit/evaluated on the EXACT SAME
    # combiner_fit/id_test row indices as the original 5-feature run (reused
    # from judge_feature_cache_mtbench.pt's `splits`, not re-derived).
    splits = existing_cache["splits"]
    if limit is None and len(splits) != n:
        raise RuntimeError(f"existing cache splits length {len(splits)} != this run's n={n}")
    # When --limit truncates `rows` below the full 1284, indices from the
    # full-size `splits` beyond the truncated range are dropped rather than
    # indexed out of bounds - this only matters for small ad hoc --limit
    # smoke tests of this script's plumbing (never for the real, full run,
    # where every split index is already < n by construction).
    combiner_fit_idx = torch.tensor([i for i, s in enumerate(splits) if s == "combiner_fit" and i < n], dtype=torch.long)
    id_test_idx = torch.tensor([i for i, s in enumerate(splits) if s == "id_test" and i < n], dtype=torch.long)

    phi6 = torch.cat([phi, swap_consistent_t.unsqueeze(-1)], dim=-1)
    phi6_fit, correct6_fit = phi6[combiner_fit_idx], correct_orig_t[combiner_fit_idx].float()
    phi6_test, correct6_test = phi6[id_test_idx], correct_orig_t[id_test_idx]

    combiner6 = LogisticRegressionCombiner().fit(phi6_fit, correct6_fit)
    combiner6_scores_test = combiner6.score(phi6_test)
    pos6 = combiner6_scores_test[correct6_test]
    neg6 = combiner6_scores_test[~correct6_test]
    combiner_6feature_auroc_id_test = auroc(pos6, neg6)

    # Refit the ORIGINAL 5-feature combiner on the existing cache's own phi,
    # using the SAME combiner_fit/id_test indices, so we have its per-example
    # id_test score vector to pair against the 6-feature combiner's for
    # DeLong (judge_experiment_results.json only stored the summary AUROC,
    # not the score array itself). Deterministic LBFGS fit on identical data
    # -> reproduces the original 0.5735 AUROC (checked below, not assumed).
    phi5_fit = existing_cache["phi"][combiner_fit_idx]
    correct5_fit = existing_cache["correct"][combiner_fit_idx].float()
    phi5_test = existing_cache["phi"][id_test_idx]
    correct5_test = existing_cache["correct"][id_test_idx]
    combiner5 = LogisticRegressionCombiner().fit(phi5_fit, correct5_fit)
    combiner5_scores_test = combiner5.score(phi5_test)
    pos5 = combiner5_scores_test[correct5_test]
    neg5 = combiner5_scores_test[~correct5_test]
    combiner_5feature_auroc_reproduced = auroc(pos5, neg5)
    print(f"reproduced 5-feature combiner id_test AUROC: {combiner_5feature_auroc_reproduced:.4f} "
          f"(originally recorded: {existing_results['combiner_auroc_id_test']:.4f})")

    # DeLong paired test: 6-feature combiner vs original 5-feature combiner,
    # both scored on the identical id_test examples/correctness labels.
    if not torch.equal(correct6_test, correct5_test):
        raise RuntimeError("correctness labels for id_test disagree between this run and the existing cache - "
                            "cannot run a valid paired DeLong test.")
    delong = delong_test(correct6_test, combiner6_scores_test, combiner5_scores_test)

    if delong.p_value < 0.05:
        combiner_winner = "6-feature (+ swap-consistency)" if delong.auc_a > delong.auc_b else "5-feature (original)"
    else:
        combiner_winner = "no significant difference"

    print()
    print("=== Results summary ===")
    print(f"{'total examples':<40} {n}")
    print(f"{'swap-consistency rate':<40} {swap_consistency_rate:.4f}")
    print(f"{'original-order accuracy':<40} {original_order_accuracy:.4f}")
    print(f"{'swapped-order accuracy':<40} {swapped_order_accuracy:.4f}")
    print(f"{'swap-consistency AUROC (predicts correctness)':<40} {swap_consistency_auroc:.4f}")
    print(f"{'6-feature combiner AUROC (id_test)':<40} {combiner_6feature_auroc_id_test:.4f}")
    print(f"{'5-feature combiner AUROC (id_test, reproduced)':<40} {combiner_5feature_auroc_reproduced:.4f}")
    print(f"{'5-feature combiner AUROC (id_test, recorded)':<40} {existing_results['combiner_auroc_id_test']:.4f}")
    print(f"{'DeLong z (6-feature vs 5-feature)':<40} {delong.z:.4f}")
    print(f"{'DeLong p-value':<40} {delong.p_value:.4g}")
    print(f"{'combiner winner':<40} {combiner_winner}")

    torch.save(
        {
            "phi_orig": phi,
            "correct_orig": correct_orig_t,
            "correct_swapped": correct_swap_t,
            "swap_consistent": swap_consistent_t,
            "splits": splits,
            "predicted_winner_orig": predicted_winner_orig,
            "predicted_winner_swapped": predicted_winner_swapped,
            "human_winner": human_winner,
            "question_id": question_id,
        },
        out_cache_path,
    )
    print()
    print("saved per-example cache to", out_cache_path)

    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)
        print("run completed - removed intermediate checkpoint", CHECKPOINT_PATH)

    results = {
        "judge_model": cjv.MODEL_NAME,
        "dataset": "lmsys/mt_bench_human_judgments",
        "dataset_split": "human",
        "filter": "turn==1 and winner in {model_a, model_b}",
        "n_examples": n,
        "n_combiner_fit": len(combiner_fit_idx),
        "n_id_test": len(id_test_idx),
        "swap_consistency_rate": swap_consistency_rate,
        "original_order_accuracy": original_order_accuracy,
        "swapped_order_accuracy": swapped_order_accuracy,
        "swap_consistency_auroc": swap_consistency_auroc,
        "combiner_6feature_auroc_id_test": combiner_6feature_auroc_id_test,
        "combiner_5feature_auroc_id_test": existing_results["combiner_auroc_id_test"],
        "combiner_5feature_auroc_id_test_reproduced": combiner_5feature_auroc_reproduced,
        "delong_6feature_vs_5feature_z": delong.z,
        "delong_6feature_vs_5feature_p_value": delong.p_value,
        "combiner_winner": combiner_winner,
    }
    with open(out_results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print()
    print("saved results summary to", out_results_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit", type=int, default=None,
        help="use only the first N filtered examples (fixed-size subset, chosen up front) instead of the full filtered set",
    )
    args = parser.parse_args()
    main(args.limit)
