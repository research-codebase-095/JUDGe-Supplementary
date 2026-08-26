"""Repeats scripts/collect_judge_verdicts.py's real LLM-judge reliability
experiment with a substantially stronger judge model. The 0.5B run (judge
accuracy 0.515, barely above chance) can't distinguish "our method doesn't
help" from "there's no judge skill here to predict" - this script reruns the
identical pipeline (same dataset/filter, same prompt template, same
token-id-extraction approach, same featurize() call, same three_way_split
protocol, same DeLong-test evaluation) with a bigger, more capable judge
model, so the reliability-estimation question has real judge skill to work
with. Only the model name (and, correspondingly, the confirmed verdict
token ids and checkpoint path) changes.

Default judge model: Qwen/Qwen2.5-7B-Instruct. If that model proves
genuinely infeasible to load/run in this environment (not just slow - a
hard memory/disk/download failure that persists across retries), pass
--model Qwen/Qwen2.5-3B-Instruct explicitly as the documented fallback;
this script does not silently substitute a smaller model on its own.

Usage:
    python scripts/collect_judge_verdicts_7b.py [--model Qwen/Qwen2.5-7B-Instruct] [--limit N]
"""

import argparse
import json
import os
import re
import sys
import time

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from deployment_reliability.combiner import LogisticRegressionCombiner  # noqa: E402
from deployment_reliability.features import featurize  # noqa: E402
from deployment_reliability.llm_backbone import load_frozen_causal_lm  # noqa: E402
from deployment_reliability.significance import delong_test  # noqa: E402
from deployment_reliability.splits import three_way_split  # noqa: E402

DATA_DIR = os.path.join(REPO_ROOT, "data")
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

# Verdict token ids - re-verified for THIS script's tokenizer(s), not assumed
# from the 0.5B run. Confirmed by direct encode() calls against the actual
# Qwen2.5-7B-Instruct and Qwen2.5-3B-Instruct tokenizers (both Qwen2.5-family,
# same tokenizer as the 0.5B model): tokenizer.encode("A", add_special_tokens=False)
# == [32], tokenizer.encode("B", add_special_tokens=False) == [33] for both -
# identical to the 0.5B run's TOKEN_ID_A/TOKEN_ID_B. Also re-confirmed by
# sanity_check_tokens() below (the same top-5-logit check the 0.5B script
# runs) before the full loop starts for whichever model is actually loaded.
TOKEN_ID_A = 32
TOKEN_ID_B = 33

JUDGE_INSTRUCTIONS = (
    "You are comparing two AI assistant responses to the same instruction. "
    "Decide which response is better overall.\n\n"
    "Instruction:\n{instruction}\n\n"
    "Response A:\n{response_a}\n\n"
    "Response B:\n{response_b}\n\n"
    "Which response is better, A or B? Answer with a single letter, A or B."
)


def build_prompt(tokenizer, instruction: str, response_a: str, response_b: str) -> str:
    user_content = JUDGE_INSTRUCTIONS.format(instruction=instruction, response_a=response_a, response_b=response_b)
    messages = [{"role": "user", "content": user_content}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@torch.inference_mode()
def verdict_logits(model, tokenizer, prompt_text: str) -> torch.Tensor:
    """One forward pass; returns the raw logit vector (vocab_size,) at the
    final prompt position - the verdict token's next-token distribution,
    the LLM-judge-domain counterpart of collect_llm_logits.py's logits[:-1]
    next-token setup, here for a single position instead of a whole chunk."""
    input_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"]
    outputs = model(input_ids=input_ids)
    return outputs.logits[0, -1, :]


def sanity_check_tokens(model, tokenizer, rows, n_check: int = 5) -> None:
    """Prints the top-5 highest-probability tokens at the verdict position
    for a handful of real examples, so the TOKEN_ID_A/TOKEN_ID_B assumption
    is checked against actual model behavior before being trusted for the
    full run - not just assumed from the standalone encode() check."""
    print(f"=== Sanity check: top-5 verdict-position tokens for {n_check} examples ===")
    for i in range(min(n_check, len(rows))):
        row = rows[i]
        instruction = row["conversation_a"][0]["content"]
        response_a = row["conversation_a"][1]["content"]
        response_b = row["conversation_b"][1]["content"]
        prompt = build_prompt(tokenizer, instruction, response_a, response_b)
        logits = verdict_logits(model, tokenizer, prompt)
        probs = torch.softmax(logits, dim=-1)
        top5 = torch.topk(probs, k=5)
        print(f"-- example {i} (question_id={row['question_id']}) --")
        for rank in range(5):
            tid = int(top5.indices[rank].item())
            p = float(top5.values[rank].item())
            tok_str = tokenizer.decode([tid])
            flag = "  <-- A/B" if tid in (TOKEN_ID_A, TOKEN_ID_B) else ""
            print(f"    #{rank + 1}: token_id={tid:>6}  token={tok_str!r:<12}  p={p:.4f}{flag}")
        print(f"    p(A)={probs[TOKEN_ID_A].item():.4f}  p(B)={probs[TOKEN_ID_B].item():.4f}")
    print()


def load_filtered_dataset():
    from datasets import load_dataset

    ds = load_dataset("lmsys/mt_bench_human_judgments", split="human")
    print(f"loaded lmsys/mt_bench_human_judgments 'human' split: {len(ds)} rows")
    filtered = ds.filter(lambda r: r["turn"] == 1 and r["winner"] in ("model_a", "model_b"))
    print(f"filtered to turn==1 and winner in {{model_a, model_b}}: {len(filtered)} rows")
    return filtered


CHECKPOINT_EVERY = 100  # examples between checkpoint saves - see main()'s docstring note on why this exists


def model_tag(model_name: str) -> str:
    """Filesystem-safe short tag for a HF model name, used only to keep this
    script's checkpoint file distinct per judge model - e.g. so a partial
    7B-model checkpoint can never be mistakenly resumed against a 3B fallback
    run (or vice versa) just because the row count happens to match."""
    return re.sub(r"[^a-zA-Z0-9]+", "_", model_name).strip("_").lower()


def main(model_name: str, limit: int | None) -> None:
    out_cache_path = os.path.join(DATA_DIR, "judge_feature_cache_mtbench_7b.pt")
    out_results_path = os.path.join(DATA_DIR, "judge_experiment_results_7b.json")
    checkpoint_path = os.path.join(DATA_DIR, f"_judge_checkpoint_mtbench_7b_{model_tag(model_name)}.pt")

    filtered = load_filtered_dataset()
    rows = list(filtered)
    if limit is not None:
        rows = rows[:limit]
        print(f"using a fixed-size subset of {len(rows)} examples (--limit {limit})")
    else:
        print(f"using the full filtered set: {len(rows)} examples")

    # Checkpointing (identical pattern to collect_judge_verdicts.py, added
    # from the start this time rather than after a first crash): state is
    # keyed only by *how many rows have been processed* (`rows` is a
    # deterministic prefix of the same HF dataset filter every time), so
    # resuming is only valid if `--limit` and `--model` are unchanged between
    # runs against the same checkpoint file - the model-tagged checkpoint
    # filename above enforces the `--model` half of that.
    all_phi, all_correct, predicted_winner, human_winner, question_id = [], [], [], [], []
    start_idx = 0
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path)
        if ckpt.get("total_rows") == len(rows):
            all_phi = list(ckpt["phi"].unbind(0)) if len(ckpt["correct"]) > 0 else []
            all_correct = list(ckpt["correct"])
            predicted_winner = ckpt["predicted_winner"]
            human_winner = ckpt["human_winner"]
            question_id = ckpt["question_id"]
            start_idx = len(all_correct)
            print(f"resuming from checkpoint {checkpoint_path}: {start_idx}/{len(rows)} examples already done")
        else:
            print(f"checkpoint at {checkpoint_path} was for a different row count "
                  f"({ckpt.get('total_rows')} != {len(rows)}) - ignoring it, starting fresh")

    print(f"loading frozen {model_name}...")
    model, tokenizer = load_frozen_causal_lm(model_name)

    if start_idx == 0:
        sanity_check_tokens(model, tokenizer, rows, n_check=5)

    def save_checkpoint() -> None:
        torch.save(
            {
                "phi": torch.stack(all_phi, dim=0) if all_phi else torch.empty(0, 5),
                "correct": all_correct,
                "predicted_winner": predicted_winner,
                "human_winner": human_winner,
                "question_id": question_id,
                "total_rows": len(rows),
            },
            checkpoint_path,
        )

    t0 = time.time()
    for i in range(start_idx, len(rows)):
        row = rows[i]
        instruction = row["conversation_a"][0]["content"]
        response_a = row["conversation_a"][1]["content"]
        response_b = row["conversation_b"][1]["content"]
        prompt = build_prompt(tokenizer, instruction, response_a, response_b)
        logits = verdict_logits(model, tokenizer, prompt)

        pred = "model_a" if logits[TOKEN_ID_A].item() > logits[TOKEN_ID_B].item() else "model_b"
        correct = pred == row["winner"]

        phi = featurize(logits, normalize_l2=True)
        all_phi.append(phi)
        all_correct.append(correct)
        predicted_winner.append(pred)
        human_winner.append(row["winner"])
        question_id.append(row["question_id"])

        if (i + 1) % 50 == 0 or (i + 1) == len(rows):
            elapsed = time.time() - t0
            running_acc = sum(all_correct) / len(all_correct)
            print(f"{i + 1}/{len(rows)}  elapsed={elapsed:.0f}s  running accuracy={running_acc:.4f}")
        if (i + 1) % CHECKPOINT_EVERY == 0:
            save_checkpoint()
            print(f"checkpoint saved at {i + 1}/{len(rows)}")

    phi = torch.stack(all_phi, dim=0)
    correct = torch.tensor(all_correct, dtype=torch.bool)
    n = len(correct)
    overall_accuracy = correct.float().mean().item()
    print(f"done in {time.time() - t0:.1f}s. total examples: {n}  overall judge accuracy: {overall_accuracy:.4f}")

    if overall_accuracy in (0.5, 1.0):
        print("WARNING: overall accuracy is exactly 0.5 or 1.0 - this is a red flag for broken "
              "token-extraction logic, not a real finding. Inspect the sanity-check output above before trusting this run.")

    combiner_idx, cal_idx, test_idx = three_way_split(n, seed=0)
    splits = [""] * n
    for idx, name in ((combiner_idx, "combiner_fit"), (cal_idx, "threshold_cal"), (test_idx, "id_test")):
        for i in idx.tolist():
            splits[i] = name

    torch.save(
        {
            "phi": phi,
            "correct": correct,
            "splits": splits,
            "predicted_winner": predicted_winner,
            "human_winner": human_winner,
            "question_id": question_id,
        },
        out_cache_path,
    )
    print("saved cache to", out_cache_path)

    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print("run completed - removed intermediate checkpoint", checkpoint_path)

    # Fit on combiner_fit, evaluate MSP vs combiner AUROC on id_test - exactly
    # make_paper_figures.py's evaluate_llm protocol, identical to the 0.5B run.
    phi_fit, correct_fit = phi[combiner_idx], correct[combiner_idx].float()
    phi_test, correct_test = phi[test_idx], correct[test_idx]

    combiner = LogisticRegressionCombiner().fit(phi_fit, correct_fit)
    msp_scores_test = phi_test[:, 0]  # column 0 = msp, per features.DEFAULT_FEATURE_NAMES
    combiner_scores_test = combiner.score(phi_test)

    delong = delong_test(correct_test, combiner_scores_test, msp_scores_test)
    msp_auroc, combiner_auroc = delong.auc_b, delong.auc_a

    if delong.p_value < 0.05:
        winner = "combiner" if combiner_auroc > msp_auroc else "MSP"
    else:
        winner = "no significant difference"

    print()
    print("=== Results summary ===")
    print(f"{'judge model':<28} {model_name}")
    print(f"{'total examples used':<28} {n}")
    print(f"{'combiner_fit / threshold_cal / id_test':<28} {len(combiner_idx)} / {len(cal_idx)} / {len(test_idx)}")
    print(f"{'overall judge accuracy':<28} {overall_accuracy:.4f}")
    print(f"{'MSP AUROC (id_test)':<28} {msp_auroc:.4f}")
    print(f"{'Combiner AUROC (id_test)':<28} {combiner_auroc:.4f}")
    print(f"{'DeLong p-value':<28} {delong.p_value:.4g}")
    print(f"{'DeLong z':<28} {delong.z:.4f}")
    print(f"{'winner':<28} {winner}")

    results = {
        "judge_model": model_name,
        "requested_judge_model": DEFAULT_MODEL_NAME,
        "fallback_used": model_name != DEFAULT_MODEL_NAME,
        "dataset": "lmsys/mt_bench_human_judgments",
        "dataset_split": "human",
        "filter": "turn==1 and winner in {model_a, model_b}",
        "n_total_examples": n,
        "n_combiner_fit": len(combiner_idx),
        "n_threshold_cal": len(cal_idx),
        "n_id_test": len(test_idx),
        "overall_judge_accuracy": overall_accuracy,
        "msp_auroc_id_test": msp_auroc,
        "combiner_auroc_id_test": combiner_auroc,
        "delong_z": delong.z,
        "delong_p_value": delong.p_value,
        "winner": winner,
    }
    with open(out_results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print()
    print("saved results summary to", out_results_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL_NAME,
        help="judge model to load (default: Qwen/Qwen2.5-7B-Instruct; documented fallback: Qwen/Qwen2.5-3B-Instruct)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="use only the first N filtered examples (fixed-size subset, chosen up front) instead of the full filtered set",
    )
    args = parser.parse_args()
    main(args.model, args.limit)
