"""Runs a genuine LLM-as-judge reliability experiment on real human-labeled
judge data (lmsys/mt_bench_human_judgments), following the exact same
methodology as scripts/collect_llm_logits.py + scripts/make_paper_figures.py -
featurize() over raw logits, LogisticRegressionCombiner fit on a
combiner_fit split, MSP-vs-combiner AUROC compared on id_test via paired
DeLong significance testing. Caches to data/judge_feature_cache_mtbench.pt
and writes a results summary to data/judge_experiment_results.json.

Why this script exists: every other experiment in this paper evaluates
ordinary classification / next-token prediction. None of them is an actual
LLM-judge task, even though the paper's framing (and its unevaluated
proposed Section 7) is about judge reliability specifically. This gives the
paper one real data point on that exact task, not a new methodology.

Task setup: Qwen2.5-0.5B-Instruct is given an instruction plus two candidate
responses (A and B) to it, and asked to pick the better one with a single
letter. The "prediction" is extracted from ONE forward pass - the raw logits
at the final prompt position (the verdict token's next-token distribution),
compared directly at the bare token ids for "A" (32) and "B" (33) under this
tokenizer/chat-template combination (confirmed by direct encode() calls and
by the top-5 token sanity check this script runs before the full loop).
featurize() is then computed over those same full-vocabulary logits, exactly
as for every other backbone in this project - only the "token being
predicted" changes (a verdict letter instead of a WikiText continuation
token), so phi(z)'s definition and the rest of the pipeline are unchanged.

Usage:
    python scripts/collect_judge_verdicts.py
"""

import argparse
import json
import os
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
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
TOKEN_ID_A = 32  # tokenizer.encode("A", add_special_tokens=False) for this model/template
TOKEN_ID_B = 33  # tokenizer.encode("B", add_special_tokens=False) for this model/template

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


CHECKPOINT_PATH = os.path.join(DATA_DIR, "_judge_checkpoint_mtbench.pt")
CHECKPOINT_EVERY = 100  # examples between checkpoint saves - see main()'s docstring note on why this exists


def main(limit: int | None) -> None:
    out_cache_path = os.path.join(DATA_DIR, "judge_feature_cache_mtbench.pt")
    out_results_path = os.path.join(DATA_DIR, "judge_experiment_results.json")

    filtered = load_filtered_dataset()
    rows = list(filtered)
    if limit is not None:
        rows = rows[:limit]
        print(f"using a fixed-size subset of {len(rows)} examples (--limit {limit})")
    else:
        print(f"using the full filtered set: {len(rows)} examples")

    # Checkpointing: a first full-1284 attempt at this run was killed by the
    # environment partway through (~1050/1284, no Python error - an external
    # kill, likely a background-task duration limit) with zero progress saved,
    # since the original version of this script only wrote output at the very
    # end. Rather than guess at the exact time budget and pick a subset size
    # to fit under it, this checkpoint/resume mechanism makes the run robust
    # to being killed and restarted, so the full filtered set can still be
    # used. Checkpoint state is keyed only by *how many rows have been
    # processed* (`rows` is a deterministic prefix of the same HF dataset
    # filter every time), so resuming is only valid if `--limit` is
    # unchanged between runs of the same checkpoint file.
    all_phi, all_correct, predicted_winner, human_winner, question_id = [], [], [], [], []
    start_idx = 0
    if os.path.exists(CHECKPOINT_PATH):
        ckpt = torch.load(CHECKPOINT_PATH)
        if ckpt.get("total_rows") == len(rows):
            all_phi = list(ckpt["phi"].unbind(0)) if len(ckpt["correct"]) > 0 else []
            all_correct = list(ckpt["correct"])
            predicted_winner = ckpt["predicted_winner"]
            human_winner = ckpt["human_winner"]
            question_id = ckpt["question_id"]
            start_idx = len(all_correct)
            print(f"resuming from checkpoint {CHECKPOINT_PATH}: {start_idx}/{len(rows)} examples already done")
        else:
            print(f"checkpoint at {CHECKPOINT_PATH} was for a different row count "
                  f"({ckpt.get('total_rows')} != {len(rows)}) - ignoring it, starting fresh")

    print(f"loading frozen {MODEL_NAME}...")
    model, tokenizer = load_frozen_causal_lm(MODEL_NAME)

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
            CHECKPOINT_PATH,
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

    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)
        print("run completed - removed intermediate checkpoint", CHECKPOINT_PATH)

    # Fit on combiner_fit, evaluate MSP vs combiner AUROC on id_test - exactly
    # make_paper_figures.py's evaluate_llm protocol. id_test here is n~=50% of
    # the filtered set (three_way_split's default (0.4, 0.1, 0.5) ratios),
    # large enough on its own (not the "too small, use full non-fit set"
    # fallback the task description allows for) - reported below.
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
    print(f"{'total examples used':<28} {n}")
    print(f"{'combiner_fit / threshold_cal / id_test':<28} {len(combiner_idx)} / {len(cal_idx)} / {len(test_idx)}")
    print(f"{'overall judge accuracy':<28} {overall_accuracy:.4f}")
    print(f"{'MSP AUROC (id_test)':<28} {msp_auroc:.4f}")
    print(f"{'Combiner AUROC (id_test)':<28} {combiner_auroc:.4f}")
    print(f"{'DeLong p-value':<28} {delong.p_value:.4g}")
    print(f"{'DeLong z':<28} {delong.z:.4f}")
    print(f"{'winner':<28} {winner}")

    results = {
        "judge_model": MODEL_NAME,
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
        "--limit", type=int, default=None,
        help="use only the first N filtered examples (fixed-size subset, chosen up front) instead of the full filtered set",
    )
    args = parser.parse_args()
    main(args.limit)
