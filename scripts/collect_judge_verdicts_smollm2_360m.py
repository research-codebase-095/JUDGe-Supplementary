"""Third judge-model-family run, take two: HuggingFaceTB/SmolLM2-360M-Instruct
in place of the 1.7B sibling used in collect_judge_verdicts_smollm2.py, which
was killed after 400/1284 examples (~45s/example, projecting to ~16h total -
too slow to be useful). Otherwise identical methodology (same dataset/filter,
same prompt template, same single-forward-pass raw-logit verdict extraction,
same featurize()/three_way_split/DeLong protocol, same dynamic token-id
resolution rather than hardcoded Qwen-specific ids).

Model choice rationale: 360M is ~4.7x smaller than the 1.7B attempt, so at
roughly linear CPU-inference scaling should run at ~10s/example (~3.5h total)
rather than ~16h - and it is close in size to the already-tested Qwen2.5-0.5B
-Instruct judge (500M), making it the best apples-to-apples cross-family
comparison in the "small judge" regime this paper already characterizes,
rather than an arbitrarily-sized pick. Same ungated, AutoModelForCausalLM/
AutoTokenizer-compatible family already proven to load via
load_frozen_causal_lm (the 1.7B sibling's tokenizer/chat-template path is
already validated).

Distinct output/checkpoint filenames (suffixed _360m) so this run cannot
collide with the killed 1.7B attempt's partial checkpoint or its leftover
5-example smoke-test cache.

Usage:
    python scripts/collect_judge_verdicts_smollm2_360m.py
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
MODEL_NAME = "HuggingFaceTB/SmolLM2-360M-Instruct"

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
    final prompt position - the verdict token's next-token distribution."""
    input_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"]
    outputs = model(input_ids=input_ids)
    return outputs.logits[0, -1, :]


def resolve_token_ids(tokenizer) -> tuple[int, int]:
    """Derives TOKEN_ID_A/TOKEN_ID_B from this specific tokenizer via direct
    encode() calls, rather than assuming any other model's ids. Fails loudly
    if "A"/"B" is not a single token under this tokenizer."""
    enc_a = tokenizer.encode("A", add_special_tokens=False)
    enc_b = tokenizer.encode("B", add_special_tokens=False)
    print(f"encode('A', add_special_tokens=False) = {enc_a}")
    print(f"encode('B', add_special_tokens=False) = {enc_b}")
    assert len(enc_a) == 1, f"'A' is not a single token under {MODEL_NAME}'s tokenizer: {enc_a}"
    assert len(enc_b) == 1, f"'B' is not a single token under {MODEL_NAME}'s tokenizer: {enc_b}"
    token_id_a, token_id_b = enc_a[0], enc_b[0]
    assert token_id_a != token_id_b, "TOKEN_ID_A and TOKEN_ID_B resolved to the same id"
    print(f"resolved TOKEN_ID_A={token_id_a}  TOKEN_ID_B={token_id_b}")
    print()
    return token_id_a, token_id_b


def sanity_check_tokens(model, tokenizer, rows, token_id_a: int, token_id_b: int, n_check: int = 5) -> None:
    """Prints the top-5 highest-probability tokens at the verdict position
    for a handful of real examples, so the resolved TOKEN_ID_A/TOKEN_ID_B are
    checked against actual model behavior before being trusted for the full
    run."""
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
            flag = "  <-- A/B" if tid in (token_id_a, token_id_b) else ""
            print(f"    #{rank + 1}: token_id={tid:>6}  token={tok_str!r:<12}  p={p:.4f}{flag}")
        print(f"    p(A)={probs[token_id_a].item():.4f}  p(B)={probs[token_id_b].item():.4f}")
    print()


def load_filtered_dataset():
    from datasets import load_dataset

    ds = load_dataset("lmsys/mt_bench_human_judgments", split="human")
    print(f"loaded lmsys/mt_bench_human_judgments 'human' split: {len(ds)} rows")
    filtered = ds.filter(lambda r: r["turn"] == 1 and r["winner"] in ("model_a", "model_b"))
    print(f"filtered to turn==1 and winner in {{model_a, model_b}}: {len(filtered)} rows")
    return filtered


CHECKPOINT_PATH = os.path.join(DATA_DIR, "_judge_checkpoint_mtbench_smollm2_360m.pt")
CHECKPOINT_EVERY = 100  # examples between checkpoint saves - see collect_judge_verdicts.py's comment for why this exists


def main(limit: int | None) -> None:
    out_cache_path = os.path.join(DATA_DIR, "judge_feature_cache_mtbench_smollm2_360m.pt")
    out_results_path = os.path.join(DATA_DIR, "judge_experiment_results_smollm2_360m.json")

    filtered = load_filtered_dataset()
    rows = list(filtered)
    if limit is not None:
        rows = rows[:limit]
        print(f"using a fixed-size subset of {len(rows)} examples (--limit {limit})")
    else:
        print(f"using the full filtered set: {len(rows)} examples")

    all_phi, all_correct, predicted_winner, human_winner, question_id = [], [], [], [], []
    start_idx = 0
    token_id_a = token_id_b = None
    if os.path.exists(CHECKPOINT_PATH):
        ckpt = torch.load(CHECKPOINT_PATH)
        if ckpt.get("total_rows") == len(rows):
            all_phi = list(ckpt["phi"].unbind(0)) if len(ckpt["correct"]) > 0 else []
            all_correct = list(ckpt["correct"])
            predicted_winner = ckpt["predicted_winner"]
            human_winner = ckpt["human_winner"]
            question_id = ckpt["question_id"]
            token_id_a = ckpt["token_id_a"]
            token_id_b = ckpt["token_id_b"]
            start_idx = len(all_correct)
            print(f"resuming from checkpoint {CHECKPOINT_PATH}: {start_idx}/{len(rows)} examples already done, "
                  f"token_id_a={token_id_a} token_id_b={token_id_b} (carried over from checkpoint)")
        else:
            print(f"checkpoint at {CHECKPOINT_PATH} was for a different row count "
                  f"({ckpt.get('total_rows')} != {len(rows)}) - ignoring it, starting fresh")

    print(f"loading frozen {MODEL_NAME}...")
    t_load0 = time.time()
    model, tokenizer = load_frozen_causal_lm(MODEL_NAME)
    print(f"model+tokenizer loaded in {time.time() - t_load0:.1f}s")

    if token_id_a is None:
        token_id_a, token_id_b = resolve_token_ids(tokenizer)
        sanity_check_tokens(model, tokenizer, rows, token_id_a, token_id_b, n_check=5)

    def save_checkpoint() -> None:
        torch.save(
            {
                "phi": torch.stack(all_phi, dim=0) if all_phi else torch.empty(0, 5),
                "correct": all_correct,
                "predicted_winner": predicted_winner,
                "human_winner": human_winner,
                "question_id": question_id,
                "token_id_a": token_id_a,
                "token_id_b": token_id_b,
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

        pred = "model_a" if logits[token_id_a].item() > logits[token_id_b].item() else "model_b"
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
            "token_id_a": token_id_a,
            "token_id_b": token_id_b,
        },
        out_cache_path,
    )
    print("saved cache to", out_cache_path)

    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)
        print("run completed - removed intermediate checkpoint", CHECKPOINT_PATH)

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
        "token_id_a": token_id_a,
        "token_id_b": token_id_b,
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
