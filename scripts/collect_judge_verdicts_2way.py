"""Reviewer ablation: computes phi(z) over the 2-way {A, B} verdict-token
logit subvector instead of the full ~150k-way vocabulary, for all three
judge configs already in the paper (Qwen2.5-0.5B/1.5B-Instruct, SmolLM2-360M-
Instruct). Same dataset/filter/prompt/single-forward-pass protocol as
collect_judge_verdicts.py / collect_judge_verdicts_1p5b.py /
collect_judge_verdicts_smollm2_360m.py - only the slice of the logit vector
that featurize() is computed over changes (full vocab -> [logit_A, logit_B]).

Why this matters: for a pairwise judge, the decision-relevant distribution is
over {A, B}. Entropy/margin/energy/L2-norm computed over the full vocabulary
are dominated by "how peaked is the next token" formatting/continuation
structure, not verdict-specific uncertainty, and this may be why the five
features were found near-perfectly collinear (0/642 disagreement) and near-
chance at ranking correctness - both could be an artifact of the full-vocab
choice, not a property of judge uncertainty itself. This script produces the
data needed to check that directly; a companion analysis (see bottom of this
file / judge_characterization_2way.py) compares AUROC and disagreement rate
against the already-reported full-vocab numbers.

predicted_winner/correct are numerically identical to the full-vocab runs
(both are argmax(logit_A, logit_B) - the full vocab was never used for the
prediction itself, only for phi). Only phi changes here.

Each of the three configs is run sequentially in one process, with its own
checkpoint file, so a partial run resumes per-config rather than restarting
the whole three-model sweep from scratch.

Usage:
    python scripts/collect_judge_verdicts_2way.py
"""

from __future__ import annotations

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

JUDGE_INSTRUCTIONS = (
    "You are comparing two AI assistant responses to the same instruction. "
    "Decide which response is better overall.\n\n"
    "Instruction:\n{instruction}\n\n"
    "Response A:\n{response_a}\n\n"
    "Response B:\n{response_b}\n\n"
    "Which response is better, A or B? Answer with a single letter, A or B."
)

CONFIGS = [
    {
        "name": "Qwen2.5-0.5B-Instruct (judge, 2way)",
        "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
        "token_id_a": 32,
        "token_id_b": 33,
        "limit": None,
        "checkpoint": "_judge_checkpoint_mtbench_2way_qwen05b.pt",
        "out_cache": "judge_feature_cache_mtbench_2way.pt",
        "out_results": "judge_experiment_results_2way.json",
    },
    {
        "name": "Qwen2.5-1.5B-Instruct (judge, 2way)",
        "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
        "token_id_a": 32,
        "token_id_b": 33,
        "limit": None,
        "checkpoint": "_judge_checkpoint_mtbench_2way_qwen15b.pt",
        "out_cache": "judge_feature_cache_mtbench_1p5b_2way.pt",
        "out_results": "judge_experiment_results_1p5b_2way.json",
    },
    {
        "name": "SmolLM2-360M-Instruct (judge, 2way, n=400 subset)",
        "model_name": "HuggingFaceTB/SmolLM2-360M-Instruct",
        "token_id_a": None,
        "token_id_b": None,
        "limit": 400,
        "checkpoint": "_judge_checkpoint_mtbench_2way_smollm2_360m.pt",
        "out_cache": "judge_feature_cache_mtbench_smollm2_360m_2way.pt",
        "out_results": "judge_experiment_results_smollm2_360m_2way.json",
    },
]


def build_prompt(tokenizer, instruction: str, response_a: str, response_b: str) -> str:
    user_content = JUDGE_INSTRUCTIONS.format(instruction=instruction, response_a=response_a, response_b=response_b)
    messages = [{"role": "user", "content": user_content}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@torch.inference_mode()
def verdict_logits(model, tokenizer, prompt_text: str) -> torch.Tensor:
    input_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"]
    outputs = model(input_ids=input_ids)
    return outputs.logits[0, -1, :]


def resolve_token_ids(tokenizer, expected_a: int | None, expected_b: int | None) -> tuple[int, int]:
    enc_a = tokenizer.encode("A", add_special_tokens=False)
    enc_b = tokenizer.encode("B", add_special_tokens=False)
    assert len(enc_a) == 1 and len(enc_b) == 1, f"A/B not single tokens: {enc_a}, {enc_b}"
    token_id_a, token_id_b = enc_a[0], enc_b[0]
    if expected_a is not None:
        assert token_id_a == expected_a, f"token_id_a mismatch: {token_id_a} != {expected_a}"
    if expected_b is not None:
        assert token_id_b == expected_b, f"token_id_b mismatch: {token_id_b} != {expected_b}"
    return token_id_a, token_id_b


def load_filtered_dataset():
    from datasets import load_dataset

    ds = load_dataset("lmsys/mt_bench_human_judgments", split="human")
    filtered = ds.filter(lambda r: r["turn"] == 1 and r["winner"] in ("model_a", "model_b"))
    return filtered


def run_one_config(cfg: dict) -> None:
    checkpoint_path = os.path.join(DATA_DIR, cfg["checkpoint"])
    out_cache_path = os.path.join(DATA_DIR, cfg["out_cache"])
    out_results_path = os.path.join(DATA_DIR, cfg["out_results"])

    if os.path.exists(out_cache_path):
        print(f"[skip] {cfg['name']}: {cfg['out_cache']} already exists")
        return

    print("=" * 100)
    print(f"=== {cfg['name']} ===")
    print("=" * 100)

    filtered = load_filtered_dataset()
    rows = list(filtered)
    if cfg["limit"] is not None:
        rows = rows[: cfg["limit"]]
    print(f"using {len(rows)} filtered examples")

    all_phi2, all_correct, predicted_winner, human_winner, question_id = [], [], [], [], []
    start_idx = 0
    token_id_a, token_id_b = cfg["token_id_a"], cfg["token_id_b"]
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path)
        if ckpt.get("total_rows") == len(rows):
            all_phi2 = list(ckpt["phi2"].unbind(0)) if len(ckpt["correct"]) > 0 else []
            all_correct = list(ckpt["correct"])
            predicted_winner = ckpt["predicted_winner"]
            human_winner = ckpt["human_winner"]
            question_id = ckpt["question_id"]
            token_id_a, token_id_b = ckpt["token_id_a"], ckpt["token_id_b"]
            start_idx = len(all_correct)
            print(f"resuming from checkpoint: {start_idx}/{len(rows)} examples already done")
        else:
            print("checkpoint row count mismatch - starting fresh")

    print(f"loading frozen {cfg['model_name']}...")
    t_load0 = time.time()
    model, tokenizer = load_frozen_causal_lm(cfg["model_name"])
    print(f"model+tokenizer loaded in {time.time() - t_load0:.1f}s")

    if token_id_a is None or token_id_b is None:
        token_id_a, token_id_b = resolve_token_ids(tokenizer, cfg["token_id_a"], cfg["token_id_b"])
        print(f"resolved token_id_a={token_id_a} token_id_b={token_id_b}")

    def save_checkpoint() -> None:
        torch.save(
            {
                "phi2": torch.stack(all_phi2, dim=0) if all_phi2 else torch.empty(0, 5),
                "correct": all_correct,
                "predicted_winner": predicted_winner,
                "human_winner": human_winner,
                "question_id": question_id,
                "token_id_a": token_id_a,
                "token_id_b": token_id_b,
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

        logit_a, logit_b = logits[token_id_a].item(), logits[token_id_b].item()
        pred = "model_a" if logit_a > logit_b else "model_b"
        correct = pred == row["winner"]

        z2 = torch.tensor([logit_a, logit_b], dtype=torch.float32)
        phi2 = featurize(z2, normalize_l2=True)
        all_phi2.append(phi2)
        all_correct.append(correct)
        predicted_winner.append(pred)
        human_winner.append(row["winner"])
        question_id.append(row["question_id"])

        if (i + 1) % 50 == 0 or (i + 1) == len(rows):
            elapsed = time.time() - t0
            running_acc = sum(all_correct) / len(all_correct)
            print(f"{i + 1}/{len(rows)}  elapsed={elapsed:.0f}s  running accuracy={running_acc:.4f}")
        if (i + 1) % 100 == 0:
            save_checkpoint()
            print(f"checkpoint saved at {i + 1}/{len(rows)}")

    phi2 = torch.stack(all_phi2, dim=0)
    correct = torch.tensor(all_correct, dtype=torch.bool)
    n = len(correct)
    overall_accuracy = correct.float().mean().item()
    print(f"done in {time.time() - t0:.1f}s. total examples: {n}  overall judge accuracy: {overall_accuracy:.4f}")

    combiner_idx, cal_idx, test_idx = three_way_split(n, seed=0)
    splits = [""] * n
    for idx, name in ((combiner_idx, "combiner_fit"), (cal_idx, "threshold_cal"), (test_idx, "id_test")):
        for i in idx.tolist():
            splits[i] = name

    torch.save(
        {
            "phi": phi2,
            "correct": correct,
            "splits": splits,
            "predicted_winner": predicted_winner,
            "human_winner": human_winner,
            "question_id": question_id,
            "token_id_a": token_id_a,
            "token_id_b": token_id_b,
            "restricted_to_verdict_tokens": True,
        },
        out_cache_path,
    )
    print("saved cache to", out_cache_path)

    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    phi_fit, correct_fit = phi2[combiner_idx], correct[combiner_idx].float()
    phi_test, correct_test = phi2[test_idx], correct[test_idx]

    combiner = LogisticRegressionCombiner().fit(phi_fit, correct_fit)
    msp_scores_test = phi_test[:, 0]
    combiner_scores_test = combiner.score(phi_test)

    delong = delong_test(correct_test, combiner_scores_test, msp_scores_test)
    msp_auroc, combiner_auroc = delong.auc_b, delong.auc_a

    print()
    print("=== Results summary (2-way verdict-token-restricted phi) ===")
    print(f"{'total examples used':<28} {n}")
    print(f"{'combiner_fit / threshold_cal / id_test':<28} {len(combiner_idx)} / {len(cal_idx)} / {len(test_idx)}")
    print(f"{'overall judge accuracy':<28} {overall_accuracy:.4f}")
    print(f"{'MSP AUROC (id_test)':<28} {msp_auroc:.4f}")
    print(f"{'Combiner AUROC (id_test)':<28} {combiner_auroc:.4f}")
    print(f"{'DeLong p-value':<28} {delong.p_value:.4g}")
    print(f"{'DeLong z':<28} {delong.z:.4f}")

    results = {
        "judge_model": cfg["model_name"],
        "restricted_to_verdict_tokens": True,
        "n_total_examples": n,
        "n_combiner_fit": len(combiner_idx),
        "n_threshold_cal": len(cal_idx),
        "n_id_test": len(test_idx),
        "overall_judge_accuracy": overall_accuracy,
        "msp_auroc_id_test": msp_auroc,
        "combiner_auroc_id_test": combiner_auroc,
        "delong_z": delong.z,
        "delong_p_value": delong.p_value,
    }
    with open(out_results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("saved results summary to", out_results_path)
    print()


def main() -> None:
    for cfg in CONFIGS:
        run_one_config(cfg)


if __name__ == "__main__":
    main()
