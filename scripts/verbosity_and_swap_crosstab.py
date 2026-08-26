"""Reviewer follow-up (GAP 4/5a): two independent checks, both from data
already on disk plus one HF reload, no new model inference.

Part (a) -- verbosity bias: is either judge's predicted winner correlated
with which response is simply longer? Response text lives only in the raw
HF dataset (not in the cached tensors), so we reload
lmsys/mt_bench_human_judgments via collect_judge_verdicts.load_filtered_dataset()
-- the exact same filter (turn==1, winner in {model_a,model_b}) that produced
every judge cache's row order -- and join by row index to the cached
predicted_winner/human_winner arrays, restricted to id_test.

Part (b) -- does position-flippiness (swap-consistency, judge2026.tex's
98.9%-flip finding) co-occur with the confidence failure modes already
reported in Table 5 (high-confidence-wrong, low-confidence-correct)? Uses
data/judge_swap_consistency_cache.pt (0.5B only), restricted to its OWN
id_test split (n=642, verified bit-identical to judge_feature_cache_mtbench.pt's
phi/question_id/splits) so denominators match Table 5's. MSP orientation
uses the same pooled bootstrap-stabilized convention as Table 3/Table 5
(oriented_pooled(), not id_test's own verify_feature_directions, which would
be circular) -- computed here from the ORIGINAL judge_feature_cache_mtbench.pt
via load_judge_config, then applied to the swap cache's phi_orig column 0,
since the two are confirmed identical arrays.

No new model inference anywhere in this script.

Usage:
    python scripts/verbosity_and_swap_crosstab.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch
from scipy.stats import binomtest, fisher_exact

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

import collect_judge_verdicts as cjv  # noqa: E402
from judge_characterization import load_judge_config, oriented_pooled  # noqa: E402
from deployment_reliability.features import DEFAULT_FEATURE_NAMES  # noqa: E402

DATA_DIR = os.path.join(REPO_ROOT, "data")

QWEN_CONFIGS = [
    ("judge_feature_cache_mtbench.pt", "Qwen2.5-0.5B-Instruct"),
    ("judge_feature_cache_mtbench_1p5b.pt", "Qwen2.5-1.5B-Instruct"),
]


def word_count(text: str) -> int:
    return len(text.split())


def part_a() -> None:
    print("=" * 78)
    print("PART A: verbosity bias -- does predicted winner track response length?")
    print("=" * 78)
    rows = list(cjv.load_filtered_dataset())  # 1284 rows, fixed deterministic order
    wc_a = np.array([word_count(r["conversation_a"][1]["content"]) for r in rows])
    wc_b = np.array([word_count(r["conversation_b"][1]["content"]) for r in rows])
    longer = np.where(wc_a > wc_b, "model_a", np.where(wc_b > wc_a, "model_b", "tie"))
    n_ties = int((longer == "tie").sum())
    print(f"word-count ties: {n_ties}/{len(rows)}")

    for cache_filename, name in QWEN_CONFIGS:
        cache = torch.load(os.path.join(DATA_DIR, cache_filename))
        splits = np.array(cache["splits"])
        m_test = splits == "id_test"
        pred = np.array(cache["predicted_winner"])[m_test]
        human = np.array(cache["human_winner"])[m_test]
        longer_test = longer[m_test]
        nontie = longer_test != "tie"

        pred_matches = pred[nontie] == longer_test[nontie]
        human_matches = human[nontie] == longer_test[nontie]
        n = int(nontie.sum())
        pred_rate = float(pred_matches.mean())
        human_rate = float(human_matches.mean())

        # McNemar (paired, same examples): discordant pairs where pred and human disagree
        # on whether they match the longer response.
        b = int(((pred_matches) & (~human_matches)).sum())
        c = int(((~pred_matches) & (human_matches)).sum())
        mcnemar = binomtest(min(b, c), b + c, 0.5) if (b + c) > 0 else None
        p = mcnemar.pvalue if mcnemar is not None else float("nan")

        print(f"{name}: n={n} (id_test, non-tied) pred-matches-longer={pred_rate:.4f} "
              f"human-matches-longer={human_rate:.4f} McNemar(b={b},c={c}) p={p:.4g}")


def part_b() -> None:
    print()
    print("=" * 78)
    print("PART B: swap-consistency vs. confidence failure modes (0.5B, id_test)")
    print("=" * 78)
    swap = torch.load(os.path.join(DATA_DIR, "judge_swap_consistency_cache.pt"))
    orig_config = load_judge_config("judge_feature_cache_mtbench.pt", "Qwen2.5-0.5B-Instruct")
    msp_oriented_id_test = oriented_pooled(orig_config)[:, DEFAULT_FEATURE_NAMES.index("msp")].numpy()

    splits = np.array(swap["splits"])
    m_test = splits == "id_test"
    assert int(m_test.sum()) == 642, f"expected 642 id_test rows, got {m_test.sum()}"

    # Confirm phi_orig[m_test] column-0 values line up 1:1 with orig_config["phi"] column 0
    # (both come from the identical underlying cache; sanity-check before trusting the join).
    swap_msp_raw = swap["phi_orig"][m_test][:, 0].numpy()
    orig_msp_raw = orig_config["phi"][:, 0].numpy()
    assert np.allclose(swap_msp_raw, orig_msp_raw), "swap cache and original cache MSP columns don't align"

    correct_test = orig_config["correct"].numpy().astype(bool)
    swap_consistent = swap["swap_consistent"][m_test].numpy().astype(bool)

    n_swap_consistent = int(swap_consistent.sum())
    print(f"swap-consistent examples in id_test: {n_swap_consistent}/642 ({100*n_swap_consistent/642:.2f}%)")

    q75, q25 = np.percentile(msp_oriented_id_test, 75), np.percentile(msp_oriented_id_test, 25)
    rule1 = (~correct_test) & (msp_oriented_id_test >= q75)   # high-confidence-wrong
    rule2 = (correct_test) & (msp_oriented_id_test <= q25)    # low-confidence-correct

    for rule_name, rule_mask in [("high-confidence-wrong", rule1), ("low-confidence-correct", rule2)]:
        n_rule = int(rule_mask.sum())
        n_rule_and_swap = int((rule_mask & swap_consistent).sum())
        # 2x2 table: rows = rule_mask (yes/no), cols = swap_consistent (yes/no)
        table = [
            [n_rule_and_swap, n_rule - n_rule_and_swap],
            [int(swap_consistent.sum()) - n_rule_and_swap,
             642 - n_rule - (int(swap_consistent.sum()) - n_rule_and_swap)],
        ]
        odds_ratio, p_fisher = fisher_exact(table)
        print(f"{rule_name}: n={n_rule}, swap-consistent within rule={n_rule_and_swap}/{n_rule} "
              f"({100*n_rule_and_swap/n_rule:.2f}%), Fisher exact p={p_fisher:.4g}, table={table}")


if __name__ == "__main__":
    part_a()
    part_b()
