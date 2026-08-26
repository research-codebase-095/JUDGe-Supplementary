"""Answers a reviewer question about judge2026.tex: the paper treats
agreement with a single human MT-Bench label as "correctness," but each
label reflects one of over 60 annotators' individual judgment, not an
adjudicated or majority-vote decision. How reliable is that reference itself?

lmsys/mt_bench_human_judgments's `human` split does, in places, have more
than one annotator judging the same (question, model-pair) instance (with
model_a/model_b slot order not necessarily consistent across annotators for
the same pair), so an inter-annotator agreement rate is directly computable
from the same filtered rows every other judge script already uses - no new
data collection, no model inference.

Methodology: canonicalize each instance by (question_id,
frozenset({model_a, model_b})) so two rows judging the same model pair on
the same question are matched regardless of which model was slotted "A" vs
"B"; restrict to model pairs with >=2 DISTINCT annotator ids. Report both
item-level agreement (does every annotator on that instance pick the same
winning MODEL, unanimously) and pairwise agreement (of all distinct-annotator
pairs within an instance, what fraction agree) - these answer sightly
different questions and neither subsumes the other.

Usage:
    python scripts/inter_annotator_agreement.py
"""

from __future__ import annotations

import collections
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from collect_judge_verdicts import load_filtered_dataset  # noqa: E402


def main() -> None:
    rows = list(load_filtered_dataset())
    print(f"total filtered rows: {len(rows)}")

    key_to_records: dict[tuple, list[tuple[str, str]]] = collections.defaultdict(list)
    for r in rows:
        winning_model = r["model_a"] if r["winner"] == "model_a" else r["model_b"]
        key = (r["question_id"], frozenset([r["model_a"], r["model_b"]]))
        key_to_records[key].append((r["judge"], winning_model))

    n_unique_keys = len(key_to_records)
    overlapping = {k: v for k, v in key_to_records.items() if len({j for j, _ in v}) > 1}
    print(f"n unique (question_id, model-pair) keys: {n_unique_keys}")
    print(f"n with >=2 DISTINCT judge annotators: {len(overlapping)}")

    if not overlapping:
        print("NO inter-annotator overlap exists in this dataset.")
        return

    agree = sum(1 for records in overlapping.values() if len({w for _, w in records}) == 1)
    print(f"item-level agreement rate: {agree}/{len(overlapping)} = {agree/len(overlapping):.4f}")

    pairwise_total = 0
    pairwise_agree = 0
    for records in overlapping.values():
        for i in range(len(records)):
            for j in range(i + 1, len(records)):
                if records[i][0] == records[j][0]:
                    continue  # same annotator id appearing twice for this key - not a distinct pair
                pairwise_total += 1
                pairwise_agree += int(records[i][1] == records[j][1])
    print(f"pairwise agreement rate: {pairwise_agree}/{pairwise_total} = {pairwise_agree/pairwise_total:.4f}")

    sizes = collections.Counter(len({j for j, _ in v}) for v in overlapping.values())
    print(f"distribution of distinct-annotator-count per overlapping key: {dict(sorted(sizes.items()))}")


if __name__ == "__main__":
    main()
