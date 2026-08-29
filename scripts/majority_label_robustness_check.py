"""Reviewer follow-up: does the paper's main combiner-vs-best-signal AUROC
comparison (Appendix A.6's headline null result -- the combiner does not
reliably beat the single best signal) hold up if restricted to the 351
multiply-annotated MT-Bench (question_id, model-pair) instances, using a
MAJORITY label instead of the single-annotator label the paper's "correct"
currently means?

This is the robustness check for the "no adjudication step" caveat in the
paper's Full limitations list: MT-Bench labels come from a single annotator
per instance, so this asks whether the combiner-vs-best-signal gap is an
artifact of that per-instance label noise.

Data-availability check first: judge_feature_cache_mtbench*.pt stores
`question_id` per row (in the same deterministic row order the paper's other
scripts already establish -- collect_judge_verdicts_continuous.py's own
alignment check against judge_feature_cache_mtbench.pt is the precedent this
reuses), but NOT model_a/model_b/judge identity. Those live only in the raw
HF dataset (lmsys/mt_bench_human_judgments), loaded via
collect_judge_verdicts.load_filtered_dataset() -- the exact same call
inter_annotator_agreement.py itself makes. This is a dataset *load* (no
model forward pass, no GPU), the same operation the paper's own
inter_annotator_agreement.py already relies on, so cross-referencing it
against the cache is not new model inference.

Method:
  1. Reconstruct rows = load_filtered_dataset() (deterministic order).
  2. Build the same (question_id, frozenset({model_a,model_b})) -> records
     key used in inter_annotator_agreement.py; keep keys with >=2 distinct
     judges (the paper's 351).
  3. For each judge cache (0.5B, 1.5B), verify row-index alignment against
     `rows` via question_id, then find which id_test row-indices fall inside
     an overlapping key.
  4. For those rows, compute a MAJORITY winning-model label per key
     (strict majority across the key's annotators; ties -- exactly split votes
     with no single winner -- are reported and excluded, not arbitrarily
     broken).
  5. Recompute "correct" for those rows against the majority label instead of
     that row's own single annotator, and compare combiner vs. best-single-
     feature AUROC on that relabeled subset, reusing
     judge_characterization.py's LogisticRegressionCombiner /
     best_single_feature / _rank_based_auroc (imported, not reimplemented).

Usage:
    python scripts/majority_label_robustness_check.py
"""
from __future__ import annotations

import collections
import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data")
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from collect_judge_verdicts import load_filtered_dataset  # noqa: E402
from deployment_reliability.significance import _rank_based_auroc, delong_test  # noqa: E402

import judge_characterization as jc  # noqa: E402


def main() -> None:
    rows = list(load_filtered_dataset())
    n = len(rows)
    print(f"n filtered rows = {n}")

    # Step 2: overlapping keys, same definition as inter_annotator_agreement.py
    key_of_row = []
    key_to_row_idxs: dict[tuple, list[int]] = collections.defaultdict(list)
    key_to_records: dict[tuple, list[tuple[str, str]]] = collections.defaultdict(list)
    for i, r in enumerate(rows):
        key = (r["question_id"], frozenset([r["model_a"], r["model_b"]]))
        key_of_row.append(key)
        key_to_row_idxs[key].append(i)
        winning_model = r["model_a"] if r["winner"] == "model_a" else r["model_b"]
        key_to_records[key].append((r["judge"], winning_model))

    overlapping_keys = {k for k, v in key_to_records.items() if len({j for j, _ in v}) > 1}
    print(f"n unique keys = {len(key_to_records)}, n overlapping (>=2 distinct judges) = {len(overlapping_keys)}")

    # Majority label per overlapping key (ties = no majority)
    majority_label: dict[tuple, str | None] = {}
    n_ties = 0
    for k in overlapping_keys:
        votes = collections.Counter(w for _, w in key_to_records[k])
        top = votes.most_common()
        if len(top) > 1 and top[0][1] == top[1][1]:
            majority_label[k] = None  # tie, no majority
            n_ties += 1
        else:
            majority_label[k] = top[0][0]
    print(f"n overlapping keys with a clear majority = {len(overlapping_keys) - n_ties}, ties (no majority) = {n_ties}")

    overlapping_row_idxs = set()
    for k in overlapping_keys:
        overlapping_row_idxs.update(key_to_row_idxs[k])
    print(f"n rows (not keys) belonging to overlapping keys = {len(overlapping_row_idxs)}")

    def analyze_config(cache_filename: str, display_name: str) -> None:
        print("\n" + "=" * 90)
        print(display_name, cache_filename)
        path = os.path.join(DATA_DIR, cache_filename)
        cache = torch.load(path, weights_only=False)
        cache_qid = np.array(cache["question_id"])
        assert len(cache_qid) == n, f"length mismatch: cache has {len(cache_qid)}, rows has {n}"
        # alignment check: cache row i should correspond to rows[i]
        row_qid = np.array([r["question_id"] for r in rows])
        n_match = int((cache_qid == row_qid).sum())
        print(f"row alignment check (question_id): {n_match}/{n} match "
              f"({'OK, exact' if n_match == n else 'MISMATCH -- cannot proceed safely'})")
        if n_match != n:
            print("  -> ABORTING this config: cannot trust index-based cross-reference.")
            return

        splits = np.array(cache["splits"])
        m_test = splits == "id_test"
        test_idxs = np.where(m_test)[0]
        test_overlap_idxs = [i for i in test_idxs if i in overlapping_row_idxs]
        print(f"n id_test rows total = {int(m_test.sum())}")
        print(f"n id_test rows in a multiply-annotated (overlapping) key = {len(test_overlap_idxs)}")

        # of those, how many have a clear (non-tied) majority label
        usable_idxs = [i for i in test_overlap_idxs if majority_label[key_of_row[i]] is not None]
        print(f"n of those with a clear majority (non-tied) label = {len(usable_idxs)}")

        if len(usable_idxs) < 30:
            print(f"  -> n={len(usable_idxs)} is too small for a meaningful AUROC re-analysis (need both enough "
                  f"correct AND incorrect examples with reasonable bootstrap stability); reporting count only.")
            return

        # Build majority-corrected "correct" labels for these rows
        majority_correct = []
        for i in usable_idxs:
            r = rows[i]
            predicted_winner = cache["predicted_winner"][i]  # 'model_a' / 'model_b', row-local slot
            predicted_model = r["model_a"] if predicted_winner == "model_a" else r["model_b"]
            majority_correct.append(predicted_model == majority_label[key_of_row[i]])
        majority_correct = np.array(majority_correct, dtype=bool)
        n_correct = int(majority_correct.sum())
        print(f"majority-label subset: n={len(usable_idxs)}, n_correct={n_correct}, n_incorrect={len(usable_idxs)-n_correct}")

        if n_correct < 5 or (len(usable_idxs) - n_correct) < 5:
            print("  -> too few correct or incorrect examples in this subset for a stable AUROC; skipping AUROC computation.")
            return

        # Full-id_test combiner + best-feature scores (fit on combiner_fit / selected on threshold_cal,
        # exactly as judge_characterization.py does), then restrict to usable_idxs.
        config = jc.load_judge_config(cache_filename, display_name)
        combiner_scores_full = config["combiner"].score(config["phi"])  # full id_test, in id_test row order
        best_name, best_scores_full = jc.best_single_feature(config)

        # map global row index -> position within id_test ordering used by config["phi"]
        test_idx_list = list(test_idxs)
        pos_in_test = {global_i: pos for pos, global_i in enumerate(test_idx_list)}
        sub_pos = [pos_in_test[i] for i in usable_idxs]

        combiner_sub = combiner_scores_full[sub_pos].numpy()
        best_sub = best_scores_full[sub_pos].numpy()

        auroc_combiner = _rank_based_auroc(combiner_sub[majority_correct], combiner_sub[~majority_correct])
        auroc_best = _rank_based_auroc(best_sub[majority_correct], best_sub[~majority_correct])
        print(f"best single feature (selected disjointly): {best_name}")
        print(f"AUROC combiner  (majority-label subset, n={len(usable_idxs)}): {auroc_combiner:.4f}")
        print(f"AUROC best-feat (majority-label subset, n={len(usable_idxs)}): {auroc_best:.4f}")
        print(f"gap (combiner - best): {auroc_combiner - auroc_best:+.4f}")

        # majority_correct is the SAME label vector for both scores on this subset,
        # so this IS a valid DeLong pairing (unlike the order-averaging check's
        # differing-label case in order_averaging_auroc_check.py).
        dl = delong_test(
            torch.from_numpy(majority_correct), torch.from_numpy(combiner_sub), torch.from_numpy(best_sub)
        )
        print(f"DeLong (combiner vs best-feat, majority label, n={len(usable_idxs)}): z={dl.z:.3f} p={dl.p_value:.3e}")

        # For reference: same rows/scores but graded against the ORIGINAL single-annotator label
        orig_correct_sub = config["correct"][sub_pos].numpy().astype(bool)
        auroc_combiner_orig = _rank_based_auroc(combiner_sub[orig_correct_sub], combiner_sub[~orig_correct_sub])
        auroc_best_orig = _rank_based_auroc(best_sub[orig_correct_sub], best_sub[~orig_correct_sub])
        print(f"[reference, same subset, ORIGINAL single-annotator label] "
              f"AUROC combiner={auroc_combiner_orig:.4f}  AUROC best-feat={auroc_best_orig:.4f}  "
              f"gap={auroc_combiner_orig - auroc_best_orig:+.4f}")

    analyze_config("judge_feature_cache_mtbench.pt", "Qwen2.5-0.5B-Instruct (judge)")
    analyze_config("judge_feature_cache_mtbench_1p5b.pt", "Qwen2.5-1.5B-Instruct (judge)")


if __name__ == "__main__":
    main()
