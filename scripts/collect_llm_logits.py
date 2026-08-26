"""Runs a frozen causal LM once over the real WikiText-2 validation corpus and
caches per-token features + correctness + split tags to
data/llm_feature_cache_<tag>.pt, so notebooks/14_llm_extension.ipynb doesn't
need to re-run inference every time it's opened.

Deliberately caches phi(z) (5 floats/token), not raw per-token logits: GPT-2's
vocabulary is 50,257 wide versus ImageNet's ~1,000, so caching raw logits the
way collect_logits.py does for vision backbones (trivial there, ~8MB for
~2,000 images) would mean ~12GB for a comparable number of tokens here - a
real, practical consequence of DESIGN.md 14.4's vocabulary-scaling discussion,
not a corner cut. Correctness is still derivable from the cached quantities
alone (phi already encodes msp/margin/entropy/energy/l2norm; the correctness
label itself is stored directly rather than requiring the argmax of a logit
vector that no longer exists in the cache).

Usage:
    python scripts/collect_llm_logits.py                                     # gpt2 (default, unchanged)
    python scripts/collect_llm_logits.py --model EleutherAI/pythia-160m --tag pythia160m --auto

`--auto` routes model loading through `load_frozen_causal_lm`
(AutoModelForCausalLM/AutoTokenizer) instead of `load_frozen_gpt2` - added for
STUDY_PLAN.md 3.6 item 1b (a second LLM backbone), so the discrimination-
quality finding can be checked against a structurally different model
(GPT-NeoX-family, trained on the Pile rather than WebText), not just GPT-2.
Without `--auto`, behavior for the default gpt2 case is unchanged.

No download_eval_data.py step needed first - WikiText-2 and the model are both
fetched directly by this script (no registration wall on either, for gpt2 or
pythia-160m).
"""

import argparse
import os
import sys
import time
import urllib.request

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from deployment_reliability.features import featurize  # noqa: E402
from deployment_reliability.llm_backbone import (  # noqa: E402
    load_frozen_causal_lm,
    load_frozen_gpt2,
    logits_for_token_chunk,
)
from deployment_reliability.splits import three_way_split  # noqa: E402

DATA_DIR = os.path.join(REPO_ROOT, "data")

WIKITEXT2_VALID_URL = (
    "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/valid.txt"
)
CHUNK_SIZE = 1024  # GPT-2's context window; also used for Pythia-160m (context 2048) for direct cross-model comparability


def download_wikitext2_valid() -> str:
    cache_path = os.path.join(DATA_DIR, "wikitext2_valid.txt")
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            return f.read()
    print("downloading WikiText-2 validation split...")
    text = urllib.request.urlopen(WIKITEXT2_VALID_URL, timeout=60).read().decode("utf-8")
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        f.write(text)
    return text


def main(model_name: str, tag: str, use_auto: bool) -> None:
    out_path = os.path.join(DATA_DIR, f"llm_feature_cache_{tag}.pt")
    text = download_wikitext2_valid()
    print(f"loaded WikiText-2 validation split: {len(text)} chars")

    print(f"loading frozen {model_name}...")
    model, tokenizer = load_frozen_causal_lm(model_name) if use_auto else load_frozen_gpt2(model_name)

    full_ids = tokenizer(text)["input_ids"]
    n_chunks = len(full_ids) // CHUNK_SIZE
    print(f"total tokens: {len(full_ids)}  processing {n_chunks} chunks of {CHUNK_SIZE} tokens = {n_chunks * CHUNK_SIZE} tokens")

    all_phi, all_correct = [], []
    t0 = time.time()
    for i in range(n_chunks):
        chunk = torch.tensor(full_ids[i * CHUNK_SIZE : (i + 1) * CHUNK_SIZE])
        logits = logits_for_token_chunk(model, chunk)
        # position j's logits predict token j+1 - standard causal-LM next-token setup
        preds = logits[:-1].argmax(dim=-1)
        true_next = chunk[1:]
        correct = preds == true_next
        phi = featurize(logits[:-1], normalize_l2=True)  # DESIGN.md 14.4's sqrt(C) correction, real vocab scale
        all_phi.append(phi)
        all_correct.append(correct)
        if i % 20 == 0:
            print(f"{(i + 1) * CHUNK_SIZE}/{n_chunks * CHUNK_SIZE}  elapsed={time.time() - t0:.0f}s")

    phi = torch.cat(all_phi, dim=0)
    correct = torch.cat(all_correct, dim=0)
    print(f"done in {time.time() - t0:.1f}s. total token predictions: {len(correct)}  overall accuracy: {correct.float().mean().item():.4f}")

    n = len(correct)
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
        },
        out_path,
    )
    print("saved cache to", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt2", help="HF model name/path (default: gpt2)")
    parser.add_argument(
        "--tag", default=None, help="cache filename tag -> data/llm_feature_cache_<tag>.pt (default: derived from --model)"
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="load via load_frozen_causal_lm (AutoModelForCausalLM/AutoTokenizer) instead of load_frozen_gpt2",
    )
    args = parser.parse_args()
    tag = args.tag or args.model.split("/")[-1].replace("-", "")
    main(args.model, tag, args.auto)
