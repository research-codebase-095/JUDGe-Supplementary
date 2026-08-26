"""Runs frozen ResNet-50 once more over exactly the same real images already
used by data/logit_cache_resnet50.pt (same files/splits/labels, reused
directly rather than re-sampled) and additionally captures penultimate-layer
(2048-d, pre-fc) features via backbone.py's logits_and_features_for_images,
caching them to data/mahalanobis_feature_cache_resnet50.pt.

STUDY_PLAN.md 3.6 item 2: this is what lets mahalanobis.MahalanobisScorer
be fit on real combiner_fit features and tested against real imagenet_a/
imagenet_o data, directly checking whether feature-space distance recovers
any of the near-zero correctness signal DESIGN.md 15/20 and STUDY_PLAN.md
6.2 document for logit-only signals on ImageNet-A.

Also re-derives logits/argmax-correctness from this same forward pass (rather
than reusing the cached logits verbatim) purely as an internal consistency
check - see the assertion below - not because the cached logits are
distrusted.

Usage: python scripts/collect_features_mahalanobis.py

Requires data/logit_cache_resnet50.pt to already exist (scripts/collect_logits.py resnet50).
"""

import os
import sys
import time

import torch
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from deployment_reliability.backbone import load_frozen_resnet50, logits_and_features_for_images  # noqa: E402

DATA_DIR = os.path.join(REPO_ROOT, "data")
LOGIT_CACHE_PATH = os.path.join(DATA_DIR, "logit_cache_resnet50.pt")
OUT_PATH = os.path.join(DATA_DIR, "mahalanobis_feature_cache_resnet50.pt")
BATCH_SIZE = 32

# Subdirectories collect_logits.py samples from - used to remap absolute file
# paths recorded by a *different* machine/checkout onto this environment's
# own data/ directory (the cache stores whatever absolute path the original
# collection run happened to have; this run's data/ tree is byte-identical
# but may live at a different absolute prefix).
_DATA_SUBDIR_MARKERS = ("imagenette2-160", "imagenet-a", "imagenet-o")


def _remap_to_local_data_dir(path: str) -> str:
    normalized = path.replace("\\", "/")
    for marker in _DATA_SUBDIR_MARKERS:
        idx = normalized.find(marker)
        if idx != -1:
            return os.path.join(DATA_DIR, *normalized[idx:].split("/"))
    raise ValueError(f"cannot remap cached file path onto a known data/ subdirectory: {path}")


def main() -> None:
    assert os.path.exists(LOGIT_CACHE_PATH), "run scripts/collect_logits.py resnet50 first"
    cache = torch.load(LOGIT_CACHE_PATH)
    files = [_remap_to_local_data_dir(f) for f in cache["files"]]
    labels = cache["labels"]
    splits = cache["splits"]
    cached_logits = cache["logits"]
    bad_indices = set(cache["bad_indices"])

    print("loading frozen ResNet-50...")
    model, preprocess, categories = load_frozen_resnet50()

    all_logits, all_features = [], []
    t0 = time.time()
    for start in range(0, len(files), BATCH_SIZE):
        batch_files = files[start : start + BATCH_SIZE]
        imgs = []
        for i, f in enumerate(batch_files):
            if (start + i) in bad_indices:
                imgs.append(Image.new("RGB", (224, 224)))  # matches collect_logits.py's zero-tensor placeholder path
                continue
            imgs.append(Image.open(f).convert("RGB"))
        logits, features = logits_and_features_for_images(model, preprocess, imgs, architecture="resnet50")
        all_logits.append(logits)
        all_features.append(features)
        if (start // BATCH_SIZE) % 20 == 0:
            print(f"{start + len(batch_files)}/{len(files)}  elapsed={time.time() - t0:.0f}s")

    logits = torch.cat(all_logits, dim=0)
    features = torch.cat(all_features, dim=0)
    print(f"done in {time.time() - t0:.1f}s. features shape: {tuple(features.shape)}")

    # Internal consistency check: this fresh forward pass's logits should
    # match the cached ones (up to the "bad" placeholder images, which were
    # zero-tensors in collect_logits.py and blank white images here, and so
    # legitimately differ) - confirms the file/label/split alignment is
    # correct, not silently shuffled by the remap above.
    good_mask = torch.ones(len(files), dtype=torch.bool)
    for idx in bad_indices:
        good_mask[idx] = False
    max_abs_diff = (logits[good_mask] - cached_logits[good_mask]).abs().max().item()
    print(f"max |logit diff| vs. cached logits on good (non-placeholder) images: {max_abs_diff:.6f}")
    assert max_abs_diff < 1e-3, "recomputed logits diverge from the cached ones - file/split alignment may be wrong"

    torch.save(
        {
            "features": features,
            "logits": logits,
            "labels": labels,
            "splits": splits,
            "bad_indices": cache["bad_indices"],
        },
        OUT_PATH,
    )
    print("saved cache to", OUT_PATH)


if __name__ == "__main__":
    main()
