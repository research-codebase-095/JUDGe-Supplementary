"""Runs a frozen backbone over a genuine OOD dataset (iNaturalist, Places365,
DTD - classes the ImageNet-1k head was never trained to predict) and caches
logits to data/logit_cache_<dataset>_<backbone>.pt, for the AUROC(id vs OOD)/
FPR@95%TPR metrics in DESIGN.md 11.3 - the same evaluate-only role
`imagenet_o` already plays in the small-scale caches (no true class label is
meaningful here, since these datasets' classes aren't in the 1000-way
ImageNet head at all; labels are stored as -1, mirroring collect_logits.py's
existing imagenet_o convention exactly).

Subsampling, and why: iNaturalist's validation split alone is ~100,000
images and Places365's is 36,500 - running every image through a CPU-only
backbone for a metric (AUROC/FPR@95%TPR) that stabilizes with a few thousand
samples would spend most of its compute budget on statistical noise
reduction well past the point of diminishing return, at direct cost to the
higher-priority full-scale ImageNet-1k id_test run. This script therefore
subsamples each OOD dataset to `--n` images (default 3000, or all available
if fewer), sampled with a fixed seed for reproducibility - a stated, reasoned
scope decision, not a silent shortcut. DTD (5,640 images total) is small
enough to run in full without subsampling changing the cost meaningfully.

Usage: python scripts/collect_logits_ood.py <dataset> <backbone> [--n 3000]
    dataset in {places365, dtd, inaturalist}
"""

from __future__ import annotations

import argparse
import glob
import os
import random
import sys
import time

import torch
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from deployment_reliability.backbone import (  # noqa: E402
    load_frozen_convnext_tiny,
    load_frozen_resnet50,
    load_frozen_vit_b16,
)

DATA_DIR = os.path.join(REPO_ROOT, "data")

BACKBONE_LOADERS = {
    "resnet50": load_frozen_resnet50,
    "vit_b16": load_frozen_vit_b16,
    "convnext_tiny": load_frozen_convnext_tiny,
}
BATCH_SIZES = {"resnet50": 32, "vit_b16": 16, "convnext_tiny": 16}

DATASET_ROOTS = {
    "places365": os.path.join(DATA_DIR, "places365_val_256", "val_256"),
    "dtd": os.path.join(DATA_DIR, "dtd", "images"),
    "inaturalist": os.path.join(DATA_DIR, "inaturalist2021_val"),
}
DEFAULT_N = {"places365": 3000, "dtd": 100_000, "inaturalist": 3000}  # dtd: no subsampling (full dataset < 6000)


def list_images(root: str, n: int, seed: int = 0) -> list[str]:
    files = sorted(glob.glob(os.path.join(root, "**", "*.jpg"), recursive=True))
    files += sorted(glob.glob(os.path.join(root, "**", "*.JPEG"), recursive=True))
    files += sorted(glob.glob(os.path.join(root, "**", "*.jpeg"), recursive=True))
    files += sorted(glob.glob(os.path.join(root, "**", "*.png"), recursive=True))
    files = sorted(set(files))
    if len(files) > n:
        rng = random.Random(seed)
        files = rng.sample(files, n)
        files.sort()
    return files


def main(dataset: str, backbone_name: str, n: int) -> None:
    if dataset not in DATASET_ROOTS:
        raise ValueError(f"unknown dataset {dataset!r}; choose from {sorted(DATASET_ROOTS)}")
    if backbone_name not in BACKBONE_LOADERS:
        raise ValueError(f"unknown backbone {backbone_name!r}; choose from {sorted(BACKBONE_LOADERS)}")

    root = DATASET_ROOTS[dataset]
    if not os.path.isdir(root):
        raise FileNotFoundError(f"{root} not found - run scripts/download_ood_suite.py first")

    files = list_images(root, n)
    print(f"{dataset}: using {len(files)} images from {root}")

    print(f"loading frozen {backbone_name}...")
    model, preprocess, categories = BACKBONE_LOADERS[backbone_name]()
    assert len(categories) == 1000

    batch_size = BATCH_SIZES[backbone_name]
    all_logits, bad_indices = [], []
    t0 = time.time()
    for start in range(0, len(files), batch_size):
        batch_files = files[start : start + batch_size]
        imgs = []
        for i, f in enumerate(batch_files):
            try:
                imgs.append(preprocess(Image.open(f).convert("RGB")))
            except Exception:
                bad_indices.append(start + i)
                imgs.append(torch.zeros(3, 224, 224))
        batch = torch.stack(imgs)
        with torch.inference_mode():
            logits = model(batch)
        all_logits.append(logits)
        if (start // batch_size) % 20 == 0:
            print(f"{start + len(batch_files)}/{len(files)}  elapsed={time.time() - t0:.0f}s  bad={len(bad_indices)}")

    all_logits = torch.cat(all_logits, dim=0)
    print(f"done in {time.time() - t0:.1f}s. bad_indices: {bad_indices}")

    out_path = os.path.join(DATA_DIR, f"logit_cache_{dataset}_{backbone_name}.pt")
    torch.save(
        {
            "logits": all_logits,
            "labels": torch.full((len(all_logits),), -1, dtype=torch.long),  # no meaningful ImageNet label - mirrors imagenet_o
            "splits": [dataset] * len(all_logits),
            "files": files,
            "bad_indices": bad_indices,
        },
        out_path,
    )
    print("saved cache to", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=sorted(DATASET_ROOTS))
    parser.add_argument("backbone", nargs="?", default="resnet50")
    parser.add_argument("--n", type=int, default=None)
    args = parser.parse_args()
    main(args.dataset, args.backbone, args.n or DEFAULT_N[args.dataset])
