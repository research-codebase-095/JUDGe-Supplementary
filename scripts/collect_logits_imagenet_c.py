"""Runs a frozen backbone over ImageNet-C (Hendrycks & Dietterich, 2019) and
caches logits + labels to data/logit_cache_imagenet_c_<backbone>.pt, for the
DESIGN.md 11.1 distribution-shift graceful-degradation test - evaluate-only,
the already-frozen combiner/thresholds are never fit on this data (10.5).

Expected on-disk layout (the standard public ImageNet-C release structure,
as extracted by scripts/download_imagenet_c.py from the Zenodo tars):
    data/imagenet-c/<corruption_name>/<severity 1-5>/<WNID>/<image>.JPEG
e.g. data/imagenet-c/gaussian_noise/3/n01440764/ILSVRC2012_val_00000293.JPEG

Labels come from the WNID directory name, routed through the SAME
WNID -> torchvision class index mapping used everywhere else in this repo
(scripts/imagenet1k_labels.py's build_wnid_to_class_index(), not a
separately invented one) - the WNID is explicit in the path here, unlike
the raw-validation-set case, so there is no raw-ILSVRC-ID indirection to
get wrong.

Subsampling: full ImageNet-C (4 standard categories x ~4 corruptions each x
5 severities x 50,000 images) is on the order of tens of millions of images
- far beyond what a CPU-only run in this project can process. This script
samples `--n-per-severity` images per (corruption, severity) cell (default
200) with a fixed seed, which is enough to estimate per-cell accuracy/AURC
to within a few percentage points while keeping total inference cost
bounded - a stated, reasoned scope decision (matching the same reasoning
collect_logits_ood.py gives for subsampling iNaturalist/Places365).

Usage: python scripts/collect_logits_imagenet_c.py <backbone> [--n-per-severity 200] [--corruptions blur digital ...]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
import time

import torch
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from deployment_reliability.backbone import (  # noqa: E402
    load_frozen_convnext_tiny,
    load_frozen_resnet50,
    load_frozen_vit_b16,
)
from imagenet1k_labels import build_wnid_to_class_index  # noqa: E402

DATA_DIR = os.path.join(REPO_ROOT, "data")
IMAGENET_C_DIR = os.path.join(DATA_DIR, "imagenet-c")

BACKBONE_LOADERS = {
    "resnet50": load_frozen_resnet50,
    "vit_b16": load_frozen_vit_b16,
    "convnext_tiny": load_frozen_convnext_tiny,
}
BATCH_SIZES = {"resnet50": 32, "vit_b16": 16, "convnext_tiny": 16}


def discover_cells() -> list[tuple[str, str]]:
    """Returns [(corruption_name, severity), ...] for whatever's actually on
    disk under data/imagenet-c/ - doesn't hardcode the corruption-name list,
    since which categories were downloaded (scripts/download_imagenet_c.py's
    `categories` arg) varies run to run."""
    cells = []
    if not os.path.isdir(IMAGENET_C_DIR):
        return cells
    for corruption in sorted(os.listdir(IMAGENET_C_DIR)):
        corruption_dir = os.path.join(IMAGENET_C_DIR, corruption)
        if not os.path.isdir(corruption_dir):
            continue
        for severity in sorted(os.listdir(corruption_dir)):
            severity_dir = os.path.join(corruption_dir, severity)
            if os.path.isdir(severity_dir) and severity in {"1", "2", "3", "4", "5"}:
                cells.append((corruption, severity))
    return cells


def main(backbone_name: str, n_per_severity: int, corruptions_filter: list[str] | None) -> None:
    if backbone_name not in BACKBONE_LOADERS:
        raise ValueError(f"unknown backbone {backbone_name!r}; choose from {sorted(BACKBONE_LOADERS)}")

    cells = discover_cells()
    if corruptions_filter:
        cells = [(c, s) for c, s in cells if c in corruptions_filter]
    if not cells:
        raise FileNotFoundError(
            f"no (corruption, severity) directories found under {IMAGENET_C_DIR} - "
            "run scripts/download_imagenet_c.py first"
        )
    print(f"found {len(cells)} (corruption, severity) cells")

    wnid_to_class_index = build_wnid_to_class_index()

    rng = random.Random(0)
    all_files, all_labels, all_groups = [], [], []
    for corruption, severity in cells:
        cell_dir = os.path.join(IMAGENET_C_DIR, corruption, severity)
        wnids = sorted(d for d in os.listdir(cell_dir) if os.path.isdir(os.path.join(cell_dir, d)))
        # Two-stage sampling: first take a small per-class quota (so the cell
        # isn't dominated by whichever classes happen to be listed first),
        # THEN cap the pooled result down to n_per_severity total. A naive
        # single-stage "max(1, n_per_severity // len(wnids)) per class"
        # quota rounds up to 1 whenever n_per_severity < len(wnids) (the
        # common case: 200 target images over up to 1000 classes), which
        # would silently sample ~1000 images per cell instead of ~200 - a
        # real bug caught here before this script was ever run on real data.
        per_class_quota = max(1, n_per_severity // max(len(wnids), 1))
        pool_files, pool_labels = [], []
        for wnid in wnids:
            if wnid not in wnid_to_class_index:
                continue
            class_idx = wnid_to_class_index[wnid]
            candidates = sorted(glob.glob(os.path.join(cell_dir, wnid, "*")))
            if not candidates:
                continue
            chosen = rng.sample(candidates, min(len(candidates), per_class_quota))
            pool_files.extend(chosen)
            pool_labels.extend([class_idx] * len(chosen))

        if len(pool_files) > n_per_severity:
            idx = rng.sample(range(len(pool_files)), n_per_severity)
            cell_files = [pool_files[i] for i in idx]
            cell_labels = [pool_labels[i] for i in idx]
        else:
            cell_files, cell_labels = pool_files, pool_labels

        all_files.extend(cell_files)
        all_labels.extend(cell_labels)
        all_groups.extend([f"{corruption}_sev{severity}"] * len(cell_files))
        print(f"  {corruption} severity={severity}: {len(cell_files)} images sampled (target {n_per_severity})")

    print(f"total images: {len(all_files)}")
    print(f"loading frozen {backbone_name}...")
    model, preprocess, categories = BACKBONE_LOADERS[backbone_name]()
    assert len(categories) == 1000

    batch_size = BATCH_SIZES[backbone_name]
    all_logits, bad_indices = [], []
    t0 = time.time()
    for start in range(0, len(all_files), batch_size):
        batch_files = all_files[start : start + batch_size]
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
        if (start // batch_size) % 50 == 0:
            print(f"{start + len(batch_files)}/{len(all_files)}  elapsed={time.time() - t0:.0f}s  bad={len(bad_indices)}")

    all_logits = torch.cat(all_logits, dim=0)
    print(f"done in {time.time() - t0:.1f}s. bad_indices: {bad_indices}")

    out_path = os.path.join(DATA_DIR, f"logit_cache_imagenet_c_{backbone_name}.pt")
    torch.save(
        {
            "logits": all_logits,
            "labels": torch.tensor(all_labels),
            "groups": all_groups,
            "files": all_files,
            "bad_indices": bad_indices,
        },
        out_path,
    )
    print("saved cache to", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("backbone", nargs="?", default="resnet50")
    parser.add_argument("--n-per-severity", type=int, default=200)
    parser.add_argument("--corruptions", nargs="*", default=None)
    args = parser.parse_args()
    main(args.backbone, args.n_per_severity, args.corruptions)
