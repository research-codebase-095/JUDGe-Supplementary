"""Runs a frozen backbone once over the FULL ImageNet-1k validation set
(50,000 real, labeled images - not a proxy subset) and caches logits + labels
to data/logit_cache_imagenet1k_<backbone>.pt, mirroring collect_logits.py's
caching convention exactly (same field names, same
BACKBONE_LOADERS/DEFAULT_FEATURE_NAMES-compatible layout) but kept as a
separate script and a separate cache file rather than folding into
collect_logits.py, for two reasons stated explicitly rather than left implicit:

1. Scale: this is a ~50,000-image, tens-of-minutes-per-backbone CPU run,
   completely different in cost from collect_logits.py's few-hundred-image
   Imagenette/ImageNet-A/O collection - conflating the two would make the
   fast path slow by default.
2. Split-protocol decision (documented in full in DESIGN.md and
   PROGRESS_REPORT.md's full-scale-evaluation section): this project already
   has fitted combiner_fit/threshold_cal splits from Imagenette
   (collect_logits.py, used by notebooks 06-13) with real, previously-reported
   results built on them. Re-splitting the full ImageNet-1k validation set
   into a fresh combiner_fit/threshold_cal/id_test partition (DESIGN.md
   10.5's option (a)) would require refitting the combiner and thresholds,
   silently invalidating every previously reported number that depends on
   the existing fit. Instead (DESIGN.md 10.5's option (b), the one this
   project chose): the existing Imagenette-fitted combiner and thresholds are
   reused UNCHANGED, and the entire 50,000-image ImageNet-1k validation set
   is treated as one large, reporting-only "id_test_full_scale" split - never
   used for fitting anything, exactly like this project's existing
   `imagenet_a`/`imagenet_o` shift/OOD splits are evaluate-only per 10.5.
   Caveat, stated rather than glossed over: the existing combiner/thresholds
   were fit on Imagenette, a 10-class simplified subset where per-class
   confusion is structurally easier than the full 1000-way task: this doesn't
   bias the fit itself (the five features are computed purely from logit
   geometry, never from which classes are present), but it does mean the
   *feature distributions* the combiner/thresholds were calibrated against
   may not exactly match full 1000-way ImageNet's - itself an empirical
   question this full-scale run is positioned to answer, not assume.

Usage: python scripts/collect_logits_imagenet1k.py [resnet50|vit_b16|convnext_tiny] [--max-images N]

Requires scripts/download_eval_data.py to have downloaded the ImageNet-1k
devkit + validation images first.
"""

from __future__ import annotations

import argparse
import glob
import os
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
from imagenet1k_labels import load_val_labels, val_filename_to_index  # noqa: E402

DATA_DIR = os.path.join(REPO_ROOT, "data")
VAL_DIR = os.path.join(DATA_DIR, "imagenet1k_val")

BACKBONE_LOADERS = {
    "resnet50": load_frozen_resnet50,
    "vit_b16": load_frozen_vit_b16,
    "convnext_tiny": load_frozen_convnext_tiny,
}

BATCH_SIZES = {"resnet50": 32, "vit_b16": 16, "convnext_tiny": 16}


def main(backbone_name: str, max_images: int | None, batch_size: int | None, checkpoint_every: int) -> None:
    if backbone_name not in BACKBONE_LOADERS:
        raise ValueError(f"unknown backbone {backbone_name!r}; choose from {sorted(BACKBONE_LOADERS)}")

    out_path = os.path.join(DATA_DIR, f"logit_cache_imagenet1k_{backbone_name}.pt")
    ckpt_path = out_path + ".partial"

    print("loading ImageNet-1k validation labels (raw ILSVRC ID -> WNID -> torchvision class index)...")
    val_labels = load_val_labels()

    files = sorted(glob.glob(os.path.join(VAL_DIR, "*.JPEG")))
    if len(files) != 50000:
        print(f"WARNING: expected 50000 val images, found {len(files)} - proceeding with what's present.")
    if max_images is not None:
        files = files[:max_images]
    labels = [val_labels[val_filename_to_index(f) - 1] for f in files]

    print(f"loading frozen {backbone_name}...")
    model, preprocess, categories = BACKBONE_LOADERS[backbone_name]()
    assert len(categories) == 1000

    batch_size = batch_size or BATCH_SIZES[backbone_name]

    start_idx = 0
    all_logits_list: list[torch.Tensor] = []
    bad_indices: list[int] = []
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path)
        if ckpt["files"] == files[: len(ckpt["files"])]:
            all_logits_list = [ckpt["logits"]]
            bad_indices = ckpt["bad_indices"]
            start_idx = ckpt["logits"].shape[0]
            print(f"resuming from checkpoint at {start_idx}/{len(files)}")
        else:
            print("checkpoint file list mismatch (different max_images?) - starting fresh")

    t0 = time.time()
    for start in range(start_idx, len(files), batch_size):
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
        all_logits_list.append(logits)

        done = start + len(batch_files)
        if (start // batch_size) % 20 == 0:
            elapsed = time.time() - t0
            rate = (done - start_idx) / max(elapsed, 1e-6)
            eta = (len(files) - done) / max(rate, 1e-6)
            print(f"{done}/{len(files)}  elapsed={elapsed:.0f}s  rate={rate:.1f} img/s  eta={eta / 60:.1f}min  bad={len(bad_indices)}")

        if checkpoint_every and done % checkpoint_every < batch_size:
            torch.save(
                {"logits": torch.cat(all_logits_list, dim=0), "files": files[:done], "bad_indices": bad_indices},
                ckpt_path,
            )

    all_logits = torch.cat(all_logits_list, dim=0)
    print(f"done in {time.time() - t0:.1f}s. bad_indices: {bad_indices}")

    torch.save(
        {
            "logits": all_logits,
            "labels": torch.tensor(labels[: len(all_logits)]),
            "splits": ["id_test_full_scale"] * len(all_logits),
            "files": files[: len(all_logits)],
            "bad_indices": bad_indices,
        },
        out_path,
    )
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    print("saved cache to", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("backbone", nargs="?", default="resnet50")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=3200, help="save a resumable checkpoint every N images (0 disables)")
    args = parser.parse_args()
    main(args.backbone, args.max_images, args.batch_size, args.checkpoint_every)
