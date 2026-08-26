"""Runs a frozen backbone once over real Imagenette/ImageNet-A/ImageNet-O
images and caches logits + labels + split tags to data/logit_cache_<backbone>.pt,
so notebooks/07_real_evaluation.ipynb doesn't need to re-run inference (several
minutes on CPU) every time it's opened.

Usage: python scripts/collect_logits.py [resnet50|vit_b16|convnext_tiny]

Requires scripts/download_eval_data.py to have been run first.
"""

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

from deployment_reliability.backbone import (  # noqa: E402
    load_frozen_convnext_tiny,
    load_frozen_resnet50,
    load_frozen_vit_b16,
)

DATA_DIR = os.path.join(REPO_ROOT, "data")

IMAGENETTE_SYNSETS = [
    "n01440764", "n02102040", "n02979186", "n03000684", "n03028079",
    "n03394916", "n03417042", "n03425413", "n03445777", "n03888257",
]

BACKBONE_LOADERS = {
    "resnet50": load_frozen_resnet50,
    "vit_b16": load_frozen_vit_b16,
    "convnext_tiny": load_frozen_convnext_tiny,
}

# resnet50 uses a larger sample (it's the fastest of the three on CPU); vit_b16
# and convnext_tiny use smaller samples purely as cross-architecture parity
# checks, not full re-runs of the same evaluation (DESIGN.md 7's transfer claim
# doesn't need re-proving at the same scale, just re-checking with the same
# mechanism on a structurally different backbone each time).
SAMPLE_SIZES = {
    "resnet50": dict(combiner_fit=150, threshold_cal=30, id_test=150, imagenet_a_per_class=10, imagenet_o_all=True, batch=32),
    "vit_b16": dict(combiner_fit=50, threshold_cal=20, id_test=30, imagenet_a_per_class=1, imagenet_o_all=False, batch=16),
    "convnext_tiny": dict(combiner_fit=50, threshold_cal=20, id_test=30, imagenet_a_per_class=1, imagenet_o_all=False, batch=16),
}


def list_files_per_class(root, synsets, syn_to_idx, n_per_class, seed_offset):
    files, labels = [], []
    for synset in synsets:
        idx = syn_to_idx[synset]
        candidates = sorted(glob.glob(os.path.join(root, synset, "*")))
        rng = random.Random(hash(synset) + seed_offset)
        rng.shuffle(candidates)
        chosen = candidates[:n_per_class]
        files.extend(chosen)
        labels.extend([idx] * len(chosen))
    return files, labels


def list_files_disjoint(root, synsets, syn_to_idx, n_per_class, skip_n_per_class, seed_offset):
    files, labels = [], []
    for synset in synsets:
        idx = syn_to_idx[synset]
        candidates = sorted(glob.glob(os.path.join(root, synset, "*")))
        rng = random.Random(hash(synset) + seed_offset)
        rng.shuffle(candidates)
        chosen = candidates[skip_n_per_class : skip_n_per_class + n_per_class]
        files.extend(chosen)
        labels.extend([idx] * len(chosen))
    return files, labels


def main(backbone_name: str) -> None:
    sizes = SAMPLE_SIZES[backbone_name]
    out_path = os.path.join(DATA_DIR, f"logit_cache_{backbone_name}.pt")

    with open(os.path.join(REPO_ROOT, "notebooks", "assets", "imagenet_class_index.json")) as f:
        class_index = json.load(f)
    syn_to_idx = {v[0]: int(k) for k, v in class_index.items()}

    if backbone_name not in BACKBONE_LOADERS:
        raise ValueError(f"unknown backbone {backbone_name!r}; choose from {sorted(BACKBONE_LOADERS)}")
    print(f"loading frozen {backbone_name}...")
    model, preprocess, categories = BACKBONE_LOADERS[backbone_name]()

    random.seed(0)
    imagenette_train = os.path.join(DATA_DIR, "imagenette2-160", "train")
    imagenette_val = os.path.join(DATA_DIR, "imagenette2-160", "val")

    combiner_files, combiner_labels = list_files_per_class(
        imagenette_train, IMAGENETTE_SYNSETS, syn_to_idx, sizes["combiner_fit"], seed_offset=1
    )
    threshold_files, threshold_labels = list_files_disjoint(
        imagenette_train, IMAGENETTE_SYNSETS, syn_to_idx, sizes["threshold_cal"], sizes["combiner_fit"], seed_offset=1
    )
    test_files, test_labels = list_files_per_class(
        imagenette_val, IMAGENETTE_SYNSETS, syn_to_idx, sizes["id_test"], seed_offset=2
    )

    imagenet_a_root = os.path.join(DATA_DIR, "imagenet-a")
    a_synsets = sorted(d for d in os.listdir(imagenet_a_root) if os.path.isdir(os.path.join(imagenet_a_root, d)))
    a_files, a_labels = list_files_per_class(
        imagenet_a_root, a_synsets, syn_to_idx, sizes["imagenet_a_per_class"], seed_offset=3
    )

    imagenet_o_root = os.path.join(DATA_DIR, "imagenet-o")
    o_synsets = sorted(d for d in os.listdir(imagenet_o_root) if os.path.isdir(os.path.join(imagenet_o_root, d)))
    o_files = []
    for synset in o_synsets:
        files_here = sorted(glob.glob(os.path.join(imagenet_o_root, synset, "*")))
        if sizes["imagenet_o_all"]:
            o_files.extend(files_here)
        else:
            rng = random.Random(hash(synset) + 4)
            rng.shuffle(files_here)
            o_files.extend(files_here[:2])

    print(
        f"combiner_fit={len(combiner_files)} threshold_cal={len(threshold_files)} "
        f"id_test={len(test_files)} imagenet_a={len(a_files)} imagenet_o={len(o_files)}"
    )

    all_files = combiner_files + threshold_files + test_files + a_files + o_files
    all_splits = (
        ["combiner_fit"] * len(combiner_files)
        + ["threshold_cal"] * len(threshold_files)
        + ["id_test"] * len(test_files)
        + ["imagenet_a"] * len(a_files)
        + ["imagenet_o"] * len(o_files)
    )
    all_labels = combiner_labels + threshold_labels + test_labels + a_labels + [-1] * len(o_files)

    batch_size = sizes["batch"]
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
        if (start // batch_size) % 20 == 0:
            print(f"{start + len(batch_files)}/{len(all_files)}  elapsed={time.time() - t0:.0f}s  bad={len(bad_indices)}")

    all_logits = torch.cat(all_logits, dim=0)
    print(f"done in {time.time() - t0:.1f}s. bad_indices: {bad_indices}")

    torch.save(
        {
            "logits": all_logits,
            "labels": torch.tensor(all_labels),
            "splits": all_splits,
            "files": all_files,
            "bad_indices": bad_indices,
        },
        out_path,
    )
    print("saved cache to", out_path)


if __name__ == "__main__":
    backbone = sys.argv[1] if len(sys.argv) > 1 else "resnet50"
    main(backbone)
