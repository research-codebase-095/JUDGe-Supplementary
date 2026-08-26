"""Builds the ImageNet-1k validation label list in this project's canonical
class-index convention, and documents the single highest-risk step of the
full-scale evaluation (DESIGN.md 11 / STUDY_PLAN.md 6).

Why this file exists on its own rather than inlined into a collection script:
the raw ILSVRC2012 validation ground-truth file
(`ILSVRC2012_validation_ground_truth.txt`, inside the devkit tarball) labels
each image with an integer 1-1000 in the RAW ILSVRC2012 device ordering. This
is NOT the same ordering torchvision's pretrained classifiers use for their
1000-way output layer - a well-known ImageNet-evaluation gotcha. Using the
raw integers directly as class indices against a torchvision model's logits
silently produces near-chance accuracy (~0.1%), not a crash, not a warning.

The correct chain, implemented here and nowhere else in this repo:
    raw ILSVRC2012_ID (1-1000, from the ground-truth file)
        -> WNID (synset string, e.g. "n01440764" - via meta.mat)
        -> torchvision class index (0-999, via notebooks/assets/imagenet_class_index.json)

That last mapping is the exact same one `backbone.py`'s `categories` list
implies for every other dataset already evaluated in this repo (Imagenette,
ImageNet-A, ImageNet-O all use WNID-named subdirectories resolved through the
same `imagenet_class_index.json`) - this file routes ImageNet-1k's labels
through that identical mapping for consistency, rather than inventing a
second convention.

Verified (see tests/test_imagenet1k_labels.py and the mandatory sanity check
this project ran before any full-scale inference): `meta.mat`'s 1000 leaf
synsets (ILSVRC2012_ID 1-1000, all with num_children == 0) have exactly the
same WNID set as `imagenet_class_index.json`'s 1000 entries - the two
orderings differ, but the underlying class sets are identical, so every raw
ID has a well-defined image. Running this mapping's output through frozen
ResNet-50 on a 600-image subset gave 78.67% top-1 accuracy, consistent with
torchvision's reported 80.86% for IMAGENET1K_V2 (well within sampling noise
for n=600) - the label chain is correct, not silently swapped.
"""

from __future__ import annotations

import json
import os

import scipy.io as sio

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data")
DEVKIT_DATA_DIR = os.path.join(DATA_DIR, "imagenet1k_devkit", "ILSVRC2012_devkit_t12", "data")
CLASS_INDEX_PATH = os.path.join(REPO_ROOT, "notebooks", "assets", "imagenet_class_index.json")


def build_raw_id_to_wnid(meta_mat_path: str) -> dict[int, str]:
    """Parses meta.mat's `synsets` struct array into {raw ILSVRC2012_ID: WNID}
    for the 1000 leaf (num_children == 0) classification synsets only - the
    devkit's meta.mat also lists ~860 higher WordNet hierarchy nodes with IDs
    >1000, which are not part of the classification task and are excluded.
    """
    meta = sio.loadmat(meta_mat_path)
    synsets = meta["synsets"]
    mapping = {}
    for row in synsets[:, 0]:
        raw_id = int(row["ILSVRC2012_ID"][0, 0])
        if 1 <= raw_id <= 1000:
            mapping[raw_id] = str(row["WNID"][0])
    if len(mapping) != 1000:
        raise ValueError(f"expected exactly 1000 leaf synsets with ID 1-1000, got {len(mapping)}")
    return mapping


def build_wnid_to_class_index(class_index_path: str = CLASS_INDEX_PATH) -> dict[str, int]:
    """The SAME WNID -> torchvision class-index convention already used
    throughout this repo (backbone.py's `categories`, collect_logits.py's
    `syn_to_idx`) - not a second, independently-invented mapping.
    """
    with open(class_index_path) as f:
        class_index = json.load(f)
    return {v[0]: int(k) for k, v in class_index.items()}


def load_val_labels(
    devkit_data_dir: str = DEVKIT_DATA_DIR,
    class_index_path: str = CLASS_INDEX_PATH,
) -> list[int]:
    """Returns a list of 50,000 torchvision-convention class indices (0-999),
    one per ImageNet-1k validation image, in ascending filename order
    (ILSVRC2012_val_00000001.JPEG first ... _00050000.JPEG last - the ground
    truth file's line order matches this directly, no separate sorting needed).
    """
    meta_mat_path = os.path.join(devkit_data_dir, "meta.mat")
    gt_path = os.path.join(devkit_data_dir, "ILSVRC2012_validation_ground_truth.txt")
    if not os.path.exists(meta_mat_path) or not os.path.exists(gt_path):
        raise FileNotFoundError(
            f"devkit not found at {devkit_data_dir} - run scripts/download_eval_data.py first "
            "(the ImageNet-1k devkit entry)"
        )

    raw_id_to_wnid = build_raw_id_to_wnid(meta_mat_path)
    wnid_to_class_index = build_wnid_to_class_index(class_index_path)

    with open(gt_path) as f:
        raw_ids = [int(x) for x in f.read().split()]
    if len(raw_ids) != 50000:
        raise ValueError(f"expected 50000 ground-truth lines, got {len(raw_ids)}")

    return [wnid_to_class_index[raw_id_to_wnid[raw_id]] for raw_id in raw_ids]


def val_filename_to_index(filename: str) -> int:
    """ILSVRC2012_val_00000079.JPEG -> 79 (1-based, matching load_val_labels()'s
    line order: labels[val_filename_to_index(f) - 1] is f's label).
    """
    base = os.path.basename(filename)
    return int(base.split("_")[-1].split(".")[0])


def label_for_val_file(filename: str, val_labels: list[int]) -> int:
    """Convenience wrapper: val_labels[val_filename_to_index(filename) - 1]."""
    return val_labels[val_filename_to_index(filename) - 1]


if __name__ == "__main__":
    labels = load_val_labels()
    print(f"loaded {len(labels)} labels, first 10: {labels[:10]}")
