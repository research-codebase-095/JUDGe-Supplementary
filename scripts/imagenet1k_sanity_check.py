"""The mandatory blocking sanity check for the ImageNet-1k label pipeline
(scripts/imagenet1k_labels.py), run BEFORE any full-scale (50,000-image)
inference: extracts a small subset of real validation images, runs them
through frozen ResNet-50, and checks accuracy lands near torchvision's
reported 80.86% for IMAGENET1K_V2 weights (already cited in DESIGN.md
15.2/STUDY_PLAN.md 6.1) rather than near chance (~0.1%) - which is exactly
what a raw-ILSVRC-ID-used-as-class-index labeling bug would silently produce,
with no crash or warning (see imagenet1k_labels.py's docstring).

Produces data/logit_cache_imagenet1k_sanity_resnet50.pt, a small (few-hundred
image) cache that tests/test_imagenet1k_labels.py checks against as a
permanent regression test - kept deliberately separate from and much smaller
than the full logit_cache_imagenet1k_resnet50.pt (scripts/collect_logits_imagenet1k.py)
so the regression test runs in seconds, not tens of minutes, and so the full
50,000-image cache isn't a prerequisite for the test suite to pass.

Usage: python scripts/imagenet1k_sanity_check.py [--n 600]

Can run directly against data/imagenet1k_val/*.JPEG if already extracted, or
(if that directory is empty / partial, e.g. because the full download is
still in progress) streams the first N images straight out of the partially-
or fully-downloaded data/_downloads/ILSVRC2012_img_val.tar - tar entries are
stored in ascending filename order, so the first N images fully written to
disk are always readable even mid-download.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import tarfile
import time

import torch
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from deployment_reliability.backbone import load_frozen_resnet50  # noqa: E402
from imagenet1k_labels import load_val_labels, val_filename_to_index  # noqa: E402

DATA_DIR = os.path.join(REPO_ROOT, "data")
VAL_DIR = os.path.join(DATA_DIR, "imagenet1k_val")
VAL_TAR = os.path.join(DATA_DIR, "_downloads", "ILSVRC2012_img_val.tar")
OUT_PATH = os.path.join(DATA_DIR, "logit_cache_imagenet1k_sanity_resnet50.pt")

# torchvision's own reported IMAGENET1K_V2 top-1 accuracy (DESIGN.md 15.2 /
# STUDY_PLAN.md 6.1). A correct label pipeline on a few-hundred-image
# subset should land close to this, well outside sampling noise of a
# near-chance (~0.1%) mislabeling bug.
EXPECTED_ACCURACY = 0.8086


def _images_from_extracted_dir(n: int) -> list[str]:
    files = sorted(glob.glob(os.path.join(VAL_DIR, "*.JPEG")))
    return files[:n]


def _images_from_partial_tar(n: int, tmp_dir: str) -> list[str]:
    os.makedirs(tmp_dir, exist_ok=True)
    extracted = []
    with tarfile.open(VAL_TAR, "r") as tf:
        for member in tf:
            if not (member.isfile() and member.name.endswith(".JPEG")):
                continue
            data = tf.extractfile(member).read()
            dest = os.path.join(tmp_dir, os.path.basename(member.name))
            with open(dest, "wb") as out:
                out.write(data)
            extracted.append(dest)
            if len(extracted) >= n:
                break
    return extracted


def main(n: int) -> None:
    val_labels = load_val_labels()

    files = _images_from_extracted_dir(n)
    source = "extracted imagenet1k_val/"
    if len(files) < n:
        if not os.path.exists(VAL_TAR):
            raise FileNotFoundError(
                f"neither {VAL_DIR} (has {len(files)} images) nor {VAL_TAR} exist - "
                "run scripts/download_eval_data.py first"
            )
        source = "streamed from partial/full tar"
        files = _images_from_partial_tar(n, os.path.join(DATA_DIR, "_sanity_check_tmp"))

    print(f"using {len(files)} images ({source})")
    labels = [val_labels[val_filename_to_index(f) - 1] for f in files]

    print("loading frozen resnet50...")
    model, preprocess, categories = load_frozen_resnet50()
    assert len(categories) == 1000

    t0 = time.time()
    all_logits = []
    batch_size = 32
    for start in range(0, len(files), batch_size):
        batch_files = files[start : start + batch_size]
        imgs = torch.stack([preprocess(Image.open(f).convert("RGB")) for f in batch_files])
        with torch.inference_mode():
            logits = model(imgs)
        all_logits.append(logits)
    all_logits = torch.cat(all_logits, dim=0)
    labels_t = torch.tensor(labels)
    preds = all_logits.argmax(dim=1)
    accuracy = (preds == labels_t).float().mean().item()
    print(f"accuracy on {len(files)} images: {accuracy * 100:.2f}%  (elapsed {time.time() - t0:.1f}s)")

    torch.save({"logits": all_logits, "labels": labels_t, "files": files, "accuracy": accuracy}, OUT_PATH)
    print("saved sanity-check cache to", OUT_PATH)

    # Mandatory blocking check (see module docstring): fail loudly, don't
    # proceed to a full-scale run, if accuracy lands anywhere near chance.
    if accuracy < 0.5:
        raise SystemExit(
            f"SANITY CHECK FAILED: accuracy {accuracy * 100:.2f}% is far below the expected "
            f"~{EXPECTED_ACCURACY * 100:.0f}% - the label mapping is almost certainly wrong "
            "(raw ILSVRC IDs used directly instead of routed through meta.mat -> WNID -> "
            "torchvision class index). Do NOT proceed to full-scale inference until this is fixed."
        )
    print(f"PASS: within expected range of torchvision's reported {EXPECTED_ACCURACY * 100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=600)
    args = parser.parse_args()
    main(args.n)
