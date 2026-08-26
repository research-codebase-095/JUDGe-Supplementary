"""Regression tests for the ImageNet-1k label-mapping pipeline
(scripts/imagenet1k_labels.py) - the single highest-risk step of the
full-scale evaluation (DESIGN.md 11 / STUDY_PLAN.md 6, PROGRESS_REPORT.md's
full-scale-evaluation section).

Two things are locked in here, matching this project's other real-data test
conventions (tests/test_combiner.py's real-ResNet50-data tests: assert-exists
on the relevant cache/data file with a clear "run script X first" message,
rather than a network call or multi-minute inference run inside the test
suite itself):

1. The mapping's internal consistency (raw ILSVRC ID -> WNID -> torchvision
   class index) - runs directly off the devkit files, no image data needed.
2. The mandatory blocking sanity check's result: real ResNet-50 inference on
   a real image subset, labeled through this mapping, should land near
   torchvision's reported 80.86% (IMAGENET1K_V2), not near chance (~0.1%,
   the signature of raw-ILSVRC-ID-used-as-class-index, DESIGN.md's known
   ImageNet-evaluation gotcha). Uses the small, fast
   logit_cache_imagenet1k_sanity_resnet50.pt artifact (scripts/imagenet1k_sanity_check.py)
   rather than the full 50,000-image cache, so this test runs in
   milliseconds, not tens of minutes, and doesn't require the full-scale
   download/inference to have completed.
"""

import os

import pytest
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")

import sys  # noqa: E402

sys.path.insert(0, SCRIPTS_DIR)

DEVKIT_DATA_DIR = os.path.join(REPO_ROOT, "data", "imagenet1k_devkit", "ILSVRC2012_devkit_t12", "data")
SANITY_CACHE = os.path.join(REPO_ROOT, "data", "logit_cache_imagenet1k_sanity_resnet50.pt")

# torchvision's own reported IMAGENET1K_V2 top-1 accuracy (DESIGN.md 15.2 /
# STUDY_PLAN.md 6.1). The sanity-check subset is small (a few hundred
# images), so this bound is generous - it exists to catch a label-mapping
# bug (which produces ~0.1% accuracy, not a small drift), not to validate
# ResNet-50's exact reported accuracy at this sample size.
MIN_EXPECTED_ACCURACY = 0.60


def test_meta_mat_leaf_synsets_match_torchvision_class_index():
    if not os.path.exists(DEVKIT_DATA_DIR):
        pytest.skip("ImageNet-1k devkit not found - run scripts/download_eval_data.py first")

    from imagenet1k_labels import build_raw_id_to_wnid, build_wnid_to_class_index

    meta_mat_path = os.path.join(DEVKIT_DATA_DIR, "meta.mat")
    raw_id_to_wnid = build_raw_id_to_wnid(meta_mat_path)
    wnid_to_class_index = build_wnid_to_class_index()

    # Exactly 1000 leaf classification synsets, IDs 1-1000 with no gaps.
    assert set(raw_id_to_wnid.keys()) == set(range(1, 1001))

    # The devkit's WNID set is EXACTLY the same set already used by
    # backbone.py/collect_logits.py for every other dataset in this repo -
    # not a second, independently-invented class vocabulary.
    assert set(raw_id_to_wnid.values()) == set(wnid_to_class_index.keys())

    # Spot-check a couple of well-known entries against their expected WNIDs
    # (kit fox is ILSVRC2012_ID 1 in the raw devkit ordering).
    assert raw_id_to_wnid[1] == "n02119789"


def test_load_val_labels_returns_50000_valid_class_indices():
    if not os.path.exists(DEVKIT_DATA_DIR):
        pytest.skip("ImageNet-1k devkit not found - run scripts/download_eval_data.py first")

    from imagenet1k_labels import load_val_labels

    labels = load_val_labels()
    assert len(labels) == 50000
    assert all(0 <= label <= 999 for label in labels)
    # Real ImageNet-1k val labels should exercise close to the full 1000-way
    # class set, not collapse onto a handful of indices (a coarse sanity
    # check that the mapping isn't degenerate).
    assert len(set(labels)) > 900


def test_val_filename_to_index_parses_ilsvrc_naming():
    from imagenet1k_labels import val_filename_to_index

    assert val_filename_to_index("ILSVRC2012_val_00000001.JPEG") == 1
    assert val_filename_to_index("ILSVRC2012_val_00050000.JPEG") == 50000
    assert val_filename_to_index("/some/dir/ILSVRC2012_val_00000079.JPEG") == 79


def test_sanity_check_resnet50_accuracy_confirms_correct_label_mapping():
    # This is the permanent regression test for the mandatory blocking check
    # described in scripts/imagenet1k_sanity_check.py's docstring: a wrong
    # label mapping (raw ILSVRC ID used directly as a torchvision class
    # index) produces accuracy near chance (~0.1%) with no crash - silently
    # wrong, not loudly broken. This test would have caught that failure
    # mode had it occurred; it did not (see PROGRESS_REPORT.md's full-scale
    # evaluation section for the actual measured accuracy).
    if not os.path.exists(SANITY_CACHE):
        pytest.skip(
            "ImageNet-1k sanity-check cache not found - run "
            "scripts/imagenet1k_sanity_check.py first"
        )
    cache = torch.load(SANITY_CACHE)
    logits, labels = cache["logits"], cache["labels"]
    assert logits.shape[0] == labels.shape[0]
    assert logits.shape[0] >= 100, "sanity-check subset too small to be statistically meaningful"

    preds = logits.argmax(dim=-1)
    accuracy = (preds == labels).float().mean().item()
    assert accuracy >= MIN_EXPECTED_ACCURACY, (
        f"ImageNet-1k label-mapping sanity check FAILED: accuracy {accuracy:.4f} is far below "
        f"the expected ~0.81 (torchvision IMAGENET1K_V2) - this is the signature of the raw "
        f"ILSVRC-ID-used-as-class-index bug documented in imagenet1k_labels.py, not normal "
        f"sampling noise."
    )
    # Also confirm this isn't accidentally near-chance in a way the bound
    # above wouldn't obviously catch (redundant but cheap, and directly
    # names the failure mode this test exists to prevent).
    assert accuracy > 0.05, "accuracy is near 1/1000 chance level - label mapping is broken"
