"""Runs frozen ResNet-50 twice per batch (a forward pass to get the input
gradient, a perturbed second forward pass) over exactly the same real images
already used by data/logit_cache_resnet50.pt (same files/splits/labels,
reused directly rather than re-sampled), producing ODIN (Liang et al., 2018)
scores and caching them to data/odin_cache_resnet50.pt.

STUDY_PLAN.md 3.6's OOD-detection comparison table previously carried ODIN
as a qualitative claim only ("this design wins on cost for comparable-or-
worse separation; ODIN kept only as an evaluation baseline") - this script
is what makes that a real, same-protocol number instead.

ODIN algorithm (the paper's own two mechanisms, reimplemented directly from
the paper rather than a third-party port):
  1. Forward pass with the input tensor's gradient enabled (NOT
     torch.inference_mode(), unlike every other collector in this project -
     input-pixel gradients are exactly the thing ODIN needs and every other
     script here has no reason to compute).
  2. y_hat = argmax(raw logits) - the model's own predicted class, treated
     as a fixed target for the next step (not the true label - ODIN never
     uses ground truth, consistent with being a training-free, label-free
     detector).
  3. loss = CrossEntropy(logits / T, y_hat), T=1000 (the paper's own
     reported default, not tuned here - this project's job is a same-
     protocol comparison, not re-deriving ODIN's own hyperparameters).
  4. Perturb the input AGAINST the loss gradient (i.e. in the direction that
     INCREASES the predicted class's temperature-scaled confidence):
     x_perturbed = x - epsilon * (sign(grad_x(loss)) / channel_std),
     epsilon=0.0014 (the paper's own reported default for their most
     comparable setting). The per-channel division by the preprocessing
     normalization std (ImageNet: [0.229, 0.224, 0.225]) matches the
     official ODIN implementation, which applies epsilon in *pixel* space
     even though the perturbed tensor lives in normalized space - dividing
     the sign by std is what keeps epsilon's stated magnitude meaningful in
     that space. (A first version of this script omitted this division,
     making its effective perturbation ~4-4.5x smaller than the paper's
     epsilon=0.0014 actually produces in the reference pipeline - found via
     an independent numerical audit, fixed here, and the real cache
     regenerated.)
  5. A second forward pass on x_perturbed (no gradient needed here), scored
     by the max temperature-scaled softmax probability - the final ODIN
     score, higher = more in-distribution-looking.

Deliberately NOT tuning T/epsilon on this project's own data - using the
paper's stated defaults is what makes this a fair "as-published" comparison
rather than a version quietly optimized to look better or worse than it
would out of the box.

Usage: python scripts/collect_odin_scores.py

Requires data/logit_cache_resnet50.pt to already exist (scripts/collect_logits.py resnet50).
"""

import os
import sys
import time

import torch
import torch.nn.functional as F
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from deployment_reliability.backbone import load_frozen_resnet50  # noqa: E402

DATA_DIR = os.path.join(REPO_ROOT, "data")
LOGIT_CACHE_PATH = os.path.join(DATA_DIR, "logit_cache_resnet50.pt")
OUT_PATH = os.path.join(DATA_DIR, "odin_cache_resnet50.pt")
BATCH_SIZE = 32

TEMPERATURE = 1000.0
EPSILON = 0.0014
# ImageNet normalization std (ResNet50_Weights.IMAGENET1K_V2's own preprocess,
# confirmed directly: mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]) - the
# reference ODIN implementation divides the perturbation's sign by this per
# channel before scaling by epsilon, since epsilon is stated in pixel space
# but the perturbed tensor lives in normalized space.
NORM_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

_DATA_SUBDIR_MARKERS = ("imagenette2-160", "imagenet-a", "imagenet-o")


def _remap_to_local_data_dir(path: str) -> str:
    normalized = path.replace("\\", "/")
    for marker in _DATA_SUBDIR_MARKERS:
        idx = normalized.find(marker)
        if idx != -1:
            return os.path.join(DATA_DIR, *normalized[idx:].split("/"))
    raise ValueError(f"cannot remap cached file path onto a known data/ subdirectory: {path}")


def odin_scores_for_batch(model, batch: torch.Tensor) -> torch.Tensor:
    """One batch through the full ODIN procedure. `batch` is already
    preprocessed (normalized, stacked) - NOT wrapped in torch.inference_mode
    by the caller, since this function needs a live autograd graph for the
    first forward pass.
    """
    batch = batch.clone().requires_grad_(True)
    logits = model(batch)
    y_hat = logits.argmax(dim=-1).detach()
    loss = F.cross_entropy(logits / TEMPERATURE, y_hat)
    (grad,) = torch.autograd.grad(loss, batch)

    with torch.no_grad():
        perturbed = batch - EPSILON * (grad.sign() / NORM_STD.to(grad.device, grad.dtype))
        perturbed_logits = model(perturbed)
        scores = F.softmax(perturbed_logits / TEMPERATURE, dim=-1).amax(dim=-1)
    return scores.detach()


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

    all_scores = []
    t0 = time.time()
    for start in range(0, len(files), BATCH_SIZE):
        batch_files = files[start : start + BATCH_SIZE]
        imgs = []
        for i, f in enumerate(batch_files):
            if (start + i) in bad_indices:
                imgs.append(torch.zeros(3, 224, 224))
                continue
            imgs.append(preprocess(Image.open(f).convert("RGB")))
        batch = torch.stack(imgs, dim=0)
        scores = odin_scores_for_batch(model, batch)
        all_scores.append(scores)
        if (start // BATCH_SIZE) % 10 == 0:
            print(f"{start + len(batch_files)}/{len(files)}  elapsed={time.time() - t0:.0f}s")

    scores = torch.cat(all_scores, dim=0)
    print(f"done in {time.time() - t0:.1f}s. scores shape: {tuple(scores.shape)}")

    torch.save(
        {
            "scores": scores,
            "logits": cached_logits,
            "labels": labels,
            "splits": splits,
            "bad_indices": cache["bad_indices"],
            "temperature": TEMPERATURE,
            "epsilon": EPSILON,
        },
        OUT_PATH,
    )
    print("saved cache to", OUT_PATH)


if __name__ == "__main__":
    main()
