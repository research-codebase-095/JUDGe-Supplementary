"""Imagenette-scoped SelectiveNet-style comparison (STUDY_PLAN.md 3.6, item
6 - the largest of the six items that section's synthesis named, addressed
last).

STUDY_PLAN.md 3.6's own comparison table names SelectiveNet (Geifman &
El-Yaniv, 2019) as a real rival: "wins on frozen-model compatibility and has
a proof SelectiveNet lacks; SelectiveNet likely wins on raw accuracy, since
selection is learned jointly with the backbone rather than fit post-hoc."
That "likely" was never checked - this module is the direct check.

Confirmed CPU-only environment, no CUDA: this trains ONLY a lightweight
selection head (a single linear layer) plus a lightweight classification
head (a single linear layer) jointly, on top of the ALREADY-FROZEN backbone's
penultimate features (`mahalanobis.py`'s feature-extraction path reused
directly, not recomputed) - NOT full backbone retraining, which the original
SelectiveNet paper does and which is infeasible here even at Imagenette
scale on CPU. Because the backbone stays frozen and features are precomputed
once, training this head is cheap: gradient descent over a fixed (N, 2048)
feature tensor, not a single additional CNN forward/backward pass.

A real, disclosed methodological difference from a plain "same task, two
scoring methods" comparison: this project's existing post-hoc combiner
scores TRUST IN THE FROZEN BACKBONE'S OWN 1000-way prediction (unchanged,
general-purpose ImageNet classifier); the SelectiveNet-style head trained
here makes ITS OWN FRESH 10-way Imagenette-specific prediction from the same
frozen features. Both are scored against the same real ground-truth
Imagenette labels on the same held-out `id_test` split, and both risk-
coverage curves answer the same real deployment question ("if I reject the
bottom (1-coverage) fraction by this method's own score, what's my error
rate on what's left") - but the SelectiveNet head's classifier is
task-specific where the combiner's is not, a genuine confound this project
discloses rather than hides (see STUDY_PLAN.md 3.6/DESIGN.md's write-up).

Deliberately a SIMPLIFIED SelectiveNet: the original paper's full objective
includes a separate auxiliary classification head (trained on all points, to
prevent representation collapse under a frozen shared backbone) in addition
to the selective head; since the backbone here is already frozen and never
updated by this training at all, representation collapse of the SHARED
backbone is structurally impossible, so the auxiliary head's original
purpose does not apply and it is omitted - a disclosed simplification of the
original architecture, not a hidden one. The coverage-constrained selective
risk loss itself (Geifman & El-Yaniv, 2019, Eq. 3) IS implemented faithfully.
"""

from __future__ import annotations

import torch


class SelectiveNetHead(torch.nn.Module):
    """Two lightweight linear heads over fixed, frozen backbone features:
    `classifier` (a fresh, task-specific num_classes-way prediction) and
    `selector` (a scalar g(x) in (0,1), "confidence this input should be
    selected/predicted on," per Geifman & El-Yaniv, 2019).
    """

    def __init__(self, feature_dim: int, num_classes: int) -> None:
        super().__init__()
        self.classifier = torch.nn.Linear(feature_dim, num_classes)
        self.selector = torch.nn.Linear(feature_dim, 1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        class_logits = self.classifier(features)
        g = torch.sigmoid(self.selector(features)).squeeze(-1)
        return class_logits, g


def train_selective_net_head(
    features: torch.Tensor,
    labels: torch.Tensor,
    target_coverage: float = 0.7,
    lambda_coverage: float = 32.0,
    epochs: int = 300,
    lr: float = 0.05,
    seed: int = 0,
) -> SelectiveNetHead:
    """Fit a `SelectiveNetHead` via the coverage-constrained selective risk
    objective (Geifman & El-Yaniv, 2019, Eq. 3):

        L(f,g) = r_hat(f,g) + lambda * max(0, target_coverage - phi_hat(g))^2

    where `phi_hat(g) = mean(g(x_i))` is empirical coverage and
    `r_hat(f,g) = mean(g(x_i) * ell(f(x_i), y_i)) / phi_hat(g)` is the
    empirical selective risk (cross-entropy loss, weighted by the selection
    score, normalized by how much mass was actually selected). Full-batch
    Adam over the fixed feature tensor - cheap, since nothing here re-runs
    the frozen backbone.

    `labels` must already be a contiguous 0..num_classes-1 index space (use
    `remap_labels_to_contiguous` below if starting from raw ImageNet-1k
    synset indices, as `mahalanobis_feature_cache_resnet50.pt` stores).
    """
    torch.manual_seed(seed)
    num_classes = int(labels.max().item()) + 1
    head = SelectiveNetHead(features.shape[-1], num_classes)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)

    for _ in range(epochs):
        optimizer.zero_grad()
        class_logits, g = head(features)
        per_example_loss = torch.nn.functional.cross_entropy(class_logits, labels, reduction="none")
        phi_hat = g.mean()
        r_hat = (g * per_example_loss).mean() / phi_hat.clamp_min(1e-6)
        coverage_penalty = lambda_coverage * torch.clamp(target_coverage - phi_hat, min=0.0) ** 2
        loss = r_hat + coverage_penalty
        loss.backward()
        optimizer.step()

    head.eval()
    return head


@torch.no_grad()
def selective_net_predict(head: SelectiveNetHead, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (predicted_class, selection_score) for a fitted head - the
    selection score `g` is used as the ranking score for a risk-coverage
    curve (`router.risk_coverage_curve(g, correct)`), the exact same
    protocol this project already uses for the post-hoc combiner's score `S`.
    """
    class_logits, g = head(features)
    return class_logits.argmax(dim=-1), g


def remap_labels_to_contiguous(labels: torch.Tensor, reference_labels: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Map arbitrary integer class labels (e.g. raw ImageNet-1k synset
    indices, as cached by scripts/collect_features_mahalanobis.py) onto a
    contiguous 0..K-1 space `torch.nn.functional.cross_entropy` requires.

    `reference_labels` fixes the class ordering (e.g. combiner_fit's labels,
    so a later `id_test` remap uses the SAME index assignment even if
    id_test happens not to contain every class) - defaults to `labels`
    itself when not given. Returns (remapped_labels, original_classes) where
    `original_classes[i]` is the original label value now mapped to index i.
    """
    original_classes = torch.unique(reference_labels if reference_labels is not None else labels)
    lookup = {int(c.item()): i for i, c in enumerate(original_classes)}
    remapped = torch.tensor([lookup[int(v.item())] for v in labels], dtype=torch.int64)
    return remapped, original_classes
