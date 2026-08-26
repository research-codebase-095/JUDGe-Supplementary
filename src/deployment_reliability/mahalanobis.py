"""Mahalanobis-lite feature-space evidence signal (STUDY_PLAN.md 3.6, item 2 -
the highest-priority item in that section's synthesis, since it is the one
most likely to change what the synthesis paragraph honestly says).

STUDY_PLAN.md 3.6's comparison table names Mahalanobis (Lee et al., 2018) as
likely carrying a stronger raw OOD signal than anything this project's
logit-only feature set (`features.py`) can extract, at the cost of needing
feature access (not just logits) - a real, disclosed trade-off, not
previously checked directly. This module is that direct check: a class-
conditional Gaussian model (shared covariance, the same simplifying
restriction the original Lee et al. formulation uses) fit once on
penultimate-layer features from `backbone.py`'s new
`logits_and_features_for_images`, closed-form (no gradient steps, consistent
with DESIGN.md 3.2's computational-cost discipline - one covariance inverse
at fit time, one quadratic form per scored point).

Deliberately NOT the full Lee et al. (2018) method: no input-perturbation
step (their "ODIN-style" preprocessing noise), no multi-layer ensembling
across intermediate feature maps - just the single-layer, single-pass
version, since anything requiring extra forward/backward passes would
violate DESIGN.md 3.2 the same way ODIN itself does (STUDY_PLAN.md §3.6's own table already
notes this cost trade-off for ODIN). Called "Mahalanobis-lite" throughout
this project's docs specifically to keep that scope honest.

This is an explicitly OPTIONAL, additive evidence dimension: nothing in
`features.py`/`evidence.py`/`providers.py`'s single-tensor (logits-only)
interface changes because this module exists, and `reliability.py`'s
`estimate_reliability_state`/`ReliabilityPipeline` only compute this signal
when a caller explicitly passes both a fitted `MahalanobisScorer` and the
matching penultimate features - the default logit-only path is untouched.
"""

from __future__ import annotations

import torch


class MahalanobisScorer:
    """Class-conditional Gaussians over penultimate-layer features, with a
    single covariance shared across all classes (Lee et al., 2018's
    simplifying assumption - a full per-class covariance would need far more
    fit data per class than this project's small combiner_fit splits have).

    `fit(features, labels)` must be called on a labeled reference split
    (`combiner_fit`, per DESIGN.md 10.5 - this scorer is fit exactly like
    `combiner.py`'s combiners and `normalization.py`'s ReferenceNormalizer,
    never on `imagenet_a`/`imagenet_o`/`id_test`). Both `features` and
    `labels` must come from the same batch of inputs, in the same order, as
    produced by `backbone.py`'s `logits_and_features_for_images` and the
    matching ground-truth class labels.
    """

    def __init__(self, eps: float = 1e-6) -> None:
        self.eps = eps
        self.classes: torch.Tensor | None = None  # (K,) actual label ids seen at fit time
        self.class_means: torch.Tensor | None = None  # (K, D)
        self.precision: torch.Tensor | None = None  # (D, D) inverse of the shared covariance

    def fit(self, features: torch.Tensor, labels: torch.Tensor) -> "MahalanobisScorer":
        labels = labels.to(dtype=torch.int64, device=features.device)
        classes = torch.unique(labels)
        d = features.shape[-1]

        means = []
        centered_chunks = []
        for c in classes.tolist():
            feats_c = features[labels == c]
            mean_c = feats_c.mean(dim=0)
            means.append(mean_c)
            centered_chunks.append(feats_c - mean_c)

        self.classes = classes
        self.class_means = torch.stack(means, dim=0)

        centered = torch.cat(centered_chunks, dim=0)
        n = centered.shape[0]
        # Pooled within-class scatter / (n - K): the standard shared-covariance
        # estimator (Lee et al., 2018 §2.1), not a naive single global
        # covariance over the unclassed features (which would be inflated by
        # between-class separation, not just within-class spread).
        dof = max(n - len(classes), 1)
        cov = (centered.T @ centered) / dof
        cov = cov + self.eps * torch.eye(d, dtype=cov.dtype, device=cov.device)
        self.precision = torch.linalg.inv(cov)
        return self

    def _sq_distances_to_all_classes(self, features: torch.Tensor) -> torch.Tensor:
        """(N, K) squared Mahalanobis distance from each point to every class mean."""
        if self.class_means is None or self.precision is None:
            raise RuntimeError("MahalanobisScorer.fit() must be called before scoring")
        means = self.class_means.to(device=features.device, dtype=features.dtype)
        precision = self.precision.to(device=features.device, dtype=features.dtype)
        diff = features.unsqueeze(1) - means.unsqueeze(0)  # (N, K, D)
        sq = torch.einsum("nkd,de,nke->nk", diff, precision, diff)
        return sq.clamp_min(0.0)

    def nearest_class_distance(self, features: torch.Tensor) -> torch.Tensor:
        """Minimum Mahalanobis distance to any class's mean - lower means the
        point sits in a region the reference data actually populated; higher
        means it's far from every class the scorer was fit on (the raw
        OOD-style quantity, before the sign flip `score()` applies).
        """
        return self._sq_distances_to_all_classes(features).amin(dim=-1).sqrt()

    def nearest_class(self, features: torch.Tensor) -> torch.Tensor:
        """argmin class label (in the original label space passed to `fit`) -
        a Mahalanobis-only nearest-class-mean classifier. Diagnostic only:
        this project's actual predictions still come from the frozen
        backbone's own softmax argmax, never from this scorer.
        """
        idx = self._sq_distances_to_all_classes(features).argmin(dim=-1)
        return self.classes.to(device=features.device)[idx]

    def score(self, features: torch.Tensor) -> torch.Tensor:
        """Trust-oriented score: negative nearest-class distance, so HIGHER =
        more trustworthy/typical, the same orientation convention
        `FEATURE_DIRECTIONS` and every `evidence.py` operator use, even
        though this scorer lives outside `phi(z)` and is not itself listed
        in `DEFAULT_FEATURE_NAMES`.
        """
        return -self.nearest_class_distance(features)
