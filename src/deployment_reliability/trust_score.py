"""Trust Score (Jiang et al., 2018) - a second feature-space evidence signal,
alongside `mahalanobis.py`'s Mahalanobis-lite, reimplemented here to give
STUDY_PLAN.md 3.6's OOD-detection comparison table a real, same-protocol
number for this method instead of the qualitative "likely stronger raw OOD
signal... traded away for architecture-agnosticism" claim it previously
carried unchecked.

The original method: fit a k-NN density estimate per class on a labeled
reference split, then score a test point by the ratio of its distance to the
nearest *different*-class region over its distance to its own predicted
class's region - high when a point sits comfortably inside its predicted
class's neighborhood and far from every rival class, low when a point is
about as close to a rival class as to its own.

Reuses exactly the same `data/mahalanobis_feature_cache_resnet50.pt`
penultimate-layer (2048-d) ResNet-50 features `mahalanobis.py` uses - no new
data collection, since Trust Score needs the identical feature-space
representation, just a different distance/density model (per-class k-NN
rather than a shared-covariance Gaussian).

Deliberately the simplified, single-layer version, same scope discipline as
`mahalanobis.py`: no multi-layer ensembling, no input-perturbation
preprocessing - a pure post-hoc, single-forward-pass feature-space distance.
"""

from __future__ import annotations

import torch


class TrustScorer:
    """Per-class k-NN density model over penultimate-layer features (Jiang et
    al., 2018's Trust Score, high-density-set variant with k=1, i.e. plain
    nearest-neighbor distance per class - the paper's own simplification for
    when a full density-level-set estimate is too expensive to fit).

    `fit(features, labels)` must be called on a labeled reference split
    (`combiner_fit`, matching `MahalanobisScorer`'s protocol - never fit on
    `imagenet_a`/`imagenet_o`/`id_test`).
    """

    def __init__(self, k: int = 1) -> None:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self.k = k
        self.classes: torch.Tensor | None = None  # (K,) actual label ids seen at fit time
        self._class_features: list[torch.Tensor] | None = None  # one (n_c, D) tensor per class

    def fit(self, features: torch.Tensor, labels: torch.Tensor) -> "TrustScorer":
        labels = labels.to(dtype=torch.int64, device=features.device)
        classes = torch.unique(labels)
        self.classes = classes
        self._class_features = [features[labels == c] for c in classes.tolist()]
        return self

    def _distance_to_every_class(self, features: torch.Tensor) -> torch.Tensor:
        """(N, K) k-th-nearest-neighbor Euclidean distance from each point to
        every class's reference features - the per-class density proxy Trust
        Score is built on (smaller = denser/closer to that class's region).
        """
        if self._class_features is None:
            raise RuntimeError("TrustScorer.fit() must be called before scoring")
        dists = []
        for class_feats in self._class_features:
            # (N, n_c) pairwise Euclidean distances to this class's reference points.
            d = torch.cdist(features, class_feats.to(device=features.device, dtype=features.dtype))
            k = min(self.k, d.shape[-1])
            kth, _ = d.kthvalue(k, dim=-1)
            dists.append(kth)
        return torch.stack(dists, dim=-1)

    def score(self, features: torch.Tensor, predicted_labels: torch.Tensor) -> torch.Tensor:
        """Trust Score = distance to the nearest OTHER class / distance to the
        PREDICTED class - the ratio the original paper defines, requiring the
        backbone's own predicted label (not just the feature vector) as an
        explicit second input, unlike `MahalanobisScorer.score` (which is
        prediction-agnostic, scoring typicality against every class).

        `predicted_labels` may include class ids outside the label space
        `fit` was called with (e.g. a 1000-way ImageNet prediction scored
        against a `TrustScorer` fit only on Imagenette's 10 synsets, the
        real situation this project's `combiner_fit`/`id_test` split
        protocol produces) - for those, "distance to own predicted class"
        is undefined (there is no reference density for a class the scorer
        was never fit on), so `own_dist` is treated as infinite for that
        point, driving its score to 0 ("no trust": the model predicted a
        class this scorer has no evidence about, the least trustworthy
        interpretation available, not an error).

        Higher = more trustworthy (closer to own predicted class, farther
        from every rival) - same "higher is more trustworthy" orientation
        convention as `MahalanobisScorer.score` and every `evidence.py`
        operator.
        """
        if self.classes is None:
            raise RuntimeError("TrustScorer.fit() must be called before scoring")
        predicted_labels = predicted_labels.to(dtype=torch.int64, device=features.device)
        dists = self._distance_to_every_class(features)  # (N, K)
        fitted_classes = self.classes.to(features.device)
        n_classes = len(fitted_classes)

        clamped = predicted_labels.clamp(fitted_classes[0].item(), fitted_classes[-1].item())
        class_index = torch.searchsorted(fitted_classes, clamped).clamp(0, n_classes - 1)
        known_class = predicted_labels == fitted_classes[class_index]

        own_dist = dists.gather(1, class_index.unsqueeze(1)).squeeze(1)
        own_dist = torch.where(known_class, own_dist, torch.full_like(own_dist, float("inf")))

        # Exclude the predicted class's own column from the "nearest other
        # class" search only when that class is actually known to the
        # scorer; for an unknown predicted class, class_index points at an
        # arbitrary nearby column that is still a legitimate "other class"
        # and must stay eligible for the min.
        masked = dists.clone()
        own_column = masked.gather(1, class_index.unsqueeze(1)).squeeze(1)
        excluded_value = torch.where(known_class, torch.full_like(own_column, float("inf")), own_column)
        masked.scatter_(1, class_index.unsqueeze(1), excluded_value.unsqueeze(1))
        nearest_other_dist = masked.amin(dim=-1)

        return nearest_other_dist / own_dist.clamp_min(torch.finfo(features.dtype).tiny)
