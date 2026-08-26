"""Tier-2 feature normalization: calibrate phi(z) against a reference distribution
of training/in-distribution logits (DESIGN.md 5.6).

Distinct from combiner.py's WeightedLinearCombiner normalization: this fits
once on a proper reference/calibration set and is meant to be reused across
many later batches, rather than re-fit per batch. Min-max normalization
(as WeightedLinearCombiner uses) saturates out-of-range values to the
boundary, which reads as "confident" for any feature where OOD inputs
produce unusually *high* raw values (e.g. energy/L2-norm on pixel noise -
see notebooks/06's Check 5). Z-scoring against a reference distribution, and
then treating distance from that reference's mean (|z-score|) as *evidence
against* trust rather than raw signed value, prevents that saturation-to-
confident failure mode: an anomalously high or low value both signal "far
from what the model was calibrated on."
"""

from __future__ import annotations

import torch


class ReferenceNormalizer:
    """z-scores phi(z) columns against a reference/calibration distribution.

    fit() should be called once on a proper reference set of phi(z) vectors
    (ideally hundreds+ of in-distribution examples, not the batch being
    scored) - fitting and scoring on the same small batch defeats the point,
    since the batch's own outliers then define its own normalization range.
    """

    def __init__(self) -> None:
        self._mean: torch.Tensor | None = None
        self._std: torch.Tensor | None = None

    def fit(self, phi_reference: torch.Tensor) -> "ReferenceNormalizer":
        self._mean = phi_reference.mean(dim=0)
        self._std = phi_reference.std(dim=0).clamp_min(1e-6)
        return self

    def transform(self, phi: torch.Tensor) -> torch.Tensor:
        """Return z-scores: (phi - reference_mean) / reference_std."""
        if self._mean is None or self._std is None:
            raise RuntimeError("ReferenceNormalizer.fit() must be called before transform()")
        mean = self._mean.to(device=phi.device, dtype=phi.dtype)
        std = self._std.to(device=phi.device, dtype=phi.dtype)
        return (phi - mean) / std

    def anomaly_score(self, phi: torch.Tensor) -> torch.Tensor:
        """Per-feature |z-score|: how far each value sits from the reference distribution,
        in either direction. High in either direction should count as *less* trustworthy,
        which is what makes this different from just calling transform() and averaging -
        transform() alone still lets an anomalously-high energy/norm read as "confident."
        """
        return self.transform(phi).abs()
