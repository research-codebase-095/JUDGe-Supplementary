"""Data-splitting protocol for combiner fitting, threshold calibration, and reporting (DESIGN.md 10.5)."""

from __future__ import annotations

import torch


def three_way_split(
    n: int,
    ratios: tuple[float, float, float] = (0.4, 0.1, 0.5),
    seed: int | None = None,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Partition n indices into (combiner_fit, threshold_cal, held_out_test) index tensors.

    Mirrors DESIGN.md 10.5: the combiner-fit split is used only to fit S's
    weights (section 8), the threshold-cal split only to fit tau_hi/tau_lo
    (section 10), and the held-out test split is the only one whose numbers
    should ever be reported (section 11). Shift/OOD sets are never passed
    through this function - per 10.5 they are test-only, unconditionally, and
    should be evaluated directly against an already-fitted combiner and router.

    `device` places the returned index tensors directly on the device of the
    phi(z)/logits tensor they'll index (e.g. "cuda") - torch.randperm always
    generates on CPU internally, so this is an explicit .to() afterward, not
    an assumption that CPU indices always work (advanced indexing requires
    matching devices).
    """
    if not abs(sum(ratios) - 1.0) < 1e-6:
        raise ValueError(f"ratios must sum to 1.0, got {ratios}")
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    permutation = torch.randperm(n, generator=generator)
    if device is not None:
        permutation = permutation.to(device=device)
    n_combiner = int(round(n * ratios[0]))
    n_threshold = int(round(n * ratios[1]))
    combiner_idx = permutation[:n_combiner]
    threshold_idx = permutation[n_combiner : n_combiner + n_threshold]
    test_idx = permutation[n_combiner + n_threshold :]
    return combiner_idx, threshold_idx, test_idx
