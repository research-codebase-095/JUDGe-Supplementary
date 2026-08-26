"""Optional post-hoc calibration of S's probability semantics, per DESIGN.md section 9.

Sits between the confidence function (section 8) and thresholding (section 10).
Recalibrates S so it can be read as P(correct) without changing its ranking
(temperature/Platt scaling are monotonic transforms; isotonic regression is not,
by design - see DESIGN.md 9's non-monotonic-miscalibration fallback).
"""

from __future__ import annotations

import torch

_EPS = 1e-6


class TemperatureScaling:
    """S_cal = sigmoid(logit(S) / T), fit by minimizing NLL (DESIGN.md 9).

    The default recommendation: cheapest, most standard, and - because it's a
    single monotonic scalar - preserves S's ranking exactly. `log_t`'s dtype
    and device are set from `s` inside fit(), not hardcoded at construction,
    so this works unchanged on CPU or GPU inputs.
    """

    def __init__(self) -> None:
        self.log_t: torch.Tensor | None = None  # T = exp(log_t) keeps T > 0
        self._fitted = False

    def fit(self, s: torch.Tensor, y: torch.Tensor, max_iter: int = 200, lr: float = 0.1) -> "TemperatureScaling":
        self.log_t = torch.zeros(1, dtype=s.dtype, device=s.device, requires_grad=True)
        y = y.to(dtype=s.dtype, device=s.device)
        z = torch.logit(s.clamp(_EPS, 1 - _EPS))
        optimizer = torch.optim.LBFGS([self.log_t], lr=lr, max_iter=max_iter, line_search_fn="strong_wolfe")

        def closure():
            optimizer.zero_grad()
            t = torch.exp(self.log_t)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(z / t, y)
            loss.backward()
            return loss

        optimizer.step(closure)
        self._fitted = True
        return self

    def transform(self, s: torch.Tensor) -> torch.Tensor:
        if not self._fitted or self.log_t is None:
            raise RuntimeError("TemperatureScaling.fit() must be called before transform()")
        with torch.no_grad():
            t = torch.exp(self.log_t).to(device=s.device, dtype=s.dtype)
            z = torch.logit(s.clamp(_EPS, 1 - _EPS))
            return torch.sigmoid(z / t)


class PlattScaling:
    """S_cal = sigmoid(a * S + b), fit by minimizing NLL (DESIGN.md 9).

    Slightly more flexible than temperature scaling (2 free parameters vs. 1),
    still a monotonic transform provided the fitted slope `a` is positive.
    `a`/`b`'s dtype and device are set from `s` inside fit(), not hardcoded at
    construction.
    """

    def __init__(self) -> None:
        self.a: torch.Tensor | None = None
        self.b: torch.Tensor | None = None
        self._fitted = False

    def fit(self, s: torch.Tensor, y: torch.Tensor, max_iter: int = 200, lr: float = 0.1) -> "PlattScaling":
        self.a = torch.ones(1, dtype=s.dtype, device=s.device, requires_grad=True)
        self.b = torch.zeros(1, dtype=s.dtype, device=s.device, requires_grad=True)
        y = y.to(dtype=s.dtype, device=s.device)
        optimizer = torch.optim.LBFGS([self.a, self.b], lr=lr, max_iter=max_iter, line_search_fn="strong_wolfe")

        def closure():
            optimizer.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(self.a * s + self.b, y)
            loss.backward()
            return loss

        optimizer.step(closure)
        self._fitted = True
        return self

    def transform(self, s: torch.Tensor) -> torch.Tensor:
        if not self._fitted or self.a is None or self.b is None:
            raise RuntimeError("PlattScaling.fit() must be called before transform()")
        with torch.no_grad():
            a = self.a.to(device=s.device, dtype=s.dtype)
            b = self.b.to(device=s.device, dtype=s.dtype)
            return torch.sigmoid(a * s + b)


class IsotonicCalibration:
    """Non-parametric monotonic calibration via pool-adjacent-violators (DESIGN.md 9).

    Fallback when reliability diagrams show non-monotonic miscalibration that
    temperature/Platt scaling can't fix. Needs more calibration data than the
    parametric options to avoid overfitting.
    """

    def __init__(self) -> None:
        self._x: torch.Tensor | None = None
        self._y: torch.Tensor | None = None

    def fit(self, s: torch.Tensor, y: torch.Tensor) -> "IsotonicCalibration":
        order = torch.argsort(s)
        self._x = s[order]
        self._y = _pool_adjacent_violators(y.to(dtype=s.dtype, device=s.device)[order])
        return self

    def transform(self, s: torch.Tensor) -> torch.Tensor:
        if self._x is None or self._y is None:
            raise RuntimeError("IsotonicCalibration.fit() must be called before transform()")
        x = self._x.to(device=s.device, dtype=s.dtype)
        y_fitted = self._y.to(device=s.device, dtype=s.dtype)
        # Step-function lookup: each query maps to the fitted value of the
        # largest fitted breakpoint not exceeding it.
        idx = torch.searchsorted(x, s.contiguous(), right=True) - 1
        idx = idx.clamp(0, len(y_fitted) - 1)
        return y_fitted[idx]


def _pool_adjacent_violators(y: torch.Tensor) -> torch.Tensor:
    """Standard O(n) PAVA: the closest non-decreasing sequence to y in least-squares sense."""
    values: list[float] = []
    weights: list[float] = []
    for level in y.tolist():
        values.append(level)
        weights.append(1.0)
        while len(values) > 1 and values[-2] > values[-1]:
            v2, w2 = values.pop(), weights.pop()
            v1, w1 = values.pop(), weights.pop()
            merged_w = w1 + w2
            values.append((v1 * w1 + v2 * w2) / merged_w)
            weights.append(merged_w)
    expanded: list[float] = []
    for v, w in zip(values, weights):
        expanded.extend([v] * int(round(w)))
    return torch.tensor(expanded, dtype=y.dtype, device=y.device)
