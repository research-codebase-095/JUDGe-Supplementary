"""Confidence function (score combiner): phi(z) -> S in [0,1], per DESIGN.md section 8.

Deliberately excludes calibration (section 9) and thresholding (section 10),
which are separate pipeline stages (section 4).
"""

from __future__ import annotations

import math

import torch

from .features import DEFAULT_FEATURE_DIRECTIONS


class WeightedLinearCombiner:
    """(a) Fixed-weight heuristic combiner (DESIGN.md 8a).

    Orients each feature by its DEFAULT_FEATURE_DIRECTIONS sign (so a
    lower-is-better feature like entropy is flipped before it's combined with
    higher-is-better features - without this, entropy would silently
    contribute in the wrong direction), applies per-feature min-max
    normalization fit on a calibration split, then averages the oriented,
    normalized features with equal weight and squashes with a sigmoid. Zero
    training beyond fitting the normalization range - the reference baseline /
    ablation floor for the learned combiners below.

    Equal-weight min-max averaging is still naive in a way direction-fixing
    doesn't solve: an out-of-reference-range value (e.g. energy/L2-norm on an
    OOD input) can still saturate to the normalized maximum and read as
    "confident" rather than "anomalous." `ReferenceNormalizer`
    (`normalization.py`) is the DESIGN.md 5.6 Tier-2 building block for a more
    principled fix, once a real calibration set is available.
    """

    def __init__(self, feature_directions: torch.Tensor | None = None) -> None:
        # Stored as given (may be None); resolved to phi's device/dtype/width
        # lazily in fit(), so this never hardcodes CPU/float32 or a fixed
        # feature count independent of what's actually passed in.
        self._feature_directions_override = feature_directions
        self.feature_directions: torch.Tensor | None = None
        self._min: torch.Tensor | None = None
        self._max: torch.Tensor | None = None

    def _resolve_feature_directions(self, phi: torch.Tensor) -> torch.Tensor:
        if self._feature_directions_override is not None:
            directions = self._feature_directions_override
        else:
            if phi.shape[-1] != DEFAULT_FEATURE_DIRECTIONS.numel():
                raise ValueError(
                    f"phi has {phi.shape[-1]} features but no feature_directions was given and "
                    f"DEFAULT_FEATURE_DIRECTIONS covers {DEFAULT_FEATURE_DIRECTIONS.numel()} - pass "
                    "feature_directions explicitly for a non-default feature set."
                )
            directions = DEFAULT_FEATURE_DIRECTIONS
        return directions.to(device=phi.device, dtype=phi.dtype)

    def fit(self, phi: torch.Tensor) -> "WeightedLinearCombiner":
        self.feature_directions = self._resolve_feature_directions(phi)
        oriented = phi * self.feature_directions
        self._min = oriented.amin(dim=0)
        self._max = oriented.amax(dim=0)
        return self

    def score(self, phi: torch.Tensor) -> torch.Tensor:
        if self._min is None or self._max is None or self.feature_directions is None:
            raise RuntimeError("WeightedLinearCombiner.fit() must be called before score()")
        oriented = phi * self.feature_directions
        span = (self._max - self._min).clamp_min(1e-8)
        normalized = ((oriented - self._min) / span).clamp(0.0, 1.0)
        # Center the equal-weight average at logit 0 (score 0.5) and spread it
        # across the sigmoid's sensitive range rather than clustering near 0.5.
        return torch.sigmoid(normalized.mean(dim=-1) * 4.0 - 2.0)


class LogisticRegressionCombiner:
    """(b) Learned linear combiner via logistic regression - primary recommendation (DESIGN.md 8b).

    Fits weights w, b on binary correctness labels y = 1[argmax(z) == true_label].
    Still a linear score, but now the weights are learned - satisfies "handcrafted
    features, learnable combination" without ever touching the backbone. Trained
    with full-batch LBFGS since the parameter count (num_features + 1 bias) is tiny.

    Parameter count, dtype, and device are all inferred from `phi` inside fit()
    rather than fixed at construction time, so this works unchanged whether
    phi(z) has DESIGN.md's 5 recommended features, an ablated subset, or a
    future extension's feature set - and whether phi lives on CPU or GPU.

    L2-regularized (ridge) by default: DESIGN.md 15.6 found that on a smaller
    fit sample (ConvNeXt-Tiny, n=500) combined with strong MSP/entropy
    multicollinearity (DESIGN.md 15.2, r=-0.92), the unregularized fit produced
    weight magnitudes an order of magnitude larger than on ResNet-50/ViT-B/16 -
    not the classical perfectly-separable-data degeneracy (checked directly:
    no fitted scores saturated near 0/1), but consistent with a shallow, poorly
    constrained loss direction along the near-redundant feature axis that a
    small ridge penalty resolves without materially changing predictions on
    well-conditioned data. `l2=0.0` recovers the original unregularized fit.

    Also fits a Laplace approximation to the epistemic uncertainty of (weight,
    bias) itself (STUDY_PLAN.md Objective 6; DESIGN.md's uncertainty-
    quantification section), exposed via `epistemic_std()`/`score_with_uncertainty()`.
    This is closed-form - one Hessian evaluation at the already-fitted optimum,
    no additional forward passes and no refitting/ensembling - so it stays
    within the same computational-cost constraint (DESIGN.md 3.2) the rest of
    this package is held to.
    """

    def __init__(self, class_balanced: bool = True, l2: float = 1e-2) -> None:
        self.class_balanced = class_balanced
        self.l2 = l2
        self.weight: torch.Tensor | None = None
        self.bias: torch.Tensor | None = None
        self._fitted = False
        self._covariance: torch.Tensor | None = None  # (d+1, d+1) Laplace posterior covariance over [weight, bias]

    def fit(self, phi: torch.Tensor, y: torch.Tensor, max_iter: int = 200, lr: float = 0.5) -> "LogisticRegressionCombiner":
        y = y.to(dtype=phi.dtype, device=phi.device)
        n = phi.shape[0]
        num_features = phi.shape[-1]
        self.weight = torch.zeros(num_features, dtype=phi.dtype, device=phi.device, requires_grad=True)
        self.bias = torch.zeros(1, dtype=phi.dtype, device=phi.device, requires_grad=True)

        if self.class_balanced:
            # Counters the majority-correct imbalance noted in DESIGN.md 8b
            # (a frozen backbone is right far more often than it's wrong).
            n_pos = y.sum().clamp_min(1.0)
            n_neg = (y.numel() - n_pos).clamp_min(1.0)
            sample_weight = torch.where(y == 1, y.numel() / (2 * n_pos), y.numel() / (2 * n_neg))
        else:
            sample_weight = torch.ones_like(y)

        optimizer = torch.optim.LBFGS(
            [self.weight, self.bias], lr=lr, max_iter=max_iter, line_search_fn="strong_wolfe"
        )

        def mean_loss(weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
            logits = phi @ weight + bias
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, y, weight=sample_weight, reduction="mean"
            )
            if self.l2:
                # Ridge penalty on the weights only - the bias sets the overall
                # base rate and shouldn't be shrunk toward zero.
                loss = loss + self.l2 * weight.pow(2).sum()
            return loss

        def closure():
            optimizer.zero_grad()
            loss = mean_loss(self.weight, self.bias)
            loss.backward()
            return loss

        optimizer.step(closure)
        self._fitted = True
        self._fit_laplace_covariance(mean_loss, n, num_features, phi.dtype, phi.device)
        return self

    def _fit_laplace_covariance(self, mean_loss, n: int, num_features: int, dtype, device) -> None:
        # Laplace approximation: posterior covariance = inverse Hessian of the
        # SUMMED negative log-posterior at the MAP estimate. `mean_loss` is
        # what fit() actually minimizes (mean BCE + l2*||w||^2), so its
        # Hessian must be rescaled by n to recover the summed quantity -
        # PyTorch's reduction="mean" divides by n, not by sum(sample_weight),
        # so multiplying by n exactly undoes that division regardless of
        # class_balanced reweighting. Note this Hessian is therefore of the
        # class-balanced-reweighted objective actually being fit, not a
        # strict unweighted-likelihood Bayesian posterior - self-consistent
        # with what the point estimate (self.weight, self.bias) itself is,
        # rather than a different model's uncertainty.
        theta_star = torch.cat([self.weight.detach(), self.bias.detach()]).requires_grad_(False)

        def loss_of_theta(theta: torch.Tensor) -> torch.Tensor:
            w, b = theta[:num_features], theta[num_features:]
            return mean_loss(w, b)

        hessian_mean = torch.autograd.functional.hessian(loss_of_theta, theta_star)
        hessian = hessian_mean * n
        try:
            covariance = torch.linalg.inv(hessian)
        except RuntimeError:
            # Singular Hessian (e.g. l2=0.0 under exact multicollinearity) -
            # fall back to a pseudo-inverse rather than raising, since
            # epistemic_std()/score_with_uncertainty() are optional add-ons
            # that shouldn't break fit() itself.
            covariance = torch.linalg.pinv(hessian)
        self._covariance = covariance

    def score(self, phi: torch.Tensor) -> torch.Tensor:
        if not self._fitted:
            raise RuntimeError("LogisticRegressionCombiner.fit() must be called before score()")
        with torch.no_grad():
            return torch.sigmoid(phi @ self.weight + self.bias)

    def epistemic_std(self, phi: torch.Tensor) -> torch.Tensor:
        """Standard deviation of the predicted LOGIT (pre-sigmoid) under the
        Laplace-approximated posterior over (weight, bias) - the epistemic
        uncertainty of the combiner's own fit, not the aleatoric uncertainty
        `score()` already reports. Distinct from `uncertainty`
        (`ReliabilityState`/`normalized_entropy`), which is about the
        backbone's class-prediction spread, not the combiner's parameter
        uncertainty. Shape (...,), matching `score()`'s output shape.
        """
        if not self._fitted or self._covariance is None:
            raise RuntimeError("LogisticRegressionCombiner.fit() must be called before epistemic_std()")
        with torch.no_grad():
            ones = torch.ones(*phi.shape[:-1], 1, dtype=phi.dtype, device=phi.device)
            x_aug = torch.cat([phi, ones], dim=-1)  # (..., d+1)
            covariance = self._covariance.to(dtype=phi.dtype, device=phi.device)
            # quadratic form x^T Sigma x, batched over leading dims
            var = (x_aug @ covariance * x_aug).sum(dim=-1).clamp_min(0.0)
            return var.sqrt()

    def score_with_uncertainty(self, phi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (score, epistemic_std) - `score()` unchanged, plus
        `epistemic_std()` - both from the same forward pass's logit, no
        duplicated computation.
        """
        if not self._fitted:
            raise RuntimeError("LogisticRegressionCombiner.fit() must be called before score_with_uncertainty()")
        return self.score(phi), self.epistemic_std(phi)

    def score_mackay_adjusted(self, phi: torch.Tensor) -> torch.Tensor:
        """Predictive probability averaged over the Laplace posterior via
        MacKay's (1992) closed-form probit approximation:
        sigmoid(mean_logit / sqrt(1 + pi*var_logit/8)). Pulls the score
        toward 0.5 in proportion to epistemic uncertainty - a high-confidence
        point estimate backed by a poorly-determined fit (e.g. an input far
        from combiner_fit's feature-space coverage) is reported as less
        extreme than `score()` alone would say, without needing sampling or
        an ensemble.
        """
        if not self._fitted:
            raise RuntimeError("LogisticRegressionCombiner.fit() must be called before score_mackay_adjusted()")
        with torch.no_grad():
            mean_logit = phi @ self.weight + self.bias
            std = self.epistemic_std(phi)
            kappa = (1.0 + math.pi * std.pow(2) / 8.0).sqrt()
            return torch.sigmoid(mean_logit / kappa)
