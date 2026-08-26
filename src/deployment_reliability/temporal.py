"""Temporal reliability dynamics: track a reliability signal over a stream of
steps (video frames, decoding steps, control-loop ticks, ...) instead of
looking at one static value.

Motivation: a single scalar r_t in isolation can't distinguish "always been
borderline" from "was confident and is now declining" - and a declining
trend is often the more actionable signal (e.g. in a robot control loop or
video stream, the point where the trajectory turns is the point to
intervene, not the point the raw score first crosses some fixed threshold).

Everything in this module operates on a bare stream of floats, with no
assumption about what produced them (reliability.py's correctness,
evidence.py's concentration, a raw MSP value, anything else in [0, 1] or
unbounded) - that's what makes it domain-agnostic. It also means it has been
unit-tested only on synthetic sequences (tests/test_temporal.py): this repo
has no video, robot-control, or streaming-decode dataset to validate the
tracker against real temporal drift, so treat it as mechanically-correct-
but-empirically-unvalidated - the same caveat as providers.py's stub
classes.
"""

from __future__ import annotations

from collections import deque


class TemporalReliabilityTracker:
    """Rolling-window tracker over a stream of scalar reliability scores.

    `window` controls both the trend estimate's span and the EMA's effective
    memory (the smoothing constant is derived from `window`).
    """

    def __init__(self, window: int = 10, decline_rate_threshold: float = -0.02) -> None:
        if window < 2:
            raise ValueError("window must be >= 2 to compute a trend")
        self.window = window
        self.decline_rate_threshold = decline_rate_threshold
        self._history: deque[float] = deque(maxlen=window)
        self._ema: float | None = None
        self._alpha = 2.0 / (window + 1)  # standard EMA smoothing constant for this window

    def update(self, score: float) -> None:
        score = float(score)
        self._history.append(score)
        self._ema = score if self._ema is None else self._alpha * score + (1 - self._alpha) * self._ema

    @property
    def ema(self) -> float:
        if self._ema is None:
            raise RuntimeError("update() must be called at least once before reading ema")
        return self._ema

    def trend(self) -> float:
        """Least-squares slope of the score over the current window (units:
        score change per step). Requires at least 2 observations."""
        n = len(self._history)
        if n < 2:
            raise RuntimeError("trend() requires at least 2 update() calls")
        xs = range(n)
        ys = list(self._history)
        mean_x = (n - 1) / 2.0
        mean_y = sum(ys) / n
        num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
        den = sum((x - mean_x) ** 2 for x in xs)
        return num / den if den != 0.0 else 0.0

    def is_declining(self) -> bool:
        """True if the current window's trend is dropping faster than
        `decline_rate_threshold` (a negative number; more negative = requires
        a steeper decline to trigger)."""
        return self.trend() < self.decline_rate_threshold

    def reset(self) -> None:
        self._history.clear()
        self._ema = None


def lyapunov_candidate(ema: float, target: float = 1.0) -> float:
    """One candidate Lyapunov-style function V(D_t) = (target - ema_t)^2 over
    the tracker's own EMA statistic, D_t := ema_t (STUDY_PLAN.md 3.6 item
    4 - an EMPIRICAL check, not a committed proof; see
    `empirical_lyapunov_transitions` and DESIGN.md 26.3 for the honest,
    checked-both-ways outcome, not a theorem).

    `target=1.0` treats "maximally reliable" (the top of this project's [0,1]
    combiner-score range) as the system's equilibrium point: V=0 exactly
    when the EMA sits at that equilibrium, and V grows quadratically with
    distance from it in either direction (though in practice the EMA never
    exceeds 1.0 for a combiner score bounded in [0,1], so this only ever
    tracks decline from equilibrium here, not overshoot past it - a
    quadratic form is used anyway rather than a linear (1 - ema), matching
    the standard Lyapunov-function convention of a positive-definite bowl
    around equilibrium, not because overshoot is expected in this specific
    application).
    """
    return (target - ema) ** 2


def empirical_lyapunov_transitions(
    score_sequences, window: int = 10, target: float = 1.0
) -> tuple[list[float], list[float]]:
    """Replay each of `score_sequences` (a list of independent scalar-score
    sequences - e.g. one per corruption-ramp image, or one per WikiText-2
    chunk) through a FRESH `TemporalReliabilityTracker(window=window)`, and
    collect every consecutive (V(D_t), V(D_{t+1})) pair from the resulting
    EMA stream, using `lyapunov_candidate` above.

    This is the raw data collection step for STUDY_PLAN.md 3.6 item 4's
    empirical Lyapunov check: whether E[V(D_{t+1})|D_t] <= V(D_t) tends to
    hold. That expectation is a POPULATION-level statement, not a per-
    transition guarantee - individual transitions are expected to violate it
    under any noisy score stream (a single step of noise can always push V
    up even in a genuinely stable system); the actual check bins the
    returned (V_t, V_tp1) pairs by V_t (e.g. deciles) and compares each
    bin's mean V_tp1 against its mean V_t, the direct empirical estimate of
    the conditional expectation the drift condition is stated in terms of.
    See `tests/test_temporal.py`'s Lyapunov tests and DESIGN.md 26.3 for the
    real, checked-both-ways result (corruption ramp vs. real WikiText-2
    token sequences) - this function only produces the transition data, it
    does not itself decide whether the condition "holds."

    Each sequence resets its own tracker (`window`'s history never leaks
    across sequences - e.g. across different images' corruption ramps, or
    across different WikiText-2 chunks, which per test_llm_extension.py are
    independent forward passes with no real continuity between them anyway).

    Returns (V_t, V_tp1) as two parallel plain Python lists (this module has
    no numpy/torch dependency elsewhere and keeps that property here -
    callers convert to whatever array type they need).
    """
    v_t, v_tp1 = [], []
    for scores in score_sequences:
        tracker = TemporalReliabilityTracker(window=window)
        emas = []
        for s in scores:
            tracker.update(s)
            emas.append(tracker.ema)
        for i in range(len(emas) - 1):
            v_t.append(lyapunov_candidate(emas[i], target=target))
            v_tp1.append(lyapunov_candidate(emas[i + 1], target=target))
    return v_t, v_tp1


def lyapunov_drift_boundary(alpha: float, mu: float, sigma2: float) -> tuple[float, float]:
    """The formal drift-condition boundary (DESIGN.md 26.5's Proposition and
    Proof), not just the empirical check `empirical_lyapunov_transitions`
    above provides.

    For e_t := target - ema_t and d_t := target - s_t (the per-step raw-score
    deviation from target, with conditional mean `mu` and variance `sigma2`,
    assumed constant here - a stationary-innovations approximation), the EMA
    recursion e_t = alpha*d_t + (1-alpha)*e_{t-1} gives an EXACT expression
    for E[V(D_{t+1})|e_t] in terms of e_t, alpha, mu, sigma2 (`alpha` is the
    tracker's own EMA smoothing constant, `2/(window+1)`). Requiring
    E[V(D_{t+1})|e_t] <= V(D_t) = e_t^2 reduces to a quadratic inequality in
    e_t; this function returns its two real roots (e_lo <= e_hi). The drift
    condition is PROVABLY satisfied whenever `e_t <= e_lo or e_t >= e_hi`
    (outside the interval) - a genuine, checkable sufficient condition, not
    an empirical tendency.

    In this project's actual usage (target=1.0, scores in [0,1]), e_t =
    target - ema_t is always >= 0 by construction (ema is a convex
    combination of scores <= 1), so only e_hi (equivalently V_hi = e_hi**2)
    is reachable/relevant in practice - e_lo is a real root of the same
    quadratic but describes a regime (e_t < 0, i.e. ema_t > target) this
    project's own score range never enters. Callers in this domain should
    use `e_hi**2` as "the V(D_t) value above which the tracker's drift
    condition is guaranteed", not `e_lo`.

    Verified (DESIGN.md 26.5): matches direct Monte Carlo simulation of the
    exact drift condition across multiple (mu, sigma2, e_t) configurations,
    and the mu/sigma2 estimated directly from real WikiText-2 combiner
    scores predicts, via this function, almost exactly where the real
    empirical per-decile drift changes sign.
    """
    a = 2.0 - alpha
    b = -2.0 * (1.0 - alpha) * mu
    c = -alpha * (sigma2 + mu * mu)
    discriminant = b * b - 4.0 * a * c
    sqrt_disc = discriminant**0.5
    e_lo = (-b - sqrt_disc) / (2.0 * a)
    e_hi = (-b + sqrt_disc) / (2.0 * a)
    return e_lo, e_hi
