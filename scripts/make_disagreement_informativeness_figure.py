"""Generates the "disagreement without informativeness" figure requested to make
the paper's central mechanism (Table 3 x Table 2: disagreement with MSP is
necessary but not sufficient for a combiner to help) visually legible in one
plot, rather than requiring a reader to cross-reference two tables.

X-axis: pairwise rank-discordance vs. MSP (%), per config, per feature (same
values as Table 3, extended to all four non-MSP features rather than just
energy/L2-norm, and to ResNet-50).
Y-axis: that feature's own individual AUROC, oriented by its own per-config
empirically-verified direction (verify_feature_directions) - NOT Table 2's
fixed global convention, which deliberately keeps some entries sub-chance to
expose the sign-reversal problem. This figure asks a different question
(is the disagreeing signal informative at all, correctly oriented), so it
uses the correctly-oriented AUROC throughout.

No new model inference - reuses cached phi/correct tensors, same
corrected_phi() approach as scripts/pairwise_rank_discordance.py.

Usage:
    python scripts/make_disagreement_informativeness_figure.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from deployment_reliability.features import (  # noqa: E402
    DEFAULT_FEATURE_DIRECTIONS,
    DEFAULT_FEATURE_NAMES,
    featurize,
    verify_feature_directions,
)
from deployment_reliability.router import auroc  # noqa: E402

from judge_characterization import load_judge_config, oriented_pooled  # noqa: E402

DATA_DIR = os.path.join(REPO_ROOT, "data")
FIG_DIR = os.path.join(REPO_ROOT, "figures")

COLOR_RESNET50 = "#2a78d6"
COLOR_QWEN05B = "#3f9b5c"
COLOR_SMOLLM2 = "#eb6834"
COLOR_QWEN15B = "#8b5fbf"
INK_PRIMARY = "#0b0b0b"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"

FEATURE_LABELS = {
    "logit_margin": "Margin",
    "normalized_entropy": "Entropy",
    "energy_score": "Energy",
    "logit_l2_norm": "$L_2$ norm",
}
FEATURE_MARKERS = {
    "logit_margin": "o",
    "normalized_entropy": "s",
    "energy_score": "^",
    "logit_l2_norm": "D",
}


def corrected_phi(phi: torch.Tensor, correct: torch.Tensor) -> torch.Tensor:
    d = verify_feature_directions(phi, correct)
    signs = torch.tensor([1.0 if d[n] else -1.0 for n in DEFAULT_FEATURE_NAMES])
    return phi * DEFAULT_FEATURE_DIRECTIONS * signs


def pairwise_discordance(a: np.ndarray, b: np.ndarray, n_sample: int = 200_000, seed: int = 0) -> float:
    n = len(a)
    rng = np.random.default_rng(seed)
    if n * (n - 1) // 2 <= n_sample:
        ii, jj = np.triu_indices(n, k=1)
    else:
        ii = rng.integers(0, n, size=n_sample)
        jj = rng.integers(0, n, size=n_sample)
        mask = ii != jj
        ii, jj = ii[mask], jj[mask]
    da, db = a[ii] - a[jj], b[ii] - b[jj]
    return float((((da > 0) & (db < 0)) | ((da < 0) & (db > 0))).mean())


def load_vision() -> tuple[torch.Tensor, torch.Tensor]:
    full = torch.load(os.path.join(DATA_DIR, "logit_cache_imagenet1k_resnet50.pt"))
    phi = featurize(full["logits"])
    correct = full["logits"].argmax(dim=-1) == full["labels"]
    return phi, correct


def collect_points() -> list[dict]:
    judge_configs = [
        ("Qwen2.5-0.5B (judge)", COLOR_QWEN05B, "judge_feature_cache_mtbench.pt", "Qwen2.5-0.5B-Instruct (judge)"),
        ("Qwen2.5-1.5B (judge)", COLOR_QWEN15B, "judge_feature_cache_mtbench_1p5b.pt", "Qwen2.5-1.5B-Instruct (judge)"),
        # SmolLM2 intentionally excluded: Tables 1-5 report it separately in
        # Appendix A.6 (app:smollm2), not as a headline row/figure -- this
        # figure's scope must match tab:disagreement's, which is Qwen+ResNet-50 only.
    ]
    points = []
    # Judge configs: pooled bootstrap-stabilized orientation (disjoint from
    # id_test -- see judge_characterization.py), not id_test's own
    # verify_feature_directions, which is circular against id_test-computed
    # discordance/AUROC (scripts/direction_split_robustness_check.py).
    for name, color, cache_filename, display_name in judge_configs:
        config = load_judge_config(cache_filename, display_name)
        phi_c = oriented_pooled(config)
        correct_bool = config["correct"].bool()
        msp_col = phi_c[:, 0].numpy()
        for feat in ("logit_margin", "normalized_entropy", "energy_score", "logit_l2_norm"):
            idx = DEFAULT_FEATURE_NAMES.index(feat)
            col = phi_c[:, idx]
            a = auroc(col[correct_bool], col[~correct_bool])
            disc = pairwise_discordance(msp_col, col.numpy()) * 100
            points.append({"domain": name, "color": color, "feature": feat, "discordance": disc, "auroc": a})

    # Vision: no pooled split available (separate calibration sample, per
    # Table 1's dagger footnote) -- id_test-verified direction, unaffected.
    phi, correct = load_vision()
    phi_c = corrected_phi(phi, correct)
    correct_bool = torch.as_tensor(correct).bool()
    msp_col = phi_c[:, 0].numpy()
    for feat in ("logit_margin", "normalized_entropy", "energy_score", "logit_l2_norm"):
        idx = DEFAULT_FEATURE_NAMES.index(feat)
        col = phi_c[:, idx]
        a = auroc(col[correct_bool], col[~correct_bool])
        disc = pairwise_discordance(msp_col, col.numpy()) * 100
        points.append({"domain": "ResNet-50 (vision)", "color": COLOR_RESNET50, "feature": feat, "discordance": disc, "auroc": a})
    return points


def make_figure(points: list[dict], out_path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.edgecolor": INK_MUTED,
            "text.color": INK_PRIMARY,
            "axes.labelcolor": INK_PRIMARY,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
        }
    )

    fig, ax = plt.subplots(figsize=(5.0, 3.4))

    for p in points:
        ax.scatter(
            p["discordance"], p["auroc"],
            color=p["color"], marker=FEATURE_MARKERS[p["feature"]],
            s=46, edgecolors="white", linewidths=0.6, zorder=3,
        )

    ax.axhline(0.5, color=GRIDLINE, linewidth=1.2, zorder=1)
    ax.text(1, 0.5, "chance", fontsize=7, color=INK_MUTED, va="bottom", ha="left")

    ax.set_xlabel("Pairwise rank-discordance vs. MSP (%)")
    ax.set_ylabel("Feature's own AUROC\n(correctly oriented)")
    ax.set_xlim(-3, 88)
    ax.set_ylim(0.46, 0.90)
    ax.grid(True, color=GRIDLINE, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    domain_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=c, markeredgecolor="white",
               markersize=7, label=name)
        for name, c in [
            ("ResNet-50 (vision)", COLOR_RESNET50),
            ("Qwen2.5-0.5B (judge)", COLOR_QWEN05B),
            ("Qwen2.5-1.5B (judge)", COLOR_QWEN15B),
        ]
    ]
    feature_handles = [
        Line2D([0], [0], marker=FEATURE_MARKERS[f], color=INK_MUTED, linestyle="none",
               markersize=6, label=FEATURE_LABELS[f])
        for f in ("logit_margin", "normalized_entropy", "energy_score", "logit_l2_norm")
    ]
    legend1 = ax.legend(handles=domain_handles, loc="upper left", frameon=False, title="Domain", handlelength=1.2)
    ax.add_artist(legend1)
    ax.legend(handles=feature_handles, loc="upper right", frameon=False, title="Feature", handlelength=1.2)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, format="pdf")
    plt.close(fig)


def main() -> None:
    points = collect_points()
    print(f"{'Domain':<24}{'Feature':<20}{'Discordance':>13}{'AUROC':>9}")
    for p in points:
        print(f"{p['domain']:<24}{p['feature']:<20}{p['discordance']:>12.2f}%{p['auroc']:>9.4f}")
    out_path = os.path.join(FIG_DIR, "disagreement_vs_informativeness.pdf")
    make_figure(points, out_path)
    print(f"\nsaved figure to {out_path}")


if __name__ == "__main__":
    main()
