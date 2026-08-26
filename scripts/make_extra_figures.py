"""Reviewer-requested figures beyond figures/risk_coverage.pdf (make_paper_figures.py):
a calibration/reliability diagram, an ROC curve, and a feature correlation
heatmap. Reuses existing cached data/combiner-fit protocol (same as
make_paper_figures.py / judge_characterization.py) - no new inference.

Representative configs: GPT-2 (LLM, base) and Qwen2.5-0.5B-Instruct (judge),
matching the two backbones already figured in risk_coverage.pdf plus the
paper's headline judge pilot.

Usage: python scripts/make_extra_figures.py
Output: figures/reliability.pdf, figures/roc.pdf, figures/correlation_heatmap.pdf
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from deployment_reliability.combiner import LogisticRegressionCombiner  # noqa: E402
from deployment_reliability.features import DEFAULT_FEATURE_NAMES  # noqa: E402
from deployment_reliability.router import auroc  # noqa: E402

DATA_DIR = os.path.join(REPO_ROOT, "data")
FIGURES_DIR = os.path.join(REPO_ROOT, "figures")

COLOR_RESNET50 = "#2a78d6"
COLOR_GPT2 = "#eb6834"
COLOR_JUDGE = "#3f9b5c"
INK_PRIMARY = "#0b0b0b"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"


def load_llm(tag: str):
    cache = torch.load(os.path.join(DATA_DIR, f"llm_feature_cache_{tag}.pt"))
    phi, correct, splits = cache["phi"], cache["correct"], np.array(cache["splits"])
    m_fit = torch.from_numpy(splits == "combiner_fit")
    m_test = torch.from_numpy(splits == "id_test")
    combiner = LogisticRegressionCombiner().fit(phi[m_fit], correct[m_fit].float())
    phi_test, correct_test = phi[m_test], correct[m_test]
    return phi_test[:, 0], combiner.score(phi_test), correct_test, phi[m_fit]


def load_judge(cache_filename: str):
    cache = torch.load(os.path.join(DATA_DIR, cache_filename))
    phi, correct, splits = cache["phi"], cache["correct"], np.array(cache["splits"])
    m_fit = torch.from_numpy(splits == "combiner_fit")
    m_test = torch.from_numpy(splits == "id_test")
    combiner = LogisticRegressionCombiner().fit(phi[m_fit], correct[m_fit].float())
    phi_test, correct_test = phi[m_test], correct[m_test]
    return phi_test[:, 0], combiner.score(phi_test), correct_test, phi[m_fit]


def reliability_bins(scores: torch.Tensor, correct: torch.Tensor, n_bins: int = 10):
    y = correct.float()
    bin_edges = torch.linspace(0, 1, n_bins + 1)
    centers, confs, accs, counts = [], [], [], []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (scores >= lo) & (scores < hi) if i < n_bins - 1 else (scores >= lo) & (scores <= hi)
        if mask.sum() == 0:
            continue
        centers.append(((lo + hi) / 2).item())
        confs.append(scores[mask].mean().item())
        accs.append(y[mask].mean().item())
        counts.append(int(mask.sum().item()))
    return np.array(centers), np.array(confs), np.array(accs), np.array(counts)


def roc_points(scores: torch.Tensor, correct: torch.Tensor):
    order = torch.argsort(scores, descending=True)
    y = correct[order].numpy().astype(bool)
    n_pos, n_neg = y.sum(), (~y).sum()
    tpr = np.concatenate([[0.0], np.cumsum(y) / max(n_pos, 1)])
    fpr = np.concatenate([[0.0], np.cumsum(~y) / max(n_neg, 1)])
    return fpr, tpr


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 9, "axes.labelsize": 10, "axes.titlesize": 10,
        "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.edgecolor": INK_MUTED, "text.color": INK_PRIMARY,
        "axes.labelcolor": INK_PRIMARY, "xtick.color": INK_MUTED, "ytick.color": INK_MUTED,
    })
    os.makedirs(FIGURES_DIR, exist_ok=True)

    gpt2_msp, gpt2_comb, gpt2_correct, gpt2_phi_fit = load_llm("gpt2")
    judge_msp, judge_comb, judge_correct, judge_phi_fit = load_judge("judge_feature_cache_mtbench.pt")

    configs = [
        ("GPT-2 (LLM)", COLOR_GPT2, gpt2_msp, gpt2_comb, gpt2_correct),
        ("Qwen2.5-0.5B (judge)", COLOR_JUDGE, judge_msp, judge_comb, judge_correct),
    ]

    # --- (a) reliability diagram ---
    fig, axes = plt.subplots(1, 2, figsize=(6.7, 2.7))
    for ax, (name, color, msp, comb, correct) in zip(axes, configs):
        ax.plot([0, 1], [0, 1], color=GRIDLINE, linewidth=1.0, zorder=0)
        c_c, conf_c, acc_c, n_c = reliability_bins(msp, correct)
        c_g, conf_g, acc_g, n_g = reliability_bins(comb, correct)
        ax.plot(conf_c, acc_c, "--", color=color, linewidth=1.4, marker="o", markersize=3, label="MSP")
        ax.plot(conf_g, acc_g, "-", color=color, linewidth=1.8, marker="s", markersize=3, label="Combiner")
        ax.set_title(name)
        ax.set_xlabel("Predicted confidence")
        ax.set_ylabel("Empirical accuracy")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, color=GRIDLINE, linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.legend(loc="upper left", frameon=False, handlelength=2.0)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "reliability.pdf"), format="pdf")
    plt.close(fig)
    print("saved figures/reliability.pdf")

    # --- (b) ROC curve ---
    fig, axes = plt.subplots(1, 2, figsize=(6.7, 2.7))
    for ax, (name, color, msp, comb, correct) in zip(axes, configs):
        ax.plot([0, 1], [0, 1], color=GRIDLINE, linewidth=1.0, zorder=0)
        fpr_msp, tpr_msp = roc_points(msp, correct)
        fpr_comb, tpr_comb = roc_points(comb, correct)
        auc_msp = auroc(msp[correct.bool()], msp[~correct.bool()])
        auc_comb = auroc(comb[correct.bool()], comb[~correct.bool()])
        ax.plot(fpr_msp, tpr_msp, "--", color=color, linewidth=1.4, label=f"MSP (AUROC {auc_msp:.3f})")
        ax.plot(fpr_comb, tpr_comb, "-", color=color, linewidth=1.8, label=f"Combiner (AUROC {auc_comb:.3f})")
        ax.set_title(name)
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, color=GRIDLINE, linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.legend(loc="lower right", frameon=False, handlelength=2.0)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "roc.pdf"), format="pdf")
    plt.close(fig)
    print("saved figures/roc.pdf")

    # --- (c) feature correlation heatmap (Qwen2.5-0.5B judge, id_test) ---
    cache = torch.load(os.path.join(DATA_DIR, "judge_feature_cache_mtbench.pt"))
    splits = np.array(cache["splits"])
    m_test = splits == "id_test"
    phi_test_np = cache["phi"][m_test].numpy()
    corr = np.corrcoef(phi_test_np, rowvar=False)
    names = list(DEFAULT_FEATURE_NAMES)
    short = ["MSP", "Margin", "Entropy", "Energy", "L2 norm"]

    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(short)))
    ax.set_yticks(range(len(short)))
    ax.set_xticklabels(short, rotation=45, ha="right")
    ax.set_yticklabels(short)
    for i in range(len(short)):
        for j in range(len(short)):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                     color="white" if abs(corr[i, j]) > 0.6 else INK_PRIMARY, fontsize=7.5)
    ax.set_title("Qwen2.5-0.5B judge: feature correlation")
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.ax.tick_params(labelsize=7.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "correlation_heatmap.pdf"), format="pdf")
    plt.close(fig)
    print("saved figures/correlation_heatmap.pdf")

    print()
    print("Qwen2.5-0.5B judge (id_test) correlation matrix (raw features):")
    header = "".join(f"{s:>10}" for s in short)
    print(" " * 10 + header)
    for i, s in enumerate(short):
        print(f"{s:<10}" + "".join(f"{corr[i, j]:>10.3f}" for j in range(len(short))))


if __name__ == "__main__":
    main()
