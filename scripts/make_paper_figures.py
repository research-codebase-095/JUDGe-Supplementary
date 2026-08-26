"""Builds the results table and risk-coverage figure for the JUDGe @ NeurIPS
2026 workshop paper, from this project's EXISTING, already-validated cached
data - no new inference, no new fitting beyond what evaluate_full_scale.py and
collect_llm_logits.py already do, and no numbers invented: everything here is
recomputed straight from data/*.pt so it can be checked against README.md's
"Headline results" and DESIGN.md's 24.3/14.5/29.4 sections.

Backbones covered, and why exactly these three: the vision backbone
(ResNet-50) is scored on the real full-scale ImageNet-1k validation cache
(data/logit_cache_imagenet1k_resnet50.pt, n=50,000), reproducing
DESIGN.md 24.3's headline AURC numbers exactly, using the same protocol as
scripts/evaluate_full_scale.py: the LogisticRegressionCombiner is fit ONCE on
the small-scale Imagenette combiner_fit split
(data/logit_cache_resnet50.pt) and evaluated, never refit, on the full-scale
set. GPT-2 and Pythia-160m are scored on their own cached per-token phi/
correct/splits (data/llm_feature_cache_<tag>.pt, DESIGN.md 14.5/29.4),
combiner fit on their own combiner_fit split and evaluated on their own
id_test split (n=128,898 each), reproducing DESIGN.md 14.5's AURC=0.3182 and
29.4's AUROC=0.8134 exactly.

ViT-B/16 and ConvNeXt-Tiny are deliberately NOT included here: README.md's
cross-architecture numbers (AURC 0.04819/0.04265) were computed against a
10,000-image full-scale ImageNet-1k subsample (DESIGN.md 24.6) whose raw
logit cache (data/logit_cache_imagenet1k_{vit_b16,convnext_tiny}.pt) is not
present in this environment's data/ - only the small-scale (n=30 id_test)
Imagenette caches are. Recomputing on the small-scale cache would silently
produce a DIFFERENT, non-headline number under the same backbone name, so
those two backbones are left out of this script's table rather than
presenting a number that looks like, but is not, the published one.

Usage: python scripts/make_paper_figures.py
Output: figures/risk_coverage.pdf, plus the results table printed to stdout.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from deployment_reliability.combiner import LogisticRegressionCombiner  # noqa: E402
from deployment_reliability.features import featurize, msp  # noqa: E402
from deployment_reliability.router import aurc, risk_coverage_curve  # noqa: E402

DATA_DIR = os.path.join(REPO_ROOT, "data")
FIGURES_DIR = os.path.join(REPO_ROOT, "figures")

# dataviz skill's validated categorical palette (references/palette.md):
# slot 1 (blue) / slot 2 (orange), fixed hue order, not cycled.
COLOR_RESNET50 = "#2a78d6"
COLOR_GPT2 = "#eb6834"
INK_PRIMARY = "#0b0b0b"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"


def evaluate_vision_full_scale(backbone: str) -> dict:
    """Fits the combiner on the small-scale combiner_fit split, evaluates MSP
    and combiner scores on the real full-scale id_test_full_scale split -
    exactly scripts/evaluate_full_scale.py's protocol (DESIGN.md 24.3)."""
    small_cache = torch.load(os.path.join(DATA_DIR, f"logit_cache_{backbone}.pt"))
    splits = np.array(small_cache["splits"])
    m_fit = torch.from_numpy(splits == "combiner_fit")
    phi_fit = featurize(small_cache["logits"][m_fit])
    correct_fit = (small_cache["logits"][m_fit].argmax(dim=-1) == small_cache["labels"][m_fit]).float()
    combiner = LogisticRegressionCombiner().fit(phi_fit, correct_fit)

    full_cache = torch.load(os.path.join(DATA_DIR, f"logit_cache_imagenet1k_{backbone}.pt"))
    logits, labels = full_cache["logits"], full_cache["labels"]
    correct = logits.argmax(dim=-1) == labels
    phi = featurize(logits)

    return {
        "name": backbone,
        "domain": "Vision (ImageNet-1k, full scale)",
        "n": int(logits.shape[0]),
        "msp_scores": msp(logits),
        "combiner_scores": combiner.score(phi),
        "correct": correct,
    }


def evaluate_llm(tag: str, display_name: str) -> dict:
    """Fits the combiner on the LLM cache's own combiner_fit split, evaluates
    MSP (phi column 0, per features.DEFAULT_FEATURE_NAMES) and combiner
    scores on its own id_test split - DESIGN.md 14.5/29.4's protocol."""
    cache = torch.load(os.path.join(DATA_DIR, f"llm_feature_cache_{tag}.pt"))
    phi, correct, splits = cache["phi"], cache["correct"], np.array(cache["splits"])
    m_fit = torch.from_numpy(splits == "combiner_fit")
    m_test = torch.from_numpy(splits == "id_test")

    combiner = LogisticRegressionCombiner().fit(phi[m_fit], correct[m_fit].float())
    phi_test, correct_test = phi[m_test], correct[m_test]

    return {
        "name": display_name,
        "domain": "LLM (WikiText-2, token-level)",
        "n": int(m_test.sum()),
        "msp_scores": phi_test[:, 0],  # column 0 = msp, per features.DEFAULT_FEATURE_NAMES
        "combiner_scores": combiner.score(phi_test),
        "correct": correct_test,
    }


def print_results_table(results: list[dict]) -> None:
    header = f"{'Architecture':<16} {'Domain':<32} {'n':>8} {'MSP AURC':>10} {'Combiner AURC':>15}"
    print(header)
    print("-" * len(header))
    for r in results:
        aurc_msp = aurc(r["msp_scores"], r["correct"])
        aurc_combiner = aurc(r["combiner_scores"], r["correct"])
        r["aurc_msp"], r["aurc_combiner"] = aurc_msp, aurc_combiner
        print(f"{r['name']:<16} {r['domain']:<32} {r['n']:>8} {aurc_msp:>10.5f} {aurc_combiner:>15.5f}")


def make_figure(results_for_figure: list[dict], out_path: str) -> None:
    """One risk-coverage figure, MSP (dashed) vs. combiner (solid), colored
    by backbone - the cross-modality claim (vision + LLM) in a single panel,
    per the dataviz skill's "color follows the entity" rule (fixed hue order,
    linestyle as the secondary MSP-vs-combiner encoding, one shared axis)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.edgecolor": INK_MUTED,
            "text.color": INK_PRIMARY,
            "axes.labelcolor": INK_PRIMARY,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
        }
    )

    fig, ax = plt.subplots(figsize=(3.35, 2.6))  # single-column NeurIPS-style width

    colors = {"resnet50": COLOR_RESNET50, "gpt2": COLOR_GPT2}
    labels = {"resnet50": "ResNet-50 (ImageNet-1k)", "gpt2": "GPT-2 (WikiText-2)"}

    for r in results_for_figure:
        color = colors[r["name"]]
        coverage_msp, risk_msp = risk_coverage_curve(r["msp_scores"], r["correct"])
        coverage_comb, risk_comb = risk_coverage_curve(r["combiner_scores"], r["correct"])
        ax.plot(
            coverage_msp.numpy(), risk_msp.numpy(),
            color=color, linestyle="--", linewidth=1.4,
            label=f"{labels[r['name']]} - MSP",
        )
        ax.plot(
            coverage_comb.numpy(), risk_comb.numpy(),
            color=color, linestyle="-", linewidth=1.8,
            label=f"{labels[r['name']]} - Combiner",
        )

    ax.set_xlabel("Coverage")
    ax.set_ylabel("Risk")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(bottom=0.0)
    ax.grid(True, color=GRIDLINE, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(loc="upper left", frameon=False, handlelength=2.4)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, format="pdf")
    plt.close(fig)


def main() -> None:
    resnet50 = evaluate_vision_full_scale("resnet50")
    gpt2 = evaluate_llm("gpt2", "gpt2")
    pythia = evaluate_llm("pythia160m", "Pythia-160m")

    resnet50["name"], gpt2["name"] = "ResNet-50", "GPT-2"
    print("=== Results table (computed from cached data, this run) ===")
    print_results_table([resnet50, gpt2, pythia])

    print()
    print("=== Consistency check against README.md / DESIGN.md ===")
    print(f"ResNet-50 full-scale AURC (combiner/MSP): published 0.04966 / 0.07487  "
          f"computed {resnet50['aurc_combiner']:.5f} / {resnet50['aurc_msp']:.5f}")
    print(f"GPT-2 id_test AURC (combiner): published 0.3182  computed {gpt2['aurc_combiner']:.5f}")
    print(f"Pythia-160m id_test AURC (combiner): no published AURC (only AUROC=0.8134); "
          f"computed {pythia['aurc_combiner']:.5f} for reference")

    out_path = os.path.join(FIGURES_DIR, "risk_coverage.pdf")
    # Restore the "resnet50"/"gpt2" keys make_figure expects for color/label lookup.
    make_figure([{**resnet50, "name": "resnet50"}, {**gpt2, "name": "gpt2"}], out_path)
    print()
    print(f"saved figure to {out_path}")


if __name__ == "__main__":
    main()
