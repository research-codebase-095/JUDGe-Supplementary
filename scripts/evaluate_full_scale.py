"""Computes the DESIGN.md 11.3 metrics (risk-coverage/AURC/E-AURC, selective
accuracy at {100,90,80,70,50}% coverage, Execute-band catch-rate, ECE,
AUROC(id vs OOD)/FPR@95%TPR where an OOD cache is available) against the
FULL-SCALE ImageNet-1k validation set and, where downloaded, ImageNet-C /
the OOD suite - reusing the existing library code
(deployment_reliability.combiner/router/calibration) rather than
reimplementing any metric, per this project's stated evaluation discipline.

Split-protocol (see scripts/collect_logits_imagenet1k.py's docstring for the
full reasoning): the combiner and thresholds are fit ONCE, on the existing
small-scale Imagenette-based combiner_fit/threshold_cal splits
(data/logit_cache_<backbone>.pt) - never on any full-scale or shift/OOD data.
The full ImageNet-1k validation set, ImageNet-C, and the OOD suite are then
scored with that already-frozen combiner/calibration/router, exactly the
same evaluate-only discipline this project already applies to
imagenet_a/imagenet_o (DESIGN.md 10.5).

Usage: python scripts/evaluate_full_scale.py [resnet50|vit_b16|convnext_tiny]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from deployment_reliability.calibration import TemperatureScaling  # noqa: E402
from deployment_reliability.combiner import LogisticRegressionCombiner  # noqa: E402
from deployment_reliability.features import featurize, msp  # noqa: E402
from deployment_reliability.router import (  # noqa: E402
    auroc,
    aurc,
    bonferroni_clopper_pearson_thresholds,
    risk_coverage_curve,
    route,
    two_threshold_risk_coverage,
)

DATA_DIR = os.path.join(REPO_ROOT, "data")


def ece(scores: torch.Tensor, correct: torch.Tensor, n_bins: int = 10) -> float:
    bins = torch.linspace(0, 1, n_bins + 1)
    total = 0.0
    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        # Half-open [lo, hi) for every bin except the last, which is closed
        # on the right ([lo, hi]) so a score of exactly 1.0 (e.g. a
        # perfectly-confident sigmoid output) still lands in a bin instead
        # of silently falling outside all of them and being dropped from
        # both the numerator and the implicit weight denominator.
        m = (scores >= lo) & (scores <= hi) if i == n_bins - 1 else (scores >= lo) & (scores < hi)
        if m.sum() == 0:
            continue
        conf = scores[m].mean()
        acc = correct[m].float().mean()
        total += m.float().mean() * (conf - acc).abs()
    return float(total.item())


def selective_accuracy_at_coverages(scores: torch.Tensor, correct: torch.Tensor, targets=(1.0, 0.9, 0.8, 0.7, 0.5)) -> dict:
    coverage, risk = risk_coverage_curve(scores, correct)
    out = {}
    for cov_target in targets:
        hits = (coverage >= cov_target).nonzero()
        idx = hits[0, 0] if len(hits) else -1
        out[cov_target] = {
            "selective_accuracy": 1 - risk[idx].item(),
            "actual_coverage": coverage[idx].item(),
        }
    return out


def fit_pipeline(small_cache_path: str):
    """Fits the combiner/calibration/thresholds ONCE on the existing
    small-scale Imagenette splits - the only fitting step in this whole
    script (DESIGN.md 10.5)."""
    cache = torch.load(small_cache_path)
    logits, labels = cache["logits"], cache["labels"]
    splits_arr = np.array(cache["splits"])

    def mask(name):
        return torch.from_numpy(splits_arr == name)

    m_fit, m_cal = mask("combiner_fit"), mask("threshold_cal")
    correct = logits.argmax(dim=-1) == labels
    phi = featurize(logits)

    combiner = LogisticRegressionCombiner().fit(phi[m_fit], correct[m_fit].float())
    s_cal_split = combiner.score(phi[m_cal])
    temp_cal = TemperatureScaling().fit(s_cal_split, correct[m_cal].float())
    # Primary thresholds: the naive two_threshold_risk_coverage, unchanged from
    # earlier runs. DESIGN.md 23.6 disclosed that this selects from up to n_cal
    # candidates with no correction for that selection step - a real theoretical
    # gap, checked directly and found not to violate a proper bound on real,
    # independent full-scale data (this script's own bucket_report below).
    tau_hi, tau_lo = two_threshold_risk_coverage(s_cal_split, correct[m_cal], execute_risk=0.05, verify_risk=0.2)
    # Also compute the fully rigorous Bonferroni-corrected alternative for the
    # record (DESIGN.md 23.6's second finding: at this calibration split's size,
    # the formally-guaranteed version shrinks the Execute band dramatically -
    # reported here, not silently swapped in as the default, since that would
    # replace a checked-safe, practically usable threshold with a technically
    # stronger but far less useful one).
    tau_hi_bonf, tau_lo_bonf = bonferroni_clopper_pearson_thresholds(
        s_cal_split, correct[m_cal], execute_risk=0.05, verify_risk=0.2
    )

    return {
        "combiner": combiner,
        "temp_cal": temp_cal,
        "tau_hi": tau_hi,
        "tau_lo": tau_lo,
        "tau_hi_bonferroni": tau_hi_bonf,
        "tau_lo_bonferroni": tau_lo_bonf,
        "n_combiner_fit": int(m_fit.sum()),
        "n_threshold_cal": int(m_cal.sum()),
    }


def score_dataset(pipeline: dict, logits: torch.Tensor, labels: torch.Tensor):
    phi = featurize(logits)
    correct = logits.argmax(dim=-1) == labels
    s = pipeline["combiner"].score(phi)
    s_cal = pipeline["temp_cal"].transform(s)
    return phi, correct, s, s_cal


def bucket_report(s: torch.Tensor, correct: torch.Tensor, tau_hi: float, tau_lo: float) -> dict:
    decisions = route(s, tau_hi, tau_lo)
    decisions_t = torch.tensor([{"Execute": 2, "Verify": 1, "HITL": 0}[d] for d in decisions])
    out = {}
    for name, code in [("Execute", 2), ("Verify", 1), ("HITL", 0)]:
        bm = decisions_t == code
        n = int(bm.sum())
        risk = 1 - correct[bm].float().mean().item() if n else float("nan")
        out[name] = {"n": n, "fraction": n / len(decisions_t), "empirical_risk": risk}
    return out


def fpr_at_95tpr(id_scores: torch.Tensor, ood_scores: torch.Tensor) -> float:
    """Fraction of OOD inputs scoring >= the threshold that keeps 95% of ID
    inputs above it (DESIGN.md 11.3) - lower is better (fewer OOD inputs
    slip through at a 95%-ID-retention operating point)."""
    thresh = torch.quantile(id_scores, 0.05)
    return float((ood_scores >= thresh).float().mean().item())


def ood_report(pipeline: dict, backbone: str, id_scores: torch.Tensor, id_logits: torch.Tensor) -> dict:
    """Scores every downloaded OOD cache (logit_cache_<dataset>_<backbone>.pt)
    against the id_test_full_scale scores, using the SAME already-frozen
    combiner - never refit on OOD data (DESIGN.md 10.5)."""
    out = {}
    for dataset in ("places365", "dtd", "inaturalist"):
        cache_path = os.path.join(DATA_DIR, f"logit_cache_{dataset}_{backbone}.pt")
        if not os.path.exists(cache_path):
            continue
        cache = torch.load(cache_path)
        ood_logits = cache["logits"]
        phi_ood = featurize(ood_logits)
        s_ood = pipeline["combiner"].score(phi_ood)
        msp_ood = msp(ood_logits)
        msp_id = msp(id_logits)
        out[dataset] = {
            "n": int(ood_logits.shape[0]),
            "auroc_combiner": auroc(id_scores, s_ood),
            "auroc_msp": auroc(msp_id, msp_ood),
            "fpr_at_95tpr_combiner": fpr_at_95tpr(id_scores, s_ood),
            "fpr_at_95tpr_msp": fpr_at_95tpr(msp_id, msp_ood),
            "mean_score_id": float(id_scores.mean().item()),
            "mean_score_ood": float(s_ood.mean().item()),
        }
        print(
            f"  [{dataset:12s}] n={out[dataset]['n']:5d}  "
            f"AUROC(combiner)={out[dataset]['auroc_combiner']:.4f}  "
            f"AUROC(msp)={out[dataset]['auroc_msp']:.4f}  "
            f"FPR@95%TPR(combiner)={out[dataset]['fpr_at_95tpr_combiner']:.4f}"
        )
    return out


def imagenet_c_report(pipeline: dict, backbone: str) -> dict:
    """If ImageNet-C logits have been collected (scripts/collect_logits_imagenet_c.py),
    reports selective accuracy / AURC per corruption category+severity against
    the same already-frozen combiner - DESIGN.md 11.1's distribution-shift
    graceful-degradation test."""
    cache_path = os.path.join(DATA_DIR, f"logit_cache_imagenet_c_{backbone}.pt")
    if not os.path.exists(cache_path):
        return {}
    cache = torch.load(cache_path)
    logits, labels, groups = cache["logits"], cache["labels"], cache["groups"]
    phi = featurize(logits)
    correct = logits.argmax(dim=-1) == labels
    s = pipeline["combiner"].score(phi)
    out = {}
    for group in sorted(set(groups)):
        gmask = torch.tensor([g == group for g in groups])
        acc = correct[gmask].float().mean().item()
        aurc_g = aurc(s[gmask], correct[gmask])
        out[group] = {"n": int(gmask.sum()), "accuracy": acc, "aurc": aurc_g}
        print(f"  [{group:20s}] n={out[group]['n']:6d}  accuracy={acc:.4f}  AURC={aurc_g:.5f}")
    return out


def main(backbone: str) -> None:
    small_cache_path = os.path.join(DATA_DIR, f"logit_cache_{backbone}.pt")
    full_cache_path = os.path.join(DATA_DIR, f"logit_cache_imagenet1k_{backbone}.pt")
    assert os.path.exists(small_cache_path), f"{small_cache_path} not found - run scripts/collect_logits.py {backbone} first"
    assert os.path.exists(full_cache_path), (
        f"{full_cache_path} not found - run scripts/collect_logits_imagenet1k.py {backbone} first"
    )

    print(f"=== Full-scale evaluation: {backbone} ===")
    pipeline = fit_pipeline(small_cache_path)
    print(
        f"combiner fit on {pipeline['n_combiner_fit']} Imagenette images, "
        f"thresholds/calibration fit on {pipeline['n_threshold_cal']} - both UNCHANGED from small-scale runs"
    )
    print(f"tau_hi={pipeline['tau_hi']:.4f}  tau_lo={pipeline['tau_lo']:.4f}")

    full_cache = torch.load(full_cache_path)
    logits, labels = full_cache["logits"], full_cache["labels"]
    n = logits.shape[0]
    print(f"full-scale ImageNet-1k id_test_full_scale: n={n}")

    phi, correct, s, s_cal = score_dataset(pipeline, logits, labels)
    overall_acc = correct.float().mean().item()
    print(f"overall top-1 accuracy: {overall_acc:.4f}")

    results = {"backbone": backbone, "n": n, "overall_accuracy": overall_acc, "tau_hi": pipeline["tau_hi"], "tau_lo": pipeline["tau_lo"]}

    # Risk-coverage / AURC / E-AURC
    aurc_combiner = aurc(s, correct)
    aurc_msp = aurc(msp(logits), correct)
    aurc_optimal = aurc(correct.float(), correct)  # oracle ordering: all-correct-first
    results["aurc"] = {"combiner": aurc_combiner, "msp": aurc_msp, "oracle": aurc_optimal}
    results["e_aurc"] = {"combiner": aurc_combiner - aurc_optimal, "msp": aurc_msp - aurc_optimal}
    print(f"AURC: combiner={aurc_combiner:.5f}  msp={aurc_msp:.5f}  oracle={aurc_optimal:.5f}")
    print(f"E-AURC: combiner={results['e_aurc']['combiner']:.5f}  msp={results['e_aurc']['msp']:.5f}")

    # Selective accuracy at target coverages
    sel_acc = selective_accuracy_at_coverages(s, correct)
    results["selective_accuracy"] = {str(k): v for k, v in sel_acc.items()}
    for cov, v in sel_acc.items():
        print(f"  coverage>={cov}: selective_accuracy={v['selective_accuracy']:.4f} (actual coverage {v['actual_coverage']:.3f})")

    # ECE before/after calibration
    ece_before = ece(s, correct)
    ece_after = ece(s_cal, correct)
    results["ece"] = {"before_calibration": ece_before, "after_calibration": ece_after}
    print(f"ECE before calibration: {ece_before:.4f}  after: {ece_after:.4f}")

    # Execute/Verify/HITL bucket report (using thresholds fit on small-scale threshold_cal)
    buckets = bucket_report(s, correct, pipeline["tau_hi"], pipeline["tau_lo"])
    results["buckets"] = buckets
    for name, v in buckets.items():
        print(f"  {name:8s} n={v['n']:6d} ({v['fraction']:.1%})  empirical_risk={v['empirical_risk']:.4f}")

    # For the record only, not used elsewhere in this script: the fully rigorous
    # Bonferroni-corrected alternative's bucket report (DESIGN.md 23.6).
    buckets_bonf = bucket_report(s, correct, pipeline["tau_hi_bonferroni"], pipeline["tau_lo_bonferroni"])
    results["buckets_bonferroni_corrected"] = buckets_bonf
    print(f"For comparison, Bonferroni-corrected thresholds (tau_hi={pipeline['tau_hi_bonferroni']:.4f}  tau_lo={pipeline['tau_lo_bonferroni']:.4f}):")
    for name, v in buckets_bonf.items():
        print(f"  {name:8s} n={v['n']:6d} ({v['fraction']:.1%})  empirical_risk={v['empirical_risk']:.4f}")

    print("OOD suite (AUROC(id vs OOD), FPR@95%TPR - evaluate-only, never fit on):")
    results["ood"] = ood_report(pipeline, backbone, s, logits)
    if not results["ood"]:
        print("  (no OOD caches found - run scripts/collect_logits_ood.py first)")

    print("ImageNet-C (distribution shift - evaluate-only):")
    results["imagenet_c"] = imagenet_c_report(pipeline, backbone)
    if not results["imagenet_c"]:
        print("  (no ImageNet-C cache found - run scripts/collect_logits_imagenet_c.py first)")

    out_path = os.path.join(DATA_DIR, f"full_scale_results_{backbone}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print("saved results to", out_path)

    md_path = os.path.join(DATA_DIR, f"full_scale_results_{backbone}.md")
    with open(md_path, "w") as f:
        f.write(render_markdown_summary(results))
    print("saved paste-ready markdown summary to", md_path)


def render_markdown_summary(results: dict) -> str:
    """Formats the results dict as a paste-ready markdown table, matching
    DESIGN.md 15.2's existing small-scale results-table style - written so
    the eventual doc write-up is a copy/adapt step, not a re-derive-every-
    number-by-hand step. Not itself a claim about what belongs in the docs -
    the actual DESIGN.md/PROGRESS_REPORT.md/STUDY_PLAN.md prose still
    needs to state the split-protocol decision, investigate anything
    surprising, and compare against the small-scale numbers explicitly,
    none of which this function does - it only formats numbers that were
    already computed and printed above.
    """
    lines = [f"### Full-scale results: {results['backbone']} (n={results['n']}, real ImageNet-1k validation set)", ""]
    lines.append("| Metric | Result |")
    lines.append("|---|---|")
    lines.append(f"| id_test_full_scale top-1 accuracy | {results['overall_accuracy']:.4f} |")
    lines.append(f"| AURC (combiner / MSP / oracle) | {results['aurc']['combiner']:.5f} / {results['aurc']['msp']:.5f} / {results['aurc']['oracle']:.5f} |")
    lines.append(f"| E-AURC (combiner / MSP) | {results['e_aurc']['combiner']:.5f} / {results['e_aurc']['msp']:.5f} |")
    for cov, v in results["selective_accuracy"].items():
        lines.append(f"| Selective accuracy @ coverage>={cov} | {v['selective_accuracy']:.4f} (actual coverage {v['actual_coverage']:.3f}) |")
    lines.append(f"| ECE before / after temperature scaling | {results['ece']['before_calibration']:.4f} / {results['ece']['after_calibration']:.4f} |")
    for name, v in results["buckets"].items():
        lines.append(f"| {name}-band n (fraction), empirical risk | {v['n']} ({v['fraction']:.1%}), {v['empirical_risk']:.4f} |")
    if results.get("ood"):
        lines.append("")
        lines.append("| OOD dataset | n | AUROC(combiner) | AUROC(MSP) | FPR@95%TPR(combiner) |")
        lines.append("|---|---|---|---|---|")
        for dataset, v in results["ood"].items():
            lines.append(f"| {dataset} | {v['n']} | {v['auroc_combiner']:.4f} | {v['auroc_msp']:.4f} | {v['fpr_at_95tpr_combiner']:.4f} |")
    if results.get("imagenet_c"):
        lines.append("")
        lines.append("| ImageNet-C cell | n | accuracy | AURC |")
        lines.append("|---|---|---|---|")
        for group, v in results["imagenet_c"].items():
            lines.append(f"| {group} | {v['n']} | {v['accuracy']:.4f} | {v['aurc']:.5f} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("backbone", nargs="?", default="resnet50")
    args = parser.parse_args()
    main(args.backbone)
