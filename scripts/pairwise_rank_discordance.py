"""Reviewer correctness check, round 1: Table 3 ("signal disagreement")
originally counted single EXAMPLES where two standardized features have
opposite-signed z-scores with |z|>0.5 each. Section 3's Proposition is about
PAIRS of examples x1,x2 with U(x1)>U(x2) but V(x1)<V(x2) - a different
quantity entirely. This script computes the actual pairwise rank-discordance
fraction (the fraction of all example pairs on which a given pair of features
disagree in ordering) for every 2-of-5 feature combination, on the same
id_test splits already used elsewhere in the paper. No new model inference -
reuses cached phi tensors.

Reviewer correctness check, round 2 (critical): Section 3 documents that the
FIXED GLOBAL feature-direction convention (DEFAULT_FEATURE_DIRECTIONS) is
empirically wrong for some features/configs, illustrated by Table 2's own
sub-0.5 entries there. Computing Table 3's discordance under that SAME wrong
global sign, for exactly the features/configs affected, would misreport
disagreement: flipping one feature's sign turns every concordant pair into a
discordant one and vice versa. This script's `corrected_phi()` applies each
judge config's bootstrap-stabilized per-feature direction (see
scripts/judge_characterization.py's bootstrap_stabilized_direction_and_best_feature)
before computing discordance.

Reviewer correctness check, round 3 (critical): an earlier version of this
function verified each feature's direction on id_test itself (the same split
Table 3 then reports discordance on) - a genuine circularity, since checking
"does the assumed direction hold on id_test" is close to tautological with
whatever Table 2's own id_test-computed AUROC already shows for that config
(scripts/direction_split_robustness_check.py's investigation). The current
version instead verifies direction via a bootstrap-majority vote over
combiner_fit+threshold_cal, pooled -- fully disjoint from id_test, and large
enough (n=200-642, comparable to id_test itself) not to inherit
threshold_cal-alone's small-sample instability (that same investigation
found threshold_cal alone, at n=40 for SmolLM2, gives a confident-looking but
unreliable direction for MSP itself, disagreeing with id_test's own n=200
estimate). This changed three of Table 3's sixteen cells materially:
Qwen2.5-0.5B and -1.5B's vs.-$L_2$-norm entries swap toward their
already-disclosed "could equally read the other way" alternative (Section~
\ref{sec:method}'s tie-breaking-artifact note), and SmolLM2's vs.-margin/
-entropy/-energy entries drop sharply (the old id_test-verified direction
was itself largely a circularity artifact at this config, not fresh
disagreement -- see the Appendix's rewritten verdict-token-logits section).
ResNet-50 has no combiner_fit/threshold_cal split in its cache (a separate
calibration sample is used instead, per Table 1's dagger footnote) so it is
unaffected and out of scope for this specific fix.

For large n, exact all-pairs computation (O(n^2)) is replaced by a random
sample of pairs (200k draws) for a stable estimate.

Usage:
    python scripts/pairwise_rank_discordance.py
"""

from __future__ import annotations

import itertools
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
    verify_feature_directions,
)

from judge_characterization import (  # noqa: E402
    bootstrap_stabilized_direction_and_best_feature,
    load_judge_config,
)

DATA_DIR = os.path.join(REPO_ROOT, "data")
N_SAMPLE_PAIRS = 200_000
SEED = 0


def pairwise_discordance(phi: torch.Tensor, feature_names=DEFAULT_FEATURE_NAMES) -> dict:
    n = phi.shape[0]
    rng = np.random.default_rng(SEED)
    if n * (n - 1) // 2 <= N_SAMPLE_PAIRS:
        ii, jj = np.triu_indices(n, k=1)
    else:
        ii = rng.integers(0, n, size=N_SAMPLE_PAIRS)
        jj = rng.integers(0, n, size=N_SAMPLE_PAIRS)
        mask = ii != jj
        ii, jj = ii[mask], jj[mask]

    results = {}
    for a, b in itertools.combinations(range(len(feature_names)), 2):
        fa, fb = phi[:, a].numpy(), phi[:, b].numpy()
        da, db = fa[ii] - fa[jj], fb[ii] - fb[jj]
        disc = ((da > 0) & (db < 0)) | ((da < 0) & (db > 0))
        results[(feature_names[a], feature_names[b])] = float(disc.mean())
    return results


def corrected_phi_pooled(config: dict) -> torch.Tensor:
    """Orients id_test's phi by the bootstrap-stabilized per-config direction
    verified on the pooled combiner_fit+threshold_cal data (disjoint from
    id_test) - see module docstring. Judge configs only; vision uses
    corrected_phi_id_test below instead (no pooled split available)."""
    result = bootstrap_stabilized_direction_and_best_feature(config)
    signs = torch.tensor([result["stabilized_sign"][n] for n in DEFAULT_FEATURE_NAMES])
    return config["phi"] * DEFAULT_FEATURE_DIRECTIONS * signs


def corrected_phi_id_test(phi: torch.Tensor, correct: torch.Tensor) -> torch.Tensor:
    """Orients by direction verified on id_test itself. Used only for
    ResNet-50 (vision), which has no combiner_fit/threshold_cal split in its
    cache (a separate calibration sample is used instead, per Table 1's
    dagger footnote) - this is the same fallback the paper already documents
    as out of scope for the pooled fix."""
    directions_emp = verify_feature_directions(phi, correct)
    signs = torch.tensor([1.0 if directions_emp[n] else -1.0 for n in DEFAULT_FEATURE_NAMES])
    return phi * DEFAULT_FEATURE_DIRECTIONS * signs


def load_id_test(path: str) -> tuple[torch.Tensor, torch.Tensor]:
    cache = torch.load(os.path.join(DATA_DIR, path))
    splits = np.array(cache["splits"])
    mask = splits == "id_test"
    return cache["phi"][mask], cache["correct"][mask]


def load_vision_id_test() -> tuple[torch.Tensor, torch.Tensor]:
    from deployment_reliability.features import featurize

    full = torch.load(os.path.join(DATA_DIR, "logit_cache_imagenet1k_resnet50.pt"))
    phi = featurize(full["logits"])
    correct = full["logits"].argmax(dim=-1) == full["labels"]
    return phi, correct


def main() -> None:
    judge_configs = [
        ("Qwen2.5-0.5B-Instruct (judge)", "judge_feature_cache_mtbench.pt"),
        ("Qwen2.5-1.5B-Instruct (judge)", "judge_feature_cache_mtbench_1p5b.pt"),
        ("SmolLM2-360M-Instruct (judge)", "judge_feature_cache_mtbench_smollm2_360m.pt"),
    ]

    for name, cache_filename in judge_configs:
        config = load_judge_config(cache_filename, name)
        phi, correct = config["phi"], config["correct"]
        n = phi.shape[0]
        print(f"=== {name}  (n={n}) ===")

        result = bootstrap_stabilized_direction_and_best_feature(config)
        flipped = [f for f, s in result["stabilized_sign"].items() if s < 0]
        print(f"  pooled-bootstrap-stabilized flipped-from-global features: {flipped or 'none'}")
        print(f"  sign stability (frac. positive, of {result['valid']} valid resamples): " +
              ", ".join(f"{f}={result['sign_frac'][f]:.3f}" for f in DEFAULT_FEATURE_NAMES))

        print("  -- GLOBAL-CONVENTION orientation (DEFAULT_FEATURE_DIRECTIONS, matches Table 2) --")
        phi_global = phi * DEFAULT_FEATURE_DIRECTIONS
        res_global = pairwise_discordance(phi_global)
        for (fa, fb), rate in sorted(res_global.items(), key=lambda kv: kv[1]):
            print(f"    {fa:>18} vs {fb:<18} discordant pair fraction: {rate:.4f}")

        print("  -- POOLED-BOOTSTRAP-STABILIZED orientation (this is what Table 3 in the paper reports) --")
        phi_corrected = corrected_phi_pooled(config)
        res_corrected = pairwise_discordance(phi_corrected)
        for (fa, fb), rate in sorted(res_corrected.items(), key=lambda kv: kv[1]):
            print(f"    {fa:>18} vs {fb:<18} discordant pair fraction: {rate:.4f}")
        print()

    print("=== ResNet-50 (vision) -- no pooled split available, uses id_test-verified direction (unaffected by the fix) ===")
    phi, correct = load_vision_id_test()
    n = phi.shape[0]
    print(f"  n={n}")
    directions_emp = verify_feature_directions(phi, correct)
    reversed_feats = [k for k, v in directions_emp.items() if not v]
    print(f"  empirically-reversed features: {reversed_feats or 'none'}")
    phi_corrected = corrected_phi_id_test(phi, correct)
    res_corrected = pairwise_discordance(phi_corrected)
    for (fa, fb), rate in sorted(res_corrected.items(), key=lambda kv: kv[1]):
        print(f"    {fa:>18} vs {fb:<18} discordant pair fraction: {rate:.4f}")


if __name__ == "__main__":
    main()
