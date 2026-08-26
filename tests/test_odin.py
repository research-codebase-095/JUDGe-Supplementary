import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from collect_odin_scores import odin_scores_for_batch  # noqa: E402
from deployment_reliability.backbone import load_frozen_resnet50  # noqa: E402
from deployment_reliability.router import auroc  # noqa: E402

ODIN_CACHE = os.path.join(REPO_ROOT, "data", "odin_cache_resnet50.pt")


def test_odin_scores_for_batch_produces_finite_scores_in_zero_one_and_does_not_mutate_input():
    # A real forward+backward+forward pass through the actual frozen
    # ResNet-50 on random (not real image) input - checks the gradient
    # mechanics (autograd.grad on a live graph, sign-perturbation, second
    # inference-mode-free forward pass) work end to end before spending real
    # time running this over ~4000 real images.
    model, preprocess, categories = load_frozen_resnet50()
    batch = torch.randn(4, 3, 224, 224)
    original = batch.clone()

    scores = odin_scores_for_batch(model, batch)

    assert scores.shape == (4,)
    assert torch.isfinite(scores).all()
    assert (scores >= 0.0).all() and (scores <= 1.0).all()
    assert not scores.requires_grad, "returned scores should be detached, not carry a live autograd graph"
    assert torch.equal(batch, original), "the input batch tensor itself must not be mutated in place"


def test_odin_perturbation_moves_the_predicted_classs_confidence_in_the_expected_direction():
    # ODIN's whole mechanism is that the perturbation is specifically
    # constructed to INCREASE the predicted class's (temperature-scaled)
    # confidence relative to an unperturbed forward pass at the same
    # temperature - check that directly, not just that the code runs.
    import torch.nn.functional as F

    model, preprocess, categories = load_frozen_resnet50()
    torch.manual_seed(0)
    batch = torch.randn(4, 3, 224, 224)

    with torch.no_grad():
        unperturbed_logits = model(batch)
        y_hat = unperturbed_logits.argmax(dim=-1)
        unperturbed_score = F.softmax(unperturbed_logits / 1000.0, dim=-1).gather(1, y_hat.unsqueeze(1)).squeeze(1)

    perturbed_score = odin_scores_for_batch(model, batch)
    # perturbed_score is the max softmax prob post-perturbation, which by
    # construction targets the same y_hat class - should be >= the
    # unperturbed same-class score for at least most of the batch (allowing
    # for the rare case a perturbation overshoots at this epsilon).
    assert (perturbed_score >= unperturbed_score).float().mean() >= 0.5


def test_real_odin_scores_have_near_zero_dynamic_range_at_the_papers_own_default_temperature():
    # A genuine, investigated finding, not a bug: the untempered (T=1) MSP on
    # this same cached logit set spans 0.017-0.964 (a healthy range), but
    # ODIN's paper-specified T=1000 - tuned by Liang et al. for CIFAR-scale
    # networks with a much smaller typical per-image logit range - nearly
    # flattens ResNet-50's tempered softmax to uniform (1/C=0.001) for every
    # image, since this backbone's per-image logit range (~7.7) divided by
    # T=1000 is far too small to survive exponentiation. Locked in as a
    # regression test specifically so this doesn't get silently "fixed" by
    # someone re-tuning T later without disclosing the comparison changed.
    assert os.path.exists(ODIN_CACHE), "run scripts/collect_odin_scores.py first"
    cache = torch.load(ODIN_CACHE)
    scores = cache["scores"]
    assert cache["temperature"] == 1000.0
    span = (scores.max() - scores.min()).item()
    assert span < 0.001, f"expected the T=1000 score range to be near-degenerate (<0.001), got span={span:.6f}"


def test_real_odin_auroc_id_test_vs_imagenet_o_and_imagenet_a_shift():
    # The direct same-protocol numbers for STUDY_PLAN.md 3.6's OOD table,
    # replacing the qualitative "comparable-or-worse separation" claim.
    # Reported exactly as found: ood_o lands AT OR BELOW chance (a real,
    # disclosed negative result explained by the near-zero dynamic range
    # above, not a broken experiment - the raw per-image ranking that
    # survives at this precision is essentially noise); shift_a carries only
    # a weak signal, weaker than an earlier version of this script found,
    # after a numerical audit caught and fixed a missing per-channel
    # normalization-std division in the perturbation step (DESIGN.md §31) -
    # the corrected, larger effective epsilon does not help ODIN here.
    assert os.path.exists(ODIN_CACHE), "run scripts/collect_odin_scores.py first"
    cache = torch.load(ODIN_CACHE)
    scores = cache["scores"]
    splits_arr = np.array(cache["splits"])
    m_test = torch.from_numpy(splits_arr == "id_test")
    m_o = torch.from_numpy(splits_arr == "imagenet_o")
    m_a = torch.from_numpy(splits_arr == "imagenet_a")
    assert int(m_o.sum()) > 0 and int(m_a.sum()) > 0

    a_ood = auroc(scores[m_test], scores[m_o])
    a_shift = auroc(scores[m_test], scores[m_a])
    assert 0.4 < a_ood < 0.5, f"expected below-chance ood_o AUROC at T=1000, got {a_ood:.4f}"
    assert 0.48 < a_shift < 0.58, f"expected a weak shift_a signal, got {a_shift:.4f}"


def test_real_odin_does_not_recover_imagenet_a_correctness_signal_either():
    # Same-axis check as mahalanobis.py's/trust_score.py's equivalent tests:
    # does this signal separate correct from incorrect predictions WITHIN
    # imagenet_a. After the per-channel std normalization fix (DESIGN.md
    # §31), this is now measurably BELOW chance (0.37), not just near it -
    # the corrected, larger perturbation makes ODIN's within-imagenet_a
    # ranking actively anti-correlated with correctness here, a real,
    # disclosed finding rather than the milder "near-chance" the
    # under-scaled pre-fix perturbation had produced.
    assert os.path.exists(ODIN_CACHE), "run scripts/collect_odin_scores.py first"
    cache = torch.load(ODIN_CACHE)
    scores, logits, labels = cache["scores"], cache["logits"], cache["labels"]
    splits_arr = np.array(cache["splits"])
    m_a = torch.from_numpy(splits_arr == "imagenet_a")
    correct_a = logits[m_a].argmax(dim=-1) == labels[m_a]
    assert int(correct_a.sum()) > 20 and int((~correct_a).sum()) > 20

    a_corr = auroc(scores[m_a][correct_a], scores[m_a][~correct_a])
    assert 0.3 < a_corr < 0.42, f"expected below-chance corr_a, got {a_corr:.4f}"


def test_real_odin_auroc_corr_id_below_the_combiner():
    # corr_id (DESIGN.md §31, notebooks/21): does ODIN separate correct from
    # incorrect predictions on id_test itself. Real number (~0.614, after
    # the per-channel std normalization fix - weaker than the pre-fix
    # ~0.699, since the corrected, larger perturbation moves the id_test
    # ranking closer to chance too), not previously locked into a permanent
    # regression test alongside the ood_o/shift_a/corr_a checks above.
    assert os.path.exists(ODIN_CACHE), "run scripts/collect_odin_scores.py first"
    cache = torch.load(ODIN_CACHE)
    scores, logits, labels = cache["scores"], cache["logits"], cache["labels"]
    splits_arr = np.array(cache["splits"])
    m_test = torch.from_numpy(splits_arr == "id_test")
    correct_test = logits[m_test].argmax(dim=-1) == labels[m_test]

    a_corr_id = auroc(scores[m_test][correct_test], scores[m_test][~correct_test])
    assert 0.58 < a_corr_id < 0.65, f"expected corr_id near 0.614 (DESIGN.md §31), got {a_corr_id:.4f}"
