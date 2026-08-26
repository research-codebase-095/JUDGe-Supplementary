import os

import numpy as np
import torch

from deployment_reliability.router import auroc
from deployment_reliability.trust_score import TrustScorer

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FEATURE_CACHE = os.path.join(REPO_ROOT, "data", "mahalanobis_feature_cache_resnet50.pt")


def _synthetic_two_class(n_per_class=500, d=8, sep=6.0, seed=0):
    torch.manual_seed(seed)
    mean0 = torch.zeros(d)
    mean1 = torch.zeros(d)
    mean1[0] = sep
    x0 = torch.randn(n_per_class, d) + mean0
    x1 = torch.randn(n_per_class, d) + mean1
    features = torch.cat([x0, x1], dim=0)
    labels = torch.cat([torch.zeros(n_per_class, dtype=torch.int64), torch.ones(n_per_class, dtype=torch.int64)])
    return features, labels


def test_rejects_k_below_one():
    try:
        TrustScorer(k=0)
        assert False, "expected ValueError for k < 1"
    except ValueError:
        pass


def test_score_requires_fit_first():
    scorer = TrustScorer()
    try:
        scorer.score(torch.randn(3, 4), torch.zeros(3, dtype=torch.int64))
        assert False, "expected RuntimeError before fit()"
    except RuntimeError:
        pass


def test_score_is_higher_for_a_point_correctly_predicted_as_its_own_typical_class():
    features, labels = _synthetic_two_class(sep=8.0)
    scorer = TrustScorer().fit(features, labels)

    typical_class0 = features[0:1]  # a real class-0 point, correctly "predicted" as class 0
    typical_class1 = features[-1:]  # a real class-1 point, correctly "predicted" as class 1

    s_correct = scorer.score(typical_class0, predicted_labels=torch.tensor([0]))
    # Mispredict the same class-0 point as class 1: now it's far from its
    # "own" (wrong) predicted class and close to the true (other) class -
    # trust score should drop sharply relative to the correct prediction.
    s_wrong = scorer.score(typical_class0, predicted_labels=torch.tensor([1]))
    assert s_correct.item() > s_wrong.item()
    assert s_correct.item() > 1.0, "a typical point correctly matched to its own class should score above 1"


def test_score_orientation_matches_this_projects_higher_is_more_trustworthy_convention():
    features, labels = _synthetic_two_class(sep=8.0)
    scorer = TrustScorer().fit(features, labels)
    d = features.shape[-1]
    near = torch.zeros(50, d) + torch.randn(50, d) * 0.1  # tight cluster right at class-0's mean
    far = torch.randn(50, d) * 0.1 + 200.0  # tight cluster far from both class means, "predicted" as class 0
    pred_near = torch.zeros(50, dtype=torch.int64)
    pred_far = torch.zeros(50, dtype=torch.int64)
    a = auroc(scorer.score(near, pred_near), scorer.score(far, pred_far))
    assert a > 0.9, f"expected near-perfect separation between near-mean and far-outlier points, got AUROC={a:.4f}"


def test_predicted_class_outside_the_fitted_label_space_scores_as_untrustworthy():
    # The real situation id_test/imagenet_o produce: a scorer fit only on
    # Imagenette's 10 synsets, scoring a 1000-way ImageNet prediction that
    # may not be one of those 10 classes at all. No reference density exists
    # for that predicted class, so "own_dist" is undefined - treated as
    # infinite, driving the score toward 0 (maximally untrustworthy) rather
    # than raising or silently mis-indexing into an unrelated class's column.
    features, labels = _synthetic_two_class(sep=8.0)  # classes {0, 1} only
    scorer = TrustScorer().fit(features, labels)

    typical_class0 = features[0:1]
    s_out_of_scope = scorer.score(typical_class0, predicted_labels=torch.tensor([7]))
    assert torch.isfinite(s_out_of_scope).all()
    assert s_out_of_scope.item() < 0.01, f"expected near-zero trust for an out-of-scope predicted class, got {s_out_of_scope.item():.4f}"


def test_real_resnet50_features_trust_score_fit_on_combiner_fit_only():
    # Same protocol as mahalanobis.py's tests: fit only on combiner_fit.
    assert os.path.exists(FEATURE_CACHE), "run scripts/collect_features_mahalanobis.py first"
    cache = torch.load(FEATURE_CACHE)
    features, labels = cache["features"], cache["labels"]
    splits_arr = np.array(cache["splits"])
    m_fit = torch.from_numpy(splits_arr == "combiner_fit")

    scorer = TrustScorer().fit(features[m_fit], labels[m_fit])
    assert scorer.classes is not None and len(scorer.classes) <= 10  # Imagenette's 10 synsets


def test_real_resnet50_features_trust_score_separates_id_test_from_imagenet_o():
    # Direct same-protocol comparison against mahalanobis.py's equivalent
    # test (AUROC 0.966) and the best logit-only signal (0.607, DESIGN.md
    # 20.2) - a second feature-space method's real number for STUDY_PLAN.md
    # 3.6's OOD table, not the qualitative "likely stronger" claim it
    # previously carried.
    assert os.path.exists(FEATURE_CACHE), "run scripts/collect_features_mahalanobis.py first"
    cache = torch.load(FEATURE_CACHE)
    features, logits, labels = cache["features"], cache["logits"], cache["labels"]
    splits_arr = np.array(cache["splits"])
    m_fit = torch.from_numpy(splits_arr == "combiner_fit")
    m_test = torch.from_numpy(splits_arr == "id_test")
    m_o = torch.from_numpy(splits_arr == "imagenet_o")
    assert int(m_o.sum()) > 0

    scorer = TrustScorer().fit(features[m_fit], labels[m_fit])
    pred_test = logits[m_test].argmax(dim=-1)
    pred_o = logits[m_o].argmax(dim=-1)
    s_test = scorer.score(features[m_test], pred_test)
    s_o = scorer.score(features[m_o], pred_o)
    assert torch.isfinite(s_test).all() and torch.isfinite(s_o).all()
    a = auroc(s_test, s_o)
    assert a > 0.607, f"expected feature-space OOD AUROC to beat the best logit-only signal (0.607), got {a:.4f}"


def test_real_resnet50_trust_score_corr_id_matches_design_doc_headline_number():
    # DESIGN.md §31's comparison table cites Trust Score corr_id = 0.9999,
    # a real, checked number - but until this test, none of the three
    # DESIGN.md §31 headline numbers for Trust Score specifically (corr_id,
    # shift_a, corr_a) were locked into a permanent regression test the way
    # mahalanobis.py's and ODIN's equivalent numbers already are
    # (tests/test_mahalanobis.py, tests/test_odin.py) - a future change to
    # TrustScorer.score()/_distance_to_every_class() could silently drift
    # this number with nothing catching it.
    assert os.path.exists(FEATURE_CACHE), "run scripts/collect_features_mahalanobis.py first"
    cache = torch.load(FEATURE_CACHE)
    features, logits, labels = cache["features"], cache["logits"], cache["labels"]
    splits_arr = np.array(cache["splits"])
    m_fit = torch.from_numpy(splits_arr == "combiner_fit")
    m_test = torch.from_numpy(splits_arr == "id_test")

    scorer = TrustScorer().fit(features[m_fit], labels[m_fit])
    pred_test = logits[m_test].argmax(dim=-1)
    correct_test = pred_test == labels[m_test]
    s_test = scorer.score(features[m_test], pred_test)
    corr_id = auroc(s_test[correct_test], s_test[~correct_test])
    assert 0.998 < corr_id < 1.0, f"expected corr_id near 0.9999 (DESIGN.md §31), got {corr_id:.4f}"


def test_real_resnet50_trust_score_shift_a_matches_design_doc_headline_number():
    # DESIGN.md §31's comparison table cites Trust Score shift_a = 0.9028.
    assert os.path.exists(FEATURE_CACHE), "run scripts/collect_features_mahalanobis.py first"
    cache = torch.load(FEATURE_CACHE)
    features, logits, labels = cache["features"], cache["logits"], cache["labels"]
    splits_arr = np.array(cache["splits"])
    m_fit = torch.from_numpy(splits_arr == "combiner_fit")
    m_test = torch.from_numpy(splits_arr == "id_test")
    m_a = torch.from_numpy(splits_arr == "imagenet_a")

    scorer = TrustScorer().fit(features[m_fit], labels[m_fit])
    pred_test = logits[m_test].argmax(dim=-1)
    pred_a = logits[m_a].argmax(dim=-1)
    s_test = scorer.score(features[m_test], pred_test)
    s_a = scorer.score(features[m_a], pred_a)
    shift_a = auroc(s_test, s_a)
    assert 0.89 < shift_a < 0.92, f"expected shift_a near 0.9028 (DESIGN.md §31), got {shift_a:.4f}"


def test_real_resnet50_trust_score_corr_a_matches_design_doc_headline_number():
    # DESIGN.md §31's comparison table cites Trust Score corr_a = 0.4975 -
    # near chance, consistent with every other signal source tested on this
    # axis (mahalanobis.py's 0.501, ODIN's 0.4664, DESIGN.md 20.2's 0.562).
    assert os.path.exists(FEATURE_CACHE), "run scripts/collect_features_mahalanobis.py first"
    cache = torch.load(FEATURE_CACHE)
    features, logits, labels = cache["features"], cache["logits"], cache["labels"]
    splits_arr = np.array(cache["splits"])
    m_fit = torch.from_numpy(splits_arr == "combiner_fit")
    m_a = torch.from_numpy(splits_arr == "imagenet_a")

    scorer = TrustScorer().fit(features[m_fit], labels[m_fit])
    pred_a = logits[m_a].argmax(dim=-1)
    correct_a = pred_a == labels[m_a]
    assert int(correct_a.sum()) > 20 and int((~correct_a).sum()) > 20
    s_a = scorer.score(features[m_a], pred_a)
    corr_a = auroc(s_a[correct_a], s_a[~correct_a])
    assert 0.4 < corr_a < 0.6, f"expected corr_a near chance (0.4975, DESIGN.md §31), got {corr_a:.4f}"
