import torch

from deployment_reliability.normalization import ReferenceNormalizer


def test_transform_of_reference_set_is_approximately_standardized():
    torch.manual_seed(0)
    reference = torch.randn(5000, 5) * torch.tensor([1.0, 2.0, 0.5, 3.0, 10.0]) + torch.tensor(
        [0.0, 5.0, -2.0, 1.0, 100.0]
    )
    normalizer = ReferenceNormalizer().fit(reference)
    z = normalizer.transform(reference)
    assert z.mean(dim=0).abs().max().item() < 0.05
    assert (z.std(dim=0) - 1.0).abs().max().item() < 0.05


def test_anomaly_score_is_high_for_outliers_and_low_for_typical_points():
    torch.manual_seed(1)
    reference = torch.randn(2000, 3) * 1.0 + 0.0
    normalizer = ReferenceNormalizer().fit(reference)

    typical = torch.zeros(1, 3)  # right at the reference mean
    outlier_high = torch.full((1, 3), 50.0)  # far above the reference range
    outlier_low = torch.full((1, 3), -50.0)  # far below the reference range

    assert normalizer.anomaly_score(typical).max().item() < 0.5
    assert normalizer.anomaly_score(outlier_high).min().item() > 10.0
    assert normalizer.anomaly_score(outlier_low).min().item() > 10.0


def test_anomaly_score_treats_high_and_low_outliers_symmetrically():
    # This is the property that fixes the notebooks/06 Check 5 failure mode:
    # an unusually HIGH raw value (e.g. energy/L2-norm on pixel noise) should
    # register as anomalous, not as "confident."
    torch.manual_seed(2)
    reference = torch.randn(2000, 1) * 2.0 + 10.0
    normalizer = ReferenceNormalizer().fit(reference)

    # Build symmetric points from the *fitted* mean/std (not the true population
    # parameters), since a finite sample's estimates won't exactly match them.
    mean, std = normalizer._mean.item(), normalizer._std.item()
    above = torch.tensor([[mean + 6 * std]])
    below = torch.tensor([[mean - 6 * std]])
    assert torch.isclose(normalizer.anomaly_score(above), normalizer.anomaly_score(below), atol=1e-4)


def test_transform_requires_fit_before_use():
    normalizer = ReferenceNormalizer()
    try:
        normalizer.transform(torch.rand(3, 2))
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass
