import torch

from deployment_reliability.calibration import IsotonicCalibration, PlattScaling, TemperatureScaling


def _synthetic_scores_and_labels(n=500, seed=0):
    torch.manual_seed(seed)
    true_p = torch.rand(n)
    y = torch.bernoulli(true_p)
    return true_p, y


def test_temperature_scaling_preserves_ranking():
    s, y = _synthetic_scores_and_labels()
    cal = TemperatureScaling().fit(s, y)
    s_cal = cal.transform(s)
    order_before = torch.argsort(s)
    order_after = torch.argsort(s_cal)
    assert torch.equal(order_before, order_after)


def test_temperature_scaling_requires_fit_before_transform():
    cal = TemperatureScaling()
    try:
        cal.transform(torch.rand(5))
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass


def test_temperature_scaling_recovers_near_identity_on_well_calibrated_scores():
    # s itself is already P(correct) by construction, so T should land near 1
    # and the calibrated output should stay close to the input.
    s, y = _synthetic_scores_and_labels(n=5000, seed=1)
    cal = TemperatureScaling().fit(s, y)
    s_cal = cal.transform(s)
    assert (s_cal - s).abs().mean().item() < 0.1


def test_platt_scaling_preserves_ranking_when_slope_positive():
    s, y = _synthetic_scores_and_labels(seed=2)
    cal = PlattScaling().fit(s, y)
    assert cal.a.item() > 0
    s_cal = cal.transform(s)
    assert torch.equal(torch.argsort(s), torch.argsort(s_cal))


def test_isotonic_calibration_output_is_non_decreasing():
    torch.manual_seed(3)
    s = torch.rand(200)
    y = torch.bernoulli(s)
    cal = IsotonicCalibration().fit(s, y)

    query = torch.linspace(0.0, 1.0, 50)
    out = cal.transform(query)
    assert (out[1:] >= out[:-1] - 1e-6).all()


def test_isotonic_calibration_requires_fit_before_transform():
    cal = IsotonicCalibration()
    try:
        cal.transform(torch.rand(5))
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass


def test_temperature_scaling_matches_input_dtype_not_a_hardcoded_one():
    s64, y64 = _synthetic_scores_and_labels(n=200, seed=5)
    s64, y64 = s64.double(), y64.double()
    cal = TemperatureScaling().fit(s64, y64)
    assert cal.log_t.dtype == torch.float64
    assert cal.transform(s64).dtype == torch.float64


def test_platt_scaling_matches_input_dtype_not_a_hardcoded_one():
    s64, y64 = _synthetic_scores_and_labels(n=200, seed=6)
    s64, y64 = s64.double(), y64.double()
    cal = PlattScaling().fit(s64, y64)
    assert cal.a.dtype == torch.float64 and cal.b.dtype == torch.float64
    assert cal.transform(s64).dtype == torch.float64


def test_isotonic_calibration_matches_input_dtype_not_a_hardcoded_one():
    s64, y64 = _synthetic_scores_and_labels(n=200, seed=7)
    s64, y64 = s64.double(), y64.double()
    cal = IsotonicCalibration().fit(s64, y64)
    assert cal.transform(s64).dtype == torch.float64
