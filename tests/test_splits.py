import torch

from deployment_reliability.splits import three_way_split


def test_three_way_split_sizes_and_no_overlap():
    combiner_idx, threshold_idx, test_idx = three_way_split(100, ratios=(0.4, 0.1, 0.5), seed=42)
    assert len(combiner_idx) == 40
    assert len(threshold_idx) == 10
    assert len(test_idx) == 50

    all_idx = torch.cat([combiner_idx, threshold_idx, test_idx])
    assert len(all_idx.unique()) == 100


def test_three_way_split_is_reproducible_with_seed():
    a = three_way_split(100, seed=7)
    b = three_way_split(100, seed=7)
    for x, y in zip(a, b):
        assert torch.equal(x, y)


def test_three_way_split_rejects_bad_ratios():
    try:
        three_way_split(100, ratios=(0.5, 0.5, 0.5))
        assert False, "expected ValueError"
    except ValueError:
        pass
