import torch

from rotation_patterns.training import build_balanced_pair_dataset, single_cycle_derangement


def test_derangement_is_bijective_and_has_no_fixed_points() -> None:
    permutation = single_cycle_derangement(17, seed=2025)
    assert sorted(permutation.tolist()) == list(range(17))
    assert not torch.any(permutation == torch.arange(17))
    assert torch.equal(permutation, single_cycle_derangement(17, seed=2025))


def test_balanced_pairs_do_not_cross_source_partition() -> None:
    original = torch.arange(30, dtype=torch.float32).reshape(10, 3)
    rotated = original + 0.5
    sources = torch.tensor([1, 3, 5, 7])
    pairs = build_balanced_pair_dataset(original, rotated, sources, seed=3)
    assert len(pairs) == 8
    labels = [float(pairs[index][1]) for index in range(len(pairs))]
    assert labels.count(1.0) == labels.count(0.0) == 4
    allowed = {tuple(original[index].tolist()) for index in sources}
    for index in range(4, 8):
        pair, label = pairs[index]
        anchor, negative = pair[:3], pair[3:]
        assert tuple(negative.tolist()) in allowed
        assert not torch.equal(anchor, negative)

