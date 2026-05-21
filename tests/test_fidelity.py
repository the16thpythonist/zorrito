import torch

from zorrito.fidelity import estimate_fidelity, make_match_fn


def test_classification_match_exact():
    ref = torch.tensor([[0.1, 0.7, 0.2]])
    match = make_match_fn("classification", ref, tolerance=0.0)
    assert bool(match(torch.tensor([[0.1, 0.9, 0.0]])).item()) is True
    assert bool(match(torch.tensor([[0.9, 0.1, 0.0]])).item()) is False


def test_regression_tolerance_band():
    ref = torch.tensor([[1.0, 2.0]])
    match = make_match_fn("regression", ref, tolerance=0.1)
    assert bool(match(torch.tensor([[1.05, 1.95]])).item()) is True
    assert bool(match(torch.tensor([[1.05, 2.5]])).item()) is False


def test_regression_direction_up():
    """direction='up': match if pred <= ref + tolerance (any drop is fine)."""
    ref = torch.tensor([[1.0]])
    match = make_match_fn("regression", ref, tolerance=0.1, direction="up")
    assert bool(match(torch.tensor([[1.05]])).item()) is True   # tiny rise OK
    assert bool(match(torch.tensor([[0.0]])).item()) is True    # huge drop OK
    assert bool(match(torch.tensor([[1.20]])).item()) is False  # too high


def test_regression_direction_down():
    """direction='down': match if pred >= ref - tolerance (any rise is fine)."""
    ref = torch.tensor([[1.0]])
    match = make_match_fn("regression", ref, tolerance=0.1, direction="down")
    assert bool(match(torch.tensor([[0.95]])).item()) is True   # tiny drop OK
    assert bool(match(torch.tensor([[5.0]])).item()) is True    # huge rise OK
    assert bool(match(torch.tensor([[0.5]])).item()) is False   # too low


def test_direction_rejected_for_classification():
    import pytest
    ref = torch.tensor([[0.1, 0.7, 0.2]])
    with pytest.raises(ValueError, match="only meaningful for objective='regression'"):
        make_match_fn("classification", ref, tolerance=0.0, direction="up")


def test_estimate_fidelity_all_match():
    x_template = torch.zeros(3, 2)
    node_sel = torch.tensor([True, True, True])
    feat_sel = torch.tensor([True, True])

    def forward_fn(y):
        return torch.tensor([[0.0, 1.0]])

    def noise(n_rows):
        return torch.randn(n_rows, 2)

    ref = forward_fn(x_template)
    match = make_match_fn("classification", ref, tolerance=0.0)
    fid = estimate_fidelity(forward_fn, x_template, node_sel, feat_sel, noise, match, samples=10)
    assert fid == 1.0


def test_estimate_fidelity_noise_drives_disagreement():
    """When NOTHING is selected, fidelity should reflect how often a noisy
    input still predicts the same class. We rig the model to flip on positive
    inputs and verify the empirical match rate matches expectations."""
    x_template = torch.zeros(1, 1)
    node_sel = torch.tensor([False])
    feat_sel = torch.tensor([False])

    def forward_fn(y):
        # class 1 if value > 0 else 0
        return torch.tensor([[0.0, 1.0]]) if y.item() > 0 else torch.tensor([[1.0, 0.0]])

    # Reference: with zero input, model predicts class 0
    ref = forward_fn(x_template)
    match = make_match_fn("classification", ref, tolerance=0.0)

    # Noise sampler emits a fixed positive value (so the model flips)
    def noise(n_rows):
        return torch.ones(n_rows, 1)

    fid = estimate_fidelity(forward_fn, x_template, node_sel, feat_sel, noise, match, samples=20)
    assert fid == 0.0  # model flips every time
