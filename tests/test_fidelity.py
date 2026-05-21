"""
Unit tests for the match predicates and Monte Carlo estimator in
:mod:`zorrito.fidelity`.
"""
import pytest
import torch

from zorrito.fidelity import estimate_fidelity, make_match_fn


# == MATCH FUNCTIONS ==


class TestMakeMatchFn:
    """Tests for the per-sample match-predicate factory."""

    def test_classification_match_exact(self):
        """Classification match returns True iff argmax classes agree."""
        ref = torch.tensor([[0.1, 0.7, 0.2]])
        match = make_match_fn("classification", ref, tolerance=0.0)
        assert bool(match(torch.tensor([[0.1, 0.9, 0.0]])).item()) is True
        assert bool(match(torch.tensor([[0.9, 0.1, 0.0]])).item()) is False

    def test_regression_tolerance_band(self):
        """Regression symmetric band: every coordinate within ``tolerance``."""
        ref = torch.tensor([[1.0, 2.0]])
        match = make_match_fn("regression", ref, tolerance=0.1)
        assert bool(match(torch.tensor([[1.05, 1.95]])).item()) is True
        assert bool(match(torch.tensor([[1.05, 2.5]])).item()) is False

    def test_regression_direction_up(self):
        """``direction='up'`` matches if ``pred <= ref + tolerance`` (any drop is fine)."""
        ref = torch.tensor([[1.0]])
        match = make_match_fn("regression", ref, tolerance=0.1, direction="up")
        assert bool(match(torch.tensor([[1.05]])).item()) is True   # tiny rise OK
        assert bool(match(torch.tensor([[0.0]])).item()) is True    # huge drop OK
        assert bool(match(torch.tensor([[1.20]])).item()) is False  # too high

    def test_regression_direction_down(self):
        """``direction='down'`` matches if ``pred >= ref - tolerance`` (any rise is fine)."""
        ref = torch.tensor([[1.0]])
        match = make_match_fn("regression", ref, tolerance=0.1, direction="down")
        assert bool(match(torch.tensor([[0.95]])).item()) is True   # tiny drop OK
        assert bool(match(torch.tensor([[5.0]])).item()) is True    # huge rise OK
        assert bool(match(torch.tensor([[0.5]])).item()) is False   # too low

    def test_direction_rejected_for_classification(self):
        """``direction != 'both'`` is rejected for classification objectives."""
        ref = torch.tensor([[0.1, 0.7, 0.2]])
        with pytest.raises(ValueError, match="only meaningful for objective='regression'"):
            make_match_fn("classification", ref, tolerance=0.0, direction="up")


# == FIDELITY MONTE CARLO ==


class TestEstimateFidelity:
    """Tests for the Monte Carlo fidelity estimator."""

    def test_estimate_fidelity_all_match(self):
        """
        When everything is selected, no noise leaks in: fidelity is 1.0 even
        when the model is input-sensitive and the sampler is destructive. The
        ``forward_fn`` intentionally depends on the input so that a bug
        flipping ``keep_mask`` (e.g. AND vs OR) would corrupt the output and
        break the test.
        """
        x_template = torch.full((3, 2), 0.5)
        node_sel = torch.tensor([True, True, True])
        feat_sel = torch.tensor([True, True])

        def forward_fn(y):
            # Output depends on input: class 1 wins iff sum(y) > 0.
            s = y.sum()
            return torch.stack([-s, s]).unsqueeze(0)

        def noise(n_rows):
            # Large-magnitude negative noise would flip the prediction if it leaked.
            return torch.full((n_rows, 2), -10.0)

        ref = forward_fn(x_template)  # class 1 (sum = 3.0)
        match = make_match_fn("classification", ref, tolerance=0.0)
        fid = estimate_fidelity(
            forward_fn, x_template, node_sel, feat_sel, noise, match, samples=10
        )
        assert fid == 1.0

    def test_estimate_fidelity_noise_drives_disagreement(self):
        """
        When NOTHING is selected, fidelity should reflect how often a noisy
        input still predicts the same class. We rig the model to flip on
        positive inputs and verify the empirical match rate matches
        expectations.
        """
        x_template = torch.zeros(1, 1)
        node_sel = torch.tensor([False])
        feat_sel = torch.tensor([False])

        def forward_fn(y):
            # Class 1 if value > 0 else 0.
            return torch.tensor([[0.0, 1.0]]) if y.item() > 0 else torch.tensor([[1.0, 0.0]])

        # Reference: with zero input, model predicts class 0.
        ref = forward_fn(x_template)
        match = make_match_fn("classification", ref, tolerance=0.0)

        # Noise sampler emits a fixed positive value (so the model flips).
        def noise(n_rows):
            return torch.ones(n_rows, 1)

        fid = estimate_fidelity(
            forward_fn, x_template, node_sel, feat_sel, noise, match, samples=20
        )
        assert fid == 0.0  # Model flips every time.
