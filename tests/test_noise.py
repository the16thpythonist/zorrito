import torch

from zorrito.noise import (
    build_sampler,
    empirical_column_sampler,
    empirical_row_sampler,
)


def test_sampler_shape():
    pool = torch.randn(50, 7)
    sample = empirical_column_sampler(pool)
    out = sample(12)
    assert out.shape == (12, 7)


def test_sampler_values_come_from_columns():
    """Every drawn value must appear in its column's original pool."""
    pool = torch.tensor(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
        ]
    )
    sample = empirical_column_sampler(pool)
    out = sample(100)
    col0_pool = set(pool[:, 0].tolist())
    col1_pool = set(pool[:, 1].tolist())
    assert set(out[:, 0].tolist()).issubset(col0_pool)
    assert set(out[:, 1].tolist()).issubset(col1_pool)


def test_sampler_rejects_1d_input():
    import pytest

    with pytest.raises(ValueError):
        empirical_column_sampler(torch.randn(10))


def test_sampler_deterministic_with_generator():
    pool = torch.randn(20, 4)
    gen_a = torch.Generator().manual_seed(42)
    gen_b = torch.Generator().manual_seed(42)
    out_a = empirical_column_sampler(pool, generator=gen_a)(15)
    out_b = empirical_column_sampler(pool, generator=gen_b)(15)
    assert torch.equal(out_a, out_b)


def test_row_sampler_returns_real_rows():
    """Every row of a row-sampler's output must equal some row of the pool."""
    pool = torch.tensor(
        [
            [1.0, 0.0, 0.0],   # "carbon"
            [0.0, 1.0, 0.0],   # "nitrogen"
            [0.0, 0.0, 1.0],   # "oxygen"
        ]
    )
    sample = empirical_row_sampler(pool)
    out = sample(50)
    pool_rows = {tuple(row.tolist()) for row in pool}
    out_rows = {tuple(row.tolist()) for row in out}
    assert out_rows.issubset(pool_rows)
    # also: every row sums to 1 (one-hot preserved)
    assert torch.allclose(out.sum(dim=1), torch.ones(50))


def test_column_sampler_can_break_one_hot_validity():
    """Sanity check on the motivation: per-column sampling produces invalid
    one-hot rows when fed one-hot data, while per-row preserves the structure."""
    one_hot_pool = torch.eye(7).repeat(10, 1)  # 70 valid 7-d one-hot rows
    col = empirical_column_sampler(one_hot_pool, generator=torch.Generator().manual_seed(0))(200)
    row = empirical_row_sampler(one_hot_pool, generator=torch.Generator().manual_seed(0))(200)
    # column sampler: usually some rows are not exactly one-hot
    col_valid = (col.sum(dim=1) == 1).float().mean().item()
    row_valid = (row.sum(dim=1) == 1).float().mean().item()
    assert row_valid == 1.0
    assert col_valid < 0.9   # the motivating failure mode


def test_build_sampler_dispatches():
    pool = torch.randn(8, 3)
    s_col = build_sampler(pool, mode="column")
    s_row = build_sampler(pool, mode="row")
    assert s_col(5).shape == (5, 3)
    assert s_row(5).shape == (5, 3)

    import pytest

    with pytest.raises(ValueError):
        build_sampler(pool, mode="not_a_mode")  # type: ignore[arg-type]
