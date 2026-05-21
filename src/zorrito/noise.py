from __future__ import annotations

from typing import Callable, Literal

import torch


NoiseMode = Literal["column", "row"]


def empirical_column_sampler(
    feature_matrix: torch.Tensor,
    generator: torch.Generator | None = None,
) -> Callable[[int], torch.Tensor]:
    """Per-column independent empirical sampler.

    For each output cell (i, f), draw a row index uniformly from the pool and
    return that row's column-f value. Columns are sampled independently, so
    the resulting row is generally NOT a valid row from the pool — only the
    per-column marginals match. This is the paper's default; it works well
    for continuous features but produces chimeric rows for one-hot / categorical
    features (see noise.py docstring on `empirical_row_sampler`).
    """
    if feature_matrix.dim() != 2:
        raise ValueError(f"feature_matrix must be 2D, got shape {tuple(feature_matrix.shape)}")
    pool = feature_matrix.detach()
    n_pool, n_feats = pool.shape
    device = pool.device

    def sample(n_rows: int) -> torch.Tensor:
        idx = torch.randint(
            0, n_pool, (n_rows, n_feats), device=device, generator=generator
        )
        col = torch.arange(n_feats, device=device).expand(n_rows, n_feats)
        return pool[idx, col]

    return sample


def empirical_row_sampler(
    feature_matrix: torch.Tensor,
    generator: torch.Generator | None = None,
) -> Callable[[int], torch.Tensor]:
    """Whole-row empirical sampler.

    For each output row, draw a row index uniformly from the pool and return
    that row in full. Every replacement row is therefore a real row from the
    pool — for categorical / one-hot features, every replacement is a valid
    category, not a chimera. This is the recommended mode for graph tasks
    over molecules and similar categorical-feature settings.
    """
    if feature_matrix.dim() != 2:
        raise ValueError(f"feature_matrix must be 2D, got shape {tuple(feature_matrix.shape)}")
    pool = feature_matrix.detach()
    n_pool, _ = pool.shape
    device = pool.device

    def sample(n_rows: int) -> torch.Tensor:
        idx = torch.randint(0, n_pool, (n_rows,), device=device, generator=generator)
        return pool[idx]

    return sample


def build_sampler(
    feature_matrix: torch.Tensor,
    mode: NoiseMode,
    generator: torch.Generator | None = None,
) -> Callable[[int], torch.Tensor]:
    if mode == "column":
        return empirical_column_sampler(feature_matrix, generator=generator)
    if mode == "row":
        return empirical_row_sampler(feature_matrix, generator=generator)
    raise ValueError(f"noise_mode must be 'column' or 'row', got {mode!r}")
