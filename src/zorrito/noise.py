"""
Empirical noise samplers used inside the fidelity Monte Carlo.

When the explainer evaluates how much a candidate (V_s, F_s) selection
"explains" a prediction, every cell *outside* the selection is overwritten
with a draw from an empirical pool. The two strategies in this module
differ in how those replacements are drawn:

- :class:`EmpiricalColumnSampler` draws each cell independently from its
  column's marginal distribution. This is the paper's default. It works
  well for continuous features but produces invalid chimeric rows when
  the pool is categorical / one-hot.
- :class:`EmpiricalRowSampler` draws whole rows from the pool. Every
  replacement is a real pool row, which preserves any per-row structure
  (e.g. one-hot validity) at the cost of being less varied.

Both samplers are callable with the signature ``(n_rows: int) -> Tensor``
of shape ``(n_rows, n_feats)``, so they are interchangeable wherever a
``Callable[[int], torch.Tensor]`` is expected. The free-function
constructors ``empirical_column_sampler`` / ``empirical_row_sampler``
exist as thin one-line factories for the closure-shaped call sites that
predate the class hierarchy.
"""
from __future__ import annotations

from typing import Callable

import torch

from zorrito._typing import NoiseMode


# == ABSTRACT BASE ==


class AbstractNoiseSampler:
    """
    Abstract base class for empirical noise samplers.

    A noise sampler is a callable that, given an integer number of rows to
    produce, returns a tensor of shape ``(n_rows, n_feats)`` whose values
    are drawn from an empirical pool. The concrete sub-classes differ only
    in the *granularity* of the draw (per-cell vs. per-row).

    Concrete sub-classes must implement :meth:`__call__`. They are free to
    pre-compute and cache any pool-derived state in ``__init__``.

    N - pool size, number of rows in the feature_matrix
    K - feature dimensionality, number of columns in the feature_matrix

    :param feature_matrix: A (N, K) tensor whose rows are valid feature
        vectors drawn from the data distribution. Acts as the empirical
        pool from which replacement values are drawn.
    :param generator: Optional torch.Generator. When provided, all draws
        from this sampler use it, making the sampler deterministic given a
        seeded generator.
    """

    def __init__(
        self,
        feature_matrix: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> None:
        if feature_matrix.dim() != 2:
            raise ValueError(
                f"feature_matrix must be 2D, got shape {tuple(feature_matrix.shape)}"
            )
        self.pool: torch.Tensor = feature_matrix.detach()
        self.generator: torch.Generator | None = generator
        self.n_pool, self.n_feats = self.pool.shape
        self.device: torch.device = self.pool.device

    def __call__(self, n_rows: int) -> torch.Tensor:
        """
        Draw a ``(n_rows, n_feats)`` tensor from the empirical pool.

        :param n_rows: The number of replacement rows to draw.

        :returns: A tensor of shape ``(n_rows, n_feats)``.
        """
        raise NotImplementedError(
            'Concrete noise samplers must implement the "__call__" method'
        )


# == CONCRETE SAMPLERS ==


class EmpiricalColumnSampler(AbstractNoiseSampler):
    """
    Per-column independent empirical sampler.

    For each output cell ``(i, f)`` a row index is drawn uniformly from the
    pool and that row's column-``f`` value is used. Columns are sampled
    independently, so the resulting row is generally NOT a valid pool row
    — only the per-column marginals match. This is the paper's default and
    works well for continuous features; for one-hot / categorical features
    it produces chimeric rows (see :class:`EmpiricalRowSampler`).
    """

    def __call__(self, n_rows: int) -> torch.Tensor:
        # idx: (n_rows, n_feats) of row indices into the pool, drawn
        # independently per (output_row, feature_column) pair.
        idx = torch.randint(
            0,
            self.n_pool,
            (n_rows, self.n_feats),
            device=self.device,
            generator=self.generator,
        )
        # col: (n_rows, n_feats) — a broadcast column-index grid so that
        # advanced indexing pulls the correct cell from each chosen row.
        col = torch.arange(self.n_feats, device=self.device).expand(n_rows, self.n_feats)
        return self.pool[idx, col]


class EmpiricalRowSampler(AbstractNoiseSampler):
    """
    Whole-row empirical sampler.

    For each output row a row index is drawn uniformly from the pool and
    that row is copied wholesale. Every replacement row is therefore a
    real pool row — for categorical / one-hot features every replacement
    is a valid category, not a chimera. This is the recommended mode for
    graph tasks over molecules and similar settings.
    """

    def __call__(self, n_rows: int) -> torch.Tensor:
        # idx: (n_rows,) of row indices into the pool.
        idx = torch.randint(
            0, self.n_pool, (n_rows,), device=self.device, generator=self.generator
        )
        return self.pool[idx]


# == FACTORY FUNCTIONS ==
# These thin free functions exist so that older closure-style call sites
# ("sampler = empirical_column_sampler(pool); sampler(n)") keep working
# unchanged after the class refactor. They are equivalent to constructing
# the corresponding sampler class directly.


def empirical_column_sampler(
    feature_matrix: torch.Tensor,
    generator: torch.Generator | None = None,
) -> AbstractNoiseSampler:
    """
    Construct an :class:`EmpiricalColumnSampler` over the given pool.

    :param feature_matrix: A 2D tensor whose rows are valid feature vectors.
    :param generator: Optional torch.Generator for determinism.

    :returns: A callable sampler instance.
    """
    return EmpiricalColumnSampler(feature_matrix, generator=generator)


def empirical_row_sampler(
    feature_matrix: torch.Tensor,
    generator: torch.Generator | None = None,
) -> AbstractNoiseSampler:
    """
    Construct an :class:`EmpiricalRowSampler` over the given pool.

    :param feature_matrix: A 2D tensor whose rows are valid feature vectors.
    :param generator: Optional torch.Generator for determinism.

    :returns: A callable sampler instance.
    """
    return EmpiricalRowSampler(feature_matrix, generator=generator)


def build_sampler(
    feature_matrix: torch.Tensor,
    mode: NoiseMode,
    generator: torch.Generator | None = None,
) -> AbstractNoiseSampler:
    """
    Dispatch helper that selects the concrete sampler implementation by
    string identifier.

    :param feature_matrix: A 2D tensor whose rows are valid feature vectors.
    :param mode: One of ``"column"`` or ``"row"``.
    :param generator: Optional torch.Generator for determinism.

    :returns: A callable sampler instance of the requested kind.
    """
    if mode == "column":
        return EmpiricalColumnSampler(feature_matrix, generator=generator)
    if mode == "row":
        return EmpiricalRowSampler(feature_matrix, generator=generator)
    raise ValueError(f"noise_mode must be 'column' or 'row', got {mode!r}")
