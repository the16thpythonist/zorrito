"""
Match-predicates and the RDT-Fidelity Monte Carlo estimator.

This module owns the *evaluation* half of Zorro: given a candidate
(V_s, F_s) selection, how often does the model still produce a prediction
that "matches" the original? The "match" condition depends on the
prediction task (classification vs. regression) and, for regression, on
the chosen :data:`zorrito._typing.Direction`. The four valid combinations
are:

============== ============== =====================================
objective       direction      match condition
============== ============== =====================================
classification both           ``argmax(pred) == argmax(ref)``
regression     both           ``|pred - ref| <= tolerance``
regression     up             ``pred <= ref + tolerance``
regression     down           ``pred >= ref - tolerance``
============== ============== =====================================

The module also defines :class:`Explanation`, the public return type that
gathers a node mask, a feature mask, the Monte Carlo fidelity estimate,
the greedy step trace, and (for node tasks) the original-graph indices
of the extracted computational subgraph.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch

from zorrito._typing import Direction, Objective


# == EXPLANATION RESULT ==


@dataclass
class Explanation:
    """
    The result of a single Zorro greedy search.

    Both masks always have the *full* shape for the relevant axis. When
    the search was restricted to one axis (``select="nodes_only"`` /
    ``select="features_only"``), the frozen axis's mask is all-True —
    nothing was excluded along that axis.

    :param node_mask: Bool tensor of shape ``(V,)``. ``True`` at positions
        the greedy selected (or, for a frozen axis, every position). For
        node tasks the indexing is into the extracted computational
        subgraph; :attr:`subgraph_nodes` maps those local indices back to
        the original graph.
    :param feature_mask: Bool tensor of shape ``(K,)`` over feature columns.
    :param fidelity: The final RDT-Fidelity reached by the greedy search,
        in ``[0, 1]``.
    :param trace: Ordered log of greedy steps. The first entry is always
        ``("init", -1, initial_fidelity)``; subsequent entries are tuples
        ``(kind, index, fidelity_after_step)`` with ``kind`` in
        ``{"node", "feature"}``.
    :param subgraph_nodes: For node tasks, the original-graph indices of
        the extracted L-hop subgraph, ordered by their local index. For
        graph tasks this is ``None``.

    V - number of nodes (in the subgraph for node tasks, in the full graph
        for graph tasks)
    K - feature dimensionality
    """

    node_mask: torch.Tensor
    feature_mask: torch.Tensor
    fidelity: float
    trace: list[tuple[str, int, float]] = field(default_factory=list)
    subgraph_nodes: torch.Tensor | None = None

    def selected_node_indices(self) -> torch.Tensor:
        """
        :returns: A ``LongTensor`` of the positions where
            :attr:`node_mask` is ``True``.
        """
        return self.node_mask.nonzero(as_tuple=False).flatten()

    def selected_feature_indices(self) -> torch.Tensor:
        """
        :returns: A ``LongTensor`` of the positions where
            :attr:`feature_mask` is ``True``.
        """
        return self.feature_mask.nonzero(as_tuple=False).flatten()


# == MATCH FUNCTIONS ==


def make_match_fn(
    objective: Objective,
    reference_output: torch.Tensor,
    tolerance: float,
    direction: Direction = "both",
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Build a per-sample match predicate from the reference forward-pass output.

    The returned closure consumes a single forward-pass output tensor and
    returns a 0-dimensional boolean tensor indicating whether that output
    counts as a "match" against the captured reference. The four valid
    ``(objective, direction)`` combinations and the corresponding match
    conditions are documented in the module docstring; ``classification``
    only supports ``direction="both"``.

    :param objective: Either ``"classification"`` or ``"regression"``.
    :param reference_output: The model's output on the unperturbed input.
        For classification this is reduced to its argmax class index; for
        regression the raw output is kept and a tolerance band is applied.
    :param tolerance: Half-width of the tolerance band for regression. Must
        be non-negative. Ignored for classification.
    :param direction: For regression, which side(s) of the tolerance band
        count as a match. Must be ``"both"`` for classification.

    :returns: A closure ``(out: Tensor) -> Tensor`` returning a 0-d bool.
    """
    if objective == "classification":
        if direction != "both":
            raise ValueError(
                "direction is only meaningful for objective='regression'; "
                f"got direction={direction!r} for classification"
            )
        # ref_class: (B,) for graph tasks, scalar for node tasks. Captured
        # once and reused by every match call.
        ref_class = reference_output.argmax(dim=-1)

        def match(out: torch.Tensor) -> torch.Tensor:
            pred = out.argmax(dim=-1)
            return (pred == ref_class).all().reshape(())

        return match

    if objective == "regression":
        # ref: same shape as out, detached so backprop never crosses it.
        ref = reference_output.detach()

        if direction == "both":
            def match(out: torch.Tensor) -> torch.Tensor:
                return ((out - ref).abs() <= tolerance).all().reshape(())
        elif direction == "up":
            def match(out: torch.Tensor) -> torch.Tensor:
                return ((out - ref) <= tolerance).all().reshape(())
        elif direction == "down":
            def match(out: torch.Tensor) -> torch.Tensor:
                return ((out - ref) >= -tolerance).all().reshape(())
        else:
            raise ValueError(f"unknown direction: {direction!r}")

        return match

    raise ValueError(f"unknown objective: {objective!r}")


# == FIDELITY MONTE CARLO ==


def estimate_fidelity(
    forward_fn: Callable[[torch.Tensor], torch.Tensor],
    x_template: torch.Tensor,
    node_selected: torch.Tensor,
    feature_selected: torch.Tensor,
    noise_sampler: Callable[[int], torch.Tensor],
    match_fn: Callable[[torch.Tensor], torch.Tensor],
    samples: int,
) -> float:
    """
    Monte Carlo estimate of RDT-Fidelity for a given selection ``(V_s, F_s)``.

    For each of ``samples`` trials this function starts from ``x_template``,
    overwrites every ``(node, feature)`` cell *outside* the selection with
    a fresh draw from ``noise_sampler``, runs ``forward_fn`` on the
    perturbed input, and counts how often the result still satisfies
    ``match_fn``. The fidelity estimate is the empirical match rate.

    :param forward_fn: The model wrapper. Consumes a perturbed ``(V, K)``
        feature matrix and returns the prediction tensor that ``match_fn``
        was built against.
    :param x_template: The original ``(V, K)`` feature matrix that
        non-perturbed cells are copied from.
    :param node_selected: Bool tensor of shape ``(V,)``; cells with
        ``True`` are kept from ``x_template``.
    :param feature_selected: Bool tensor of shape ``(K,)``; cells with
        ``True`` are kept from ``x_template``.
    :param noise_sampler: Callable yielding ``(V, K)`` noise tensors.
    :param match_fn: A predicate as produced by :func:`make_match_fn`.
    :param samples: Number of Monte Carlo trials.

    :returns: The empirical match rate in ``[0, 1]``.
    """
    n_nodes, n_feats = x_template.shape
    # keep_mask: (V, K) bool — True only where both the node and the feature
    # are in the selection. This is the outer product of the two 1D masks.
    keep_mask = node_selected.view(-1, 1) & feature_selected.view(1, -1)

    correct = 0
    for _ in range(samples):
        noise = noise_sampler(n_nodes)
        # y: (V, K) — kept cells from template, perturbed cells from noise.
        y = torch.where(keep_mask, x_template, noise)
        out = forward_fn(y)
        if bool(match_fn(out).item()):
            correct += 1
    return correct / samples
