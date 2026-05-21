"""
Shared test helpers — tiny synthetic graphs and toy models that are reused
across the unit-test files.

Helpers live here rather than in ``conftest.py`` so that they can be
imported as ordinary classes/functions (``from .util import TinyGCN``).
``conftest.py`` is reserved for pytest fixtures.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool


# == TOY MODELS ==


class TinyGCN(nn.Module):
    """
    Two-layer GCN for node-level tests.

    Standard ``conv -> relu -> conv`` stack with no pooling. Used as the
    smoke-test target for ``Zorro(task="node")``.

    :param in_channels: Input feature dimensionality.
    :param hidden: Hidden dimensionality of the intermediate layer.
    :param out_channels: Number of output classes / regression dimensions.
    """

    def __init__(self, in_channels: int, hidden: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, out_channels)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index).relu()
        return self.conv2(h, edge_index)


class TinyGraphGCN(nn.Module):
    """
    Two-layer GCN with global mean pooling for graph-level tests.

    Used as the smoke-test target for ``Zorro(task="graph")``. The ``batch``
    argument follows the standard PyG convention.
    """

    def __init__(self, in_channels: int, hidden: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.lin = nn.Linear(hidden, out_channels)

    def forward(self, x, edge_index, batch):
        h = self.conv1(x, edge_index).relu()
        h = self.conv2(h, edge_index).relu()
        h = global_mean_pool(h, batch)
        return self.lin(h)


class SingleFeatureModel(nn.Module):
    """
    Graph-level model whose output depends ONLY on feature column ``k`` of
    the mean-pooled representation.

    Used to plant a known signal that the greedy explainer must recover.
    Class 1 wins iff ``mean(x[:, k]) > 0``; every other column has zero
    weight, so a correct explainer reports feature ``k`` as the entire
    explanation.

    :param k: The single feature column the model attends to.
    :param n_feats: Feature dimensionality of the input.
    :param n_classes: Number of output classes; default 2.
    """

    def __init__(self, k: int, n_feats: int, n_classes: int = 2) -> None:
        super().__init__()
        self.lin = nn.Linear(n_feats, n_classes, bias=False)
        with torch.no_grad():
            self.lin.weight.zero_()
            self.lin.weight[1, k] = 10.0

    def forward(self, x, edge_index, batch):
        h = global_mean_pool(x, batch)
        return self.lin(h)


# == SYNTHETIC GRAPHS ==


def make_random_node_graph(
    seed: int = 0,
    n_nodes: int = 12,
    n_feats: int = 8,
    edge_prob: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build a small random graph with Gaussian node features.

    The graph topology is sampled with a Bernoulli edge probability, and the
    node features are drawn from a standard normal. Used as the input for
    smoke tests; the structure is irrelevant as long as it produces a
    well-formed PyG edge index.

    :param seed: torch RNG seed used for both features and edges.
    :param n_nodes: Number of nodes in the graph.
    :param n_feats: Feature dimensionality.
    :param edge_prob: Per-directed-edge sampling probability.

    :returns: A tuple ``(x, edge_index)`` of shape ``(V, K)`` and ``(2, E)``.
    """
    torch.manual_seed(seed)
    x = torch.randn(n_nodes, n_feats)
    edges = []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j and torch.rand(1).item() < edge_prob:
                edges.append([i, j])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return x, edge_index
