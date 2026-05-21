"""End-to-end tests for Zorro on synthetic graphs.

The setup: a tiny dataset where the prediction depends on a known subset of
nodes/features. We verify the explainer (a) terminates, (b) reaches the
threshold, (c) returns a well-shaped Explanation.
"""
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool

from zorrito import Zorro


class TinyGCN(nn.Module):
    def __init__(self, in_channels: int, hidden: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, out_channels)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index).relu()
        return self.conv2(h, edge_index)


class TinyGraphGCN(nn.Module):
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


def _make_random_node_graph(seed: int = 0):
    torch.manual_seed(seed)
    n_nodes = 12
    n_feats = 8
    x = torch.randn(n_nodes, n_feats)
    edges = []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j and torch.rand(1).item() < 0.25:
                edges.append([i, j])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return x, edge_index


def test_node_explainer_runs():
    torch.manual_seed(0)
    x, edge_index = _make_random_node_graph(seed=0)
    model = TinyGCN(in_channels=x.size(1), hidden=8, out_channels=3)
    model.eval()

    explainer = Zorro(
        model=model,
        task="node",
        samples=20,
        top_k=3,
        seed=42,
        num_hops=2,
    )
    result = explainer.explain(
        x=x,
        edge_index=edge_index,
        node_idx=0,
        fidelity_threshold=0.5,
        max_explanations=1,
    )
    assert len(result) == 1
    expl = result[0]
    assert expl.node_mask.dtype == torch.bool
    assert expl.feature_mask.dtype == torch.bool
    assert expl.fidelity >= 0.0
    assert expl.subgraph_nodes is not None
    assert expl.node_mask.shape[0] == expl.subgraph_nodes.shape[0]


def test_graph_explainer_runs():
    torch.manual_seed(0)
    x, edge_index = _make_random_node_graph(seed=1)
    model = TinyGraphGCN(in_channels=x.size(1), hidden=8, out_channels=3)
    model.eval()

    explainer = Zorro(
        model=model,
        task="graph",
        samples=20,
        top_k=3,
        seed=42,
    )
    result = explainer.explain(
        x=x,
        edge_index=edge_index,
        fidelity_threshold=0.5,
        max_explanations=1,
    )
    assert len(result) == 1
    expl = result[0]
    assert expl.node_mask.shape[0] == x.size(0)
    assert expl.feature_mask.shape[0] == x.size(1)
    assert expl.subgraph_nodes is None  # graph task does not extract a subgraph


def test_regression_tolerance_branch():
    torch.manual_seed(0)
    x, edge_index = _make_random_node_graph(seed=2)

    # Regression: 1-output GCN
    class RegGCN(nn.Module):
        def __init__(self, in_channels, hidden):
            super().__init__()
            self.conv1 = GCNConv(in_channels, hidden)
            self.conv2 = GCNConv(hidden, 1)

        def forward(self, x, edge_index):
            h = self.conv1(x, edge_index).relu()
            return self.conv2(h, edge_index)

    model = RegGCN(in_channels=x.size(1), hidden=8)
    model.eval()

    explainer = Zorro(
        model=model,
        task="node",
        objective="regression",
        tolerance=0.5,
        samples=20,
        top_k=3,
        seed=42,
    )
    result = explainer.explain(
        x=x,
        edge_index=edge_index,
        node_idx=3,
        fidelity_threshold=0.5,
        max_explanations=1,
    )
    assert len(result) == 1
    assert result[0].fidelity >= 0.0


def test_explain_rejects_node_idx_for_graph_task():
    import pytest

    x, edge_index = _make_random_node_graph(seed=0)
    model = TinyGraphGCN(in_channels=x.size(1), hidden=8, out_channels=3)
    explainer = Zorro(model=model, task="node", samples=5)
    with pytest.raises(ValueError, match="node_idx is required"):
        explainer.explain(x=x, edge_index=edge_index, node_idx=None)


def test_graph_task_defaults_to_row_noise():
    """Graph tasks should default to row-mode sampling (so categorical
    features stay valid)."""
    model = TinyGraphGCN(in_channels=4, hidden=4, out_channels=2)
    explainer = Zorro(model=model, task="graph", samples=5)
    assert explainer.noise_mode == "row"


def test_node_task_defaults_to_column_noise():
    model = TinyGCN(in_channels=4, hidden=4, out_channels=2)
    explainer = Zorro(model=model, task="node", samples=5)
    assert explainer.noise_mode == "column"


def test_custom_noise_pool_is_used():
    """If noise_pool is given to __init__, it overrides x for sampling."""
    torch.manual_seed(0)
    x, edge_index = _make_random_node_graph(seed=0)
    model = TinyGraphGCN(in_channels=x.size(1), hidden=8, out_channels=2)
    model.eval()

    # different-shape custom pool: 100 rows with the same feature dim
    custom_pool = torch.randn(100, x.size(1))
    explainer = Zorro(
        model=model,
        task="graph",
        noise_pool=custom_pool,
        noise_mode="row",
        samples=10,
        top_k=2,
        seed=1,
    )
    result = explainer.explain(x=x, edge_index=edge_index, fidelity_threshold=0.5)
    assert len(result) == 1
    assert result[0].node_mask.shape[0] == x.size(0)


def test_select_nodes_only_freezes_feature_mask():
    torch.manual_seed(0)
    x, edge_index = _make_random_node_graph(seed=0)
    model = TinyGraphGCN(in_channels=x.size(1), hidden=8, out_channels=2)
    model.eval()
    explainer = Zorro(
        model=model,
        task="graph",
        select="nodes_only",
        samples=10,
        top_k=3,
        seed=42,
    )
    result = explainer.explain(x=x, edge_index=edge_index, fidelity_threshold=0.5)
    assert result[0].feature_mask.all().item() is True


def test_direction_validation():
    """direction='up'/'down' only allowed with regression."""
    import pytest
    x, edge_index = _make_random_node_graph(seed=0)
    model = TinyGraphGCN(in_channels=x.size(1), hidden=8, out_channels=2)
    with pytest.raises(ValueError, match="direction is only meaningful"):
        Zorro(model=model, task="graph", objective="classification", direction="up", samples=5)


def test_direction_accepted_for_regression():
    x, edge_index = _make_random_node_graph(seed=0)

    class RegGraphGCN(nn.Module):
        def __init__(self, in_channels, hidden):
            super().__init__()
            self.conv1 = GCNConv(in_channels, hidden)
            self.conv2 = GCNConv(hidden, 1)

        def forward(self, x, edge_index, batch):
            h = self.conv1(x, edge_index).relu()
            h = self.conv2(h, edge_index)
            return global_mean_pool(h, batch)

    model = RegGraphGCN(in_channels=x.size(1), hidden=8)
    model.eval()
    explainer = Zorro(
        model=model,
        task="graph",
        objective="regression",
        direction="down",
        tolerance=0.5,
        samples=10,
        top_k=3,
        seed=42,
    )
    result = explainer.explain(x=x, edge_index=edge_index, fidelity_threshold=0.3)
    assert len(result) == 1


def test_select_features_only_freezes_node_mask():
    torch.manual_seed(0)
    x, edge_index = _make_random_node_graph(seed=0)
    model = TinyGraphGCN(in_channels=x.size(1), hidden=8, out_channels=2)
    model.eval()
    explainer = Zorro(
        model=model,
        task="graph",
        select="features_only",
        samples=10,
        top_k=3,
        seed=42,
    )
    result = explainer.explain(x=x, edge_index=edge_index, fidelity_threshold=0.5)
    assert result[0].node_mask.all().item() is True
