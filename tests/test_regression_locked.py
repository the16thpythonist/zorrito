"""
Behavioral characterization tests for the Zorro explainer.

These tests are NOT correctness tests for the explainer (those live in
test_explainer.py). They are *snapshot* tests of the exact selected-mask
indices, fidelity values, and greedy trace produced by the current
implementation on three small, deterministic scenarios. They exist to
catch unintended behavioral drift during the in-house-style refactor:
if a refactor silently changes the sampler seed flow, the greedy
candidate order, the initial-ranking semantics, or any tensor op, at
least one of these three tests will fail loudly.

The scenarios are intentionally tiny and use planted signals where the
"correct" answer is obvious and unique:

- a graph-classification model whose output depends on a single feature
  column, with features_only search (locks the feature-axis greedy)
- a node-classification model whose output depends on the target node's
  own feature, with features_only search (locks the feature-axis greedy
  through the subgraph extraction path)
- the same node-classification scenario with nodes_only search (locks
  the node-axis greedy)
"""
from __future__ import annotations

import torch
import torch.nn as nn

from zorrito import Zorro

from .util import SingleFeatureModel


# == PLANTED-SIGNAL MODELS ==
# ``SingleFeatureModel`` for the graph-task scenarios is imported from
# ``util``; the node-task counterpart is only used here and stays local.


class _NodeIdentityModel(nn.Module):
    """Node-level model whose per-node output depends only on that node's own
    feature column k. No message passing, so the target node's own row is
    the only thing that matters.
    """

    def __init__(self, k: int, n_feats: int, n_classes: int = 2) -> None:
        super().__init__()
        self.lin = nn.Linear(n_feats, n_classes, bias=False)
        with torch.no_grad():
            self.lin.weight.zero_()
            self.lin.weight[1, k] = 10.0

    def forward(self, x, edge_index):
        return self.lin(x)


# -- shared fixtures --


def _graph_planted_setup():
    torch.manual_seed(123)
    n_nodes, n_feats, k = 8, 5, 2
    x = torch.zeros(n_nodes, n_feats)
    x[:, k] = 1.0
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    pool = torch.randn(40, n_feats)
    pool[:, k] = -1.0
    model = SingleFeatureModel(k=k, n_feats=n_feats)
    model.eval()
    return x, edge_index, pool, model, k


def _node_planted_setup():
    torch.manual_seed(123)
    n_nodes, n_feats, k = 6, 4, 1
    x = torch.zeros(n_nodes, n_feats)
    x[:, k] = 1.0
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 3, 4, 4, 5],
         [1, 0, 2, 1, 3, 2, 4, 3, 5, 4]],
        dtype=torch.long,
    )
    pool = torch.randn(40, n_feats)
    pool[:, k] = -1.0
    model = _NodeIdentityModel(k=k, n_feats=n_feats)
    model.eval()
    return x, edge_index, pool, model, k


# -- the locked-down characterization tests --


def test_locked_graph_features_only():
    """Lock the graph-task feature-axis greedy outcome on the planted signal."""
    x, edge_index, pool, model, k = _graph_planted_setup()

    explainer = Zorro(
        model=model,
        task="graph",
        select="features_only",
        noise_pool=pool,
        noise_mode="column",
        samples=30,
        top_k=3,
        seed=42,
    )
    [expl] = explainer.explain(
        x=x, edge_index=edge_index, fidelity_threshold=0.9,
    )

    assert expl.selected_feature_indices().tolist() == [2]
    assert expl.fidelity == 1.0
    assert expl.trace == [("init", -1, 0.0), ("feature", 2, 1.0)]
    # frozen axis is all-True
    assert bool(expl.node_mask.all().item()) is True
    assert expl.subgraph_nodes is None


def test_locked_node_features_only():
    """Lock the node-task feature-axis greedy outcome including subgraph map."""
    x, edge_index, pool, model, k = _node_planted_setup()

    explainer = Zorro(
        model=model,
        task="node",
        select="features_only",
        noise_pool=pool,
        noise_mode="column",
        samples=30,
        top_k=3,
        seed=42,
        num_hops=2,
    )
    [expl] = explainer.explain(
        x=x, edge_index=edge_index, node_idx=0, fidelity_threshold=0.9,
    )

    assert expl.selected_feature_indices().tolist() == [1]
    assert expl.fidelity == 1.0
    assert expl.trace == [("init", -1, 0.0), ("feature", 1, 1.0)]
    assert expl.subgraph_nodes is not None
    assert expl.subgraph_nodes.tolist() == [0, 1, 2]
    assert bool(expl.node_mask.all().item()) is True


def test_locked_node_nodes_only():
    """Lock the node-task node-axis greedy outcome."""
    x, edge_index, pool, model, k = _node_planted_setup()

    explainer = Zorro(
        model=model,
        task="node",
        select="nodes_only",
        noise_pool=pool,
        noise_mode="column",
        samples=30,
        top_k=3,
        seed=42,
        num_hops=2,
    )
    [expl] = explainer.explain(
        x=x, edge_index=edge_index, node_idx=0, fidelity_threshold=0.9,
    )

    # node 0 in local-subgraph indices is the target
    assert expl.selected_node_indices().tolist() == [0]
    assert expl.fidelity == 1.0
    assert expl.trace == [("init", -1, 0.0), ("node", 0, 1.0)]
    assert expl.subgraph_nodes is not None
    assert expl.subgraph_nodes.tolist() == [0, 1, 2]
    assert bool(expl.feature_mask.all().item()) is True
