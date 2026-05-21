"""
Subgraph extraction helper used by the node-task path of the explainer.

For node-classification explanations Zorro only needs to perturb the
L-hop computational neighborhood of the target node — anything beyond
that cannot influence the prediction under a depth-L GNN. This module
wraps :func:`torch_geometric.utils.k_hop_subgraph` to return the four
pieces of information the explainer needs in one tuple.
"""
from __future__ import annotations

import torch
from torch_geometric.utils import k_hop_subgraph


# == SUBGRAPH EXTRACTION ==


def extract_computational_subgraph(
    node_idx: int,
    num_hops: int,
    x: torch.Tensor,
    edge_index: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
    """
    Extract the L-hop neighborhood used by a GNN to predict ``node_idx``.

    :param node_idx: The index of the target node in the original graph.
    :param num_hops: The depth ``L`` of the GNN — used as the neighborhood
        radius. Must match the layer count of the model being explained.
    :param x: Original-graph node feature matrix of shape ``(V, K)``.
    :param edge_index: Original-graph edge index of shape ``(2, E)``.

    :returns: A 4-tuple ``(x_sub, edge_index_sub, target_local, subset)``:
        ``x_sub`` is the feature matrix restricted to subgraph nodes;
        ``edge_index_sub`` is the edge index relabeled to local subgraph
        indices; ``target_local`` is the position of ``node_idx`` inside
        the subgraph; ``subset`` is the LongTensor of original-graph
        indices of the subgraph nodes, ordered by their local index.
    """
    subset, edge_index_sub, target_local, _ = k_hop_subgraph(
        node_idx=node_idx,
        num_hops=num_hops,
        edge_index=edge_index,
        relabel_nodes=True,
        num_nodes=x.size(0),
    )
    x_sub = x[subset]
    return x_sub, edge_index_sub, int(target_local.item()), subset
