from __future__ import annotations

import torch
from torch_geometric.utils import k_hop_subgraph


def extract_computational_subgraph(
    node_idx: int,
    num_hops: int,
    x: torch.Tensor,
    edge_index: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
    """Extract the L-hop neighborhood used by a GNN to predict `node_idx`.

    Returns:
        x_sub: feature matrix restricted to subgraph nodes
        edge_index_sub: edges relabeled to local indices
        target_local: position of `node_idx` inside the subgraph
        subset: original node indices, ordered by their local indices
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
