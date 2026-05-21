"""Explain one node's prediction on Cora using a 2-layer GCN.

This mirrors the original Zorro `example.ipynb`. Expect a few minutes of
runtime even on a small dataset — the initial ranking pass evaluates every
candidate feature and node, each via Monte Carlo.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv

from zorrito import Zorro


class GCN(nn.Module):
    def __init__(self, in_channels: int, hidden: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, out_channels)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index).relu()
        h = F.dropout(h, p=0.5, training=self.training)
        return self.conv2(h, edge_index)


def train(model, data, epochs: int = 200) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()


def main() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = Planetoid(root="./data/Cora", name="Cora")
    data = dataset[0].to(device)

    model = GCN(dataset.num_features, hidden=16, out_channels=dataset.num_classes).to(device)
    train(model, data)
    model.eval()

    with torch.no_grad():
        accuracy = (model(data.x, data.edge_index).argmax(dim=-1)[data.test_mask] == data.y[data.test_mask]).float().mean().item()
    print(f"Test accuracy: {accuracy:.3f}")

    node_idx = 10
    explainer = Zorro(
        model=model,
        task="node",
        device=device,
        samples=50,        # fewer samples than the paper for quicker turnaround
        top_k=10,
        num_hops=2,
        seed=42,
        log=True,
    )
    explanations = explainer.explain(
        x=data.x,
        edge_index=data.edge_index,
        node_idx=node_idx,
        fidelity_threshold=0.85,
        max_explanations=1,
    )

    expl = explanations[0]
    print(f"\nExplanation for node {node_idx}:")
    print(f"  final fidelity:       {expl.fidelity:.3f}")
    print(f"  # selected neighbors: {int(expl.node_mask.sum())} / {expl.node_mask.numel()}")
    print(f"  # selected features:  {int(expl.feature_mask.sum())} / {expl.feature_mask.numel()}")
    print(f"  original neighbors:   {expl.subgraph_nodes[expl.node_mask].tolist()}")
    print(f"  feature indices:      {expl.selected_feature_indices().tolist()[:20]}{'…' if int(expl.feature_mask.sum()) > 20 else ''}")


if __name__ == "__main__":
    main()
