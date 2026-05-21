"""
Explain one molecule's class on the MUTAG graph-classification dataset.

MUTAG is a tiny dataset (188 molecules) where each graph carries a binary
class label. This example trains a small GIN with global mean pooling
and asks Zorrito to explain a single molecule's predicted label.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_mean_pool

from zorrito import Zorro


# == MODEL ==


class GIN(nn.Module):
    """
    A small two-layer GIN with global mean pooling.

    Each GINConv layer is built on a ``Linear -> ReLU -> Linear`` MLP. The
    pooled embedding is projected to class logits by a final linear layer.

    :param in_channels: Input feature dimensionality.
    :param hidden: Hidden dimensionality used inside both GINConvs.
    :param out_channels: Number of output classes.
    """

    def __init__(self, in_channels: int, hidden: int, out_channels: int) -> None:
        super().__init__()
        nn1 = nn.Sequential(nn.Linear(in_channels, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        nn2 = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.conv1 = GINConv(nn1)
        self.conv2 = GINConv(nn2)
        self.lin = nn.Linear(hidden, out_channels)

    def forward(self, x, edge_index, batch):
        h = self.conv1(x, edge_index).relu()
        h = self.conv2(h, edge_index).relu()
        h = global_mean_pool(h, batch)
        return self.lin(h)


# == TRAINING ==


def train(model, loader, device, epochs: int = 60) -> None:
    """
    Train the model with Adam + cross-entropy for ``epochs`` epochs.

    :param model: The model to train.
    :param loader: A PyG DataLoader yielding training batches.
    :param device: torch device.
    :param epochs: Number of training epochs.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()
    for _ in range(epochs):
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = F.cross_entropy(out, batch.y)
            loss.backward()
            optimizer.step()


# == EXPLAIN ==


def main() -> None:
    """Train a GIN on MUTAG and run Zorrito on one held-out molecule."""
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = TUDataset(root="./data/MUTAG", name="MUTAG").shuffle()
    n_train = int(len(dataset) * 0.8)
    train_set, test_set = dataset[:n_train], dataset[n_train:]
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=32)

    model = GIN(dataset.num_features, hidden=32, out_channels=dataset.num_classes).to(device)
    train(model, train_loader, device)
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            pred = model(batch.x, batch.edge_index, batch.batch).argmax(dim=-1)
            correct += (pred == batch.y).sum().item()
            total += batch.num_graphs
    print(f"Test accuracy: {correct / total:.3f}")

    # Pick the first test molecule and explain it.
    target = test_set[0].to(device)
    with torch.no_grad():
        pred = model(
            target.x,
            target.edge_index,
            torch.zeros(target.num_nodes, dtype=torch.long, device=device),
        ).argmax(dim=-1).item()
    print(f"Explaining molecule with true class {int(target.y.item())} (predicted {pred})")

    explainer = Zorro(
        model=model,
        task="graph",
        device=device,
        samples=50,
        top_k=10,
        seed=42,
        log=True,
    )
    explanations = explainer.explain(
        x=target.x,
        edge_index=target.edge_index,
        fidelity_threshold=0.85,
        max_explanations=1,
    )

    expl = explanations[0]
    print(f"\nGraph-level explanation:")
    print(f"  final fidelity:      {expl.fidelity:.3f}")
    print(f"  # selected atoms:    {int(expl.node_mask.sum())} / {expl.node_mask.numel()}")
    print(f"  # selected features: {int(expl.feature_mask.sum())} / {expl.feature_mask.numel()}")
    print(f"  atom indices:        {expl.selected_node_indices().tolist()}")
    print(f"  feature indices:     {expl.selected_feature_indices().tolist()}")


if __name__ == "__main__":
    main()
