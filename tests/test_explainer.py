"""
End-to-end tests for the :class:`Zorro` explainer on synthetic graphs.

The fixtures are intentionally tiny (12 nodes, 8 features, ~3 second
total runtime) so that the suite stays fast. Where a stronger statement
is needed than "doesn't crash and shapes are right", the test plants a
known signal into the data and verifies the explainer recovers it.
"""
import pytest
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool

from zorrito import Zorro

from .util import (
    SingleFeatureModel,
    TinyGCN,
    TinyGraphGCN,
    make_random_node_graph,
)


# == NODE TASK ==


class TestZorroNode:
    """Smoke and shape tests for ``Zorro(task="node")``."""

    def test_node_explainer_runs(self):
        """The node explainer terminates, returns one well-shaped Explanation."""
        torch.manual_seed(0)
        x, edge_index = make_random_node_graph(seed=0)
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

    def test_regression_tolerance_branch(self):
        """The regression-tolerance branch runs end-to-end for a node task."""
        torch.manual_seed(0)
        x, edge_index = make_random_node_graph(seed=2)

        # Regression: 1-output GCN.
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

    def test_subgraph_nodes_roundtrip(self):
        """
        For node tasks, ``expl.subgraph_nodes`` must map local subgraph
        indices back to ORIGINAL-graph indices. A regression that silently
        set it to ``torch.arange(n_sub)`` (local indices) would still
        produce shape-valid output but mislead every downstream "which
        original node mattered?" check.
        """
        # Chain graph: 0 - 1 - 2 - 3 - 4 - 5 (undirected: edges in both directions).
        x = torch.randn(6, 4)
        edge_index = torch.tensor(
            [
                [0, 1, 1, 2, 2, 3, 3, 4, 4, 5],
                [1, 0, 2, 1, 3, 2, 4, 3, 5, 4],
            ],
            dtype=torch.long,
        )

        model = TinyGCN(in_channels=4, hidden=4, out_channels=2)
        model.eval()

        # 1-hop neighbourhood of node 5 in this chain is exactly {4, 5}.
        explainer = Zorro(model=model, task="node", num_hops=1, samples=5, top_k=2, seed=0)
        expl = explainer.explain(
            x=x, edge_index=edge_index, node_idx=5, fidelity_threshold=0.5,
        )[0]

        assert expl.subgraph_nodes is not None
        assert set(expl.subgraph_nodes.tolist()) == {4, 5}
        # Every reported index must be a valid global index.
        assert (expl.subgraph_nodes < x.size(0)).all().item()
        # Mask length and subgraph_nodes length must agree.
        assert expl.node_mask.shape[0] == expl.subgraph_nodes.shape[0]


# == GRAPH TASK ==


class TestZorroGraph:
    """Smoke and behavioural tests for ``Zorro(task="graph")``."""

    def test_graph_explainer_runs(self):
        """The graph explainer terminates and produces a well-shaped Explanation."""
        torch.manual_seed(0)
        x, edge_index = make_random_node_graph(seed=1)
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
        assert expl.subgraph_nodes is None  # Graph task does not extract a subgraph.

    def test_direction_changes_explanation(self):
        """
        ``direction='up'`` and ``direction='down'`` must produce different
        greedy paths when noise is asymmetric. Otherwise the direction
        parameter has no effect and the suite would not notice if it were
        silently dropped.
        """
        x, edge_index = make_random_node_graph(seed=0)

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

        # Strongly asymmetric noise: shifted positive so it pushes predictions UP.
        pool = torch.randn(100, x.size(1)) + 3.0

        def run(direction: str):
            explainer = Zorro(
                model=model,
                task="graph",
                objective="regression",
                direction=direction,
                tolerance=0.3,
                noise_pool=pool,
                noise_mode="row",
                samples=25,
                top_k=3,
                seed=42,
            )
            return explainer.explain(x=x, edge_index=edge_index, fidelity_threshold=0.5)[0]

        up = run("up")
        down = run("down")

        differ = (
            not torch.equal(up.node_mask, down.node_mask)
            or not torch.equal(up.feature_mask, down.feature_mask)
            or up.fidelity != down.fidelity
        )
        assert differ, (
            "direction='up' and direction='down' produced identical results; "
            "direction parameter appears to have no effect"
        )

    def test_max_explanations_returns_disjoint(self):
        """
        ``max_explanations > 1`` enumerates disjoint explanations on the
        searched axes. Guards against the ``&= ~mask`` removal logic
        flipping to ``|= mask``.
        """
        torch.manual_seed(0)
        x, edge_index = make_random_node_graph(seed=3)
        model = TinyGraphGCN(in_channels=x.size(1), hidden=8, out_channels=2)
        model.eval()

        explainer = Zorro(
            model=model,
            task="graph",
            select="features_only",
            samples=15,
            top_k=3,
            seed=42,
        )
        result = explainer.explain(
            x=x, edge_index=edge_index, fidelity_threshold=0.3, max_explanations=3,
        )

        # At least one extra explanation should be enumerated for this
        # assertion to be non-vacuous.
        assert len(result) >= 2, (
            f"expected at least 2 explanations to test disjointness, got {len(result)}"
        )

        for i in range(len(result)):
            for j in range(i + 1, len(result)):
                overlap = (result[i].feature_mask & result[j].feature_mask).any().item()
                assert not overlap, (
                    f"explanations {i} and {j} share features: "
                    f"{result[i].selected_feature_indices().tolist()} vs "
                    f"{result[j].selected_feature_indices().tolist()}"
                )


# == CONFIGURATION + VALIDATION ==


class TestZorroConfig:
    """Tests for argument validation, per-task defaults, and select-mode behavior."""

    def test_explain_rejects_node_idx_for_graph_task(self):
        """``Zorro(task='node').explain()`` rejects ``node_idx=None``."""
        x, edge_index = make_random_node_graph(seed=0)
        model = TinyGraphGCN(in_channels=x.size(1), hidden=8, out_channels=3)
        explainer = Zorro(model=model, task="node", samples=5)
        with pytest.raises(ValueError, match="node_idx is required"):
            explainer.explain(x=x, edge_index=edge_index, node_idx=None)

    def test_direction_validation(self):
        """``direction='up'`` / ``'down'`` is only allowed with regression."""
        x, edge_index = make_random_node_graph(seed=0)
        model = TinyGraphGCN(in_channels=x.size(1), hidden=8, out_channels=2)
        with pytest.raises(ValueError, match="direction is only meaningful"):
            Zorro(
                model=model, task="graph", objective="classification",
                direction="up", samples=5,
            )

    def test_graph_task_defaults_to_row_noise(self):
        """Graph tasks default to row-mode sampling (categorical features stay valid)."""
        model = TinyGraphGCN(in_channels=4, hidden=4, out_channels=2)
        explainer = Zorro(model=model, task="graph", samples=5)
        assert explainer.noise_mode == "row"

    def test_node_task_defaults_to_column_noise(self):
        """Node tasks default to column-mode sampling (paper-faithful)."""
        model = TinyGCN(in_channels=4, hidden=4, out_channels=2)
        explainer = Zorro(model=model, task="node", samples=5)
        assert explainer.noise_mode == "column"

    def test_custom_noise_pool_is_used(self):
        """
        If ``noise_pool`` is given to ``__init__``, the pool's values must
        actually flow into the model during fidelity evaluation. Verified
        with a sentinel value that ``x`` cannot produce, so any forward
        pass containing it proves the pool (not ``x``) was the noise source.
        """
        torch.manual_seed(0)
        x, edge_index = make_random_node_graph(seed=0)

        sentinel = 999.0
        custom_pool = torch.full((50, x.size(1)), sentinel)

        captured: list[torch.Tensor] = []

        class CapturingModel(nn.Module):
            def __init__(self, in_channels: int) -> None:
                super().__init__()
                self.lin = nn.Linear(in_channels, 2)

            def forward(self, x_in, edge_index, batch):
                captured.append(x_in.detach().clone())
                return global_mean_pool(self.lin(x_in), batch)

        model = CapturingModel(in_channels=x.size(1))
        model.eval()

        explainer = Zorro(
            model=model,
            task="graph",
            noise_pool=custom_pool,
            noise_mode="row",
            select="features_only",  # cells in unselected feature columns get noise
            samples=10,
            top_k=2,
            seed=1,
        )
        explainer.explain(x=x, edge_index=edge_index, fidelity_threshold=0.5)

        assert captured, "model was never called"
        saw_sentinel = any((inp == sentinel).any().item() for inp in captured)
        assert saw_sentinel, "custom noise_pool sentinel never reached the model"

    def test_select_nodes_only_freezes_feature_mask(self):
        """``select='nodes_only'`` leaves the feature axis fully kept."""
        torch.manual_seed(0)
        x, edge_index = make_random_node_graph(seed=0)
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
        expl = result[0]
        # Frozen axis must be all-True (kept).
        assert expl.feature_mask.all().item() is True
        # Searched axis must not be trivially all-True (otherwise
        # ``select='nodes_only'`` could be silently implemented as
        # "everything True" and pass).
        assert not expl.node_mask.all().item()

    def test_select_features_only_freezes_node_mask(self):
        """``select='features_only'`` leaves the node axis fully kept."""
        torch.manual_seed(0)
        x, edge_index = make_random_node_graph(seed=0)
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
        expl = result[0]
        # Frozen axis must be all-True (kept).
        assert expl.node_mask.all().item() is True
        # Searched axis must not be trivially all-True.
        assert not expl.feature_mask.all().item()


# == DETERMINISM ==


class TestZorroDeterminism:
    """Tests for the ``seed`` reproducibility contract."""

    def test_explain_deterministic_with_seed(self):
        """
        Same seed -> identical explanation (masks, fidelity, trace).

        Guards against a refactor that introduces hidden randomness outside
        the seeded generator (e.g. an accidental ``torch.randint`` without
        ``generator=``). Note: a global ``torch.manual_seed`` is set first
        so we can also detect the "seed is silently ignored" failure mode
        -- if seed had no effect, the second run would consume different
        global-RNG state than the first and masks would diverge.
        """
        torch.manual_seed(0)
        x, edge_index = make_random_node_graph(seed=0)
        model = TinyGraphGCN(in_channels=x.size(1), hidden=8, out_channels=2)
        model.eval()

        def run(seed: int):
            explainer = Zorro(
                model=model, task="graph",
                samples=15, top_k=3, seed=seed,
            )
            return explainer.explain(x=x, edge_index=edge_index, fidelity_threshold=0.5)[0]

        a = run(42)
        b = run(42)

        assert torch.equal(a.node_mask, b.node_mask)
        assert torch.equal(a.feature_mask, b.feature_mask)
        assert a.fidelity == b.fidelity
        assert a.trace == b.trace


# == PLANTED-SIGNAL RECOVERY ==


class TestZorroPlantedSignal:
    """
    Behavioural tests where the data carries a known signal. These guarantee
    that the explainer actually does something useful, not just that it
    runs.
    """

    def test_finds_planted_feature_signal(self):
        """
        The single most important end-to-end claim: when only one feature
        actually drives the prediction, the greedy explainer must find it.
        """
        n_nodes, n_feats = 8, 5
        k = 2

        x = torch.zeros(n_nodes, n_feats)
        x[:, k] = 1.0
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

        # Noise pool whose column k is always -1.0; replacing the planted
        # column with noise from this pool flips the prediction.
        pool = torch.randn(40, n_feats)
        pool[:, k] = -1.0

        model = SingleFeatureModel(k=k, n_feats=n_feats)
        model.eval()

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
        result = explainer.explain(x=x, edge_index=edge_index, fidelity_threshold=0.9)
        expl = result[0]

        selected = expl.selected_feature_indices().tolist()
        assert k in selected, (
            f"planted feature {k} not selected; got {selected}"
        )
        assert expl.fidelity >= 0.9

    def test_fidelity_threshold_follows_paper_tau(self):
        """
        ``fidelity_threshold`` is the paper's tau directly (NOT ``1 - tau``).

        Asymmetric setup: empty-selection fidelity sits around 0.5.

        - High threshold (0.9) must drive greedy to select the planted feature.
        - Low threshold (0.3) is satisfied by the empty mask and returns immediately.

        If a refactor inverts the comparison
        (``while current_fidelity < 1 - tau``), these two cases swap and the
        test breaks.
        """
        n_nodes, n_feats = 8, 4
        k = 1

        x = torch.zeros(n_nodes, n_feats)
        x[:, k] = 1.0
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

        # Symmetric noise on column k -> empty-selection fidelity ~ 0.5.
        pool = torch.randn(200, n_feats)
        pool[:, k] = torch.randn(200)  # zero mean, unit variance

        model = SingleFeatureModel(k=k, n_feats=n_feats)
        model.eval()

        def run(threshold: float):
            explainer = Zorro(
                model=model,
                task="graph",
                select="features_only",
                noise_pool=pool,
                noise_mode="column",
                samples=200,  # tight binomial std -> stable threshold comparisons
                top_k=3,
                seed=7,
            )
            return explainer.explain(
                x=x, edge_index=edge_index, fidelity_threshold=threshold,
            )[0]

        high = run(0.9)
        low = run(0.3)

        # High threshold: greedy ran and selected the planted feature.
        assert high.feature_mask[k].item() is True
        assert high.fidelity >= 0.9

        # Low threshold: empty-selection fidelity already satisfied it; no selection.
        assert low.feature_mask.sum().item() == 0
        assert low.fidelity >= 0.3
