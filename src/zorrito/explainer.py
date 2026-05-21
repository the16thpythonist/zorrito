"""
The :class:`Zorro` greedy explainer.

This module owns the orchestration half of zorrito: it wraps the model so
the fidelity Monte Carlo only sees a perturbed feature matrix, it drives
the disjoint outer loop that produces multiple non-overlapping
explanations, and it runs the greedy inner loop that builds each
explanation node-by-node and feature-by-feature.

The match-condition machinery and the Monte Carlo estimator live in
:mod:`zorrito.fidelity`. The empirical noise samplers live in
:mod:`zorrito.noise`. The subgraph extraction used for node tasks lives
in :mod:`zorrito._subgraph`. This module ties them together; it contains
no math of its own beyond bookkeeping.
"""
from __future__ import annotations

from typing import Callable

import torch
from tqdm import tqdm

from zorrito._subgraph import extract_computational_subgraph
from zorrito._typing import Direction, NoiseMode, Objective, SelectMode, Task
from zorrito.fidelity import (
    Explanation,
    estimate_fidelity,
    make_match_fn,
)
from zorrito.noise import build_sampler


# == ZORRO EXPLAINER ==


class Zorro:
    """
    A unified Zorro explainer for PyG node- and graph-level GNNs.

    Same greedy core as the original paper (Funke et al., 2021),
    reorganized so that ``task="node"`` reproduces the original
    node-classification behavior and ``task="graph"`` explains
    whole-graph predictions. Both classification and tolerance-band
    regression are supported via ``objective``.

    **TASKS**

    Two task modes are supported via the ``task`` argument:

    - ``"node"``: the explained quantity is the model's prediction for a
      single target node, identified by ``node_idx`` at
      :meth:`explain` time. The L-hop computational subgraph is
      extracted and the greedy search runs over the subgraph.
    - ``"graph"``: the explained quantity is the model's prediction for
      the whole input graph. The greedy search runs over every node in
      the graph.

    **OBJECTIVES**

    Two objectives are supported via the ``objective`` argument:

    - ``"classification"``: a sample matches when its argmax class
      equals the reference's argmax class.
    - ``"regression"``: a sample matches when its scalar output lies
      within a tolerance band of the reference; the band's shape is
      controlled by ``direction`` (symmetric / one-sided-up / one-sided-down).

    **SELECT MODES**

    The ``select`` argument controls which axes the greedy search
    explores:

    - ``"both"``: search over both nodes and features (the paper's behavior).
    - ``"nodes_only"``: only select nodes; all feature columns are
      treated as kept. Useful when feature attribution is uninformative
      or you only care about which neighbors mattered.
    - ``"features_only"``: only select features; all nodes are treated
      as kept.

    Masks in the returned :class:`Explanation` always have the full
    shape; when an axis is not searched, that mask is all-True
    (i.e. "kept").

    **NOISE SAMPLING**

    Cells outside the current selection are replaced with draws from an
    empirical pool — see :mod:`zorrito.noise`. The default pool is the
    input graph's own feature matrix; pass ``noise_pool`` to supply a
    larger dataset-wide pool. ``noise_mode`` selects between
    paper-faithful per-column independent draws (good for continuous
    features) and whole-row draws (recommended for categorical /
    one-hot features). The default is per-task: ``"column"`` for node
    tasks, ``"row"`` for graph tasks.

    **OUTPUT**

    :meth:`explain` returns a list of :class:`Explanation` instances —
    one per disjoint search run, up to ``max_explanations`` and bounded
    by the size of the searched axes.

    V - number of nodes (subgraph for node tasks, full graph for graph tasks)
    K - feature dimensionality
    E - number of edges

    :param model: The PyG-compatible model to explain. For ``task="node"``
        the model must implement ``forward(x, edge_index)``; for
        ``task="graph"`` it must implement ``forward(x, edge_index, batch)``.
    :param task: ``"node"`` or ``"graph"``; see TASKS section above.
    :param objective: ``"classification"`` or ``"regression"``; see
        OBJECTIVES section.
    :param select: ``"both"``, ``"nodes_only"``, or ``"features_only"``; see
        SELECT MODES section.
    :param noise_pool: Optional (P, K) feature matrix used as the empirical
        pool. Defaults to the input graph's own ``x``.
    :param noise_mode: ``"column"`` or ``"row"``; defaults are task-specific.
    :param device: torch device for all internal tensors. Inputs to
        :meth:`explain` are moved to this device.
    :param samples: Number of Monte Carlo samples per fidelity evaluation.
    :param top_k: Per-iteration candidate budget for the greedy loop.
    :param tolerance: Half-width of the regression tolerance band. Ignored
        for classification.
    :param direction: For regression, which side(s) of the tolerance band
        count as a match. Must be ``"both"`` for classification.
    :param num_hops: GNN depth ``L`` — radius of the extracted computational
        subgraph for node tasks. Must match the model's layer count.
    :param seed: Optional integer seed; when provided, the internal
        torch.Generator used by the noise sampler is seeded with it.
    :param log: Whether to show a tqdm progress bar during the greedy loop.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        task: Task = "node",
        objective: Objective = "classification",
        select: SelectMode = "both",
        noise_pool: torch.Tensor | None = None,
        noise_mode: NoiseMode | None = None,
        device: torch.device | str = "cpu",
        samples: int = 100,
        top_k: int = 10,
        tolerance: float = 0.0,
        direction: Direction = "both",
        num_hops: int = 2,
        seed: int | None = None,
        log: bool = False,
    ) -> None:
        if task not in ("node", "graph"):
            raise ValueError(f"task must be 'node' or 'graph', got {task!r}")
        if objective not in ("classification", "regression"):
            raise ValueError(
                f"objective must be 'classification' or 'regression', got {objective!r}"
            )
        if select not in ("both", "nodes_only", "features_only"):
            raise ValueError(
                f"select must be 'both', 'nodes_only', or 'features_only', got {select!r}"
            )
        if objective == "regression" and tolerance < 0:
            raise ValueError(f"tolerance must be >= 0 for regression, got {tolerance}")
        if direction not in ("both", "up", "down"):
            raise ValueError(f"direction must be 'both', 'up', or 'down', got {direction!r}")
        if objective == "classification" and direction != "both":
            raise ValueError(
                f"direction is only meaningful for objective='regression', got direction={direction!r}"
            )
        if noise_mode is not None and noise_mode not in ("column", "row"):
            raise ValueError(f"noise_mode must be 'column' or 'row', got {noise_mode!r}")

        # Defaults per task: column for node (paper-faithful, continuous
        # features work well), row for graph (categorical one-hot atom
        # features get chimeric under column-independent sampling).
        if noise_mode is None:
            noise_mode = "column" if task == "node" else "row"

        self.model = model
        self.task = task
        self.objective: Objective = objective
        self.select: SelectMode = select
        self.noise_pool = noise_pool
        self.noise_mode: NoiseMode = noise_mode
        self.device = torch.device(device)
        self.samples = samples
        self.top_k = top_k
        self.tolerance = tolerance
        self.direction: Direction = direction
        self.num_hops = num_hops
        self.log = log

        self._generator: torch.Generator | None
        if seed is None:
            self._generator = None
        else:
            self._generator = torch.Generator(device=self.device).manual_seed(seed)

    # == EXPLAIN ==

    def explain(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        node_idx: int | None = None,
        batch: torch.Tensor | None = None,
        fidelity_threshold: float = 0.85,
        max_explanations: int = 1,
    ) -> list[Explanation]:
        """
        Run the greedy explainer against a single input graph.

        For ``task="node"`` the L-hop computational subgraph is extracted
        around ``node_idx`` and the search runs over the subgraph. For
        ``task="graph"`` the search runs over the full graph; ``batch``
        may be supplied to signal a multi-graph batch tensor as PyG does,
        or left as ``None`` for a single-graph forward pass where an
        all-zeros batch is supplied internally.

        Up to ``max_explanations`` disjoint explanations are produced.
        After each greedy run, the elements that were selected are
        removed from the available pool on the searched axes so the next
        run is forced to find a different sufficient subset.

        :param x: ``(V, K)`` node feature matrix of the original graph.
        :param edge_index: ``(2, E)`` edge index of the original graph.
        :param node_idx: Required for ``task="node"`` — the index of the
            target node in the original graph. Ignored for ``task="graph"``.
        :param batch: Optional ``(V,)`` long tensor for ``task="graph"``;
            the standard PyG batch assignment. ``None`` is fine for a
            single graph.
        :param fidelity_threshold: The paper's ``tau`` directly — fidelity
            must reach this value for the greedy to stop. In ``(0, 1]``.
        :param max_explanations: Upper bound on the number of disjoint
            explanations returned. The actual count may be smaller if the
            search exhausts its available axes earlier.

        :returns: A list of :class:`Explanation` instances.
        """
        if not 0.0 < fidelity_threshold <= 1.0:
            raise ValueError(
                f"fidelity_threshold must be in (0, 1], got {fidelity_threshold}"
            )
        if max_explanations < 1:
            raise ValueError(f"max_explanations must be >= 1, got {max_explanations}")
        if self.task == "node" and node_idx is None:
            raise ValueError("node_idx is required for task='node'")

        self.model.eval()
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        if batch is not None:
            batch = batch.to(self.device)

        # Task-specific setup: for node tasks we restrict to the L-hop
        # computational subgraph and remember the local-to-original index
        # map; for graph tasks we work on the whole input directly.
        if self.task == "node":
            assert node_idx is not None
            x_sub, edge_index_sub, target_local, subset = extract_computational_subgraph(
                node_idx=node_idx,
                num_hops=self.num_hops,
                x=x,
                edge_index=edge_index,
            )
            forward_fn = self._build_node_forward(edge_index_sub, target_local)
            x_active = x_sub
            subgraph_nodes: torch.Tensor | None = subset.to(self.device)
        else:
            forward_fn = self._build_graph_forward(edge_index, batch)
            x_active = x
            subgraph_nodes = None

        # Capture the unperturbed forward output once and build the match
        # predicate against it. Every subsequent forward pass during the
        # Monte Carlo is compared to this reference.
        with torch.no_grad():
            reference_out = forward_fn(x_active)
        match_fn = make_match_fn(self.objective, reference_out, self.tolerance, self.direction)
        pool = self.noise_pool.to(self.device) if self.noise_pool is not None else x
        sampler = build_sampler(pool, mode=self.noise_mode, generator=self._generator)

        def fidelity_of(node_sel: torch.Tensor, feat_sel: torch.Tensor) -> float:
            with torch.no_grad():
                return estimate_fidelity(
                    forward_fn=forward_fn,
                    x_template=x_active,
                    node_selected=node_sel,
                    feature_selected=feat_sel,
                    noise_sampler=sampler,
                    match_fn=match_fn,
                    samples=self.samples,
                )

        n_nodes, n_feats = x_active.shape
        # The "available" masks track which positions are still candidates
        # across disjoint explanations — selected positions are flipped to
        # False before the next outer-loop iteration.
        available_nodes = torch.ones(n_nodes, dtype=torch.bool, device=self.device)
        available_feats = torch.ones(n_feats, dtype=torch.bool, device=self.device)

        search_nodes = self.select != "features_only"
        search_features = self.select != "nodes_only"

        explanations: list[Explanation] = []
        for _ in range(max_explanations):
            if search_nodes and not available_nodes.any():
                break
            if search_features and not available_feats.any():
                break
            expl = self._greedy_one(
                fidelity_of=fidelity_of,
                available_nodes=available_nodes.clone(),
                available_feats=available_feats.clone(),
                n_nodes=n_nodes,
                n_feats=n_feats,
                fidelity_threshold=fidelity_threshold,
                subgraph_nodes=subgraph_nodes,
            )
            explanations.append(expl)
            # Exclude what was used for disjoint enumeration — only on the
            # axes we actually searched (frozen axes stay all-True).
            if search_nodes:
                available_nodes &= ~expl.node_mask.to(self.device)
            if search_features:
                available_feats &= ~expl.feature_mask.to(self.device)
            if expl.fidelity < fidelity_threshold:
                # Search exhausted before reaching the threshold — don't
                # pile on more half-baked explanations.
                break

        return explanations

    # == INTERNALS ==

    def _build_node_forward(
        self,
        edge_index_sub: torch.Tensor,
        target_local: int,
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        """
        Build a closure that returns the model's output for the target
        node only.

        The closure captures the subgraph edge index and the target's
        local index, so the fidelity loop only needs to pass in the
        perturbed feature matrix.

        :param edge_index_sub: ``(2, E_sub)`` edge index relabeled to local
            subgraph indices.
        :param target_local: Local index of the target node inside the
            subgraph.

        :returns: A closure ``(x: Tensor) -> Tensor`` of shape ``(1, C)``.
        """
        model = self.model

        def forward_fn(x_in: torch.Tensor) -> torch.Tensor:
            out = model(x_in, edge_index_sub)
            return out[target_local].unsqueeze(0)

        return forward_fn

    def _build_graph_forward(
        self,
        edge_index: torch.Tensor,
        batch: torch.Tensor | None,
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        """
        Build a closure that returns the model's output for a graph-level
        prediction.

        Handles the common single-graph case by supplying an all-zeros
        batch vector internally when the caller did not provide one.

        :param edge_index: ``(2, E)`` edge index.
        :param batch: Optional ``(V,)`` batch assignment.

        :returns: A closure ``(x: Tensor) -> Tensor`` of shape ``(B, C)``.
        """
        model = self.model

        if batch is None:
            def forward_fn(x_in: torch.Tensor) -> torch.Tensor:
                zeros = torch.zeros(x_in.size(0), dtype=torch.long, device=x_in.device)
                return model(x_in, edge_index, zeros)
        else:
            def forward_fn(x_in: torch.Tensor) -> torch.Tensor:
                return model(x_in, edge_index, batch)

        return forward_fn

    def _greedy_one(
        self,
        fidelity_of: Callable[[torch.Tensor, torch.Tensor], float],
        available_nodes: torch.Tensor,
        available_feats: torch.Tensor,
        n_nodes: int,
        n_feats: int,
        fidelity_threshold: float,
        subgraph_nodes: torch.Tensor | None,
    ) -> Explanation:
        """
        Run one full greedy search and return the resulting explanation.

        The search runs in two phases. The initial-ranking phase scores
        every still-available element of each searched axis on its solo
        contribution to fidelity (everything else kept). The greedy phase
        then iteratively adds the single top-K-candidate that produces
        the largest fidelity gain, stopping when the threshold is reached
        or no candidate improves things.

        :param fidelity_of: A bound closure that scores a given
            ``(node_sel, feat_sel)`` pair via the Monte Carlo estimator.
        :param available_nodes: ``(V,)`` bool — positions still eligible
            for selection on the node axis.
        :param available_feats: ``(K,)`` bool — positions still eligible
            for selection on the feature axis.
        :param n_nodes: ``V``, the node-axis length.
        :param n_feats: ``K``, the feature-axis length.
        :param fidelity_threshold: The paper's ``tau`` — stop when fidelity
            reaches it.
        :param subgraph_nodes: Optional original-graph index map to attach
            to the returned :class:`Explanation`.

        :returns: A single :class:`Explanation`.
        """
        search_nodes = self.select != "features_only"
        search_features = self.select != "nodes_only"

        # Axes that are NOT searched start fully kept (all-True), so the
        # masking has no effect on those axes during fidelity evaluation.
        if search_nodes:
            node_sel = torch.zeros(n_nodes, dtype=torch.bool, device=self.device)
        else:
            node_sel = torch.ones(n_nodes, dtype=torch.bool, device=self.device)
        if search_features:
            feat_sel = torch.zeros(n_feats, dtype=torch.bool, device=self.device)
        else:
            feat_sel = torch.ones(n_feats, dtype=torch.bool, device=self.device)
        trace: list[tuple[str, int, float]] = []

        # -- 1. initial ranking ----------------------------------------------
        # Score each available element of the searched axes against
        # everything-else-kept (the original-paper baseline).
        all_feats = torch.ones(n_feats, dtype=torch.bool, device=self.device)
        all_nodes = torch.ones(n_nodes, dtype=torch.bool, device=self.device)

        node_scores = (
            self._initial_ranking(
                fidelity_of=fidelity_of,
                available=available_nodes,
                kind="node",
                fixed_feats=all_feats,
                fixed_nodes=all_nodes,
                n_nodes=n_nodes,
                n_feats=n_feats,
            )
            if search_nodes
            else None
        )
        feat_scores = (
            self._initial_ranking(
                fidelity_of=fidelity_of,
                available=available_feats,
                kind="feature",
                fixed_feats=all_feats,
                fixed_nodes=all_nodes,
                n_nodes=n_nodes,
                n_feats=n_feats,
            )
            if search_features
            else None
        )

        # -- 2. greedy loop --------------------------------------------------
        current_fidelity = fidelity_of(node_sel, feat_sel)
        trace.append(("init", -1, current_fidelity))

        pbar = tqdm(total=int(fidelity_threshold * 100), disable=not self.log, desc="greedy")
        while current_fidelity < fidelity_threshold:
            node_candidates = (
                self._top_k_indices(scores=node_scores, available=available_nodes & ~node_sel)
                if search_nodes
                else []
            )
            feat_candidates = (
                self._top_k_indices(scores=feat_scores, available=available_feats & ~feat_sel)
                if search_features
                else []
            )

            # Track the best single-element addition seen so far this
            # iteration, across both axes. Ties don't promote; only strict
            # improvements over ``current_fidelity`` count.
            best_kind: str | None = None
            best_idx: int = -1
            best_fid: float = current_fidelity

            for node_cand in node_candidates:
                trial = node_sel.clone()
                trial[node_cand] = True
                node_fid = fidelity_of(trial, feat_sel)
                if node_fid > best_fid:
                    best_fid, best_idx, best_kind = node_fid, int(node_cand), "node"

            for feat_cand in feat_candidates:
                trial = feat_sel.clone()
                trial[feat_cand] = True
                feat_fid = fidelity_of(node_sel, trial)
                if feat_fid > best_fid:
                    best_fid, best_idx, best_kind = feat_fid, int(feat_cand), "feature"

            if best_kind is None:
                # No candidate improved the fidelity — the greedy is stuck
                # and we stop early. This is the common "regression with a
                # tolerance band already satisfied at init" case.
                break

            if best_kind == "node":
                node_sel[best_idx] = True
            else:
                feat_sel[best_idx] = True
            trace.append((best_kind, best_idx, best_fid))

            if self.log:
                pbar.update(max(0, int((best_fid - current_fidelity) * 100)))
            current_fidelity = best_fid
        pbar.close()

        return Explanation(
            node_mask=node_sel.detach().cpu(),
            feature_mask=feat_sel.detach().cpu(),
            fidelity=current_fidelity,
            trace=trace,
            subgraph_nodes=subgraph_nodes.detach().cpu() if subgraph_nodes is not None else None,
        )

    def _initial_ranking(
        self,
        fidelity_of: Callable[[torch.Tensor, torch.Tensor], float],
        available: torch.Tensor,
        kind: str,
        fixed_feats: torch.Tensor,
        fixed_nodes: torch.Tensor,
        n_nodes: int,
        n_feats: int,
    ) -> torch.Tensor:
        """
        Rank each still-available element by its solo contribution to fidelity.

        For nodes: ``fidelity({v}, all_feats)``. For features:
        ``fidelity(all_nodes, {f})``. Unavailable positions get ``-inf``
        so they cannot be selected by a subsequent top-K.

        :param fidelity_of: The bound fidelity Monte Carlo closure.
        :param available: Bool mask over the relevant axis — only ``True``
            positions are scored.
        :param kind: Either ``"node"`` or ``"feature"`` — selects which
            axis is being ranked.
        :param fixed_feats: ``(K,)`` mask used as the held-fixed axis when
            ``kind="node"``.
        :param fixed_nodes: ``(V,)`` mask used as the held-fixed axis when
            ``kind="feature"``.
        :param n_nodes: ``V``, node-axis length.
        :param n_feats: ``K``, feature-axis length.

        :returns: A 1D score tensor over the ranked axis.
        """
        scores = torch.full(
            (n_nodes if kind == "node" else n_feats,),
            float("-inf"),
            device=self.device,
        )
        idxs = available.nonzero(as_tuple=False).flatten().tolist()
        for i in idxs:
            if kind == "node":
                node_sel = torch.zeros(n_nodes, dtype=torch.bool, device=self.device)
                node_sel[i] = True
                scores[i] = fidelity_of(node_sel, fixed_feats)
            else:
                feat_sel = torch.zeros(n_feats, dtype=torch.bool, device=self.device)
                feat_sel[i] = True
                scores[i] = fidelity_of(fixed_nodes, feat_sel)
        return scores

    def _top_k_indices(
        self,
        scores: torch.Tensor,
        available: torch.Tensor,
    ) -> list[int]:
        """
        Return the ``top_k`` indices from ``scores`` restricted to
        ``available`` positions.

        Positions outside the available set are masked to ``-inf`` so they
        cannot be returned. The result has at most ``top_k`` entries; it
        may be shorter when fewer positions are available.

        :param scores: 1D score tensor over the relevant axis.
        :param available: Bool mask of the same shape.

        :returns: A Python list of int indices.
        """
        masked = scores.clone()
        masked[~available] = float("-inf")
        k = min(self.top_k, int(available.sum().item()))
        if k == 0:
            return []
        return torch.topk(masked, k=k).indices.tolist()
