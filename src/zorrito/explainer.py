from __future__ import annotations

from typing import Callable, Literal

import torch
from tqdm import tqdm

from zorrito._subgraph import extract_computational_subgraph
from zorrito.fidelity import (
    Direction,
    Explanation,
    Objective,
    estimate_fidelity,
    make_match_fn,
)
from zorrito.noise import NoiseMode, build_sampler


Task = Literal["node", "graph"]
SelectMode = Literal["both", "nodes_only", "features_only"]


class Zorro:
    """Zorrito: a unified Zorro explainer for PyG node- and graph-level GNNs.

    Same greedy core as the original paper (Funke et al., 2021), reorganized
    so that `task='node'` reproduces the original behavior and `task='graph'`
    explains whole-graph predictions. Classification and tolerance-band
    regression are both supported via `objective`.

    The `select` argument controls which axes the greedy search explores:

    - ``"both"``: search over both nodes and features (the paper's behavior)
    - ``"nodes_only"``: only select nodes; all feature columns are treated as
      kept. Useful when feature attribution is uninformative or you only care
      about which neighbors mattered.
    - ``"features_only"``: only select features; all nodes are treated as kept.

    Masks in the returned :class:`Explanation` always have the full shape;
    when an axis is not searched, that mask is all-True (i.e. "kept").
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
            raise ValueError(f"objective must be 'classification' or 'regression', got {objective!r}")
        if select not in ("both", "nodes_only", "features_only"):
            raise ValueError(f"select must be 'both', 'nodes_only', or 'features_only', got {select!r}")
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

        # Defaults per task: column for node (paper-faithful, continuous TF-IDF
        # works well), row for graph (categorical one-hot atom features get
        # chimeric under column-independent sampling).
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

    # ----------------------------------------------------------------- explain
    def explain(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        node_idx: int | None = None,
        batch: torch.Tensor | None = None,
        fidelity_threshold: float = 0.85,
        max_explanations: int = 1,
    ) -> list[Explanation]:
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
            # exclude what was used for disjoint enumeration — only on the axes
            # we actually searched (frozen axes stay all-True)
            if search_nodes:
                available_nodes &= ~expl.node_mask.to(self.device)
            if search_features:
                available_feats &= ~expl.feature_mask.to(self.device)
            if expl.fidelity < fidelity_threshold:
                # search exhausted before reaching threshold; don't pile on more
                break

        return explanations

    # --------------------------------------------------------------- internals
    def _build_node_forward(
        self,
        edge_index_sub: torch.Tensor,
        target_local: int,
    ) -> Callable[[torch.Tensor], torch.Tensor]:
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
        search_nodes = self.select != "features_only"
        search_features = self.select != "nodes_only"

        # axes that are NOT searched start fully kept (all-True), so the
        # masking has no effect on those axes during fidelity evaluation
        if search_nodes:
            node_sel = torch.zeros(n_nodes, dtype=torch.bool, device=self.device)
        else:
            node_sel = torch.ones(n_nodes, dtype=torch.bool, device=self.device)
        if search_features:
            feat_sel = torch.zeros(n_feats, dtype=torch.bool, device=self.device)
        else:
            feat_sel = torch.ones(n_feats, dtype=torch.bool, device=self.device)
        trace: list[tuple[str, int, float]] = []

        # 1. initial ranking — score each available element of the searched
        # axes against everything-else-kept (the original-paper baseline)
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

        # 2. greedy loop
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

            best_kind: str | None = None
            best_idx: int = -1
            best_fid: float = current_fidelity

            for v in node_candidates:
                trial = node_sel.clone()
                trial[v] = True
                f = fidelity_of(trial, feat_sel)
                if f > best_fid:
                    best_fid, best_idx, best_kind = f, int(v), "node"

            for f_idx in feat_candidates:
                trial = feat_sel.clone()
                trial[f_idx] = True
                fv = fidelity_of(node_sel, trial)
                if fv > best_fid:
                    best_fid, best_idx, best_kind = fv, int(f_idx), "feature"

            if best_kind is None:
                # no candidate improves things — stop early
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
        """Rank each available element by its solo contribution to fidelity.

        For nodes: fidelity({v}, all_feats). For features: fidelity(all_nodes, {f}).
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
        masked = scores.clone()
        masked[~available] = float("-inf")
        k = min(self.top_k, int(available.sum().item()))
        if k == 0:
            return []
        return torch.topk(masked, k=k).indices.tolist()
