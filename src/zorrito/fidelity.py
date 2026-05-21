from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

import torch


Objective = Literal["classification", "regression"]
Direction = Literal["both", "up", "down"]


@dataclass
class Explanation:
    node_mask: torch.Tensor
    feature_mask: torch.Tensor
    fidelity: float
    trace: list[tuple[str, int, float]] = field(default_factory=list)
    subgraph_nodes: torch.Tensor | None = None

    def selected_node_indices(self) -> torch.Tensor:
        return self.node_mask.nonzero(as_tuple=False).flatten()

    def selected_feature_indices(self) -> torch.Tensor:
        return self.feature_mask.nonzero(as_tuple=False).flatten()


def make_match_fn(
    objective: Objective,
    reference_output: torch.Tensor,
    tolerance: float,
    direction: Direction = "both",
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build a per-sample match predicate from the reference forward-pass output.

    For classification, the reference is the argmax class index (a scalar for
    node tasks, a vector of length batch_size for graph tasks). For regression,
    the reference is the raw output and `direction` controls which side(s) of
    the tolerance band counts as a match:

    - "both"  — symmetric band:  |pred − ref| ≤ tolerance
    - "up"    — no upward drift: pred ≤ ref + tolerance
                (selects atoms preventing the prediction from *rising*)
    - "down"  — no downward drift: pred ≥ ref − tolerance
                (selects atoms preventing the prediction from *falling*)
    """
    if objective == "classification":
        if direction != "both":
            raise ValueError(
                "direction is only meaningful for objective='regression'; "
                f"got direction={direction!r} for classification"
            )
        ref_class = reference_output.argmax(dim=-1)

        def match(out: torch.Tensor) -> torch.Tensor:
            pred = out.argmax(dim=-1)
            return (pred == ref_class).all().reshape(())

        return match

    if objective == "regression":
        ref = reference_output.detach()

        if direction == "both":
            def match(out: torch.Tensor) -> torch.Tensor:
                return ((out - ref).abs() <= tolerance).all().reshape(())
        elif direction == "up":
            def match(out: torch.Tensor) -> torch.Tensor:
                return ((out - ref) <= tolerance).all().reshape(())
        elif direction == "down":
            def match(out: torch.Tensor) -> torch.Tensor:
                return ((out - ref) >= -tolerance).all().reshape(())
        else:
            raise ValueError(f"unknown direction: {direction!r}")

        return match

    raise ValueError(f"unknown objective: {objective!r}")


def estimate_fidelity(
    forward_fn: Callable[[torch.Tensor], torch.Tensor],
    x_template: torch.Tensor,
    node_selected: torch.Tensor,
    feature_selected: torch.Tensor,
    noise_sampler: Callable[[int], torch.Tensor],
    match_fn: Callable[[torch.Tensor], torch.Tensor],
    samples: int,
) -> float:
    """Monte Carlo estimate of RDT-Fidelity for a given (V_s, F_s).

    For each of `samples` trials, start from `x_template` (the original feature
    matrix restricted to the relevant nodes), then overwrite every (node, feature)
    cell NOT in (node_selected, feature_selected) with a fresh random draw from
    `noise_sampler`. Run `forward_fn` on the perturbed input and count matches.
    """
    n_nodes, n_feats = x_template.shape
    keep_mask = node_selected.view(-1, 1) & feature_selected.view(1, -1)

    correct = 0
    for _ in range(samples):
        noise = noise_sampler(n_nodes)
        y = torch.where(keep_mask, x_template, noise)
        out = forward_fn(y)
        if bool(match_fn(out).item()):
            correct += 1
    return correct / samples
