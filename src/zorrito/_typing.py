"""
Type aliases used throughout the zorrito package.

These are kept in a single underscore-prefixed module so that they can be
imported by every other module (``explainer.py``, ``fidelity.py``,
``noise.py``) without circular-import risk and without being shadowed by
the stdlib ``typing`` module on dotted imports.
"""
from __future__ import annotations

from typing import Literal


"""
Whether the explainer operates on a node-level prediction or a graph-level
prediction.

- ``"node"``: the explained quantity is the model's prediction for a single
  target node within a graph. The L-hop computational neighborhood of that
  node is extracted and the greedy search runs over the subgraph.
- ``"graph"``: the explained quantity is the model's prediction for an
  entire graph. The greedy search runs over all nodes in the input graph.
"""
Task = Literal["node", "graph"]


"""
Which axes the greedy search is allowed to select on.

- ``"both"``: search over both node and feature axes (the paper's default).
- ``"nodes_only"``: search only over nodes; all feature columns are treated
  as kept and the returned ``feature_mask`` is all-True.
- ``"features_only"``: search only over features; all nodes are treated as
  kept and the returned ``node_mask`` is all-True.
"""
SelectMode = Literal["both", "nodes_only", "features_only"]


"""
The kind of prediction task being explained — drives the match condition
used to score perturbed inputs in the fidelity Monte Carlo.

- ``"classification"``: the match condition is ``argmax(pred) == argmax(ref)``.
- ``"regression"``: the match condition is a tolerance band around the
  reference output; see :data:`Direction` for the symmetric / one-sided
  variants.
"""
Objective = Literal["classification", "regression"]


"""
For regression objectives, which side(s) of the tolerance band count as a
match.

- ``"both"``: symmetric band, ``|pred - ref| <= tolerance``.
- ``"up"``: one-sided, ``pred <= ref + tolerance`` (selects atoms that
  prevent the prediction from drifting *upward*).
- ``"down"``: one-sided, ``pred >= ref - tolerance`` (selects atoms that
  prevent the prediction from drifting *downward*).

For classification objectives, ``"both"`` is the only valid value; the
others raise ``ValueError``.
"""
Direction = Literal["both", "up", "down"]


"""
How the empirical noise sampler draws replacement values for perturbed
cells in the fidelity Monte Carlo.

- ``"column"``: per-cell, per-column independent draw from the pool's
  per-column marginal. Paper-faithful; good for continuous features.
- ``"row"``: whole-row draw from the pool. Every replacement row is a real
  pool row. Recommended for categorical / one-hot features where
  per-column independent draws would produce invalid chimeric rows.
"""
NoiseMode = Literal["column", "row"]
