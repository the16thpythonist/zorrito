# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-05-21

### Added

- `noxfile.py` with a `tests` session parameterized over Python 3.10, 3.11,
  and 3.12, using uv as the venv backend
  (`nox.options.default_venv_backend = "uv"`) so environments are provisioned
  with the same resolver as the rest of the project.
- `nox>=2024.4.15` added to the `dev` optional-dependency group.

## [0.1.0] - 2026-05-20

Initial release. A lean, modernized reimplementation of [Zorro](https://arxiv.org/abs/2105.08621)
for PyTorch Geometric.

### Added

- `Zorro` class with unified node-/graph-level dispatch (`task="node"` or `task="graph"`).
- Classification objective with argmax-equal match function.
- Regression objective with tolerance-band match function (`objective="regression"`,
  `tolerance=ε`).
- One-sided regression mode via `direction="up" | "down" | "both"` — explains
  which atoms prevent the prediction from drifting in a chosen direction,
  rather than only from drifting in either direction symmetrically.
- `select="both" | "nodes_only" | "features_only"` to restrict which axes the
  greedy search explores.
- `noise_mode="column" | "row"` with sensible per-task defaults: per-column
  independent (paper-faithful, good for continuous features) and whole-row
  (recommended for categorical / one-hot features such as atom types).
- Configurable `noise_pool` so the empirical noise distribution can be drawn
  from a dataset-wide pool independently of the input being explained.
- Disjoint-explanation enumeration via `max_explanations`.
- Multi-hop computational subgraph extraction for node tasks (via PyG's
  `k_hop_subgraph`).
- Pseudo-deterministic mode via the `seed` argument.
- 25 pytest tests covering the fidelity estimator, the noise samplers, and
  end-to-end explanation for both tasks and both objectives.
- Two worked examples in `examples/`: node classification on Cora,
  graph classification on MUTAG.

### Notable departures from the original Zorro reference implementation

- `fidelity_threshold` replaces the original `tau` (which was actually
  `1 - fidelity_threshold`), removing a frequent source of confusion.
- `max_explanations` replaces the original `recursion_depth`.
- Boolean masks replace the original numpy float masks; explanations are
  returned as a single `Explanation` dataclass.
- Modern Python ( `>=3.10`) and modern PyG ( `>=2.4`).
