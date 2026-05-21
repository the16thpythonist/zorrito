# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What zorrito is

A modernized reimplementation of [Zorro](https://arxiv.org/abs/2105.08621) for
PyTorch Geometric. The original Zorro retrieves valid, sparse, stable hard-mask
explanations for GNN node classifications. zorrito keeps the greedy core and
adds three things that aren't in the original repo:

1. **Graph-level** explanations (not just node-level) via `task="graph"`.
2. **Regression** support with a tolerance band (`objective="regression"`,
   `tolerance=ε`), plus an optional **one-sided** mode
   (`direction="up" | "down"`) that asks the cleaner "which atoms keep the
   prediction from rising / falling" question instead of the symmetric
   "stays within ±ε" question.
3. **Whole-row noise sampling** plus a configurable **dataset-wide pool**, so
   perturbed inputs stay on the data manifold — particularly important for
   categorical / one-hot features like atom types, where the paper's
   per-column independent noise produces chimeric "atoms."

## Repo layout

```
zorrito/
├── pyproject.toml              project metadata, deps, build config
├── README.md                   PyPI-facing docs
├── CHANGELOG.md                Keep-a-Changelog
├── LICENSE                     MIT
├── src/zorrito/
│   ├── __init__.py             exports Zorro, Explanation
│   ├── explainer.py            Zorro class — the greedy loop
│   ├── fidelity.py             match functions + Monte Carlo estimator + Explanation dataclass
│   ├── noise.py                per-column and whole-row empirical samplers
│   └── _subgraph.py            k-hop neighborhood extraction for node tasks
├── examples/
│   ├── node_cora.py            Cora node classification
│   └── graph_mutag.py          MUTAG graph classification
└── tests/
    ├── test_explainer.py       end-to-end on synthetic graphs
    ├── test_fidelity.py        match-function correctness
    └── test_noise.py           sampler shapes / determinism / row vs column
```

## Environment

This project uses uv (https://docs.astral.sh/uv/) and a local venv. Always
activate or invoke through the venv before running anything:

```bash
source .venv/bin/activate
# or call binaries directly:
.venv/bin/pytest tests/
.venv/bin/python examples/graph_mutag.py
```

Bootstrap from scratch:

```bash
uv venv --python 3.11
uv pip install -e ".[dev]"
```

## Common commands

```bash
.venv/bin/pytest tests/              # run all tests (~3 s, 25 tests)
uv build                             # build sdist + wheel into dist/
.venv/bin/python examples/graph_mutag.py    # graph-classification demo
```

## Architecture notes

### The Zorro class (`src/zorrito/explainer.py`)

One public class. Constructor takes `task`, `objective`, `select`,
`noise_pool`, `noise_mode`, `direction`, plus the usual `samples`,
`top_k`, `tolerance`, `num_hops`, `seed`, `log`. The `.explain()` method is
the single entry point and returns `list[Explanation]`.

Internals:

- `_build_node_forward` / `_build_graph_forward` wrap the user's model so the
  fidelity loop only needs to perturb `x`. For graph tasks with no batch the
  wrapper supplies a zero-batch automatically.
- `_greedy_one` runs one disjoint search: an initial ranking pass (top-K
  candidates per axis) followed by the greedy loop. Each iteration is at most
  `2 * top_k * samples` forward passes through the model.
- The `select` argument controls which axes (`"both" | "nodes_only" |
  "features_only"`) the greedy explores; frozen axes start with an all-True
  mask so they don't restrict the fidelity evaluation.

### Match functions (`src/zorrito/fidelity.py`)

`make_match_fn(objective, reference_output, tolerance, direction)` returns a
closure that takes a forward-pass output and returns a 0-d tensor scalar.
There are four match conditions; classification only supports
`direction="both"`:

| objective       | direction   | match condition                          |
|-----------------|-------------|------------------------------------------|
| classification  | both        | `argmax(pred) == argmax(ref)`            |
| regression      | both        | `|pred − ref| ≤ tolerance`               |
| regression      | up          | `pred ≤ ref + tolerance`                 |
| regression      | down        | `pred ≥ ref − tolerance`                 |

The `estimate_fidelity` function runs `samples` Monte Carlo trials, draws
noise via the supplied sampler, builds `Y = where(keep_mask, x_template,
noise)`, and counts matches.

### Noise samplers (`src/zorrito/noise.py`)

Two samplers, both built from an empirical pool (a feature matrix):

- `empirical_column_sampler` — per-cell independent draw from the per-column
  marginal. Paper-faithful; good for continuous features.
- `empirical_row_sampler` — draw an entire row at a time. Each replacement is
  a valid pool row. Recommended for categorical / one-hot features.

`build_sampler(pool, mode)` dispatches between them. zorrito's task defaults
are: `column` for node tasks, `row` for graph tasks.

### Explanation dataclass (`src/zorrito/fidelity.py`)

```python
@dataclass
class Explanation:
    node_mask: torch.Tensor       # bool, over subgraph nodes (node task) or all nodes (graph task)
    feature_mask: torch.Tensor    # bool, over feature columns
    fidelity: float
    trace: list[tuple[str, int, float]]   # ("init"|"node"|"feature", idx, fidelity_after_step)
    subgraph_nodes: torch.Tensor | None   # only for node tasks; original-graph indices
```

Helpers: `.selected_node_indices()` and `.selected_feature_indices()` return
`torch.LongTensor` of indices.

## Conventions

- **Modern Python**: target `>=3.10`. Use PEP 604 unions (`int | None`),
  `from __future__ import annotations`, and type hints throughout.
- **Docstrings**: short, one-line summary + a short paragraph explaining
  "why this exists, not what it does." Use ReST-style for params if needed;
  don't bloat short helpers.
- **No external dependencies beyond `torch`, `torch-geometric`, `numpy`,
  `tqdm`**. Examples may import RDKit / pandas (used in user-side demo
  scripts), but these are NOT runtime deps of the package.
- **No emojis** in code, comments, docs, or commit messages.
- **Boolean masks are `torch.bool` tensors**, not numpy arrays.
- **Determinism**: callers can pass `seed` to `Zorro.__init__`; the sampler
  uses that generator. Don't introduce hidden randomness elsewhere.

## What's deliberately out of scope (per design, don't add without asking)

- Exhaustive (non-greedy) search
- SoftZorro / continuous-mask variant
- Precomputed-distortion caching
- CLI scripts for batch evaluation

If a user requests one of these, push back and confirm the scope decision
before implementing.

## Important behavioural notes

- **`fidelity_threshold` is the paper's `τ` directly** (e.g. 0.85 means
  fidelity must reach 0.85). The original code used `tau = 1 − τ_paper`
  which is the single most common API gotcha; zorrito fixed it.
- For **graph tasks**, the noise pool defaults to the input graph's atoms
  unless `noise_pool` is set. For categorical features this is usually too
  narrow — pass a dataset-wide pool from training graphs.
- For **regression**, the symmetric tolerance band can give empty / stuck
  greedy results when the initial fidelity already satisfies the threshold.
  If the user expects atoms to be selected but the algorithm returns nothing,
  try (a) tightening `tolerance`, (b) raising `fidelity_threshold` closer
  to 1.0, or (c) using a one-sided `direction`.
- The model passed to `Zorro` must implement `forward(x, edge_index)` for
  node tasks or `forward(x, edge_index, batch)` for graph tasks. If the
  model needs extra arguments (e.g. `edge_attr` for GINEConv), wrap it in a
  small `nn.Module` that fixes those for the single graph being explained.

## PyPI publication

The package is set up for PyPI publication (classifiers, keywords, URLs,
license file, sdist file list) but has not been published yet. See
[DEVELOP.md](DEVELOP.md) for the full release checklist; in short:

```bash
# update pyproject.toml URLs + bump version, update CHANGELOG.md
rm -rf dist/
uv build
uv publish    # uses UV_PUBLISH_TOKEN from environment, or pass --token
```

## Tests

Tests are pytest-based and run quickly (a few seconds total). The
end-to-end tests in `test_explainer.py` use tiny synthetic graphs (12 nodes,
8 features) with small `samples=10–20` to stay fast — keep them that way.
When adding a feature, add at least one test that exercises it via the
public API in addition to any unit test.
