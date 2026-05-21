# zorrito

A lean, modernized reimplementation of [Zorro](https://arxiv.org/abs/2105.08621) —
valid, sparse, stable explanations for PyTorch Geometric GNNs.

zorrito keeps the greedy core of the paper extends the following

- **Modern** Python and Pytorch Geometric support
- **Graph-level explanations**, not just node-level
- **Regression support** via a tolerance band, with optional **one-sided**
  (direction-aware) match functions
- **Configurable noise** — whole-row sampling and dataset-wide pools, so
  perturbed inputs stay on the data manifold (matters a lot for categorical /
  one-hot features like atom types)

---

## Install

```bash
pip install zorrito
```

zorrito depends on `torch>=2.0` and `torch-geometric>=2.4`. Install those
according to the [official PyG instructions](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)
for your CUDA/PyTorch combination first.

For development:

```bash
git clone https://github.com/your-user/zorrito.git
cd zorrito
uv venv --python 3.11
uv pip install -e ".[dev]"
pytest tests/
```

See [DEVELOP.md](DEVELOP.md) for the release / publishing checklist.

## Quick start

### Node classification (Cora-style)

```python
import torch
from zorrito import Zorro

# `model` is any nn.Module that implements forward(x, edge_index)
explainer = Zorro(
    model=model,
    task="node",
    samples=100,
    top_k=10,
    seed=42,
)
explanations = explainer.explain(
    x=data.x,
    edge_index=data.edge_index,
    node_idx=10,
    fidelity_threshold=0.85,
)
expl = explanations[0]
print(expl.fidelity)                       # final RDT-Fidelity reached
print(expl.selected_node_indices())        # indices into the subgraph
print(expl.selected_feature_indices())     # indices into feature columns
print(expl.subgraph_nodes)                 # mapping back to the original graph
```

### Graph classification (MUTAG-style)

```python
from zorrito import Zorro

# `model` is any nn.Module that implements forward(x, edge_index, batch)
explainer = Zorro(
    model=model,
    task="graph",
    noise_mode="row",                # recommended for categorical features
    noise_pool=dataset_atom_pool,    # dataset-wide pool of valid atoms
    samples=100,
    seed=42,
)
explanations = explainer.explain(
    x=graph.x,
    edge_index=graph.edge_index,
    fidelity_threshold=0.95,
)
```

### Regression with a one-sided direction

```python
explainer = Zorro(
    model=model,
    task="graph",
    objective="regression",
    tolerance=0.4,                   # ε for the tolerance band
    direction="down",                # which atoms prevent the prediction
                                     # from FALLING below ref - ε
    noise_mode="row",
    noise_pool=dataset_atom_pool,
    seed=42,
)
explanations = explainer.explain(
    x=molecule.x,
    edge_index=molecule.edge_index,
    fidelity_threshold=0.95,
)
```

## Key concepts

**RDT-Fidelity.** For a candidate explanation $\mathcal{S} = (V_s, F_s)$:

$$\mathcal{F}(\mathcal{S}) = \mathbb{E}_{Z\sim\mathcal{N}}\big[\,\mathrm{match}(\Phi(Y_\mathcal{S}), \Phi(X))\,\big]$$

where $Y_\mathcal{S}$ is the original input with the unselected entries replaced
by random draws $Z$ from a noise distribution. zorrito estimates this with
`samples` Monte-Carlo trials.

**Match functions** (the per-trial yes/no test):

| objective       | direction   | match condition                          |
| --------------- | ----------- | ---------------------------------------- |
| classification  | n/a         | `argmax(Φ(Y_S)) == argmax(Φ(X))`         |
| regression      | `both`      | `|Φ(Y_S) − Φ(X)| ≤ tolerance`            |
| regression      | `up`        | `Φ(Y_S) ≤ Φ(X) + tolerance`              |
| regression      | `down`      | `Φ(Y_S) ≥ Φ(X) − tolerance`              |

For regression, `direction="up"` selects atoms that *prevent the prediction
from rising*, and `direction="down"` selects atoms that *prevent it from
falling*. This is often more informative than the symmetric tolerance band
when the two sides have different chemistry.

**Greedy algorithm.** Starting from the empty selection, at each iteration
zorrito evaluates the top-K candidate nodes and top-K candidate features (the
"k" in `top_k`), adds the single element with the largest fidelity gain, and
stops when fidelity exceeds `fidelity_threshold`. Then optionally re-runs to
enumerate disjoint alternative explanations (`max_explanations`).

**Noise distribution.** The original paper draws each noise cell independently
from its column's empirical distribution. zorrito keeps that as the default
for node tasks (continuous features), and switches to **whole-row sampling**
by default for graph tasks (categorical / one-hot features stay valid). The
pool the rows are drawn from can be set independently via `noise_pool`.

## Configuration cheat sheet

```python
Zorro(
    model,
    task               = "node" | "graph",
    objective          = "classification" | "regression",
    select             = "both" | "nodes_only" | "features_only",
    direction          = "both" | "up" | "down",     # regression only
    noise_pool         = None | torch.Tensor,        # default: x from explain()
    noise_mode         = "column" | "row",           # default: column (node), row (graph)
    device             = "cpu" | "cuda",
    samples            = 100,
    top_k              = 10,
    tolerance          = 0.0,
    num_hops           = 2,                          # node-task subgraph radius
    seed               = None,
    log                = False,
)
```

```python
explainer.explain(
    x,
    edge_index,
    node_idx           = None,                       # required for task="node"
    batch              = None,
    fidelity_threshold = 0.85,
    max_explanations   = 1,
)
```

The return is a `list[Explanation]`, where each `Explanation` exposes
`node_mask`, `feature_mask`, `fidelity`, `trace`, and (for node tasks)
`subgraph_nodes`. The masks are boolean tensors; use
`selected_node_indices()` / `selected_feature_indices()` to convert.

## Notable departures from the original paper / reference implementation

- `fidelity_threshold=0.85` is the value $\tau$ from the paper. The original
  code uses `tau=0.15` (which is `1 − τ_paper`)
- `max_explanations` replaces `recursion_depth`.
- Explanations are returned as a single `Explanation` dataclass with boolean
  masks, instead of nested numpy arrays.
- The package targets modern Python (>=3.10) and PyG (>=2.4).

## Citing the algorithm

The algorithm is from the original Zorro paper:

```bibtex
@article{funke2021zorro,
  title   = {Zorro: Valid, Sparse, and Stable Explanations in Graph Neural Networks},
  author  = {Funke, Thorben and Khosla, Megha and Rathee, Mandeep and Anand, Avishek},
  journal = {arXiv preprint arXiv:2105.08621},
  year    = {2021}
}
```

## License

MIT — see [LICENSE](LICENSE).
