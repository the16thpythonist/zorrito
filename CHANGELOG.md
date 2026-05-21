# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-05-21

A no-behavior-change refactor that brings the codebase in line with the
in-house coding style of the sibling projects `graph_attention_student`
and `chem_mat_data`. The public API is unchanged: `from zorrito import
Zorro, Explanation` and the `Zorro.__init__` / `Zorro.explain` signatures
keep their existing names, defaults, and return types.

### Added

- `src/zorrito/_typing.py` — one home for the five `Literal` aliases
  (`Task`, `SelectMode`, `Objective`, `Direction`, `NoiseMode`), each
  with a prose docstring describing meaning and allowed values.
- `src/zorrito/VERSION` — plain-text version string mirroring
  `pyproject.toml`, read at runtime by `get_version()`.
- `src/zorrito/utils.py` — package-level `PATH`, `VERSION_PATH`,
  `get_version()`, and a default silent `NULL_LOGGER`.
- `AbstractNoiseSampler` abstract base in `src/zorrito/noise.py`, with
  `EmpiricalColumnSampler` and `EmpiricalRowSampler` as concrete
  subclasses. The existing free-function constructors
  (`empirical_column_sampler`, `empirical_row_sampler`, `build_sampler`)
  are retained as thin one-line factories that return instances of the
  new classes, so the closure-style call sites continue to work.
- `tests/test_regression_locked.py` — three characterization tests that
  pin the exact selected-mask indices, fidelity values, and greedy trace
  for planted-signal scenarios. Acts as a canary against silent
  behavioral drift in future refactors.
- `tests/util.py` — shared toy models (`TinyGCN`, `TinyGraphGCN`,
  `SingleFeatureModel`) and the `make_random_node_graph` fixture
  factory, lifted out of the test files for reuse.

### Changed

- Module docstrings, ReST `:param:` / `:returns:` blocks on every public
  function and class, and `# == SECTION ==` dividers throughout the
  source tree. `Zorro`'s class docstring is now split into
  `**TASKS**` / `**OBJECTIVES**` / `**SELECT MODES**` / `**NOISE
  SAMPLING**` / `**OUTPUT**` sections.
- Test files are grouped into `Test*` classes by subject:
  `TestZorroNode`, `TestZorroGraph`, `TestZorroConfig`,
  `TestZorroDeterminism`, `TestZorroPlantedSignal`, `TestMakeMatchFn`,
  `TestEstimateFidelity`, `TestEmpiricalColumnSampler`,
  `TestEmpiricalRowSampler`. Every test has a one-line ReST docstring.
- Example scripts gain `# == MODEL ==` / `# == TRAINING ==` / `# ==
  EXPLAIN ==` section dividers and ReST docstrings on the inline helper
  classes. No behavioral change.
- `__version__` is now read from the `VERSION` file via
  `get_version()` rather than hard-coded.
- `CLAUDE.md` documents the new house-style conventions (section
  dividers, ReST docstrings, `**ALL CAPS**` headers, `Abstract*`
  prefix, dated change markers); the existing PEP 604 typing rule is
  preserved unchanged.
- The bundled test count grows from 29 to 32 (3 new characterization
  tests); total suite runtime stays at ~7 seconds.

### Not changed (regression-safety guarantees)

- The greedy loop, the disjoint outer loop, the initial-ranking pass,
  the top-K logic, and the noise-sampler math are byte-identical in
  the operations they perform.
- `Zorro.__init__` keyword names, defaults, and validation rules.
- `Zorro.explain` signature and return type.
- `Explanation` dataclass field names and helper methods.
- All 29 pre-existing tests pass without modification of their assertions.

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
