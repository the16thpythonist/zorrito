# Coding style — `graph_attention_student` and `chem_mat_data`

A reference document distilled from reading both projects. Use this as a guide
when extending either codebase or when starting a new project that should
"feel" like one of them.

## 1. Module layout

Each module follows the same skeleton:

```python
"""
Top-level module docstring — one short paragraph explaining what this module
is for (the thematic area), not what it does line by line.
"""

# 1. standard library
import os
import typing as t
from typing import List, Dict, Tuple, Optional, Union, Callable
from collections import defaultdict

# 2. third-party
import numpy as np
import torch
import rdkit.Chem as Chem

# 3. local / first-party
from chem_mat_data.utils import get_version
from chem_mat_data._typing import GraphDict


# GLOBAL VARIABLES
# ================

PATH: str = pathlib.Path(__file__).parent.absolute()
TEMPLATE_PATH = os.path.join(PATH, 'templates')


# == SECTION HEADER ==
# A few lines of prose describing what this group of definitions is about.

def some_function(...):
    ...
```

Key conventions:

- **Module docstring first**, in triple double-quotes. Short — one paragraph.
- **Three import groups** separated by blank lines: stdlib, third-party, local.
- **One import per line** when importing names: prefer
  `from pkg.mod import A` then `from pkg.mod import B` over
  `from pkg.mod import A, B`. This makes diffs and refactors cleaner.
- **Both** `import typing as t` *and* the specific names
  (`from typing import List, Dict, ...`) are imported. Use `t.X` for ad-hoc
  hints and the explicit names for hints that appear in signatures often.
  Occasionally also `import typing as typ`.
- **Global constants** with type annotations live at the top of the module:
  `PATH: str = ...`, `VERSION_PATH: str = ...`.
- **Section dividers** are comment blocks. Two flavors:
  - `# == SECTION HEADER ==` (most common, for major sections)
  - `# --- section header ---` (subsections)
  - `# SECTION HEADER` followed by a `# ===========` underline (in
    `chem_mat_data/utils.py` and similar utility modules)
  Each section header is followed by a short prose paragraph in comments.

## 2. Typing

- **Target Python ≥ 3.10**, but use the **`typing` module** spellings
  (`t.Optional[int]`, `t.List[str]`, `t.Union[A, B]`), **not** PEP 604 unions
  (`int | None`). Both codebases are pre-PEP-604.
- **Type aliases live in `_typing.py` or `typing.py`** at the package root and
  are documented with module-level docstrings that describe the meaning,
  shape, and keys of the type. Example:

  ```python
  """
  This is an alias for the main graph representation used throughout this
  application. A graph is generally represented as a dict structure...
  """
  GraphDict = t.Dict[str, np.ndarray]
  ```

- **Annotate parameters and return types** on public functions and methods.
  Private one-liners and lambdas can be untyped.
- **Default values are part of the signature**, not the docstring:
  ```python
  def foo(x: int = 10, name: str = 'default') -> None:
  ```

## 3. Docstrings

Both projects use **ReST-style** docstrings, parsed by Sphinx.

### Function / method docstrings

```python
def get_experiment_path(name: str) -> str:
    """
    Returns the absolute path to the experiment with the given ``name``

    :returns: the string absolute path
    """
```

For functions with parameters, write:

```python
def chem_prop(
    property_name: str,
    callback: Callable[[Any], Any],
) -> Callable:
    """
    Short one-line summary.

    A short paragraph explaining the *purpose* — why this function exists.

    :param property_name: The string name of the method(!) of the atom or bond
        object to use to get the property value.
    :param callback: An additional function that can be used to encode the
        extracted property value into the correct format of a list of floats.

    :returns: A function with the signature [Union[Chem.Atom, Chem.Bond]] -> List[float]
    """
```

- Use **double backticks** for inline code: `` ``name`` ``.
- Use `:param X:` and `:returns:` blocks. Don't use Google or numpy style.
- The summary line is **one sentence**, in present tense, no period mandatory
  on the very short ones.
- The body explains **why this exists**, not what each line does.
- Don't write `:param:` blocks for self-evident parameters of trivial helpers.

### Class docstrings

Classes — especially abstract base classes and mixins — get longer docstrings
that **define the interface contract**:

```python
class AbstractGraphModel(pl.LightningModule):
    """
    This is the abstract base class for implementing a graph property prediction model.

    **PREDICT GRAPHS**

    This abstract base class implements the method "predict_graphs" which is a convenience
    wrapper around the function of the model. ...

    **ARBITRARY FORWARD RETURNS**

    This abstract base class enforces a certain interface for the forward, which dictates ...

    B - batch dimensions, number of graphs
    V - number of nodes
    E - number of edges
    O - output dimension
    """
```

- **Section headers in docstrings use `**ALL CAPS BOLD**`**, separated by blank
  lines, when a class has multiple aspects to document.
- **Shape conventions** (`B - batch`, `V - nodes`, `E - edges`) listed at the
  bottom of docstrings that document tensor-heavy APIs.
- For mixins: docstring lists the **assumptions about the base class**
  (required attributes, required methods).

### Attribute documentation

Class attributes that are meant to be visible/configurable get a comment
*above* the assignment with `:attr ATTRIBUTE_NAME:`:

```python
# :attr BATCH_SIZE:
#       This is the default batch size that is being used in all the methods
#       for the model INFERENCE tasks.
BATCH_SIZE: int = 1_000
```

For module-level parameters (pycomex-style experiment scripts):

```python
# :param NUM_CHANNELS:
#       The number of explanation channels for the model.
NUM_CHANNELS: int = 2
```

## 4. Naming

| Kind                              | Convention             | Example                                        |
|-----------------------------------|------------------------|------------------------------------------------|
| Function / method                 | `snake_case`           | `data_from_graph`, `ensure_dataset`            |
| Variable                          | `snake_case`           | `file_share`, `folder_path`                    |
| Class                             | `PascalCase`           | `MoleculeProcessing`, `OneHotEncoder`          |
| Abstract class                    | `Abstract` prefix      | `AbstractGraphModel`, `AbstractFileShare`      |
| Mixin                             | `Mixin` suffix         | `MveMixin`, `RichMixin`, `StringEncoderMixin`  |
| Module-level constant             | `SCREAMING_SNAKE_CASE` | `BATCH_SIZE`, `TEMPLATE_ENV`, `NULL_LOGGER`    |
| PyTorch layer attribute           | `lay_` prefix          | `self.lay_act`, `self.lay_dense`               |
| Private helper / module           | leading `_`            | `_typing.py`, `_build_node_forward`            |
| Test class                        | `Test` prefix          | `class TestAbstractGraphModel:`                |
| Test function                     | `test_..._works`       | `test_ensure_dataset_with_folder_works`        |

Notes:

- Use **`Abstract` prefix on ABCs**, not the `ABC`/`ABCMeta` machinery from
  `abc`. Interfaces are enforced by raising `NotImplementedError()` in the
  base methods (see §5).
- `_typing.py` (chem_mat_data) vs `typing.py` (graph_attention_student) —
  both exist; the leading underscore is the newer convention to avoid
  shadowing the stdlib module on dotted imports.
- Variable names lean **verbose and explicit** (`file_share_type` over
  `fs_type`, `graph_embedding` over `g_emb`). No one-letter names except in
  short comprehensions and math-y inner loops.

## 5. Class design

### Abstract base classes

Interface enforcement via `NotImplementedError`, not `abc.ABC`:

```python
class AbstractXyzParser:
    """
    Abstract base class for xyz file parsers. ...
    """

    def __init__(self, path: str, **kwargs):
        self.path = path

    def parse(self) -> Tuple[Chem.Mol, dict]:
        """
        This method should be implemented by the concrete implementations ...
        """
        raise NotImplementedError()
```

Subclasses **call the parent `__init__` explicitly by class name**, not via
`super()`:

```python
class DefaultXyzParser(AbstractXyzParser):

    def __init__(self, path: str, **kwargs):
        AbstractXyzParser.__init__(self, path)
```

This makes the inheritance chain readable when multiple bases are involved
(common with the mixin pattern). `super()` is also acceptable, but the
explicit form is the house style.

### Mixins

Mixins are first-class. Each mixin docstring states:

1. **What conditions** the base class must satisfy (attributes, methods, base
   classes the mixin assumes).
2. **What it adds** to the host class.

```python
class MveMixin:
    """
    This Mixin expects the following conditions to be implemented:
    - The base class has to be a subclass of ``pl.LightningModule``
    - The base class has to have an attribute ``self.embedding_dim`` which
      defines the dimensionality of the graph embedding vector ...
    """
```

Composition by mixin is preferred over deep inheritance trees.

### Dunders for "acts like a dict / acts like a callable"

When something conceptually *is* a dict (e.g. a metadata cache, a file share's
metadata), implement `__getitem__`, `__contains__`, `keys()` rather than
exposing a `.data` attribute:

```python
class AbstractFileShare:
    ...
    def keys(self) -> t.List[str]:
        self.assert_metadata()
        return self.metadata.keys()

    def __contains__(self, key: str) -> bool:
        ...

    def __getitem__(self, key: str) -> t.Any:
        ...
```

For Rich-renderable objects, implement `__rich__` or `__rich_console__`.

### Singleton pattern

When a true singleton is needed (config object), use a metaclass:

```python
class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class Config(metaclass=Singleton):
    ...
```

## 6. Comments

Comments are **plentiful and explain *why***, not what.

### Block comments above code

The standard form: 2–5 lines of prose above a non-trivial block.

```python
# However, if the file does not exist already, we need to attempt to fetch it
# from the remote file share server and download it to the local file system.
else:
    if not config:
        config = Config()
```

### Dated change markers

When code is added/changed for a specific reason, the convention is a
`DD.MM.YY` date stamp comment immediately above the change:

```python
# 24.01.24
# The graph labels are not always going to be given and we want to support
# that here as well
if 'graph_labels' in graph:
    y = torch.tensor(graph['graph_labels'], dtype=dtype)
else:
    y = torch.tensor([0, ], dtype=dtype)
```

This is the project's lightweight "blame in the source" — it survives
refactors that break `git blame`.

### Inline shape annotations

For tensor-heavy code, annotate shapes inline:

```python
# out_pred: (B, O)
out_pred = info['graph_output']
# out_pred: (B, O)
out_true = data.y.view(-1, self.out_dim)
# out_var: (B, O)
out_var = info['graph_variance']
```

### What NOT to comment

Don't comment trivial assignments. Don't restate the docstring.

## 7. PyTorch / Lightning patterns

- Models inherit from `AbstractGraphModel(pl.LightningModule)`, which itself
  defines the **forward returns a dict** contract:
  - `graph_output`: required, `(B, O)`
  - `graph_*` prefix → tensors of shape `(B, ...)`
  - `node_*` prefix → tensors of shape `(B * V, ...)`
  - `edge_*` prefix → tensors of shape `(B * E, ...)`
- Hyperparameters go into `self.hparams` (Lightning convention) using
  `self.hparams.update({...})` from each mixin's `__init__`.
- Layers are stored as `self.lay_<name>` or in `nn.ModuleList()`.
- `setattr` / `getattr` are used deliberately to swap method behavior at
  runtime (e.g., `activate_mve_training` swaps `training_prediction` for an
  alternative implementation). This is a feature, not a hack — but document
  it.

## 8. Rich for terminal UX

CLI code uses `rich_click` and the `rich` library heavily:

- Define a `RichMixin` (or implement `__rich_console__` directly) for any
  object that renders to the terminal.
- Compose with `Panel`, `Table`, `Padding`, `Columns`, `Group`, `Rule`.
- Use `Style(color=..., bgcolor=...)` rather than embedding ANSI codes.
- Long help text in CLIs is rendered as a custom `RichHelp` object yielded
  paragraph-by-paragraph, not as the click-default one-shot string.

## 9. Templates

Both projects bundle a `templates/` folder and a module-level jinja2
environment:

```python
TEMPLATES_FOLDER = os.path.join(PATH, 'templates')
TEMPLATE_ENV = j2.Environment(
    loader=j2.FileSystemLoader(TEMPLATES_FOLDER),
    autoescape=j2.select_autoescape(),
)
TEMPLATE_ENV.globals.update(**{
    'zip': zip,
    'int': int,
    'enumerate': enumerate,
})
```

Use this for LaTeX reports, HTML pages, default config files, CLI help text.

## 10. Versioning

- A plain `VERSION` file next to `__init__.py`, containing a single version
  string.
- A `get_version()` helper reads it. The same string is also in
  `pyproject.toml`.
- Optional environment-variable override of the version (used in tests):
  ```python
  if version := os.environ.get('PACKAGE_VERSION_OVERWRITE', False):
      return version
  ```

## 11. Tests

- `pytest`-based. Test files live in `tests/` at the repo root.
- **Group related tests in a `Test<Subject>` class** when there are several
  tests for the same class:
  ```python
  class TestAbstractGraphModel:
      def test_saving_works(self): ...
      def test_saving_loading_works(self): ...
  ```
- Free-standing test functions are also fine for module-level functions.
- **Descriptive names**: `test_ensure_dataset_with_folder_works` — say what
  *condition* and what *behavior*.
- **Every test has a docstring** explaining what behavior is being verified.
- Filesystem tests use `tempfile.TemporaryDirectory()` as a context manager.
- Tiny synthetic data — keep individual tests under a second.

## 12. Errors and edge cases

- Raise **specific** exceptions: `FileNotFoundError`, `LookupError`,
  `ValueError`. Don't raise bare `Exception`.
- `assert_<condition>()` methods on classes are used to validate state before
  operations (e.g., `assert_metadata()`).
- `try/except` only around boundaries (file IO, network, third-party calls).
- For "trust the caller" internal code, **don't** add defensive checks. The
  abstract base class contract is the contract.

## 13. Project boilerplate

`pyproject.toml` is the single source of truth for build metadata. Both
projects use:

- `hatchling` as build backend (graph_attention_student) or whatever the
  project chose (chem_mat_data uses similar).
- Full classifiers list including Python versions, license, audience.
- Author + maintainer entries.
- Keywords list for PyPI discoverability.
- `[project.optional-dependencies]` for `dev`, `test`, etc.

A separate `VERSION` file (see §10) keeps the version readable at runtime
without parsing TOML.

## 14. What is NOT in the house style

To make the negative space explicit:

- **No PEP 604 unions** (`int | None`). Use `t.Optional[int]`.
- **No `dataclasses` for everything** — they are used (e.g.,
  `Explanation` in zorrito) but mostly the codebase prefers regular
  classes with explicit `__init__`.
- **No `abc.ABC` / `@abstractmethod`** — abstract methods raise
  `NotImplementedError` in the body.
- **No emojis** in source, comments, docs, or commit messages.
- **No f-string-only logging**. Logging uses `logger.warning(f'...')` form
  freely though — this is not a strict project rule.
- **No bare `from X import *`**.
- **No deep inheritance trees** — prefer mixins.

## 15. Cheat sheet

When adding a new module to either project:

1. Top-level docstring describing the module's *purpose*.
2. Three blank-line-separated import blocks: stdlib / third-party / local.
3. `import typing as t` plus explicit names from `typing`.
4. Module-level constants and paths next, with `:attr:` comments.
5. Section dividers (`# == SECTION ==`) with a prose paragraph each.
6. Functions and classes with ReST docstrings.
7. Abstract base + concrete subclass(es); mixins where cross-cutting.
8. Dated change markers (`# DD.MM.YY`) when adding non-obvious code paths.
9. A matching test file in `tests/`, with a `Test<Subject>` class.
