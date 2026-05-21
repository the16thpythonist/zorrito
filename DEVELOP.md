# Development guide

Notes for working on zorrito itself (not for using it — see [README.md](README.md)
for usage).

## Local setup

```bash
uv venv --python 3.11
uv pip install -e ".[dev]"
```

Always invoke through the venv:

```bash
source .venv/bin/activate
# or directly
.venv/bin/pytest tests/
.venv/bin/python examples/graph_mutag.py
```

## Running tests

```bash
.venv/bin/pytest tests/                       # all tests
.venv/bin/pytest tests/test_fidelity.py -v    # one file, verbose
.venv/bin/pytest -k "direction"               # subset by name
```

The end-to-end tests in `test_explainer.py` use tiny synthetic graphs and
small `samples=10–20` to stay fast (~3 seconds total). Keep them that way.

When adding a feature, add at least one test that exercises it through the
public `Zorro.explain()` API in addition to any unit tests on the helper
functions.

## Building

```bash
rm -rf dist/
uv build                                       # produces sdist + wheel
ls dist/                                       # zorrito-X.Y.Z.tar.gz + .whl
```

The sdist includes `src/`, `tests/`, `examples/`, `README.md`,
`CHANGELOG.md`, `LICENSE`, and `pyproject.toml`. File inclusion is configured
under `[tool.hatch.build.targets.sdist]` in `pyproject.toml`.

## Publishing a new version to PyPI

zorrito publishes via [`uv publish`](https://docs.astral.sh/uv/guides/publish/),
not twine.

### Prerequisites (one-time)

1. **Create a PyPI account** and verify your email at https://pypi.org/.
2. **Create an API token**: PyPI → Account Settings → API tokens → "Add API
   token". Scope it to the `zorrito` project once the project exists; use a
   broader scope for the very first upload.
3. **Store the token** in your environment (don't commit it):
   ```bash
   export UV_PUBLISH_TOKEN='pypi-AgEIcHlwa...'
   ```
   Or put it in `~/.pypirc` (uv reads the same file twine uses) under a
   `[pypi]` section.

### Per-release checklist

1. **Decide the version** following [SemVer](https://semver.org/):
   - patch (`0.1.0` → `0.1.1`) for bug fixes
   - minor (`0.1.0` → `0.2.0`) for new backwards-compatible features
   - major (`0.1.0` → `1.0.0`) for breaking changes
2. **Bump `version` in `pyproject.toml`.**
3. **Update `src/zorrito/__init__.py`** so `__version__` matches.
4. **Update `CHANGELOG.md`**: move entries from `[Unreleased]` into a new
   `[X.Y.Z] - YYYY-MM-DD` section. Keep the `[Unreleased]` heading at the
   top for future changes.
5. **Update `pyproject.toml` URLs** if they still point at placeholders —
   `Homepage`, `Repository`, `Issues`, `Changelog` should all be real.
6. **Run the full test suite**:
   ```bash
   .venv/bin/pytest tests/
   ```
   All tests must pass. Do not publish with failing tests.
7. **Clean the build directory and build fresh artifacts**:
   ```bash
   rm -rf dist/
   uv build
   ```
8. **Inspect the artifacts** to make sure metadata is right:
   ```bash
   ls dist/                                                       # both files present
   tar tzf dist/zorrito-*.tar.gz | head -20                       # sdist contents
   unzip -p dist/zorrito-*.whl '*/METADATA' | head -40            # wheel metadata
   ```
   Check that `Version:`, `Summary:`, classifiers, and the long description
   are correct.
9. **(Optional) Test-publish to TestPyPI first** to catch any metadata or
   token issues without burning a real PyPI version number:
   ```bash
   uv publish --publish-url https://test.pypi.org/legacy/ \
              --token "$UV_TESTPYPI_TOKEN"
   ```
   Then install from TestPyPI in a throwaway venv to verify it works:
   ```bash
   uv pip install --index-url https://test.pypi.org/simple/ zorrito
   ```
10. **Publish to PyPI**:
    ```bash
    uv publish                          # picks up UV_PUBLISH_TOKEN
    # or explicitly:
    uv publish --token "$UV_PUBLISH_TOKEN"
    ```
11. **Tag the release in git** and push the tag:
    ```bash
    git tag -a vX.Y.Z -m "Release X.Y.Z"
    git push origin vX.Y.Z
    ```
12. **Verify on PyPI**: visit https://pypi.org/project/zorrito/ and confirm
    the new version is listed with the expected metadata.

### Rollback / fixes

PyPI does not let you reupload the same version number — even after deletion.
If something is wrong with a release:

- For a broken release, **yank it** (PyPI → project page → Manage → release
  → Yank). Yanked releases stay installable by pinned-version users but are
  hidden from default resolution.
- For a fix, **bump the patch version** and publish again. Don't try to
  re-publish the same version.

## Code conventions

- Modern Python (`>=3.10`). Use PEP 604 unions (`int | None`) and
  `from __future__ import annotations`.
- Type hints throughout. The public API in `src/zorrito/__init__.py` is the
  contract — don't change signatures without bumping the version
  appropriately.
- Docstrings should be short — one-line summary, one short paragraph if
  needed. Explain why something exists, not what it does (the type hints
  already say that).
- Boolean masks are `torch.bool` tensors, not numpy arrays.
- No emojis in code, comments, docs, or commit messages.
- Determinism: if you add a code path that introduces randomness, route it
  through the `Zorro.seed` argument's generator. Don't introduce hidden
  randomness.

## Out of scope (don't add without explicit user agreement)

- Exhaustive (non-greedy) search
- SoftZorro / continuous-mask variant
- Precomputed-distortion caching
- CLI scripts for batch evaluation
