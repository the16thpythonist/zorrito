"""Nox sessions for zorrito.

Uses uv as the backend so environments are provisioned with the same resolver
as the rest of the project. Run ``nox`` to test all supported Python versions
or ``nox -s tests-3.11`` to target a single one.
"""

from __future__ import annotations

import nox

nox.options.default_venv_backend = "uv"
nox.options.reuse_existing_virtualenvs = True

PYTHON_VERSIONS = ["3.10", "3.11", "3.12"]


@nox.session(python=PYTHON_VERSIONS)
def tests(session: nox.Session) -> None:
    """Run the pytest suite against a given Python version."""
    session.install("-e", ".[dev]")
    session.run("pytest", "tests", *session.posargs)
