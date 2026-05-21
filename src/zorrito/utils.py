"""
Utility helpers shared across the zorrito package.

Currently this module owns the package's filesystem anchor (``PATH``), the
version-string accessor, and a silent default logger that other modules can
fall back to when the caller did not pass one in.
"""
from __future__ import annotations

import logging
import os
import pathlib


# == GLOBAL VARIABLES ==
# The absolute filesystem path to this package's source directory. Other
# package files can be located relative to this anchor without depending on
# the current working directory.

PATH: str = str(pathlib.Path(__file__).parent.absolute())

# The plain-text VERSION file mirrors pyproject.toml's ``version`` field and
# is the runtime source of truth used by ``get_version()``.
VERSION_PATH: str = os.path.join(PATH, 'VERSION')

# A logger that swallows every record. Suitable as a default value for code
# paths that take an optional ``log`` argument when the caller did not supply
# one — keeps internal logging silent unless explicitly enabled.
NULL_LOGGER: logging.Logger = logging.Logger('NULL')
NULL_LOGGER.addHandler(logging.NullHandler())


# == MISC FUNCTIONS ==


def get_version() -> str:
    """
    Returns the version string of the package by reading it from the VERSION
    file.

    The VERSION file is a single-line plain-text file that mirrors the
    ``version`` field in pyproject.toml. Reading it at runtime avoids both
    hard-coding a string into the source and depending on a build step that
    rewrites the constant.

    :returns: the version string (e.g. ``"0.2.0"``).
    """
    with open(VERSION_PATH) as file:
        return file.read().replace(' ', '').replace('\n', '')
