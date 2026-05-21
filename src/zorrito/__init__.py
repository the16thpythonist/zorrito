"""
As the __init__ module of the package, this determines what functions and
classes should be globally importable from the package name alone.

The two names exposed at the package level are :class:`Zorro` (the
explainer) and :class:`Explanation` (its return type). Everything else
remains accessible through its submodule.
"""
from zorrito.explainer import Zorro as Zorro
from zorrito.fidelity import Explanation as Explanation
from zorrito.utils import get_version


__all__ = ["Zorro", "Explanation"]
__version__ = get_version()
