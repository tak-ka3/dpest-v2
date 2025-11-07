"""
Backward-compatible namespace for auto-generated dist functions.

Historically each SVT implementation required a manually-written ``*_dist`` helper.
Now the :mod:`dpest.algorithms.registry` takes care of that automatically, but this
module continues to expose the same symbols for existing import paths.
"""

from .registry import get_registered_dist_functions

_DIST_FUNCS = get_registered_dist_functions()
globals().update(_DIST_FUNCS)
__all__ = sorted(_DIST_FUNCS.keys())
