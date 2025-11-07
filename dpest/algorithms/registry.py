"""
Utility helpers for automatically exposing compile()-based algorithms as dist_func.

Algorithm implementations can opt-in via the ``@auto_dist`` decorator, which
registers an accompanying ``*_dist`` function that handles:
    1. Converting raw numeric arrays into deterministic ``Dist`` inputs.
    2. Compiling the algorithm with ``dpest.engine.compile``.
    3. Executing the compiled function and returning its output distributions.
"""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable, Dict, Iterable, List, Sequence

import numpy as np

from ..core import Dist
from ..engine import compile

_DIST_FUNCTIONS: Dict[str, Callable[..., Sequence[Dist]]] = {}


def _materialize_queries(data: Any) -> List[Dist]:
    """
    Convert various input formats into deterministic ``Dist`` objects.

    Accepts:
        - list/tuple/np.ndarray of scalars
        - already-constructed Dist objects (returned as-is)
    """
    if isinstance(data, Dist):
        return [data]
    if isinstance(data, Sequence) and data and isinstance(data[0], Dist):
        return list(data)

    np_data = np.asarray(data, dtype=float)
    flattened = np_data.ravel()
    return [Dist.deterministic(float(x)) for x in flattened.tolist()]


def auto_dist(name: str | None = None) -> Callable[[Callable[..., Sequence[Dist]]], Callable[..., Sequence[Dist]]]:
    """
    Decorator that registers a compile()-based algorithm and auto-generates a dist_func.

    Usage:
        @auto_dist()
        def svt1(queries: List[Dist], eps: float = 0.1, ...):
            ...

    The generated wrapper is named ``<algo_name>_dist`` unless a custom ``name`` is provided.
    """

    def decorator(algo_fn: Callable[..., Sequence[Dist]]):
        dist_name = f"{name or algo_fn.__name__}_dist"

        @wraps(algo_fn)
        def dist_func(data, *args, **kwargs):
            queries = _materialize_queries(data)
            compiled_algo = compile(lambda q: algo_fn(q, *args, **kwargs))
            return compiled_algo(queries)

        dist_func.__name__ = dist_name
        dist_func.__doc__ = (
            f"Auto-generated dist wrapper for {algo_fn.__name__}. "
            "Converts numeric inputs to Dist objects and executes the compiled algorithm."
        )

        _DIST_FUNCTIONS[dist_name] = dist_func
        setattr(algo_fn, "_dist_func", dist_func)
        setattr(algo_fn, "_dist_name", dist_name)
        return algo_fn

    return decorator


def get_registered_dist_functions() -> Dict[str, Callable[..., Sequence[Dist]]]:
    """Return a copy of all auto-generated dist functions."""
    return dict(_DIST_FUNCTIONS)


def get_dist_function(name: str) -> Callable[..., Sequence[Dist]]:
    """Retrieve a registered dist function by name."""
    return _DIST_FUNCTIONS[name]

