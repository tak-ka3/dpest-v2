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


def _resample_distribution(dist: Dist, n_samples: int = 100000) -> Dist:
    """
    Resample a distribution using its _sample_func to get accurate probabilities
    when analytical computation with dependencies is not possible.

    Args:
        dist: Distribution with _sample_func defined
        n_samples: Number of samples to draw

    Returns:
        New distribution with probabilities estimated from samples
    """
    if not hasattr(dist, '_sample_func'):
        # Cannot resample without a sample function
        return dist

    # Draw samples
    samples = []
    for _ in range(n_samples):
        cache = {}  # Fresh cache for each sample
        sample = dist._sample(cache)
        samples.append(sample)

    samples = np.array(samples)

    # Estimate probabilities from samples
    unique_values, counts = np.unique(samples, return_counts=True)
    atoms = [(float(v), float(c) / n_samples) for v, c in zip(unique_values, counts)]

    # Create new distribution
    from ..core import Node
    result = Dist(
        atoms=atoms,
        dependencies=dist.dependencies,
        node=Node(op='Resampled', inputs=[dist.node] if dist.node else [],
                 dependencies=set(dist.dependencies), needs_sampling=False),
        skip_validation=True
    )

    # Copy the sample function
    result._sample_func = dist._sample_func
    result.normalize()

    return result


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

    Additionally, this decorator wraps the original function to check if the result
    needs sampling (when dependencies cannot be resolved analytically) and automatically
    falls back to sampling-based computation.
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

        # Wrap the original function to check for needs_sampling
        @wraps(algo_fn)
        def wrapped_algo_fn(*args, **kwargs):
            result = algo_fn(*args, **kwargs)

            # Check if the result needs sampling
            if isinstance(result, Dist) and hasattr(result, 'node') and result.node:
                if result.node.needs_sampling:
                    # Fall back to sampling-based computation
                    return _resample_distribution(result, n_samples=100000)

            return result

        return wrapped_algo_fn

    return decorator


def get_registered_dist_functions() -> Dict[str, Callable[..., Sequence[Dist]]]:
    """Return a copy of all auto-generated dist functions."""
    return dict(_DIST_FUNCTIONS)


def get_dist_function(name: str) -> Callable[..., Sequence[Dist]]:
    """Retrieve a registered dist function by name."""
    return _DIST_FUNCTIONS[name]

