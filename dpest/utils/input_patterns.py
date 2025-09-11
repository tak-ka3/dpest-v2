import numpy as np
from typing import Dict, Tuple


def generate_patterns(n: int) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Generate adjacency input patterns of length ``n``.

    Returns a dictionary mapping pattern names to tuples ``(D, D')``
    consistent with the table in the tests.  ``D`` and ``D'`` are numpy
    arrays of integers.
    """
    ones = np.ones(n, dtype=int)
    patterns = {
        "one_above": (
            ones,
            np.concatenate(([2], np.ones(n - 1, dtype=int)))
        ),
        "one_below": (
            ones,
            np.concatenate(([0], np.ones(n - 1, dtype=int)))
        ),
        "one_above_rest_below": (
            ones,
            np.concatenate(([2], np.zeros(n - 1, dtype=int)))
        ),
        "one_below_rest_above": (
            ones,
            np.concatenate(([0], np.full(n - 1, 2, dtype=int)))
        ),
        "half_half": (
            ones,
            np.concatenate((
                np.zeros((n + 1) // 2, dtype=int),
                np.full(n - (n + 1) // 2, 2, dtype=int)
            ))
        ),
        "all_above_all_below": (
            ones,
            np.full(n, 2, dtype=int)
        ),
        "x_shape": (
            np.concatenate((
                np.ones(n // 2, dtype=int),
                np.zeros(n - n // 2, dtype=int)
            )),
            np.concatenate((
                np.zeros(n // 2, dtype=int),
                np.ones(n - n // 2, dtype=int)
            ))
        ),
    }
    return {k: (np.array(a), np.array(b)) for k, (a, b) in patterns.items()}
