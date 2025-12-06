"""
SVT34Parallel - Running SVT3 and SVT4 in parallel

This algorithm runs both SVT3 and SVT4 on the same input and returns
both outputs concatenated together.
"""

from typing import List

from ..core import Dist
from .svt3 import svt3
from .svt4 import svt4
from .registry import auto_dist


@auto_dist()
def svt34_parallel(queries: List[Dist], eps: float = 0.1, t: float = 1.0, c: int = 2) -> List[Dist]:
    """
    Run SVT3 and SVT4 in parallel and concatenate outputs.

    Args:
        queries: Query result distributions
        eps: Privacy parameter (used by both SVT3 and SVT4)
        t: Threshold value
        c: Cutoff count

    Returns:
        Concatenated outputs from SVT3 and SVT4
        First len(queries) elements are from SVT3, next len(queries) are from SVT4
    """
    # Run SVT3 (returns List[Dist])
    svt3_results = svt3(queries, eps=eps, t=t, c=c)

    # Run SVT4 (returns List[Dist])
    svt4_results = svt4(queries, eps=eps, t=t, c=c)

    # Concatenate results
    return svt3_results + svt4_results
