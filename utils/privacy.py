"""Privacy-related utility functions."""

from __future__ import annotations

import numpy as np

from core import Dist


def estimate_privacy_loss(P: Dist, Q: Dist) -> float:
    """Estimate privacy loss ε between two distributions.

    This function computes an upper bound on the privacy loss ε by comparing
    probability mass or density ratios between distributions ``P`` and ``Q``.
    It supports both discrete distributions (via ``atoms``) and continuous
    distributions represented with sampled densities.
    """
    if P.atoms and Q.atoms:
        eps = 0.0
        values = {v for v, _ in P.atoms} | {v for v, _ in Q.atoms}
        for v in values:
            p_prob = next((p for val, p in P.atoms if val == v), 0.0)
            q_prob = next((p for val, p in Q.atoms if val == v), 0.0)
            if p_prob > 0 and q_prob > 0:
                ratio = max(p_prob / q_prob, q_prob / p_prob)
                eps = max(eps, np.log(ratio))
        return eps

    if P.density and Q.density and "x" in P.density and "x" in Q.density:
        p_x, p_f = P.density["x"], P.density["f"]
        q_x, q_f = Q.density["x"], Q.density["f"]
        left = max(min(p_x), min(q_x))
        right = min(max(p_x), max(q_x))
        if right <= left:
            return float("inf")
        unified_x = np.linspace(left, right, 1000)
        p_interp = np.interp(unified_x, p_x, p_f, left=1e-10, right=1e-10)
        q_interp = np.interp(unified_x, q_x, q_f, left=1e-10, right=1e-10)
        ratios = []
        for p_val, q_val in zip(p_interp, q_interp):
            if p_val > 1e-10 and q_val > 1e-10:
                ratios.append(p_val / q_val)
                ratios.append(q_val / p_val)
        if ratios:
            return float(np.log(max(ratios)))
        return float("inf")

    return float("inf")
