"""Shared helpers for algorithm implementations."""

from __future__ import annotations

from typing import List

from ..core import Dist


def expect_single_value(dists: List[Dist], name: str) -> float:
    """Extract a deterministic scalar from a one-element distribution list."""
    if not dists:
        raise ValueError(f"{name} requires at least one input value")
    dist = dists[0]
    if not dist.atoms:
        raise ValueError(f"{name} expects deterministic input, got distribution without atoms")
    return float(dist.atoms[0][0])
