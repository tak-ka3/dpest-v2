"""General-purpose helpers for sparse vector style mechanisms.

This module provides two reusable abstractions for analytic evaluations of
mechanisms that repeatedly compare noisy queries against a shared noisy
threshold:

* :class:`SharedThresholdGrid` discretizes an arbitrary ``Dist`` object into a
  convenient weighted grid while exposing utility methods such as the tail
  probability for another distribution.
* :class:`SharedThresholdIntegrator` evolves sequences of outcomes while
  keeping the dependency on the shared threshold explicit.  Users can
  implement Algorithm 1 style logic by writing small transition functions that
  emit :class:`ThresholdBranch` objects for each step of their mechanism.

The intent is that higher-level code describes the algorithm using the same
building blocks that appear in sparse vector pseudocode (sampling noises,
comparing against a threshold, aborting after a quota of TRUE answers, …)
while this module handles the bookkeeping required for the analytic
integration over the shared randomness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, MutableMapping, Tuple

import numpy as np

from ..core import Dist


def _discretize_distribution(dist: Dist) -> Tuple[np.ndarray, np.ndarray]:
    """Return support values and normalized weights for ``dist``."""

    values: list[float] = []
    weights: list[float] = []

    for value, weight in dist.atoms:
        if weight > 0.0:
            values.append(float(value))
            weights.append(float(weight))

    if dist.density:
        x_grid = dist.density.get("x")
        f_grid = dist.density.get("f")
        dx = dist.density.get("dx")
        if x_grid is not None and f_grid is not None and dx is not None:
            continuous_weights = np.asarray(f_grid, dtype=float) * float(dx)
            for value, weight in zip(np.asarray(x_grid, dtype=float), continuous_weights):
                if weight > 0.0:
                    values.append(float(value))
                    weights.append(float(weight))

    if not weights:
        raise ValueError("distribution has zero total mass")

    weight_arr = np.asarray(weights, dtype=float)
    value_arr = np.asarray(values, dtype=float)
    total = float(weight_arr.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("distribution has non-positive total mass")

    weight_arr /= total
    return value_arr, weight_arr


def _tail_probabilities(dist: Dist, thresholds: np.ndarray) -> np.ndarray:
    """Return ``P[X >= τ]`` for each ``τ`` in ``thresholds``."""

    thresholds = np.asarray(thresholds, dtype=float)
    result = np.zeros_like(thresholds, dtype=float)

    if dist.atoms:
        for value, weight in dist.atoms:
            if weight <= 0.0:
                continue
            result += float(weight) * (float(value) >= thresholds)

    if dist.density:
        x_grid = dist.density.get("x")
        f_grid = dist.density.get("f")
        dx = dist.density.get("dx")
        if x_grid is not None and f_grid is not None and dx is not None:
            x = np.asarray(x_grid, dtype=float)
            f = np.asarray(f_grid, dtype=float)
            dx_val = float(dx)
            cumulative = np.cumsum(f) * dx_val
            mass = cumulative[-1] if cumulative.size else 0.0
            if mass > 0.0:
                cdf = np.interp(thresholds, x, cumulative, left=0.0, right=mass)
                tail = mass - cdf
                result += tail

    return np.clip(result, 0.0, 1.0)


@dataclass(frozen=True)
class SharedThresholdGrid:
    """Finite grid representation of a shared noisy threshold."""

    values: np.ndarray
    weights: np.ndarray

    @classmethod
    def from_dist(cls, dist: Dist) -> "SharedThresholdGrid":
        values, weights = _discretize_distribution(dist)
        return cls(values=values, weights=weights)

    def ones(self) -> np.ndarray:
        """Return an array of ones matching the grid shape."""

        return np.ones_like(self.weights, dtype=float)

    def tail_probabilities(self, dist: Dist) -> np.ndarray:
        """Return ``P[dist >= τ]`` evaluated on the grid values ``τ``."""

        return _tail_probabilities(dist, self.values)


@dataclass
class SharedThresholdState:
    """State tracked during sequential integration."""

    prob_vec: np.ndarray
    payload: Any


@dataclass(frozen=True)
class ThresholdBranch:
    """Transition produced by a sequential step."""

    symbol: Any
    event_prob: np.ndarray
    payload: Any


TransitionFunc = Callable[
    [Tuple[Any, ...], SharedThresholdState, SharedThresholdGrid], Iterable[ThresholdBranch]
]


class SharedThresholdIntegrator:
    """Evolve sequences that depend on a shared discretized threshold."""

    def __init__(
        self,
        grid: SharedThresholdGrid,
        payload_merger: Callable[[Any, Any], Any] | None = None,
    ) -> None:
        self._grid = grid
        self._payload_merger = payload_merger or self._default_payload_merger

    @staticmethod
    def _default_payload_merger(existing: Any, new: Any) -> Any:
        if existing != new:
            raise ValueError("inconsistent payloads for identical output sequence")
        return existing

    @property
    def grid(self) -> SharedThresholdGrid:
        return self._grid

    def initial_states(self, payload: Any) -> MutableMapping[Tuple[Any, ...], SharedThresholdState]:
        """Create the initial state before any sequential steps."""

        return {(): SharedThresholdState(self._grid.ones(), payload)}

    def step(
        self,
        states: MutableMapping[Tuple[Any, ...], SharedThresholdState],
        transition: TransitionFunc,
    ) -> MutableMapping[Tuple[Any, ...], SharedThresholdState]:
        """Advance all states by applying ``transition`` once."""

        new_states: MutableMapping[Tuple[Any, ...], SharedThresholdState] = {}
        grid_shape = self._grid.weights.shape

        for seq, state in states.items():
            prob_vec = state.prob_vec
            if prob_vec.shape != grid_shape:
                raise ValueError("state probability vector shape is incompatible with grid")
            if not np.any(prob_vec > 0.0):
                continue

            for branch in transition(seq, state, self._grid):
                event_prob = np.asarray(branch.event_prob, dtype=float)
                if event_prob.shape != grid_shape:
                    raise ValueError("branch probability vector must match the grid shape")

                branch_prob_vec = prob_vec * np.clip(event_prob, 0.0, 1.0)
                if not np.any(branch_prob_vec > 0.0):
                    continue

                new_seq = seq + (branch.symbol,)
                existing = new_states.get(new_seq)
                if existing is None:
                    new_states[new_seq] = SharedThresholdState(branch_prob_vec, branch.payload)
                else:
                    existing.prob_vec += branch_prob_vec
                    existing.payload = self._payload_merger(existing.payload, branch.payload)

        return new_states

    def finalize(self, states: MutableMapping[Tuple[Any, ...], SharedThresholdState]) -> Dist:
        """Convert the accumulated sequences into a :class:`Dist`."""

        atoms = []
        for seq, state in states.items():
            probability = float(np.dot(state.prob_vec, self._grid.weights))
            if probability > 0.0:
                atoms.append((seq, probability))

        if not atoms:
            raise ValueError("no sequences with positive probability were generated")

        dist = Dist.from_atoms(atoms)
        dist.normalize()
        return dist

