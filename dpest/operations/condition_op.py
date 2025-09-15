"""
Conditional and comparison operations.

This module provides operations for handling conditional branches and
comparisons between random variables within the distribution framework.
"""

from typing import Union
import numpy as np
from scipy import interpolate

from ..core import Dist, merge_atoms, Node
from .operations import Add, Affine


class Compare:
    """Comparison operations returning indicator distributions."""

    @staticmethod
    def geq(x_dist: Dist, y: Union[Dist, float]) -> Dist:
        """Return distribution of the indicator [X >= Y].

        Args:
            x_dist: distribution of variable X.
            y: distribution or constant for Y.

        Returns:
            Dist over {0,1} representing the event X >= Y.
        """
        # If y is a distribution, reduce to comparison with zero by taking
        # the difference X - Y.
        if isinstance(y, Dist):
            diff = Add.apply(x_dist, Affine.apply(y, -1.0, 0.0))
            return Compare.geq(diff, 0.0)

        threshold = float(y)
        prob = 0.0

        # Discrete part
        if x_dist.atoms:
            for val, weight in x_dist.atoms:
                if val >= threshold:
                    prob += weight

        # Continuous part
        if x_dist.density and 'x' in x_dist.density:
            x_grid = x_dist.density['x']
            f_grid = x_dist.density['f']
            mask = x_grid >= threshold
            if np.any(mask):
                prob += np.trapz(f_grid[mask], x_grid[mask])

        prob = min(max(prob, 0.0), 1.0)
        deps = set(x_dist.dependencies)
        inputs = [getattr(x_dist, 'node', None)]
        if isinstance(y, Dist):
            deps |= y.dependencies
            inputs.append(getattr(y, 'node', None))
        inputs = [n for n in inputs if n is not None]
        node = Node(op='CompareGEQ', inputs=inputs, dependencies=set(deps))
        return Dist.from_atoms([(1.0, prob), (0.0, 1.0 - prob)],
                              dependencies=set(deps), node=node)


class Condition:
    """Conditional mixture operation.

    Given a condition distribution over {0,1} and two branch distributions,
    compute the overall mixture distribution.
    """

    @staticmethod
    def apply(cond_dist: Dist, true_dist: Dist, false_dist: Dist) -> Dist:
        """Return mixture P(E)*true + P(~E)*false.

        Args:
            cond_dist: distribution over {0,1} representing event E.
            true_dist: distribution when event is true.
            false_dist: distribution when event is false.
        """
        p_true = 0.0
        for val, weight in cond_dist.atoms:
            if val >= 0.5:
                p_true += weight
        p_true = min(max(p_true, 0.0), 1.0)
        p_false = 1.0 - p_true

        result_atoms = []
        if true_dist.atoms:
            result_atoms.extend((v, w * p_true) for v, w in true_dist.atoms)
        if false_dist.atoms:
            result_atoms.extend((v, w * p_false) for v, w in false_dist.atoms)

        result_density = {}
        if true_dist.density or false_dist.density:
            x_true = true_dist.density.get('x', np.array([])) if true_dist.density else np.array([])
            f_true = true_dist.density.get('f', np.array([])) if true_dist.density else np.array([])
            dx_true = true_dist.density.get('dx') if true_dist.density else None
            x_false = false_dist.density.get('x', np.array([])) if false_dist.density else np.array([])
            f_false = false_dist.density.get('f', np.array([])) if false_dist.density else np.array([])
            dx_false = false_dist.density.get('dx') if false_dist.density else None

            if x_true.size > 0 and x_false.size > 0:
                dx = min(dx_true, dx_false)
                min_x = min(x_true[0], x_false[0])
                max_x = max(x_true[-1], x_false[-1])
                n_points = int((max_x - min_x) / dx) + 1
                x_grid = np.linspace(min_x, max_x, n_points)
                f_true_interp = interpolate.interp1d(x_true, f_true, bounds_error=False, fill_value=0.0)
                f_false_interp = interpolate.interp1d(x_false, f_false, bounds_error=False, fill_value=0.0)
                f_mix = p_true * f_true_interp(x_grid) + p_false * f_false_interp(x_grid)
                result_density = {'x': x_grid, 'f': f_mix, 'dx': dx}
            elif x_true.size > 0:
                result_density = {'x': x_true, 'f': p_true * f_true, 'dx': dx_true}
            elif x_false.size > 0:
                result_density = {'x': x_false, 'f': p_false * f_false, 'dx': dx_false}

        result_atoms = merge_atoms(result_atoms)
        deps = cond_dist.dependencies | true_dist.dependencies | false_dist.dependencies
        inputs = [getattr(cond_dist, 'node', None),
                  getattr(true_dist, 'node', None),
                  getattr(false_dist, 'node', None)]
        inputs = [n for n in inputs if n is not None]
        node = Node(op='Condition', inputs=inputs, dependencies=set(deps))
        result = Dist(atoms=result_atoms,
                      density=result_density if result_density else None,
                      dependencies=set(deps),
                      node=node)
        result.normalize()
        return result


def compare_geq(x_dist: Dist, y: Union[Dist, float]) -> Dist:
    """Convenience function for Compare.geq."""
    return Compare.geq(x_dist, y)


def condition_mixture(cond_dist: Dist, true_dist: Dist, false_dist: Dist) -> Dist:
    """Convenience function for Condition.apply."""
    return Condition.apply(cond_dist, true_dist, false_dist)
