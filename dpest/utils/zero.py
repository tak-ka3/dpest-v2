import numpy as np

class ZeroNoisePrng:
    """Deterministic pseudo-random generator returning zeros.

    This is a lightweight replacement for the dpsniper ZeroNoisePrng used in
    some mechanism implementations. It provides ``rand``-like methods that
    always yield deterministic zero values, which is useful for testing
    purposes or when no randomness is desired.
    """

    def random(self, size=None):
        return np.zeros(size)

    def randint(self, low, high=None, size=None):
        if high is None:
            high = low
            low = 0
        return np.full(size, low)
