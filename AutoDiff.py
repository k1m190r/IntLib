"""Automatic differentiation."""

import numpy as np


class AutoDiff:
    """AutoDiff"""

    def __init__(self, x, dx=1.0):
        self.x = x
        self.dx = dx

    def __repr__(self):
        return f"dual({self.x}, {self.dx})".replace("0", "₀")

    def __iter__(self):
        s = self
        return iter((s.x, s.dx))

    def __eq__(self, other):
        [x, dx], [ox, odx] = self, _ensure_AD(other)
        return (x == ox) and (dx == odx)

    def __neq__(self, other):
        [x, dx], [ox, odx] = self, _ensure_AD(other)
        return (x != ox) or (dx != odx)

    def __neg__(self):
        AD, [x, dx] = AutoDiff, self
        return AD(-x, -dx)

    def __add__(self, other):
        AD, [x, dx], [ox, odx] = AutoDiff, self, _ensure_AD(other)
        return AD(x + ox, dx + odx)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        AD, [x, dx], [ox, odx] = AutoDiff, self, _ensure_AD(other)
        return AD(x - ox, dx - odx)

    def __rsub__(self, other):
        return (-self) + other

    def __mul__(self, other):
        AD, [x, dx], [ox, odx] = AutoDiff, self, _ensure_AD(other)
        return AD(x * ox, x * odx + dx * ox)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        AD, [x, dx], [ox, odx] = AutoDiff, self, _ensure_AD(other)
        return AD(x / ox, (dx * ox - x * odx) / (ox * ox))

    def __rtruediv__(self, other):
        s, o = self, _ensure_AD(other)
        return o / s

    def __pow__(self, n):
        AD, [x, dx] = AutoDiff, self
        return AD(x**n, n * x ** (n - 1) * dx)

    def __abs__(self):
        AD, [x, dx] = AutoDiff, self
        return AD(abs(x), dx * np.sign(x))

    def exp(self):
        """e^x"""
        AD, [x, dx], exp = AutoDiff, self, np.exp
        return AD(exp(x), dx * exp(x))

    def log(self):
        """logₑ(x)"""
        AD, [x, dx], log = AutoDiff, self, np.log
        return AD(log(x), dx / x)

    def sin(self):
        """sin(x)"""
        AD, [x, dx] = AutoDiff, self
        sin, cos = np.sin, np.cos
        return AD(sin(x), dx * cos(x))

    def cos(self):
        """cos(x)"""
        AD, [x, dx] = AutoDiff, self
        sin, cos = np.sin, np.cos
        return AD(cos(x), -dx * sin(x))

    def sqrt(self):
        """√x"""
        AD, [x, dx] = AutoDiff, self
        sqrt = np.sqrt
        return AD(sqrt(x), dx / (2 * sqrt(x)))

    def atan(self):
        """arctan(x)"""
        AD, [x, dx] = AutoDiff, self
        atan = np.arctan
        return AD(atan(x), dx / (1 + x**2))


def _ensure_AD(x, dx=0.0):
    """Ensure x is an AutoDiff object."""
    AD = AutoDiff
    return x if isinstance(x, AD) else AD(x, dx)
