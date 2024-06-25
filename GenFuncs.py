"""Generic mathematical functions. f(x) where x ∈ ℝ, or x ∈ IR or x ∈ AutoDiff"""
import numpy as np


def sin(x):
    """Generic sin(x) function."""
    if isinstance(x, (int, float, np.ndarray)):
        return np.sin(x)
    return x.sin()


def cos(x):
    """Generic cos(x) function."""
    if isinstance(x, (int, float, np.ndarray)):
        return np.cos(x)
    return x.cos()


def exp(x):
    """Generic e^x function."""
    if isinstance(x, (int, float, np.ndarray)):
        return np.exp(x)
    return x.exp()


def log(x):
    """Generic logₑ(x) function."""
    if isinstance(x, (int, float, np.ndarray)):
        return np.log(x)
    return x.log()


def sqrt(x):
    """Generic √x function."""
    if isinstance(x, (int, float, np.ndarray)):
        return np.sqrt(x)
    return x.sqrt()


def atan(x):
    """Generic arctan(x) function."""
    if isinstance(x, (int, float, np.ndarray)):
        return np.arctan(x)
    return x.atan()


def pwr(x, n: int):
    """Generic x^n function. n ∈ ℤ."""
    if isinstance(x, (int, float, np.ndarray)):
        return x**n
    return x.pow(n)


def mag(x):
    """Magnitude of x."""
    return x.mag


def mig(x):
    """Mignitude of x."""
    return x.mig


def rad(x):
    """Radius of x."""
    return x.rad


def mid(x):
    """Midpoint of x."""
    return x.mid
