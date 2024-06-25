"""Inverval arithmetic."""

from ctypes import cdll, c_int
from numba import njit
import numpy as np
from numpy import nan, inf, isnan

# Windows specific directed rounding methods

fegetround = cdll.ucrtbased.fegetround
fegetround.restype = c_int
fegetround.argtypes = []

fesetround = cdll.ucrtbased.fesetround
fesetround.restype = c_int
fesetround.argtypes = [c_int]


@njit
def round_down():
    """Set the rounding mode → -∞."""
    FE_DOWNWARD = 0x00000100
    return fesetround(FE_DOWNWARD)


@njit
def round_up():
    """Set the rounding mode → +∞."""
    FE_UPWARD = 0x00000200
    return fesetround(FE_UPWARD)


@njit
def round_nearest():
    """Set the rounding mode → nearest."""
    FE_TONEAREST = 0x00000000
    return fesetround(FE_TONEAREST)


@njit
def round_get():
    """Get the current rounding mode."""
    return fegetround()


def _make_pi_e_sqrt2():
    """make global interval values for π, e, and √2"""
    _r, r_, r = round_down, round_up, round_nearest

    _r()
    _pi = np.arctan(1) * 4
    _e = np.exp(1)
    _sq2 = np.sqrt(2)

    r_()
    pi_ = np.arctan(1) * 4 + 1 - 1
    e_ = np.exp(1) + 1 - 1
    sq2_ = np.sqrt(2)

    r()
    return (_pi, pi_), (_e, e_), (_sq2, sq2_)


# π, e, √2
_pi_, _e_, _sqrt2_ = _make_pi_e_sqrt2()


########################################################################################
# Inteval numba.njitted functions
########################################################################################


@njit
def _is_valid(_x, x_):
    """Valid if both NAN or both not NAN."""
    return isnan(_x) == isnan(x_)


@njit
def _is_extended(_x, x_):
    """_x > x_ ⟹ exteneded _x_ = [-∞, x_] ∪ [_x, +∞]"""
    return _x > x_


@njit
def _isempty(_x, x_):
    """[_x, x_] = ∅ ≡ [nan, nan]"""
    return isnan(_x) and isnan(x_)


@njit
def _has_zero(_x, x_):
    """0 ∈ _x_ = [_x, x_]"""
    em, ex = _isempty, _is_extended

    if em(_x, x_):
        return False
    if ex(_x, x_):
        # (0 ∈ [-∞, x_]) ∨ (0 ∈ [_x, +∞])
        return (0 <= x_) or (_x <= 0)
    # 0 ∈ [_x, x_]
    return (_x <= 0) and (0 <= x_)


@njit
def _contains(_s, s_, _o, o_):
    """_o_ ⊆ _s_ ≡ _s ≤ _o ∧ o_ ≤ s_"""
    em, ex, c = _isempty, _is_extended, _contains
    if em(_o, o_):  # ∅ ⊆ _s_ ≡ True
        return True
    if em(_s, s_):  # _o_ ⊆ ∅ ≡ False, if _o_ ≠ ∅
        return False
    if ex(_s, s_) and ex(_o, o_):
        return c(_s, s_, -inf, o_) and c(_s, s_, _o, +inf)
    if ex(_s, s_):
        return c(-inf, s_, _o, o_) or c(_s, +inf, _o, o_)
    if ex(_o, o_):
        return c(_s, s_, -inf, o_) or c(_s, s_, _o, +inf)
    return (_s <= _o) and (o_ <= s_)


########################################################################################
# Arithmetic operations
########################################################################################


@njit
def _add(_a, a_, _b, b_):
    """[∇(_a + _b), ∆(a_ + b_)]"""
    _r, r_, r = round_down, round_up, round_nearest
    em, ex = _isempty, _is_extended

    # _a_ + ∅  = ∅ + _b_ = ∅
    if em(_a, a_) or em(_b, b_):
        return (nan, nan)

    # extended intervals have -∞ and +∞ ⟹ [-∞, +∞]
    if ex(_a, a_) and ex(_b, b_):
        return (-inf, inf)
    _r()
    _x = _a + _b
    r_()
    x_ = a_ + b_
    r()
    return (_x, x_)


@njit
def _sub(_a, a_, _b, b_):
    """[∇(_a - _b), ∆(a_ - b_)]"""
    _r, r_, r = round_down, round_up, round_nearest
    em, ex = _isempty, _is_extended

    # _a_ - ∅  = ∅ - _b_ = ∅
    if em(_a, a_) or em(_b, b_):
        return (nan, nan)

    # extended intervals have -∞ and +∞ ⟹ [-∞, +∞]
    if ex(_a, a_) and ex(_b, b_):
        return (-inf, inf)

    _r()
    _x = _a - b_
    r_()
    x_ = a_ - _b
    r()
    return (_x, x_)


@njit
def _mul(_a, a_, _b, b_):
    """[
        ∇⌊(_a _b, _a b_, a_ _b, a_ b_),
        ∆⌈(_a _b, _a b_, a_ _b, a_ b_)
    ]"""
    _r, r_, r = round_down, round_up, round_nearest
    em, ex = _isempty, _is_extended

    # _a_ × ∅  = ∅ × _b_ = ∅
    if em(_a, a_) or em(_b, b_):
        return (nan, nan)

    # extended intervals have -∞ and +∞ ⟹ [-∞, +∞]
    if ex(_a, a_) and ex(_b, b_):
        return (-inf, inf)

    _r()
    _x = min(_a * _b, _a * b_, a_ * _b, a_ * b_)
    r_()
    x_ = max(_a * _b + 1 - 1, _a * b_ + 1 - 1, a_ * _b + 1 - 1, a_ * b_ + 1 - 1)
    r()
    return (_x, x_)


@njit
def _div_simple(_a, a_, _b, b_):
    """[
        ∇⌊(_a / _b, _a / b_, a_ / _b, a_ / b_),
        ∆⌈(_a / _b, _a / b_, a_ / _b, a_ / b_)
    ]"""
    if _isempty(_a, a_) or _isempty(_b, b_):
        return (nan, nan)
    _r, r_, r = round_down, round_up, round_nearest

    _r()
    _x = min(_a / _b, _a / b_, a_ / _b, a_ / b_)
    r_()
    x_ = max(_a / _b + 1 - 1, _a / b_ + 1 - 1, a_ / _b + 1 - 1, a_ / b_ + 1 - 1)
    r()
    return (_x, x_)


@njit
def _div(_a, a_, _b, b_):
    """Extended division."""
    if _isempty(_a, a_) or _isempty(_b, b_):
        return (nan, nan)
    _r, r_, r = round_down, round_up, round_nearest

    _x, x_ = (-inf, +inf)

    # a × [1/b_, 1/_b] if 0 ∉ b
    if not _has_zero(_b, b_):
        _x, x_ = _div_simple(_a, a_, _b, b_)

    # [-∞, +∞] if 0 ∈ a and 0 ∈ b
    if _has_zero(_a, a_) and _has_zero(_b, b_):
        pass  # _x, x_ = (-inf, +inf)

    # [a_/_b, +∞] if a_ < 0 and _b < b_ = 0
    if a_ < 0 and _b < b_ and b_ == 0:
        _r()
        _x = a_ / _b

    # [a_/_b, a_/b_] if a_ < 0 and _b < 0 < b_
    # extended
    if a_ < 0 and _b < 0 and 0 < b_:
        _r()
        _x = a_ / _b
        r_()
        x_ = a_ / b_

    # [-∞, a_/b_] if a_ < 0 and 0 = _b < b_
    if a_ < 0 and 0 == _b and _b < b_:
        r_()
        x_ = a_ / b_

    # [-∞, _a/_b] if 0 < _a and _b < b_ = 0
    if 0 < _a and _b < b_ and b_ == 0:
        r_()
        x_ = _a / _b

    # [_a/b_, _a/_b] if 0 < _a and _b < 0 < b_
    # extended
    if 0 < _a and _b < 0 and 0 < b_:
        _r()
        _x = _a / b_
        r_()
        x_ = _a / _b

    # [_a/b_, +∞] if 0 < _a and 0 = _b < b_
    if 0 < _a and 0 == _b and _b < b_:
        _r()
        _x = _a / b_

    # ∅ if 0 ∉ a and b = [0, 0]
    if not _has_zero(_a, a_) and _b == 0 and b_ == 0:
        _x, x_ = (nan, nan)  # ∅

    r()
    return (_x, x_)


########################################################################################
# Mignitude, Magnitude, Absolute, Radius, Midpoint
########################################################################################


@njit
def _mig(_a, a_):
    """Mignitude:
    ⌊{|a|: a ∈ _a_} ≡ [0 ∈ _a_]⋅0 + [0 ∉ _a_]⋅⌊(|_a|, |a_|)"""
    if _has_zero(_a, a_):
        return 0.0
    _r, r_, r = round_down, round_up, round_nearest
    _r()
    _x = abs(_a)
    r_()
    x_ = abs(a_)
    r()
    return min(_x, x_)


@njit
def _mag(_a, a_):
    """Magnitude:
    ⌈{|a|: a ∈ _a_} ≡ ⌈(|_a|, |a_|)
    """
    _r, r_, r = round_down, round_up, round_nearest
    _r()
    _x = abs(_a)
    r_()
    x_ = abs(a_)
    r()
    return max(_x, x_)


@njit
def _abs(_a, a_):
    """abs(_a_)=|_a_|={|a|: a ∈ _a_}=[mig(_a_), mag(_a_)]"""
    return (_mig(_a, a_), _mag(_a, a_))


@njit
def _rad(_a, a_):
    """Radius: ⅟₂(a_ - _a)"""
    return 0.5 * (a_ - _a)


@njit
def _mid(_a, a_):
    """Mid-point: ⅟₂(a_ + _a)"""
    return 0.5 * (a_ + _a)


########################################################################################
# Hausdorff distance, Exponential, Square root, Logarithm, Power
########################################################################################


@njit
def _dist(_a, a_, _b, b_):
    """Hausdorff distance between two intervals.
    ⌈(|_a - _b|, |a_ - b_|)"""
    return max(abs(_a - _b), abs(a_ - b_))


@njit
def _exp(_a, a_):
    """Interval exponential."""
    _r, r_, r = round_down, round_up, round_nearest
    exp = np.exp
    _r()
    _x = exp(_a)
    r_()
    x_ = exp(a_)
    r()
    return (_x, x_)


@njit
def _sqrt(_a, a_):
    """Interval square root."""
    _r, r_, r = round_down, round_up, round_nearest
    sqrt = np.sqrt
    if 0 <= _a:
        _r()
        _x = sqrt(_a)
        r_()
        x_ = sqrt(a_)
        r()
        return (_x, x_)
    raise NotImplementedError("extended sqrt")


@njit
def _log(_a, a_):
    """Interval natural logarithm."""
    _r, r_, r = round_down, round_up, round_nearest
    log = np.log
    if 0 < _a:
        _r()
        _x = log(_a)
        r_()
        x_ = log(a_)
        r()
        return (_x, x_)
    raise NotImplementedError("extended log")


@njit
def _pow(_a, a_, n: int):
    """Interval integer power."""
    _r, r_, r = round_down, round_up, round_nearest
    _x, x_ = [1, 1]  # default for n == 0

    # n ∈ ℤ⁺ is odd
    if ((n > 1)) and (n % 2) == 1:
        _r()
        _x = _a**n
        r_()
        x_ = a_**n

    # n ∈ ℤ⁺ is even
    if ((n > 1)) and (n % 2) == 0:
        _r()
        _x = _mig(_a, a_) ** n
        r_()
        x_ = _mag(_a, a_) ** n

    # n ∈ ℤ⁻ ∧ 0 ∈ x
    if (n < 0) and (not _has_zero(_a, a_)):
        _r()
        _x = (1 / a_) ** (-n)
        r_()
        x_ = (1 / _a) ** (-n)

    r()
    return (_x, x_)


########################################################################################
# Trigonometric functions
########################################################################################


@njit
def _sin(_a, a_):
    """Interval sine."""
    _r, r_, r = round_down, round_up, round_nearest
    sin, cos = np.sin, np.cos
    _pi, _ = _pi_

    r_()
    d = a_ - _a
    if d != d or d >= 2.0 * _pi:
        return (-1.0, +1.0)
    _r()
    _x = min(sin(_a), sin(a_))
    r_()
    x_ = max(sin(_a), sin(a_))
    r()
    if cos(_a) <= 0 <= cos(a_):
        return (-1.0, x_)
    if cos(_a) >= 0 >= cos(a_):
        return (_x, +1.0)
    if d >= _pi:
        return (_x, x_)
    return (_x, x_)


@njit
def _cos(_a, a_):
    """sin(x + pi / 2)"""
    _pi, pi_ = _pi_
    _b, b_ = _div(_pi, pi_, 2, 2)  # pi / 2
    _a, a_ = _add(_a, a_, _b, b_)  # x + pi / 2
    return _sin(_a, a_)  # sin(x + pi / 2)


@njit
def _atan(_a, a_):
    """Interval arctangent."""
    _r, r_, r = round_down, round_up, round_nearest
    atan = np.arctan
    _r()
    _x = atan(_a)
    r_()
    x_ = atan(a_)
    r()
    return (_x, x_)


########################################################################################
# Interval Class
########################################################################################


class Interval:
    """
    [_x, x_] ≡ [inf, sup].
    ∅ ≡ [nan, nan] ≡ isempty.
    [x, x] ≡ is_thin ≡ real.
    _x > x_ ≡ is_extended.
    """

    def __init__(self, _x=nan, x_=nan):
        s = self
        s.isempty = _isempty(_x, x_)  # ∅
        s.is_extended = _x > x_  # extended
        if isnan(x_):  # x=x, [x, x] = thin ≡ real
            x_ = _x
        assert _is_valid(_x, x_), f"Invalid Interval: {[_x, x_]}"
        self.is_thin = _x == x_
        self._x = _x
        self.x_ = x_

    def __repr__(self) -> str:
        is_ext = self.is_extended
        _x, x_ = self._x, self.x_
        # r = f"[{_x:+17.17f}, {x_:+17.17f}]".replace("0", "₀").replace("inf", "∞")
        r = (
            (
                f"[-∞, {x_:+17.17f}]∪[{_x:+17.17f}, +∞]"
                if is_ext
                else f"[{_x:+17.17f}, {x_:+17.17f}]"
            )
            .replace("0", "₀")
            .replace("inf", "∞")
        )
        # r = (
        #     f"[-∞, {x_:+g}]∪[{_x:+g}, +∞]"
        #     if is_ext
        #     else f"[{_x:+g}, {x_:+g}]"
        # ).replace("0", "₀").replace("inf", "∞")
        return "∅" if self.isempty else r

    def __iter__(self):
        s = self
        return iter((s._x, s.x_))

    @property
    def has_zero(self):
        """0 ∈ [_x, x_]"""
        return _has_zero(*self)

    @property
    def mig(self):
        """Mignitude of [_x, x_]"""
        return _mig(*self)

    @property
    def mag(self):
        """Magnitude of [_x, x_]"""
        return _mag(*self)

    @property
    def rad(self):
        """Radius of [_x, x_]"""
        return _rad(*self)

    @property
    def mid(self):
        """Midpoint of [_x, x_]"""
        return _mid(*self)

    def __eq__(self, other):
        [_s, s_], [_o, o_] = self, _ensure_interval(other)
        return (_s == _o) and (s_ == o_)

    def __neq__(self, other):
        return not self == other

    def __contains__(self, other):
        """_o_ ⊆ _s_ ≡ _s ≤ _o ∧ o_ ≤ s_"""
        [_s, s_], o = self, _ensure_interval(other)
        if o.isempty:
            return True
        [_o, o_] = o
        # return (_s <= _o) and (o_ <= s_)
        return _contains(_s, s_, _o, o_)

    def is_inclusion(self, other):
        """_o_ ⊂ _s_ ≡ _s < _o ∧ o_ < s_"""
        [_s, s_], o = self, _ensure_interval(other)
        if o.isempty:
            return True
        [_o, o_] = o
        return (_s < _o) and (o_ < s_)

    def __and__(self, other):
        """_s_ ∩ _o_"""
        I, s, o = Interval, self, _ensure_interval(other)
        [_s, s_], [_o, o_] = s, o
        return (
            I()
            if ((s_ < _o) or (o_ < _s) or (s.isempty or o.isempty))
            else I(max(_s, _o), min(s_, o_))
        )

    def __or__(self, other):
        """Hull: _s_ ⊔ _o_"""
        I, s, o = Interval, self, _ensure_interval(other)
        [_s, s_], [_o, o_] = s, o
        return (
            I(min(_s, _o), max(s_, o_))
            if not (s.isempty or o.isempty)
            else (o if s.isempty else s)
        )

    def dist(self, other):
        """Distance between _s_ and _o_."""
        [_s, s_], [_o, o_] = self, _ensure_interval(other)
        return _dist(_s, s_, _o, o_)

    def __neg__(self):
        """-[_x, x_]"""
        I, [_x, x_] = Interval, self
        return I(-x_, -_x)

    def __add__(self, other):
        I, s, o = Interval, self, _ensure_interval(other)
        return I(*_add(s._x, s.x_, o._x, o.x_))

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        I, [_s, s_], [_o, o_] = Interval, self, _ensure_interval(other)
        return I(*_sub(_s, s_, _o, o_))

    def __rsub__(self, other):
        return (-self) + other

    def __mul__(self, other):
        I, [_s, s_], [_o, o_] = Interval, self, _ensure_interval(other)
        return I(*_mul(_s, s_, _o, o_))

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        I, [_s, s_], [_o, o_] = Interval, self, _ensure_interval(other)
        return I(*_div(_s, s_, _o, o_))

    def __rtruediv__(self, other):
        s, o = self, _ensure_interval(other)
        return o / s

    def __le__(self, other):
        [_s, s_], [_o, o_] = self, _ensure_interval(other)
        return (_s <= _o) and (s_ <= o_)

    def __lt__(self, other):
        [_s, s_], [_o, o_] = self, _ensure_interval(other)
        return (_s < _o) and (s_ < o_)

    def __ge__(self, other):
        [_s, s_], [_o, o_] = self, _ensure_interval(other)
        return (_s >= _o) and (s_ >= o_)

    def __gt__(self, other):
        [_s, s_], [_o, o_] = self, _ensure_interval(other)
        return (_s > _o) and (s_ > o_)

    def __abs__(self):
        return Interval(*_abs(*self))

    def exp(self):
        """e^[_x, x_]"""
        return Interval(*_exp(*self))

    def sqrt(self):
        """√[_x, x_]"""
        return Interval(*_sqrt(*self))

    def log(self):
        """logₑ[_x, x_]"""
        return Interval(*_log(*self))

    def atan(self):
        """arctan[_x, x_]"""
        return Interval(*_atan(*self))

    def __pow__(self, n: int):
        """[_x, x_]^n, n ∈ ℤ"""
        _s, s_ = self
        return Interval(*_pow(_s, s_, n))

    def sin(self):
        """sin[_x, x_]"""
        return Interval(*_sin(*self))

    def cos(self):
        """cos[_x, x_]"""
        return Interval(*_cos(*self))


########################################################################################


def _ensure_interval(other) -> Interval:
    """Ensure other is an Interval."""
    I, o = Interval, other
    if not isinstance(o, I):
        o = I(o)
    return o
