"""
Test IntLib module.
All functions respect inclusion isotonicity.
"""

from numpy import nan, inf

from IntLib import (
    round_down,
    round_up,
    round_nearest,
    round_get,
)

from IntLib import (
    _pi_,
    _e_,
    _sqrt2_,
    _is_valid,
    _is_extended,
    _isempty,
    _has_zero,
    _contains,
    _add,
    _sub,
    _mul,
    _div_simple,
    _div,
)

T, F = True, False


def apply_cases(fn, cases):
    """Apply cases to fn."""
    for case in cases:
        assert fn(*case[0]) == case[1]


def check_inc_iso(fn, cases):
    """Check inclusion isotonicity.
    _x_ ⊆ _y_ ⇒ _x_ ∘ _z_ ⊆ _y_ ∘ _z_"""
    for case in cases:
        _x, x_, _y, y_, _z, z_ = case
        assert _contains(_y, y_, _x, x_)
        assert _contains(*fn(_y, y_, _z, z_), *fn(_x, x_, _z, z_))


def test_rounding():
    """directed rounding"""
    assert round_down() == 0
    assert round_get() == 256

    assert round_up() == 0
    assert round_get() == 512

    assert round_nearest() == 0
    assert round_get() == 0


def test_constants():
    """constants"""
    assert _pi_ == (3.141592653589793, 3.141592653589794)
    assert _e_ == (2.718281828459045, 2.7182818284590455)
    assert _sqrt2_ == (1.414213562373095, 1.4142135623730951)


def test_is_valid():
    """is valid"""
    cases = [
        # true
        ([1, 1], T),
        ([-1, 1], T),
        ([nan, nan], T),  # ∅
        ([-inf, inf], T),
        ([1, inf], T),
        ([-inf, 1], T),
        # false
        ([nan, 1], F),
        ([1, nan], F),
    ]
    apply_cases(_is_valid, cases)


def test_is_extended():
    """is_extended"""
    cases = [
        ([-1, 1], F),
        ([1, 1], F),
        ([-2, -1], F),
        ([0, 0], F),
        # extended intevals
        ([1, -1], T),
        ([-1, -2], T),
        ([2, 1], T),
    ]
    apply_cases(_is_extended, cases)


def test_isempty():
    """isempty ≡ ∅ ≡ [nan, nan]"""
    cases = [
        ([nan, nan], T),
        ([inf, inf], F),
        ([1, 1], F),
    ]
    apply_cases(_isempty, cases)


def test_has_zero():
    """Test _has_zero."""
    cases = [
        ([-1, 1], T),
        ([1, 1], F),
        ([-2, -1], F),
        ([0, 0], T),
        ([0, 1], T),
        ([nan, nan], F),  # 0 ∉ ∅
        ([1, inf], F),
        ([-1, inf], T),
        # extended inteval
        ([1, -1], F),
        ([-1, -2], T),
        ([2, 1], T),
    ]
    apply_cases(_has_zero, cases)


def test_contains():
    """Test in operator."""
    cases = [
        ([1, 1, 1, 1], T),
        ([1, 1, 0, 0], F),
        ([1, 1, 2, 2], F),
        ([1, 2, 1, 1], T),
        # ∅ = [nan, nan]
        ([nan, nan, 1, 1], F),  # 1 in ∅
        ([1, 1, nan, nan], T),  # ∅ in [1, 1]
        ([nan, nan, nan, nan], T),  # ∅ in ∅
        # s is extended inteval
        ([1, -1, 1, 1], T),  # 1 ∈ [-∞, -1] ∪ [1, ∞]
        ([1, -1, 1, 2], T),
        ([1, -1, 2, 3], T),
        ([1, -1, 0, 0], F),  # 0 ∉ [-∞, -1] ∪ [1, ∞]
        ([1, -1, -1, 1], F),  # [-1, 1] ∉ [-∞, -1] ∪ [1, ∞]
        # o is extended inteval
        ([1, 1, 1, -1], F),
        ([2, -2, 1, -1], F),
        ([1, -1, 2, -2], T),  # ([-∞, -2] ∪ [2, ∞]) ⊆ ([-∞, -1] ∪ [1, ∞])
        ([2, -2, 2, -2], T),
        ([1, -2, 2, -2], T),
        ([2, -2, 1, -2], F),
    ]
    apply_cases(_contains, cases)


def test_add():
    """Test _add."""
    cases = [
        # _s s_ _o o_
        ([1, 1, 1, 1], (2, 2)),
        ([1, 1, 1, 2], (2, 3)),
        # extended intervals
        ([-1, -2, -1, -1], (-2, -3)),
        ([1, 2, 2, 1], (3, 3)),
        ([2, 1, 4, 3], (-inf, inf)),
        ([10, 20, 2, 1], (2, 1)),  # TODO
    ]
    apply_cases(_add, cases)

    # empty set
    empty_set_cases = [
        # ∅
        ([nan, nan, 1, 2]),
        ([1, -1, nan, nan]),
    ]
    for case in empty_set_cases:
        assert _isempty(*_add(*case))

    # inclusion isotonicity
    cases = [
        # _x_     _y_    _z_
        [10, 20, 9, 21, 1, 2],
        [10, 20, 0, 30, 1, 2],
        [1, 1, 0, 2, 1, 2],
    ]
    check_inc_iso(_add, cases)


# TODO: Ratz96 Inclusion Isotone Extended Interval Arithmetic
# as applied to subtraction

# Check Subtraction
def test_sub():
    """Test _sub."""

    cases = [
        # _s s_ _o o_
        ([1, 1, 1, 1], (0, 0)),
        ([1, 1, 1, 2], (-1, 0)),
        ([1, 2, 3, 4], (-3, -1)),
        # extended intervals
        ([4, 3, 1, 2], (2, 2)),
        ([1, 2, 4, 3], (-2, -2)),
        ([2, 1, 4, 3], (-inf, inf)),
        ([10, 20, 2, 1], (21, 12)),  # TODO
    ]
    apply_cases(_sub, cases)

    # empty sets
    empty_set_cases = [
        # ∅
        ([nan, nan, 1, 2]),
        ([1, -1, nan, nan]),
    ]
    for case in empty_set_cases:
        assert _isempty(*_sub(*case))

    # inclusion isotonicity
    # _x_ ⊆ _y_ ⇒ _x_ - _z_ ⊆ _y_ - _z_
    cases = [
        # _x_     _y_    _z_
        [10, 20, 9, 21, 1, 2],
        [10, 20, 0, 30, 1, 2],
        [1, 1, 0, 2, 1, 2],
    ]
    check_inc_iso(_sub, cases)


# TODO: verify inclusion isotonicity of all operations
# _x_ ⊆ _y_ ⇒ f(_x_) ⊆ f(_y_)
# _x_ ⊆ _y_ ⇒ _x_ ∘ _z_ ⊆ _y_ ∘ _z_

# TODO: Generative testing. From set of all intervals.


# TODO: mul
def test_mul():
    """Test _mul."""
    cases = [
        # _s s_ _o o_
        ([1, 1, 1, 1], (1, 1)),
        ([1, 1, 1, 2], (1, 2)),
        ([1, 2, 3, 4], (3, 8)),
        # extended intervals
        # TODO REALLY?
        ([4, 3, 1, 2], (3, 8)),
        ([1, 2, 4, 3], (3, 8)),
        ([2, 1, 4, 3], (-inf, inf)),
    ]
    apply_cases(_mul, cases)

    # empty sets
    empty_set_cases = [
        # ∅
        ([nan, nan, 1, 2]),
        ([1, -1, nan, nan]),
    ]
    for case in empty_set_cases:
        assert _isempty(*_mul(*case))

    # inclusion isotonicity
    # _x_ ⊆ _y_ ⇒ _x_ * _z_ ⊆ _y_ * _z_
    cases = [
        # _x_     _y_    _z_
        [10, 20, 9, 21, 1, 2],
        [10, 20, 0, 30, 1, 2],
        [1, 1, 0, 2, 1, 2],
    ]
    check_inc_iso(_mul, cases)


# TODO: div_simple

# TODO: div

# TODO: mig, mag, abs, rad, mid, dist, exp, sqrt, log, pow, atan, sin, cos ...
