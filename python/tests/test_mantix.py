import pytest
import mantix
from mantix import Float, f32, f64, ROUND_NEAREST_EVEN

def test_basic_construction():
    a = Float(42.0)
    assert float(a) == 42.0
    assert a.precision == 53
    assert not a.negative

    b = Float(-100.5, precision=128)
    assert float(b) == -100.5
    assert b.precision == 128
    assert b.negative

    c = Float(0)
    assert c.is_zero()
    assert float(c) == 0.0

def test_f32_helpers():
    x = f32(1.5)
    assert x.to_f32() == 1.5
    assert x.precision == 24

    y = f32(-0.125)
    assert y.to_f32() == -0.125
    assert y.negative

def test_string_conversion():
    a = Float("3.141592653589793")
    assert abs(float(a) - 3.141592653589793) < 1e-15

def test_arithmetic_ops():
    a = Float(1.5)
    b = Float(2.5)

    assert float(a + b) == 4.0
    assert float(a - b) == -1.0
    assert float(a * b) == 3.75
    assert float(-a) == -1.5
    assert float(abs(-a)) == 1.5

def test_mixed_arithmetic():
    a = Float(10.0)
    assert float(a + 5.0) == 15.0
    assert float(a - 2) == 8.0
    assert float(a * 3) == 30.0

def test_comparisons():
    a = Float(5.0)
    b = Float(10.0)
    c = Float(5.0)

    assert a < b
    assert a <= b
    assert a <= c
    assert b > a
    assert b >= a
    assert a == c
    assert a != b

    # Mixed comparisons
    assert a < 10.0
    assert a == 5.0
    assert a > -1.0

def test_high_precision():
    a = Float(123456789.123456789, precision=256)
    b = Float(987654321.987654321, precision=256)
    c = a * b
    assert c.precision == 256
    assert c > 0
