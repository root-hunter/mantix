"""
Mantix: High-performance arbitrary-precision floating-point library for Python.
"""

from mantix._core import (
    Float,
    ROUND_NEAREST_EVEN,
    ROUND_TOWARD_ZERO,
    ROUND_TOWARD_POSITIVE,
    ROUND_TOWARD_NEGATIVE,
    fma,
    dot,
    sqrt,
)

__version__ = "0.1.0"
__all__ = [
    "Float",
    "f32",
    "f64",
    "fma",
    "dot",
    "sqrt",
    "ROUND_NEAREST_EVEN",
    "ROUND_TOWARD_ZERO",
    "ROUND_TOWARD_POSITIVE",
    "ROUND_TOWARD_NEGATIVE",
]


def f32(value) -> Float:
    """Create a 32-bit single-precision MantixFloat (24-bit significand precision)."""
    return Float(value, precision=24)


def f64(value) -> Float:
    """Create a 64-bit double-precision MantixFloat (53-bit significand precision)."""
    return Float(value, precision=53)
