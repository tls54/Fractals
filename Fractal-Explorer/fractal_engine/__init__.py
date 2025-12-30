"""
Fractal Engine - A flexible, GPU-accelerated fractal generation system.

This package provides a modular framework for generating fractals with
custom iteration logic, automatic memory optimization, and flexible colormaps.
"""

from .config import FractalConfig
from .base_iterator import FractalIterator
from .engine import FractalEngine

__all__ = ['FractalConfig', 'FractalIterator', 'FractalEngine']
