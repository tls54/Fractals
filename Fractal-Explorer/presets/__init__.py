"""
Preset fractal iterators.

This package contains pre-built iterator implementations for popular fractals.
"""

from .mandelbrot_iterator import MandelbrotIterator
from .burning_ship_iterator import BurningShipIterator

__all__ = ['MandelbrotIterator', 'BurningShipIterator']


# Registry of available iterators
ITERATOR_REGISTRY = {
    'mandelbrot': MandelbrotIterator,
    'burning_ship': BurningShipIterator,
}


def get_iterator(name: str):
    """
    Get an iterator class by name.

    Args:
        name: Name of the iterator

    Returns:
        Iterator class

    Raises:
        ValueError: If iterator name is not found
    """
    if name not in ITERATOR_REGISTRY:
        raise ValueError(
            f"Unknown iterator: {name}. "
            f"Available: {', '.join(ITERATOR_REGISTRY.keys())}"
        )
    return ITERATOR_REGISTRY[name]


def list_iterators():
    """List all available iterator names."""
    return list(ITERATOR_REGISTRY.keys())
