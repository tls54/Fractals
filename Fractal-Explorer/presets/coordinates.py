"""
Preset coordinate bounds for interesting fractal regions.

This module contains pre-defined coordinate presets for exploring different
regions of fractals. Each preset defines bounds in the complex plane that
highlight interesting features.
"""

from typing import Dict, Tuple, Any


# Type alias for coordinate bounds: (xmin, xmax, ymin, ymax)
Bounds = Tuple[float, float, float, float]


# Registry of coordinate presets organized by fractal type
COORDINATE_PRESETS: Dict[str, Dict[str, Any]] = {
    # Mandelbrot Set Presets
    'mandelbrot_full': {
        'bounds': (-2.5, 1.0, -1.25, 1.25),
        'description': 'Full view of the Mandelbrot set',
        'fractal_type': 'mandelbrot',
    },
    'mandelbrot_main': {
        'bounds': (-2.0, 0.5, -1.25, 1.25),
        'description': 'Main body of the Mandelbrot set',
        'fractal_type': 'mandelbrot',
    },
    'mandelbrot_seahorse': {
        'bounds': (-0.75, -0.73, 0.095, 0.115),
        'description': 'Seahorse valley - intricate spiral patterns',
        'fractal_type': 'mandelbrot',
    },
    'mandelbrot_elephant': {
        'bounds': (0.25, 0.35, 0.0, 0.1),
        'description': 'Elephant valley - trunk-like tendrils',
        'fractal_type': 'mandelbrot',
    },
    'mandelbrot_spiral': {
        'bounds': (-0.7, -0.6, 0.3, 0.4),
        'description': 'Beautiful spiral formations',
        'fractal_type': 'mandelbrot',
    },
    'mandelbrot_mini': {
        'bounds': (-0.16, -0.14, 1.025, 1.045),
        'description': 'Mini Mandelbrot with detailed Julia sets',
        'fractal_type': 'mandelbrot',
    },
    'mandelbrot_feather': {
        'bounds': (-1.25, -1.23, 0.02, 0.04),
        'description': 'Delicate feather-like structures',
        'fractal_type': 'mandelbrot',
    },

    # Burning Ship Presets (from existing burning_ship_gpu.py)
    'burning_ship_full': {
        'bounds': (-2.0, 1.0, -2.0, 1.0),
        'description': 'Full view of the Burning Ship fractal',
        'fractal_type': 'burning_ship',
    },
    'burning_ship_ship': {
        'bounds': (-1.8, -1.6, -0.084374, 0.028126),
        'description': 'The iconic burning ship structure (16:9)',
        'fractal_type': 'burning_ship',
    },
    'burning_ship_4k': {
        'bounds': (-2.0, 1.73333, -1.6, 0.5),
        'description': 'Burning Ship optimized for 4K/16:9',
        'fractal_type': 'burning_ship',
    },
    'burning_ship_lower': {
        'bounds': (-1.25, -0.35, -1.2563, -0.75),
        'description': 'Lower region with intricate details',
        'fractal_type': 'burning_ship',
    },
    'burning_ship_lower_zoom': {
        'bounds': (-0.8615, -0.6485, -1.2, -1.08),
        'description': 'Zoomed view of lower region',
        'fractal_type': 'burning_ship',
    },
    'burning_ship_lower_zoom_ship': {
        'bounds': (-0.84, -0.61, -0.984, -0.855),
        'description': 'Detailed ship in lower zoom region',
        'fractal_type': 'burning_ship',
    },

    # Generic/Multi-purpose Presets
    'origin': {
        'bounds': (-2.0, 2.0, -2.0, 2.0),
        'description': 'Centered on origin, 4x4 square',
        'fractal_type': 'generic',
    },
    'unit_circle': {
        'bounds': (-1.5, 1.5, -1.5, 1.5),
        'description': 'Focused on unit circle region',
        'fractal_type': 'generic',
    },
    'wide': {
        'bounds': (-3.0, 3.0, -2.0, 2.0),
        'description': 'Wide view for exploring',
        'fractal_type': 'generic',
    },
    'square': {
        'bounds': (-2.0, 2.0, -2.0, 2.0),
        'description': 'Square aspect ratio centered on origin',
        'fractal_type': 'generic',
    },
}


def get_coordinate_preset(name: str) -> Bounds:
    """
    Get coordinate bounds by preset name.

    Args:
        name: Name of the coordinate preset

    Returns:
        Tuple of (xmin, xmax, ymin, ymax)

    Raises:
        ValueError: If preset name is not found
    """
    if name not in COORDINATE_PRESETS:
        raise ValueError(
            f"Unknown coordinate preset: {name}. "
            f"Available: {', '.join(list_coordinate_presets())}"
        )
    return COORDINATE_PRESETS[name]['bounds']


def get_coordinate_info(name: str) -> Dict[str, Any]:
    """
    Get full information about a coordinate preset.

    Args:
        name: Name of the coordinate preset

    Returns:
        Dictionary with bounds, description, and fractal_type

    Raises:
        ValueError: If preset name is not found
    """
    if name not in COORDINATE_PRESETS:
        raise ValueError(
            f"Unknown coordinate preset: {name}. "
            f"Available: {', '.join(list_coordinate_presets())}"
        )
    return COORDINATE_PRESETS[name]


def list_coordinate_presets(fractal_type: str = None) -> list:
    """
    List all available coordinate preset names.

    Args:
        fractal_type: Optional filter by fractal type (e.g., 'mandelbrot', 'burning_ship', 'generic')

    Returns:
        List of preset names
    """
    if fractal_type is None:
        return sorted(COORDINATE_PRESETS.keys())
    else:
        return sorted([
            name for name, info in COORDINATE_PRESETS.items()
            if info['fractal_type'] == fractal_type
        ])


def get_presets_by_fractal(fractal_type: str) -> Dict[str, Bounds]:
    """
    Get all coordinate presets for a specific fractal type.

    Args:
        fractal_type: Type of fractal ('mandelbrot', 'burning_ship', 'generic')

    Returns:
        Dictionary mapping preset names to bounds tuples
    """
    return {
        name: info['bounds']
        for name, info in COORDINATE_PRESETS.items()
        if info['fractal_type'] == fractal_type
    }
