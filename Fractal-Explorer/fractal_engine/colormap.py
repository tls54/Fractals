"""
Colormap utilities for fractal visualization.

This module provides support for both matplotlib colormaps and custom
HSV-based color functions for enhanced visual control.
"""

import math
import colorsys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from typing import Optional, Callable, Dict, Any, Tuple


def log_color(distance: float, base: float, const: float, scale: float) -> Tuple[int, int, int]:
    """
    Logarithmic color scale using HSV color space.

    Args:
        distance: Normalized iteration count (0 to 1)
        base: Logarithm base for color calculation
        const: Hue offset constant
        scale: Hue scale multiplier

    Returns:
        RGB tuple (0-255 range)
    """
    color = -1 * math.log(distance, base)
    rgb = colorsys.hsv_to_rgb(const + scale * color, 0.8, 0.9)
    return tuple(round(i * 255) for i in rgb)


def power_color(distance: float, exp: float, const: float, scale: float) -> Tuple[int, int, int]:
    """
    Power-based color scale using HSV color space.

    Args:
        distance: Normalized iteration count (0 to 1)
        exp: Power exponent for color calculation
        const: Hue offset constant
        scale: Hue scale multiplier

    Returns:
        RGB tuple (0-255 range)
    """
    color = distance ** exp
    rgb = colorsys.hsv_to_rgb(
        const + scale * color,
        1 - 0.6 * color,
        0.9
    )
    return tuple(round(i * 255) for i in rgb)


def smooth_color(distance: float, const: float, scale: float, saturation: float = 0.85) -> Tuple[int, int, int]:
    """
    Smooth linear color scale using HSV color space.

    Args:
        distance: Normalized iteration count (0 to 1)
        const: Hue offset constant
        scale: Hue scale multiplier
        saturation: HSV saturation value (0 to 1)

    Returns:
        RGB tuple (0-255 range)
    """
    hue = (const + scale * distance) % 1.0
    rgb = colorsys.hsv_to_rgb(hue, saturation, 0.9)
    return tuple(round(i * 255) for i in rgb)


CUSTOM_COLOR_FUNCTIONS: Dict[str, Callable] = {
    'log': log_color,
    'power': power_color,
    'smooth': smooth_color,
}


def apply_custom_colormap(
    fractal_data: np.ndarray,
    max_iterations: int,
    color_function_name: str,
    params: Dict[str, Any]
) -> np.ndarray:
    """
    Apply a custom color function to fractal iteration data.

    Args:
        fractal_data: 2D array of iteration counts
        max_iterations: Maximum iteration value
        color_function_name: Name of the color function to use
        params: Dictionary of parameters for the color function

    Returns:
        RGB image array (height × width × 3) with values in 0-255 range
    """
    if color_function_name not in CUSTOM_COLOR_FUNCTIONS:
        raise ValueError(f"Unknown custom color function: {color_function_name}")

    color_func = CUSTOM_COLOR_FUNCTIONS[color_function_name]
    height, width = fractal_data.shape
    img_array = np.zeros((height, width, 3), dtype=np.uint8)

    # Normalize iteration data to [0, 1]
    normalized = fractal_data / max_iterations

    # Extract parameters with defaults
    if color_function_name == 'log':
        base = params.get('base', 0.2)
        const = params.get('const', 0.27)
        scale = params.get('scale', 1.0)

        for i in range(height):
            for j in range(width):
                if fractal_data[i, j] == max_iterations:
                    img_array[i, j] = (0, 0, 0)  # Black for points in set
                else:
                    distance = max(normalized[i, j], 1e-10)  # Avoid log(0)
                    img_array[i, j] = color_func(distance, base, const, scale)

    elif color_function_name == 'power':
        exp = params.get('exp', 0.2)
        const = params.get('const', 0.27)
        scale = params.get('scale', 1.0)

        for i in range(height):
            for j in range(width):
                if fractal_data[i, j] == max_iterations:
                    img_array[i, j] = (0, 0, 0)
                else:
                    distance = normalized[i, j]
                    img_array[i, j] = color_func(distance, exp, const, scale)

    elif color_function_name == 'smooth':
        const = params.get('const', 0.0)
        scale = params.get('scale', 1.0)
        saturation = params.get('saturation', 0.85)

        for i in range(height):
            for j in range(width):
                if fractal_data[i, j] == max_iterations:
                    img_array[i, j] = (0, 0, 0)
                else:
                    distance = normalized[i, j]
                    img_array[i, j] = color_func(distance, const, scale, saturation)

    return img_array


def apply_matplotlib_colormap(
    fractal_data: np.ndarray,
    max_iterations: int,
    colormap_name: str,
    use_log_scale: bool = True
) -> np.ndarray:
    """
    Apply a matplotlib colormap to fractal iteration data.

    Args:
        fractal_data: 2D array of iteration counts
        max_iterations: Maximum iteration value
        colormap_name: Name of the matplotlib colormap
        use_log_scale: Whether to use logarithmic scaling

    Returns:
        Normalized fractal data ready for imshow with the specified colormap
    """
    if use_log_scale:
        # Apply logarithmic scaling for better visualization
        return np.log(fractal_data + 1)
    else:
        return fractal_data


def get_available_colormaps() -> Dict[str, list]:
    """
    Get lists of available colormaps.

    Returns:
        Dictionary with 'custom' and 'matplotlib' colormap lists
    """
    return {
        'custom': list(CUSTOM_COLOR_FUNCTIONS.keys()),
        'matplotlib': [
            'hot', 'viridis', 'plasma', 'inferno', 'magma', 'cividis',
            'twilight', 'twilight_shifted', 'turbo', 'jet',
            'coolwarm', 'seismic', 'RdYlBu', 'Spectral'
        ]
    }


def create_colormap_from_colors(colors: list, name: str = 'custom') -> LinearSegmentedColormap:
    """
    Create a custom matplotlib colormap from a list of colors.

    Args:
        colors: List of color specifications (names, hex, or RGB tuples)
        name: Name for the custom colormap

    Returns:
        LinearSegmentedColormap instance
    """
    return LinearSegmentedColormap.from_list(name, colors)
