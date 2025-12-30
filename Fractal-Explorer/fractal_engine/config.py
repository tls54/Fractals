"""
Configuration dataclass for fractal generation.

This module defines FractalConfig, which serves as the single source of truth
for all fractal generation parameters.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any


@dataclass
class FractalConfig:
    """
    Complete configuration for fractal generation.

    This class contains all parameters needed to generate a fractal image,
    including complex plane bounds, resolution, iteration settings, colormaps,
    and output options.
    """

    # Complex plane bounds
    xmin: float = -2.0
    xmax: float = 2.0
    ymin: float = -2.0
    ymax: float = 2.0

    # Resolution
    width: int = 1920
    height: int = 1080

    # Iteration parameters
    max_iterations: int = 256
    escape_radius: float = 2.0

    # Colormap settings
    colormap: str = 'hot'
    use_log_scale: bool = True
    custom_colormap: Optional[str] = None
    colormap_params: Dict[str, Any] = field(default_factory=lambda: {
        'base': 0.2,
        'const': 0.27,
        'scale': 1.0
    })

    # GPU and memory settings
    device: Optional[str] = None
    tile_size: int = 8192
    disable_tiling: bool = False

    # Output settings
    output_path: Optional[Path] = None
    show_plot: bool = False
    dpi: int = 150
    save_annotated: bool = True
    annotated_width: int = 1200
    annotated_height: int = 1000

    # Aspect ratio adjustment
    adjust_aspect_ratio: Optional[str] = None

    def __post_init__(self):
        """Convert output_path to Path object if it's a string."""
        if isinstance(self.output_path, str):
            self.output_path = Path(self.output_path)

    @classmethod
    def from_preset(cls, preset_name: str, **overrides) -> 'FractalConfig':
        """
        Create a configuration from a named preset.

        Args:
            preset_name: Name of the preset to load
            **overrides: Any parameters to override from the preset

        Returns:
            FractalConfig instance with preset values and overrides applied
        """
        presets = cls.get_presets()
        if preset_name not in presets:
            raise ValueError(f"Unknown preset: {preset_name}")

        preset_params = presets[preset_name].copy()
        preset_params.update(overrides)
        return cls(**preset_params)

    @staticmethod
    def get_presets() -> Dict[str, Dict[str, Any]]:
        """
        Get all available configuration presets.

        Returns:
            Dictionary mapping preset names to parameter dictionaries
        """
        return {
            'default': {
                'width': 1920,
                'height': 1080,
                'max_iterations': 256,
            },
            '4k': {
                'width': 3840,
                'height': 2160,
                'max_iterations': 300,
                'dpi': 200,
            },
            '8k': {
                'width': 7680,
                'height': 4320,
                'max_iterations': 400,
                'dpi': 300,
            },
            '16k': {
                'width': 15360,
                'height': 8640,
                'max_iterations': 500,
                'dpi': 300,
            },
            'fast': {
                'width': 800,
                'height': 600,
                'max_iterations': 100,
                'dpi': 100,
            },
            '32k': {
                'width': 30720,
                'height': 17280,
                'max_iterations': 500,
                'dpi': 300,
            },
        }

    def get_aspect_ratio(self) -> float:
        """Calculate the aspect ratio of the output image."""
        return self.width / self.height

    def get_domain_aspect_ratio(self) -> float:
        """Calculate the aspect ratio of the complex plane domain."""
        x_range = self.xmax - self.xmin
        y_range = self.ymax - self.ymin
        return x_range / y_range

    def adjust_bounds_for_aspect_ratio(self, axis: str = 'x') -> None:
        """
        Adjust complex plane bounds to match image aspect ratio.

        Args:
            axis: Which axis to adjust ('x' or 'y')
        """
        image_aspect = self.get_aspect_ratio()
        domain_x_range = self.xmax - self.xmin
        domain_y_range = self.ymax - self.ymin

        if axis == 'x':
            new_x_range = image_aspect * domain_y_range
            x_center = (self.xmin + self.xmax) / 2
            self.xmin = x_center - new_x_range / 2
            self.xmax = x_center + new_x_range / 2
        elif axis == 'y':
            new_y_range = domain_x_range / image_aspect
            y_center = (self.ymin + self.ymax) / 2
            self.ymin = y_center - new_y_range / 2
            self.ymax = y_center + new_y_range / 2
        else:
            raise ValueError(f"Invalid axis: {axis}. Must be 'x' or 'y'")

    def __str__(self) -> str:
        """Return a human-readable string representation."""
        return (
            f"FractalConfig(\n"
            f"  Region: [{self.xmin}, {self.xmax}] × [{self.ymin}, {self.ymax}]\n"
            f"  Resolution: {self.width}×{self.height}\n"
            f"  Iterations: {self.max_iterations}\n"
            f"  Colormap: {self.custom_colormap or self.colormap}\n"
            f"  Output: {self.output_path}\n"
            f")"
        )
