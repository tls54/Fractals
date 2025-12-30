"""
Burning Ship fractal iterator.

The Burning Ship fractal is a variant of the Mandelbrot set discovered by
Michelitsch and Rossler in 1992. It uses absolute values of the components
before squaring.
"""

import torch
from typing import Tuple
from fractal_engine.base_iterator import FractalIterator


class BurningShipIterator(FractalIterator):
    """
    Iterator for the Burning Ship fractal.

    Formula: z_n+1 = (|Re(z_n)| + i|Im(z_n)|)^2 + c
    where z_0 = 0 and c is the point in the complex plane.

    The absolute value applied to both components before squaring creates
    the characteristic "burning ship" appearance.
    """

    @property
    def name(self) -> str:
        return "Burning Ship"

    @property
    def default_bounds(self) -> Tuple[float, float, float, float]:
        """Standard Burning Ship bounds showing the full ship (16:9 aspect ratio)."""
        # Vertical range: -2.0 to 1.0 = 3.0 units
        # For 16:9 aspect: horizontal range = 3.0 * (16/9) = 5.333 units
        # Centered at -0.5: -0.5 ± 2.667
        return (-3.167, 2.167, -2.0, 1.0)

    def iterate_gpu(
        self,
        z_real: torch.Tensor,
        z_imag: torch.Tensor,
        c_real: torch.Tensor,
        c_imag: torch.Tensor,
        iteration: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform one Burning Ship iteration: z = (|Re(z)| + i|Im(z)|)^2 + c

        The key difference from Mandelbrot is taking absolute values
        before squaring.
        """
        # Take absolute values of both components
        z_real_abs = torch.abs(z_real)
        z_imag_abs = torch.abs(z_imag)

        # Square in component form: (a + bi)^2 = a^2 - b^2 + 2abi
        z_real_new = z_real_abs * z_real_abs - z_imag_abs * z_imag_abs + c_real
        z_imag_new = 2 * z_real_abs * z_imag_abs + c_imag

        return z_real_new, z_imag_new


class BurningShipFull(BurningShipIterator):
    """Burning Ship iterator with full view default bounds."""

    @property
    def name(self) -> str:
        return "Burning Ship (Full View)"

    @property
    def default_bounds(self) -> Tuple[float, float, float, float]:
        return (-2.0, 1.0, -2.0, 1.0)


class BurningShipMain(BurningShipIterator):
    """Burning Ship iterator focused on the main ship structure."""

    @property
    def name(self) -> str:
        return "Burning Ship (Main Ship)"

    @property
    def default_bounds(self) -> Tuple[float, float, float, float]:
        return (-1.8, -1.6, -0.1, 0.1)


class BurningShipAntenna(BurningShipIterator):
    """Burning Ship iterator focused on the antenna detail."""

    @property
    def name(self) -> str:
        return "Burning Ship (Antenna)"

    @property
    def default_bounds(self) -> Tuple[float, float, float, float]:
        return (-1.755, -1.745, 0.02, 0.03)


class BurningShipLower(BurningShipIterator):
    """Burning Ship iterator focused on lower regions."""

    @property
    def name(self) -> str:
        return "Burning Ship (Lower Region)"

    @property
    def default_bounds(self) -> Tuple[float, float, float, float]:
        return (-1.8, -1.6, -2.0, -1.8)
