"""
Mandelbrot set iterator.

The Mandelbrot set is defined by the iteration z = z^2 + c, where c is a
complex number representing each point in the complex plane.
"""

import torch
from typing import Tuple
from fractal_engine.base_iterator import FractalIterator


class MandelbrotIterator(FractalIterator):
    """
    Iterator for the classic Mandelbrot set.

    Formula: z_n+1 = z_n^2 + c
    where z_0 = 0 and c is the point in the complex plane.
    """

    @property
    def name(self) -> str:
        return "Mandelbrot"

    @property
    def default_bounds(self) -> Tuple[float, float, float, float]:
        """Standard Mandelbrot bounds showing the full set (16:9 aspect ratio)."""
        # Vertical range: -1.25 to 1.25 = 2.5 units
        # For 16:9 aspect: horizontal range = 2.5 * (16/9) = 4.444 units
        # Centered at -0.75 (visual center of Mandelbrot): -0.75 ± 2.222
        return (-2.972, 1.472, -1.25, 1.25)

    def iterate_gpu(
        self,
        z_real: torch.Tensor,
        z_imag: torch.Tensor,
        c_real: torch.Tensor,
        c_imag: torch.Tensor,
        iteration: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform one Mandelbrot iteration: z = z^2 + c

        Complex multiplication: (a + bi)^2 = a^2 - b^2 + 2abi
        """
        # z^2 in component form
        z_real_new = z_real * z_real - z_imag * z_imag + c_real
        z_imag_new = 2 * z_real * z_imag + c_imag

        return z_real_new, z_imag_new


class MandelbrotSeahorse(MandelbrotIterator):
    """Mandelbrot iterator with seahorse valley default bounds."""

    @property
    def name(self) -> str:
        return "Mandelbrot (Seahorse Valley)"

    @property
    def default_bounds(self) -> Tuple[float, float, float, float]:
        return (-0.75, -0.735, 0.095, 0.105)


class MandelbrotElephant(MandelbrotIterator):
    """Mandelbrot iterator with elephant valley default bounds."""

    @property
    def name(self) -> str:
        return "Mandelbrot (Elephant Valley)"

    @property
    def default_bounds(self) -> Tuple[float, float, float, float]:
        return (0.27, 0.29, 0.005, 0.018)


class MandelbrotSpiral(MandelbrotIterator):
    """Mandelbrot iterator with deep spiral zoom default bounds."""

    @property
    def name(self) -> str:
        return "Mandelbrot (Deep Spiral)"

    @property
    def default_bounds(self) -> Tuple[float, float, float, float]:
        return (-0.77568377, -0.77568372, 0.13646737, 0.13646740)
