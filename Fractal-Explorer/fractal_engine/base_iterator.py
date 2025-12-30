"""
Base iterator interface for fractal generation.

All custom fractal iterators must inherit from FractalIterator and implement
the iterate_gpu method with their specific iteration logic.
"""

from abc import ABC, abstractmethod
from typing import Tuple
import torch


class FractalIterator(ABC):
    """
    Abstract base class for fractal iteration logic.

    Custom fractals should inherit from this class and implement the iterate_gpu
    method to define their specific iteration formula.
    """

    @abstractmethod
    def iterate_gpu(
        self,
        z_real: torch.Tensor,
        z_imag: torch.Tensor,
        c_real: torch.Tensor,
        c_imag: torch.Tensor,
        iteration: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform one iteration of the fractal algorithm on GPU tensors.

        Args:
            z_real: Real component of current z values (height × width tensor)
            z_imag: Imaginary component of current z values (height × width tensor)
            c_real: Real component of constant c values (height × width tensor)
            c_imag: Imaginary component of constant c values (height × width tensor)
            iteration: Current iteration number (0-indexed)

        Returns:
            Tuple of (new_z_real, new_z_imag) tensors after iteration

        Example:
            For Mandelbrot (z = z^2 + c):
            >>> z_real_new = z_real * z_real - z_imag * z_imag + c_real
            >>> z_imag_new = 2 * z_real * z_imag + c_imag
            >>> return z_real_new, z_imag_new
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this fractal type."""
        pass

    @property
    def default_bounds(self) -> Tuple[float, float, float, float]:
        """
        Return default bounds for this fractal: (xmin, xmax, ymin, ymax).

        Override this to provide sensible defaults for your fractal.
        Default is the standard complex plane view.
        """
        return (-2.0, 2.0, -2.0, 2.0)

    @property
    def default_escape_radius(self) -> float:
        """
        Return default escape radius for this fractal.

        Override this if your fractal needs a different escape radius.
        """
        return 2.0
