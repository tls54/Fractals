"""
Core fractal generation engine with GPU acceleration and memory optimization.

This module contains the main FractalEngine class that orchestrates fractal
generation using custom iterators, with automatic tiling and device selection.
"""

import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from typing import Optional

from .config import FractalConfig
from .base_iterator import FractalIterator
from .memory import (
    get_device,
    should_use_tiling,
    print_memory_info,
    clear_gpu_cache
)
from .colormap import (
    apply_custom_colormap,
    apply_matplotlib_colormap,
    CUSTOM_COLOR_FUNCTIONS
)


class FractalEngine:
    """
    GPU-accelerated fractal generation engine with automatic memory optimization.

    This engine takes any FractalIterator implementation and generates fractal
    images using PyTorch for GPU acceleration, with automatic tiling for
    memory-constrained environments.
    """

    def __init__(self, config: FractalConfig):
        """
        Initialize the fractal engine.

        Args:
            config: FractalConfig instance with generation parameters
        """
        self.config = config
        self.device = get_device(config.device)
        self.fractal_data: Optional[np.ndarray] = None

    def generate(self, iterator: FractalIterator) -> np.ndarray:
        """
        Generate a fractal using the provided iterator.

        Args:
            iterator: FractalIterator instance defining the iteration logic

        Returns:
            2D numpy array of iteration counts (height × width)
        """
        print(f"\nGenerating {iterator.name} fractal...")
        print(f"Region: [{self.config.xmin:.6f}, {self.config.xmax:.6f}] × "
              f"[{self.config.ymin:.6f}, {self.config.ymax:.6f}]")
        print(f"Resolution: {self.config.width}×{self.config.height}")
        print(f"Max iterations: {self.config.max_iterations}")

        # Adjust aspect ratio if requested
        if self.config.adjust_aspect_ratio:
            self.config.adjust_bounds_for_aspect_ratio(self.config.adjust_aspect_ratio)
            print(f"Adjusted bounds: [{self.config.xmin:.6f}, {self.config.xmax:.6f}] × "
                  f"[{self.config.ymin:.6f}, {self.config.ymax:.6f}]")

        # Determine if tiling is needed
        use_tiling, tile_size = should_use_tiling(
            self.config.width,
            self.config.height,
            self.device,
            self.config.disable_tiling
        )

        # Override tile_size if specified in config
        if self.config.tile_size and use_tiling:
            tile_size = self.config.tile_size

        print_memory_info(
            self.config.width,
            self.config.height,
            self.device,
            use_tiling,
            tile_size
        )

        start_time = time.time()

        if use_tiling:
            self.fractal_data = self._generate_tiled(iterator, tile_size)
        else:
            self.fractal_data = self._generate_single(iterator)

        elapsed_time = time.time() - start_time
        self._print_performance_stats(elapsed_time)

        return self.fractal_data

    def _generate_single(self, iterator: FractalIterator) -> np.ndarray:
        """
        Generate fractal in a single pass (no tiling).

        Args:
            iterator: FractalIterator instance

        Returns:
            2D numpy array of iteration counts
        """
        print("\nGenerating fractal (single pass)...")

        # Create coordinate grids
        x = torch.linspace(
            self.config.xmin,
            self.config.xmax,
            self.config.width,
            dtype=torch.float32
        )
        y = torch.linspace(
            self.config.ymin,
            self.config.ymax,
            self.config.height,
            dtype=torch.float32
        )

        # Create meshgrid (indexing='ij' means rows=y, cols=x)
        y_grid, x_grid = torch.meshgrid(y, x, indexing='ij')

        # Move to device
        x_grid = x_grid.to(self.device)
        y_grid = y_grid.to(self.device)

        # Initialize complex number components
        c_real = x_grid
        c_imag = y_grid
        z_real = torch.zeros_like(c_real)
        z_imag = torch.zeros_like(c_imag)

        # Initialize iteration tracking
        iterations = torch.zeros(
            (self.config.height, self.config.width),
            dtype=torch.int32,
            device=self.device
        )
        mask = torch.ones(
            (self.config.height, self.config.width),
            dtype=torch.bool,
            device=self.device
        )

        escape_radius_sq = self.config.escape_radius ** 2

        # Iteration loop with progress bar
        with tqdm(total=self.config.max_iterations, desc="Iterating") as pbar:
            for i in range(self.config.max_iterations):
                # Apply iterator's custom logic
                z_real, z_imag = iterator.iterate_gpu(
                    z_real, z_imag, c_real, c_imag, i
                )

                # Check for escape
                z_magnitude_sq = z_real * z_real + z_imag * z_imag
                escaped = (z_magnitude_sq > escape_radius_sq) & mask

                # Record iteration count for newly escaped points
                iterations[escaped] = i

                # Update mask to exclude escaped points
                mask = mask & ~escaped

                # Early exit if all points have escaped
                active_pixels = mask.sum().item()
                if active_pixels == 0:
                    print(f"\nAll pixels escaped at iteration {i}")
                    break

                pbar.update(1)
                pbar.set_postfix({'active': f'{active_pixels:,}'})

        # Points that never escaped get max_iterations
        iterations[mask] = self.config.max_iterations

        # Move result to CPU and convert to numpy
        result = iterations.cpu().numpy()

        # Clear GPU cache
        clear_gpu_cache(self.device)

        return result

    def _generate_tiled(self, iterator: FractalIterator, tile_size: int) -> np.ndarray:
        """
        Generate fractal using tiling for memory efficiency.

        Args:
            iterator: FractalIterator instance
            tile_size: Size of each tile (in pixels)

        Returns:
            2D numpy array of iteration counts
        """
        tiles_x = (self.config.width + tile_size - 1) // tile_size
        tiles_y = (self.config.height + tile_size - 1) // tile_size
        total_tiles = tiles_x * tiles_y

        print(f"\nGenerating fractal with tiling ({tiles_x}×{tiles_y} = {total_tiles} tiles)...")

        # Pre-allocate result array on CPU
        result = np.zeros((self.config.height, self.config.width), dtype=np.int32)

        # Create full coordinate arrays
        x = torch.linspace(
            self.config.xmin,
            self.config.xmax,
            self.config.width,
            dtype=torch.float32
        )
        y = torch.linspace(
            self.config.ymin,
            self.config.ymax,
            self.config.height,
            dtype=torch.float32
        )

        # Process each tile
        with tqdm(total=total_tiles, desc="Processing tiles") as tile_pbar:
            for tile_y in range(tiles_y):
                for tile_x in range(tiles_x):
                    # Calculate tile boundaries
                    x_start = tile_x * tile_size
                    x_end = min(x_start + tile_size, self.config.width)
                    y_start = tile_y * tile_size
                    y_end = min(y_start + tile_size, self.config.height)

                    # Extract coordinate slices for this tile
                    x_tile = x[x_start:x_end]
                    y_tile = y[y_start:y_end]

                    # Create meshgrid for tile
                    y_grid, x_grid = torch.meshgrid(y_tile, x_tile, indexing='ij')
                    x_grid = x_grid.to(self.device)
                    y_grid = y_grid.to(self.device)

                    # Initialize for this tile
                    c_real = x_grid
                    c_imag = y_grid
                    z_real = torch.zeros_like(c_real)
                    z_imag = torch.zeros_like(c_imag)

                    tile_height = y_end - y_start
                    tile_width = x_end - x_start

                    iterations = torch.zeros(
                        (tile_height, tile_width),
                        dtype=torch.int32,
                        device=self.device
                    )
                    mask = torch.ones(
                        (tile_height, tile_width),
                        dtype=torch.bool,
                        device=self.device
                    )

                    escape_radius_sq = self.config.escape_radius ** 2

                    # Iterate for this tile
                    for i in range(self.config.max_iterations):
                        z_real, z_imag = iterator.iterate_gpu(
                            z_real, z_imag, c_real, c_imag, i
                        )

                        z_magnitude_sq = z_real * z_real + z_imag * z_imag
                        escaped = (z_magnitude_sq > escape_radius_sq) & mask

                        iterations[escaped] = i
                        mask = mask & ~escaped

                        if mask.sum().item() == 0:
                            break

                    iterations[mask] = self.config.max_iterations

                    # Copy result to CPU
                    result[y_start:y_end, x_start:x_end] = iterations.cpu().numpy()

                    # Clear GPU memory for next tile
                    clear_gpu_cache(self.device)

                    tile_pbar.update(1)

        return result

    def visualize(self, save: bool = True, show: bool = False) -> None:
        """
        Visualize the generated fractal.

        Args:
            save: Whether to save the image to disk
            show: Whether to display the image interactively
        """
        if self.fractal_data is None:
            raise ValueError("No fractal data to visualize. Call generate() first.")

        # Create annotated figure if requested
        if self.config.save_annotated:
            self._create_annotated_figure(save=save, show=show)

        # Create pixel-perfect figure
        if save or show:
            self._create_pixelperfect_figure(save=save, show=show)

    def _create_annotated_figure(self, save: bool = True, show: bool = False) -> None:
        """Create annotated figure with axes, labels, and colorbar."""
        fig_width = self.config.annotated_width / 100
        fig_height = self.config.annotated_height / 100

        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=100)

        # Apply colormap
        if self.config.custom_colormap and self.config.custom_colormap in CUSTOM_COLOR_FUNCTIONS:
            # Use custom colormap
            img_array = apply_custom_colormap(
                self.fractal_data,
                self.config.max_iterations,
                self.config.custom_colormap,
                self.config.colormap_params
            )
            ax.imshow(
                img_array,
                extent=[self.config.xmin, self.config.xmax, self.config.ymin, self.config.ymax],
                origin='lower',
                interpolation='bilinear'
            )
        else:
            # Use matplotlib colormap
            data = apply_matplotlib_colormap(
                self.fractal_data,
                self.config.max_iterations,
                self.config.colormap,
                self.config.use_log_scale
            )
            im = ax.imshow(
                data,
                extent=[self.config.xmin, self.config.xmax, self.config.ymin, self.config.ymax],
                origin='lower',
                cmap=self.config.colormap,
                interpolation='bilinear'
            )
            plt.colorbar(im, ax=ax, label='Iterations (log scale)' if self.config.use_log_scale else 'Iterations')

        ax.set_xlabel('Real')
        ax.set_ylabel('Imaginary')
        ax.set_title(f'{self.config.width}×{self.config.height} @ {self.config.max_iterations} iterations')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save and self.config.output_path:
            annotated_path = self.config.output_path.with_stem(
                self.config.output_path.stem + '_annotated'
            )
            plt.savefig(annotated_path, dpi=100, bbox_inches='tight')
            print(f"\nAnnotated image saved to: {annotated_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def _create_pixelperfect_figure(self, save: bool = True, show: bool = False) -> None:
        """Create pixel-perfect figure without axes or padding."""
        # Calculate figure size to match exact pixel dimensions
        fig_width = self.config.width / self.config.dpi
        fig_height = self.config.height / self.config.dpi

        fig = plt.figure(figsize=(fig_width, fig_height), dpi=self.config.dpi)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')

        # Apply colormap
        if self.config.custom_colormap and self.config.custom_colormap in CUSTOM_COLOR_FUNCTIONS:
            img_array = apply_custom_colormap(
                self.fractal_data,
                self.config.max_iterations,
                self.config.custom_colormap,
                self.config.colormap_params
            )
            ax.imshow(img_array, interpolation='bilinear')
        else:
            data = apply_matplotlib_colormap(
                self.fractal_data,
                self.config.max_iterations,
                self.config.colormap,
                self.config.use_log_scale
            )
            ax.imshow(data, cmap=self.config.colormap, interpolation='bilinear')

        if save and self.config.output_path:
            plt.savefig(
                self.config.output_path,
                dpi=self.config.dpi,
                bbox_inches='tight',
                pad_inches=0
            )
            print(f"Pixel-perfect image saved to: {self.config.output_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def _print_performance_stats(self, elapsed_time: float) -> None:
        """Print performance statistics."""
        total_pixels = self.config.width * self.config.height

        print(f"\nCalculation completed in {elapsed_time:.2f} seconds")
        print(f"  Total pixels: {total_pixels:,}")
        print(f"  Pixels/second: {total_pixels/elapsed_time:,.0f}")
        print(f"  Time per pixel: {elapsed_time/total_pixels*1_000_000:.3f} µs")
