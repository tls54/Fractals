"""
Mandelbrot Set Generator - GPU Accelerated Version
Uses PyTorch with Metal Performance Shaders for M-series Macs
"""

import argparse
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import custom colour rules
try:
    from colour_rules import colour_functions
    CUSTOM_COLOURS_AVAILABLE = True
except ImportError:
    CUSTOM_COLOURS_AVAILABLE = False
    print("⚠ Warning: colour_rules.py not found. Custom colour rules disabled.")


@dataclass
class FractalParams:
    """Parameters for generating the Mandelbrot set."""

    # Complex plane bounds
    xmin: float = -2.5
    xmax: float = 1.0
    ymin: float = -1.25
    ymax: float = 1.25

    # Resolution
    width: int = 800
    height: int = 800

    # Iteration parameters
    max_iterations: int = 100
    escape_radius: float = 2.0

    # Visualization
    colormap: str = 'hot'
    use_log_scale: bool = True
    custom_colour: Optional[str] = None  # 'powerColor' or 'logColor'

    # Output
    output_path: Optional[Path] = None
    show_plot: bool = True
    dpi: int = 150
    save_annotated: bool = False
    annotated_width: int = 1200
    annotated_height: int = 1000

    def __post_init__(self):
        """Convert output_path to Path if string."""
        if self.output_path is not None and not isinstance(self.output_path, Path):
            self.output_path = Path(self.output_path)


def mandelbrot_gpu(params: FractalParams, device: str = 'mps') -> np.ndarray:
    """
    Generate Mandelbrot set using GPU acceleration.

    Args:
        params: Fractal generation parameters
        device: Device to use ('mps' for Metal on M-series, 'cuda' for NVIDIA, 'cpu' fallback)

    Returns:
        2D numpy array of iteration counts
    """
    print(f"Generating {params.width}x{params.height} Mandelbrot on {device.upper()}...")

    # Create coordinate grids on CPU first
    x = torch.linspace(params.xmin, params.xmax, params.width, dtype=torch.float32)
    y = torch.linspace(params.ymin, params.ymax, params.height, dtype=torch.float32)

    # Create meshgrid for all points in complex plane
    # Use 'ij' indexing to match orientation (rows=y, cols=x)
    y_grid, x_grid = torch.meshgrid(y, x, indexing='ij')

    # Move to GPU
    x_grid = x_grid.to(device)
    y_grid = y_grid.to(device)

    # Initialize complex number c = x + iy for all points
    c_real = x_grid
    c_imag = y_grid

    # Initialize z = 0 for all points
    z_real = torch.zeros_like(c_real)
    z_imag = torch.zeros_like(c_imag)

    # Track iteration count for each point
    iterations = torch.zeros((params.height, params.width), dtype=torch.int32, device=device)

    # Track which points have escaped
    mask = torch.ones((params.height, params.width), dtype=torch.bool, device=device)

    escape_radius_sq = params.escape_radius ** 2

    start_time = time.time()

    # Iterate the Mandelbrot formula with progress bar
    with tqdm(total=params.max_iterations, desc="GPU iterations", unit="iter") as pbar:
        for i in range(params.max_iterations):
            # Mandelbrot formula: z = z^2 + c
            # (a + bi)^2 = a^2 - b^2 + 2abi
            z_real_new = z_real * z_real - z_imag * z_imag + c_real
            z_imag_new = 2 * z_real * z_imag + c_imag

            z_real = z_real_new
            z_imag = z_imag_new

            # Check for escape: |z|^2 = real^2 + imag^2
            z_magnitude_sq = z_real * z_real + z_imag * z_imag

            # Find newly escaped points
            escaped = (z_magnitude_sq > escape_radius_sq) & mask

            # Update iteration counts for newly escaped points
            iterations[escaped] = i

            # Update mask to exclude escaped points from future calculations
            mask = mask & ~escaped

            # Update progress bar with active pixel count
            active_pixels = mask.sum().item()
            pbar.set_postfix({'active_pixels': f"{active_pixels:,}"})
            pbar.update(1)

            # Early exit if all points have escaped
            if active_pixels == 0:
                print(f"\n✓ All points escaped by iteration {i}")
                pbar.close()
                break

    # Points that never escaped get max_iterations
    iterations[mask] = params.max_iterations

    elapsed_time = time.time() - start_time

    # Move result back to CPU and convert to numpy
    result = iterations.cpu().numpy()

    # Print timing statistics
    total_pixels = params.width * params.height
    print(f"\n✓ GPU calculation completed in {elapsed_time:.2f} seconds")
    print(f"  Total pixels: {total_pixels:,}")
    print(f"  Pixels/second: {total_pixels/elapsed_time:,.0f}")
    print(f"  Time per pixel: {elapsed_time/total_pixels*1_000_000:.3f} µs")

    return result


def mandelbrot_gpu_tiled(params: FractalParams, device: str = 'mps', tile_size: int = 8192) -> np.ndarray:
    """
    Generate Mandelbrot set using GPU acceleration with tiled rendering.

    Splits the image into tiles to reduce memory usage for large renders.

    Args:
        params: Fractal generation parameters
        device: Device to use ('mps' for Metal on M-series, 'cuda' for NVIDIA, 'cpu' fallback)
        tile_size: Size of square tiles (default 8192x8192)

    Returns:
        2D numpy array of iteration counts
    """
    print(f"Generating {params.width}x{params.height} Mandelbrot on {device.upper()} using tiled rendering...")
    print(f"  Tile size: {tile_size}x{tile_size}")

    # Calculate number of tiles needed
    tiles_x = (params.width + tile_size - 1) // tile_size
    tiles_y = (params.height + tile_size - 1) // tile_size
    total_tiles = tiles_x * tiles_y

    print(f"  Grid: {tiles_x}x{tiles_y} tiles ({total_tiles} total)")

    # Allocate result array on CPU
    result = np.zeros((params.height, params.width), dtype=np.int32)

    # Create coordinate arrays on CPU
    x = np.linspace(params.xmin, params.xmax, params.width, dtype=np.float32)
    y = np.linspace(params.ymin, params.ymax, params.height, dtype=np.float32)

    overall_start_time = time.time()
    tile_num = 0

    # Process each tile
    for tile_y in range(tiles_y):
        for tile_x in range(tiles_x):
            tile_num += 1

            # Calculate tile boundaries
            y_start = tile_y * tile_size
            y_end = min(y_start + tile_size, params.height)
            x_start = tile_x * tile_size
            x_end = min(x_start + tile_size, params.width)

            tile_height = y_end - y_start
            tile_width = x_end - x_start

            print(f"\n[Tile {tile_num}/{total_tiles}] Processing region [{x_start}:{x_end}, {y_start}:{y_end}] ({tile_width}x{tile_height})")

            # Extract coordinate ranges for this tile
            x_tile = x[x_start:x_end]
            y_tile = y[y_start:y_end]

            # Create meshgrid for tile
            y_grid_tile, x_grid_tile = np.meshgrid(y_tile, x_tile, indexing='ij')

            # Convert to torch tensors and move to GPU
            x_grid = torch.from_numpy(x_grid_tile).to(device)
            y_grid = torch.from_numpy(y_grid_tile).to(device)

            # Initialize for this tile
            c_real = x_grid
            c_imag = y_grid
            z_real = torch.zeros_like(c_real)
            z_imag = torch.zeros_like(c_imag)
            iterations = torch.zeros((tile_height, tile_width), dtype=torch.int32, device=device)
            mask = torch.ones((tile_height, tile_width), dtype=torch.bool, device=device)

            escape_radius_sq = params.escape_radius ** 2

            tile_start_time = time.time()

            # Iterate the Mandelbrot formula
            with tqdm(total=params.max_iterations, desc=f"  Tile {tile_num}/{total_tiles}", unit="iter") as pbar:
                for i in range(params.max_iterations):
                    # Mandelbrot formula: z = z^2 + c
                    z_real_new = z_real * z_real - z_imag * z_imag + c_real
                    z_imag_new = 2 * z_real * z_imag + c_imag

                    z_real = z_real_new
                    z_imag = z_imag_new

                    # Check for escape
                    z_magnitude_sq = z_real * z_real + z_imag * z_imag
                    escaped = (z_magnitude_sq > escape_radius_sq) & mask

                    # Update iteration counts
                    iterations[escaped] = i
                    mask = mask & ~escaped

                    # Update progress
                    active_pixels = mask.sum().item()
                    pbar.set_postfix({'active': f"{active_pixels:,}"})
                    pbar.update(1)

                    # Early exit if all points escaped
                    if active_pixels == 0:
                        print(f"    ✓ All points escaped by iteration {i}")
                        pbar.close()
                        break

            # Points that never escaped get max_iterations
            iterations[mask] = params.max_iterations

            # Copy tile result back to CPU and into result array
            result[y_start:y_end, x_start:x_end] = iterations.cpu().numpy()

            tile_elapsed = time.time() - tile_start_time
            tile_pixels = tile_width * tile_height
            print(f"    ✓ Tile completed in {tile_elapsed:.2f}s ({tile_pixels/tile_elapsed:,.0f} pixels/sec)")

            # Clean up GPU memory for this tile
            del x_grid, y_grid, c_real, c_imag, z_real, z_imag, iterations, mask
            if device == 'mps':
                torch.mps.empty_cache()
            elif device == 'cuda':
                torch.cuda.empty_cache()

    overall_elapsed = time.time() - overall_start_time
    total_pixels = params.width * params.height

    print(f"\n✓ Tiled GPU calculation completed in {overall_elapsed:.2f} seconds")
    print(f"  Total pixels: {total_pixels:,}")
    print(f"  Overall pixels/second: {total_pixels/overall_elapsed:,.0f}")
    print(f"  Average time per tile: {overall_elapsed/total_tiles:.2f}s")

    return result


def visualize_fractal(fractal: np.ndarray, params: FractalParams) -> plt.Figure:
    """
    Create a visualization of the fractal.

    Args:
        fractal: 2D array of iteration counts
        params: Parameters used to generate the fractal

    Returns:
        Matplotlib figure object
    """
    # Calculate figure size to match exact pixel dimensions
    fig_width = params.width / params.dpi
    fig_height = params.height / params.dpi

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=params.dpi)

    # Apply custom colour if specified
    if params.custom_colour and CUSTOM_COLOURS_AVAILABLE:
        if params.custom_colour in colour_functions:
            print(f"Using custom colour rule: {params.custom_colour}")
            # Normalize iteration data
            normalized = fractal / params.max_iterations

            # Create RGB image using custom colour function
            img_array = np.zeros((params.height, params.width, 3), dtype=np.uint8)
            color_func = colour_functions[params.custom_colour]

            for i in range(params.height):
                for j in range(params.width):
                    if fractal[i, j] == params.max_iterations:
                        img_array[i, j] = (0, 0, 0)  # Black for points in set
                    else:
                        distance = normalized[i, j]
                        img_array[i, j] = color_func(distance, 0.2, 0.27, 1.0)

            ax.imshow(
                img_array,
                extent=[params.xmin, params.xmax, params.ymin, params.ymax],
                origin='lower',
                interpolation='bilinear'
            )
        else:
            print(f"⚠ Warning: Custom colour '{params.custom_colour}' not found. Using matplotlib colormap.")
            data = np.log(fractal + 1) if params.use_log_scale else fractal
            ax.imshow(
                data,
                extent=[params.xmin, params.xmax, params.ymin, params.ymax],
                cmap=params.colormap,
                origin='lower',
                interpolation='bilinear'
            )
    else:
        # Apply logarithmic scaling if requested
        if params.use_log_scale:
            data = np.log(fractal + 1)
        else:
            data = fractal

        # Create the image
        ax.imshow(
            data,
            extent=[params.xmin, params.xmax, params.ymin, params.ymax],
            cmap=params.colormap,
            origin='lower',
            interpolation='bilinear'
        )

    # Remove axes for exact pixel-perfect output
    ax.axis('off')

    # Remove all padding
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    return fig


def create_annotated_figure(fractal: np.ndarray, params: FractalParams) -> plt.Figure:
    """
    Create an annotated visualization with axes, labels, and colorbar.

    Args:
        fractal: 2D array of iteration counts
        params: Parameters used to generate the fractal

    Returns:
        Matplotlib figure object with annotations
    """
    # Use annotated dimensions
    fig_width = params.annotated_width / 100  # Use 100 DPI for annotated version
    fig_height = params.annotated_height / 100

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=100)

    # Apply logarithmic scaling if requested (only for matplotlib colormaps)
    if params.custom_colour and CUSTOM_COLOURS_AVAILABLE and params.custom_colour in colour_functions:
        # Use custom colour function
        normalized = fractal / params.max_iterations
        img_array = np.zeros((params.height, params.width, 3), dtype=np.uint8)
        color_func = colour_functions[params.custom_colour]

        for i in range(params.height):
            for j in range(params.width):
                if fractal[i, j] == params.max_iterations:
                    img_array[i, j] = (0, 0, 0)
                else:
                    distance = normalized[i, j]
                    img_array[i, j] = color_func(distance, 0.2, 0.27, 1.0)

        im = ax.imshow(
            img_array,
            extent=[params.xmin, params.xmax, params.ymin, params.ymax],
            origin='lower',
            interpolation='bilinear'
        )
        cbar_label = f'Custom: {params.custom_colour}'
    else:
        if params.use_log_scale:
            data = np.log(fractal + 1)
            cbar_label = 'log(iterations + 1)'
        else:
            data = fractal
            cbar_label = 'iterations'

        # Create the image
        im = ax.imshow(
            data,
            extent=[params.xmin, params.xmax, params.ymin, params.ymax],
            cmap=params.colormap,
            origin='lower',
            interpolation='bilinear'
        )

    # Add labels and title
    ax.set_xlabel('Real axis', fontsize=12)
    ax.set_ylabel('Imaginary axis', fontsize=12)
    ax.set_title(
        f'Mandelbrot Set\n'
        f'Region: [{params.xmin:.4f}, {params.xmax:.4f}] × [{params.ymin:.4f}, {params.ymax:.4f}]i\n'
        f'Resolution: {params.width}×{params.height}, Max iterations: {params.max_iterations}',
        fontsize=11,
        pad=15
    )

    # Increase tick resolution for better region identification
    x_range = params.xmax - params.xmin
    y_range = params.ymax - params.ymin

    # Calculate appropriate number of ticks based on range
    # Aim for tick spacing of roughly 0.1 for normal views, adjust for zoomed views
    num_x_ticks = max(10, min(20, int(x_range / 0.05)))
    num_y_ticks = max(10, min(20, int(y_range / 0.05)))

    x_ticks = np.linspace(params.xmin, params.xmax, num_x_ticks)
    y_ticks = np.linspace(params.ymin, params.ymax, num_y_ticks)

    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)

    # Format tick labels to show appropriate precision
    if x_range < 0.01:
        x_fmt = '%.6f'
    elif x_range < 0.1:
        x_fmt = '%.4f'
    elif x_range < 1:
        x_fmt = '%.3f'
    else:
        x_fmt = '%.2f'

    if y_range < 0.01:
        y_fmt = '%.6f'
    elif y_range < 0.1:
        y_fmt = '%.4f'
    elif y_range < 1:
        y_fmt = '%.3f'
    else:
        y_fmt = '%.2f'

    ax.set_xticklabels([x_fmt % tick for tick in x_ticks], rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels([y_fmt % tick for tick in y_ticks], fontsize=9)

    # Add minor ticks for even finer resolution
    ax.minorticks_on()
    ax.grid(True, which='major', alpha=0.3, linewidth=0.8)
    ax.grid(True, which='minor', alpha=0.1, linewidth=0.4)

    # Add colorbar (skip for custom colours with RGB arrays)
    if not (params.custom_colour and CUSTOM_COLOURS_AVAILABLE and params.custom_colour in colour_functions):
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(cbar_label, fontsize=11)

    plt.tight_layout()

    return fig


def save_fractal(fig: plt.Figure, output_path: Path, dpi: int):
    """
    Save the fractal visualization to a file.

    Args:
        fig: Matplotlib figure to save
        output_path: Path where to save the image
        dpi: Resolution in dots per inch
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving to {output_path}...")
    fig.savefig(output_path, dpi=dpi, pad_inches=0)
    print(f"Saved successfully!")


def get_device() -> str:
    """
    Determine the best available device for computation.

    Returns:
        Device string: 'mps' (Metal), 'cuda', or 'cpu'
    """
    if torch.backends.mps.is_available():
        print("✓ Metal Performance Shaders (MPS) available - using GPU acceleration")
        return 'mps'
    elif torch.cuda.is_available():
        print("✓ CUDA available - using NVIDIA GPU acceleration")
        return 'cuda'
    else:
        print("⚠ No GPU available - falling back to CPU")
        return 'cpu'


def estimate_memory_usage(width: int, height: int) -> int:
    """
    Estimate GPU memory usage in bytes for a single-pass render.

    Args:
        width: Image width in pixels
        height: Image height in pixels

    Returns:
        Estimated memory usage in bytes
    """
    total_pixels = width * height

    # Persistent tensors (fp32 = 4 bytes, int32 = 4 bytes, bool = 1 byte)
    memory_bytes = 0
    memory_bytes += total_pixels * 4  # x_grid
    memory_bytes += total_pixels * 4  # y_grid
    memory_bytes += total_pixels * 4  # z_real
    memory_bytes += total_pixels * 4  # z_imag
    memory_bytes += total_pixels * 4  # iterations
    memory_bytes += total_pixels * 1  # mask

    # Temporary tensors in iteration loop
    memory_bytes += total_pixels * 4  # z_real_new
    memory_bytes += total_pixels * 4  # z_imag_new
    memory_bytes += total_pixels * 4  # z_magnitude_sq
    memory_bytes += total_pixels * 1  # escaped

    # Add 20% safety margin for PyTorch overhead
    memory_bytes = int(memory_bytes * 1.2)

    return memory_bytes


def get_available_gpu_memory(device: str) -> int:
    """
    Get available GPU memory in bytes.

    Args:
        device: Device string ('mps', 'cuda', or 'cpu')

    Returns:
        Available memory in bytes, or a large value for CPU
    """
    if device == 'cuda':
        return torch.cuda.get_device_properties(0).total_memory
    elif device == 'mps':
        # MPS uses unified memory - estimate based on system RAM
        import os
        import platform
        if platform.system() == 'Darwin':
            total_memory = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
            return total_memory // 2  # Use 50% as conservative estimate
        else:
            return 16 * 1024 * 1024 * 1024  # Fallback: 16GB
    else:
        return 100 * 1024 * 1024 * 1024  # CPU: 100GB (effectively disable tiling)


def should_use_tiling(width: int, height: int, device: str, disable_tiling: bool) -> bool:
    """
    Determine whether to use tiled rendering based on memory constraints.

    Args:
        width: Image width in pixels
        height: Image height in pixels
        device: Device string ('mps', 'cuda', or 'cpu')
        disable_tiling: User override to disable tiling

    Returns:
        True if tiling should be used, False otherwise
    """
    if disable_tiling:
        return False

    estimated_usage = estimate_memory_usage(width, height)
    available_memory = get_available_gpu_memory(device)

    # Use tiling if estimated usage exceeds 70% of available memory
    threshold = 0.7 * available_memory

    use_tiling = estimated_usage > threshold

    if use_tiling:
        print(f"ℹ Auto-enabling tiled rendering:")
        print(f"  Estimated memory: {estimated_usage / (1024**3):.2f} GB")
        print(f"  Available memory: {available_memory / (1024**3):.2f} GB")
        print(f"  Threshold (70%): {threshold / (1024**3):.2f} GB")

    return use_tiling


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate the Mandelbrot set with GPU acceleration',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Region parameters
    parser.add_argument('--xmin', type=float, default=-2.5,
                       help='Minimum real value')
    parser.add_argument('--xmax', type=float, default=1.0,
                       help='Maximum real value')
    parser.add_argument('--ymin', type=float, default=-1.25,
                       help='Minimum imaginary value')
    parser.add_argument('--ymax', type=float, default=1.25,
                       help='Maximum imaginary value')

    # Resolution
    parser.add_argument('--width', type=int, default=800,
                       help='Image width in pixels')
    parser.add_argument('--height', type=int, default=800,
                       help='Image height in pixels')

    # Iteration parameters
    parser.add_argument('--max-iterations', type=int, default=100,
                       help='Maximum iterations per point')
    parser.add_argument('--escape-radius', type=float, default=2.0,
                       help='Escape radius threshold')

    # Visualization
    parser.add_argument('--colormap', type=str, default='hot',
                       help='Matplotlib colormap (e.g., hot, viridis, twilight, plasma)')
    parser.add_argument('--no-log-scale', action='store_true',
                       help='Disable logarithmic color scaling')

    # Custom colour rules
    if CUSTOM_COLOURS_AVAILABLE:
        colour_choices = list(colour_functions.keys())
        parser.add_argument('--custom-colour', type=str, choices=colour_choices,
                           help=f'Use custom colour rule: {", ".join(colour_choices)}')
    else:
        parser.add_argument('--custom-colour', type=str,
                           help='Custom colour rules not available (colour_rules.py not found)')

    # Output
    parser.add_argument('-o', '--output', type=str,
                       help='Output file path (e.g., fractal.png)')
    parser.add_argument('--no-show', action='store_true',
                       help='Don\'t display the plot window')
    parser.add_argument('--dpi', type=int, default=150,
                       help='Output image DPI')
    parser.add_argument('--save-annotated', action='store_true',
                       help='Also save an annotated version with axes and colorbar')
    parser.add_argument('--annotated-width', type=int, default=1200,
                       help='Width of annotated version in pixels')
    parser.add_argument('--annotated-height', type=int, default=1000,
                       help='Height of annotated version in pixels')

    # Presets
    parser.add_argument('--preset', type=str,
                       choices=['full', 'seahorse', 'elephant', 'spiral', '4k', '8k'],
                       help='Use a preset region (overrides x/y min/max)')

    # Performance options
    parser.add_argument('--disable-tiling', action='store_true',
                       help='Disable automatic tiling (use single-pass rendering)')

    return parser.parse_args()


def apply_preset(args: argparse.Namespace):
    """Apply preset region parameters."""
    presets = {
        'full': {
            'xmin': -2.5, 'xmax': 1.0,
            'ymin': -1.25, 'ymax': 1.25,
            'width': 800, 'height': 800,
        },
        'seahorse': {
            'xmin': -0.75, 'xmax': -0.735,
            'ymin': 0.095, 'ymax': 0.105,
            'width': 7680, 'height': 4320,
            'max_iterations': 500,
            'save_annotated': True,
        },
        'elephant': {
            'xmin': 0.27, 'xmax': 0.29,
            'ymin': 0.005, 'ymax': 0.018,
            'width': 1920, 'height': 1080,
            'max_iterations': 300,
            'save_annotated': True,
        },
        'spiral': {
            'xmin': -0.77568377, 'xmax': -0.77568372,
            'ymin': 0.13646737, 'ymax': 0.13646740,
            'width': 3840, 'height': 2160,
            'max_iterations': 1000,
            'save_annotated': True,
        },
        '4k': {
            'xmin': -2.5, 'xmax': 1.0,
            'ymin': -1.3125, 'ymax': 1.3125,
            'width': 3840, 'height': 2160,
            'max_iterations': 200,
            'save_annotated': True,
        },
        '8k': {
            'xmin': -2.5, 'xmax': 1.0,
            'ymin': -1.3125, 'ymax': 1.3125,
            'width': 7680, 'height': 4320,
            'max_iterations': 300,
            'save_annotated': True,
        },
    }

    if args.preset and args.preset in presets:
        preset = presets[args.preset]

        # Store whether width/height were explicitly provided by user
        width_specified = args.width != 800
        height_specified = args.height != 800

        args.xmin = preset['xmin']
        args.xmax = preset['xmax']
        args.ymin = preset['ymin']
        args.ymax = preset['ymax']

        # Apply optional preset parameters only if not explicitly provided by user
        if 'width' in preset and not width_specified:
            args.width = preset['width']
        if 'height' in preset and not height_specified:
            args.height = preset['height']
        if 'max_iterations' in preset:
            args.max_iterations = preset['max_iterations']
        if 'save_annotated' in preset:
            args.save_annotated = preset['save_annotated']

        print(f"Using preset: {args.preset}")
        print(f"  Region: [{args.xmin}, {args.xmax}] × [{args.ymin}, {args.ymax}]")
        print(f"  Resolution: {args.width}×{args.height}")
        print(f"  Max iterations: {args.max_iterations}")


def main():
    """Main entry point for GPU-accelerated Mandelbrot generation."""
    # Parse arguments
    args = parse_arguments()

    # Apply preset if specified
    apply_preset(args)

    # Create parameters
    params = FractalParams(
        xmin=args.xmin,
        xmax=args.xmax,
        ymin=args.ymin,
        ymax=args.ymax,
        width=args.width,
        height=args.height,
        max_iterations=args.max_iterations,
        escape_radius=args.escape_radius,
        colormap=args.colormap,
        use_log_scale=not args.no_log_scale,
        custom_colour=args.custom_colour if hasattr(args, 'custom_colour') else None,
        output_path=Path(args.output) if args.output else None,
        show_plot=not args.no_show,
        dpi=args.dpi,
        save_annotated=args.save_annotated,
        annotated_width=args.annotated_width,
        annotated_height=args.annotated_height
    )

    # Detect best device
    device = get_device()

    # Determine whether to use tiled rendering
    use_tiling = should_use_tiling(params.width, params.height, device, args.disable_tiling)

    # Generate fractal on GPU
    if use_tiling:
        fractal = mandelbrot_gpu_tiled(params, device=device, tile_size=8192)
    else:
        fractal = mandelbrot_gpu(params, device=device)

    # Visualize
    print("Creating visualization...")
    fig = visualize_fractal(fractal, params)

    # Save if output path specified
    if params.output_path:
        save_fractal(fig, params.output_path, params.dpi)

        # Save annotated version if requested
        if params.save_annotated:
            output_stem = params.output_path.stem
            output_suffix = params.output_path.suffix
            annotated_path = params.output_path.parent / f"{output_stem}_annotated{output_suffix}"

            print("Creating annotated version...")
            fig_annotated = create_annotated_figure(fractal, params)

            print(f"Saving annotated version to {annotated_path}...")
            fig_annotated.savefig(annotated_path, dpi=100, bbox_inches='tight')
            print(f"Annotated version saved successfully!")
            plt.close(fig_annotated)

    # Show plot if requested
    if params.show_plot:
        print("Displaying plot...")
        plt.show()
    else:
        plt.close(fig)

    print("Done!")


if __name__ == '__main__':
    main()
