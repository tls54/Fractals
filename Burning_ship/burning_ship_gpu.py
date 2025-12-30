#!/usr/bin/env python3
"""
Burning Ship Fractal Generator - GPU Accelerated Version
Uses PyTorch with Metal Performance Shaders for M-series Macs
"""

import argparse
import time
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import configuration and visualization from original script
from burning_ship import FractalParams, visualize_fractal, save_fractal, create_annotated_figure


def burning_ship_gpu(params: FractalParams, device: str = 'mps') -> np.ndarray:
    """
    Generate Burning Ship fractal using GPU acceleration.

    Args:
        params: Fractal generation parameters
        device: Device to use ('mps' for Metal on M-series, 'cuda' for NVIDIA, 'cpu' fallback)

    Returns:
        2D numpy array of iteration counts
    """
    print(f"Generating {params.width}x{params.height} fractal on {device.upper()}...")

    # Create coordinate grids on CPU first
    x = torch.linspace(params.xmin, params.xmax, params.width, dtype=torch.float32)
    y = torch.linspace(params.ymin, params.ymax, params.height, dtype=torch.float32)

    # Create meshgrid for all points in complex plane
    # Use 'ij' indexing to match original orientation (rows=y, cols=x)
    y_grid, x_grid = torch.meshgrid(y, x, indexing='ij')

    # Move to GPU
    x_grid = x_grid.to(device)
    y_grid = y_grid.to(device)

    # Initialize complex number c = x + iy for all points
    # We'll work with real and imaginary parts separately for efficiency
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

    # Iterate the Burning Ship formula with progress bar
    with tqdm(total=params.max_iterations, desc="GPU iterations", unit="iter") as pbar:
        for i in range(params.max_iterations):
            # Burning Ship: take absolute values of real and imaginary parts
            z_real_abs = torch.abs(z_real)
            z_imag_abs = torch.abs(z_imag)

            # Square: (a + bi)^2 = a^2 - b^2 + 2abi
            z_real_new = z_real_abs * z_real_abs - z_imag_abs * z_imag_abs + c_real
            z_imag_new = 2 * z_real_abs * z_imag_abs + c_imag

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
    result = iterations.cpu().numpy()  # No transpose needed with 'ij' indexing

    # Print timing statistics
    total_pixels = params.width * params.height
    print(f"\n✓ GPU calculation completed in {elapsed_time:.2f} seconds")
    print(f"  Total pixels: {total_pixels:,}")
    print(f"  Pixels/second: {total_pixels/elapsed_time:,.0f}")
    print(f"  Time per pixel: {elapsed_time/total_pixels*1_000_000:.3f} µs")

    return result


def burning_ship_gpu_tiled(params: FractalParams, device: str = 'mps', tile_size: int = 8192) -> np.ndarray:
    """
    Generate Burning Ship fractal using GPU acceleration with tiled rendering.

    Splits the image into tiles to reduce memory usage for large renders.

    Args:
        params: Fractal generation parameters
        device: Device to use ('mps' for Metal on M-series, 'cuda' for NVIDIA, 'cpu' fallback)
        tile_size: Size of square tiles (default 8192x8192)

    Returns:
        2D numpy array of iteration counts
    """
    print(f"Generating {params.width}x{params.height} fractal on {device.upper()} using tiled rendering...")
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

            # Iterate the Burning Ship formula
            with tqdm(total=params.max_iterations, desc=f"  Tile {tile_num}/{total_tiles}", unit="iter") as pbar:
                for i in range(params.max_iterations):
                    # Burning Ship: take absolute values
                    z_real_abs = torch.abs(z_real)
                    z_imag_abs = torch.abs(z_imag)

                    # Square: (a + bi)^2 = a^2 - b^2 + 2abi
                    z_real_new = z_real_abs * z_real_abs - z_imag_abs * z_imag_abs + c_real
                    z_imag_new = 2 * z_real_abs * z_imag_abs + c_imag

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


def adjust_aspect_ratio(xmin: float, xmax: float, ymin: float, ymax: float,
                        width: int, height: int, adjust_axis: str) -> tuple[float, float, float, float]:
    """
    Adjust domain bounds to match image aspect ratio.

    Args:
        xmin, xmax, ymin, ymax: Current domain bounds
        width, height: Image dimensions in pixels
        adjust_axis: Which axis to adjust ('x' or 'y')

    Returns:
        Adjusted (xmin, xmax, ymin, ymax)
    """
    image_aspect = width / height
    domain_x_range = xmax - xmin
    domain_y_range = ymax - ymin
    domain_aspect = domain_x_range / domain_y_range

    if abs(image_aspect - domain_aspect) < 0.001:
        # Already matched
        return xmin, xmax, ymin, ymax

    if adjust_axis == 'x':
        # Adjust X range to match image aspect ratio
        # image_aspect = new_x_range / y_range
        new_x_range = image_aspect * domain_y_range
        x_center = (xmin + xmax) / 2
        xmin = x_center - new_x_range / 2
        xmax = x_center + new_x_range / 2
        print(f"ℹ Adjusted X range to [{xmin:.6f}, {xmax:.6f}] to match {width}×{height} aspect ratio")

    elif adjust_axis == 'y':
        # Adjust Y range to match image aspect ratio
        # image_aspect = x_range / new_y_range
        new_y_range = domain_x_range / image_aspect
        y_center = (ymin + ymax) / 2
        ymin = y_center - new_y_range / 2
        ymax = y_center + new_y_range / 2
        print(f"ℹ Adjusted Y range to [{ymin:.6f}, {ymax:.6f}] to match {width}×{height} aspect ratio")

    return xmin, xmax, ymin, ymax


def check_aspect_ratio(xmin: float, xmax: float, ymin: float, ymax: float,
                       width: int, height: int) -> None:
    """
    Warn user if domain and image aspect ratios don't match.

    Args:
        xmin, xmax, ymin, ymax: Domain bounds
        width, height: Image dimensions
    """
    image_aspect = width / height
    domain_x_range = xmax - xmin
    domain_y_range = ymax - ymin
    domain_aspect = domain_x_range / domain_y_range

    if abs(image_aspect - domain_aspect) > 0.001:
        print(f"⚠ WARNING: Aspect ratio mismatch!")
        print(f"  Image aspect ratio: {image_aspect:.3f} ({width}×{height})")
        print(f"  Domain aspect ratio: {domain_aspect:.3f} ({domain_x_range:.3f}×{domain_y_range:.3f})")
        print(f"  This will result in letterboxing (white bars) in the output image.")
        print(f"  Use --adjust-aspect x or --adjust-aspect y to auto-correct.")


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

    # Temporary tensors in iteration loop (worst case all exist simultaneously)
    memory_bytes += total_pixels * 4  # z_real_abs
    memory_bytes += total_pixels * 4  # z_imag_abs
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
        # CUDA provides direct memory query
        return torch.cuda.get_device_properties(0).total_memory
    elif device == 'mps':
        # MPS uses unified memory - estimate based on system RAM
        # For M-series Macs, unified memory is shared
        # Conservative estimate: assume 50% of system RAM available for GPU
        import os
        import platform
        if platform.system() == 'Darwin':
            # Get total physical memory on macOS
            total_memory = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
            # Use 50% as conservative estimate for GPU operations
            return total_memory // 2
        else:
            # Fallback: assume 16GB available
            return 16 * 1024 * 1024 * 1024
    else:
        # CPU - return large value to effectively disable tiling by default
        return 100 * 1024 * 1024 * 1024  # 100GB


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
    """Parse command-line arguments for GPU version."""
    parser = argparse.ArgumentParser(
        description='Generate the Burning Ship fractal with GPU acceleration',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Region parameters
    parser.add_argument('--xmin', type=float, default=-2.0,
                       help='Minimum real value')
    parser.add_argument('--xmax', type=float, default=1.0,
                       help='Maximum real value')
    parser.add_argument('--ymin', type=float, default=-2.0,
                       help='Minimum imaginary value')
    parser.add_argument('--ymax', type=float, default=1.0,
                       help='Maximum imaginary value')

    # Resolution
    parser.add_argument('--width', type=int, default=800,
                       help='Image width in pixels')
    parser.add_argument('--height', type=int, default=800,
                       help='Image height in pixels')

    # Aspect ratio adjustment
    parser.add_argument('--adjust-aspect', type=str, choices=['x', 'y', 'none'],
                       help='Auto-adjust domain bounds to match image aspect ratio (x=expand X axis, y=expand Y axis)')

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
    parser.add_argument('--preset', type=str, choices=['full', 'ship', '4k', '8k', '16k', 'lower', 'lower_zoom', 'lower_zoom_ship'],
                       help='Use a preset region (overrides x/y min/max)')

    # Performance options
    parser.add_argument('--disable-tiling', action='store_true',
                       help='Disable automatic tiling (use single-pass rendering)')

    return parser.parse_args()


def apply_preset(args: argparse.Namespace):
    """Apply preset region parameters with proper aspect ratios."""
    presets = {
        'full': {
            'xmin': -2.0, 'xmax': 1.0,
            'ymin': -2.0, 'ymax': 1.0,
            'width': 800, 'height': 800,
            'save_annotated': True,
        },
        'ship': {
            'xmin': -1.8, 'xmax': -1.6,  # X range = 0.2
            'ymin': -0.084374, 'ymax': 0.028126,  # Y range = 0.1125 for 16:9
            'width': 15360*2, 'height': 8640*2,
            'max_iterations': 300,
            'save_annotated': True,
        },
        '4k': {
            'xmin': -2, 'xmax': 1.73333,  # X range = 5.333333 for 16:9
            'ymin': -1.6, 'ymax': 0.5,  # Y range = 3.0
            'width': 3840, 'height': 2160,
            'max_iterations': 200,
            'save_annotated': True,
        },
        '8k': {
            'xmin': -2, 'xmax': 1.73333,  # X range = 5.333333 for 16:9
            'ymin': -1.6, 'ymax': 0.5,  # Y range = 3.0
            'width': 7680, 'height': 4320,
            'max_iterations': 300,
            'save_annotated': True,
        },
        '16k': {
            'xmin': -2, 'xmax': 1.7333333333333334,
            'ymin': -1.6, 'ymax': 0.5,
            'width': 15360, 'height': 8640,
            'max_iterations': 500,
            'save_annotated': True,
        },
        'lower': {
            'xmin': -1.25, 'xmax': -0.35,  # X range = 0.9 
            'ymin': -1.2563, 'ymax': -0.75,  # Y range = 0.5063
            'width': 15360, 'height': 8640,  # 16:9 aspect ratio at ~16K
            'max_iterations': 500,
            'save_annotated': True,
            'no_show': True,
        },
        'lower_zoom': {
            'xmin': -0.8615, 'xmax': -0.6485 , 
            'ymin': -1.2 , 'ymax': -1.08 ,  
            'width': 15360*2, 'height': 8640*2,  # 16:9 aspect ratio at ~16K
            'max_iterations': 500,
            'save_annotated': True,
            'no_show': True,
        },    
        'lower_zoom_ship': {
            'xmin': -0.84, 'xmax': -0.61 , 
            'ymin': -0.984 , 'ymax': -0.855,  
            'width': int(15360*1), 'height': int(8640*1),  # 16:9 aspect ratio at ~16K
            'max_iterations': 500,
            'save_annotated': True,
            'no_show': True,
        },     

    }

    if args.preset and args.preset in presets:
        preset = presets[args.preset]

        # Store whether width/height were explicitly provided by user
        # Check if they differ from defaults (800x800)
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
        if 'no_show' in preset:
            args.no_show = preset['no_show']

        print(f"Using preset: {args.preset}")
        print(f"  Region: [{args.xmin}, {args.xmax}] × [{args.ymin}, {args.ymax}]")
        print(f"  Resolution: {args.width}×{args.height}")
        print(f"  Max iterations: {args.max_iterations}")


def main():
    """Main entry point for GPU-accelerated version."""
    # Parse arguments
    args = parse_arguments()

    # Apply preset if specified
    apply_preset(args)

    # Apply aspect ratio adjustment if requested
    if args.adjust_aspect and args.adjust_aspect != 'none':
        args.xmin, args.xmax, args.ymin, args.ymax = adjust_aspect_ratio(
            args.xmin, args.xmax, args.ymin, args.ymax,
            args.width, args.height, args.adjust_aspect
        )
    else:
        # Check for aspect ratio mismatch and warn
        check_aspect_ratio(args.xmin, args.xmax, args.ymin, args.ymax,
                          args.width, args.height)

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
        fractal = burning_ship_gpu_tiled(params, device=device, tile_size=8192)
    else:
        fractal = burning_ship_gpu(params, device=device)

    # Visualize (reuse from original script)
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
