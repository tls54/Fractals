#!/usr/bin/env python3
"""
Burning Ship Fractal Generator

A well-engineered implementation of the Burning Ship fractal with progress tracking,
configurable parameters, and command-line interface.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import argparse
import time

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


@dataclass
class FractalParams:
    """Parameters for generating the Burning Ship fractal."""

    # Complex plane bounds
    xmin: float = -2.0
    xmax: float = 1.0
    ymin: float = -2.0
    ymax: float = 1.0

    # Resolution
    width: int = 800
    height: int = 800

    # Iteration parameters
    max_iterations: int = 100
    escape_radius: float = 2.0

    # Visualization
    colormap: str = 'hot'
    use_log_scale: bool = True

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


def burning_ship_iteration(c: complex, max_iterations: int, escape_radius: float) -> int:
    """
    Iterate the Burning Ship formula for a single complex number.

    The formula: z_{n+1} = (|Re(z_n)| + i|Im(z_n)|)^2 + c, with z_0 = 0

    Args:
        c: Complex number representing the point in the complex plane
        max_iterations: Maximum number of iterations to perform
        escape_radius: If |z| exceeds this, we consider it escaped

    Returns:
        Number of iterations before escape (or max_iterations if bounded)
    """
    z = 0 + 0j

    for iteration in range(max_iterations):
        # Key step: Take absolute values of real and imaginary parts
        z_abs = abs(z.real) + 1j * abs(z.imag)

        # Square and add c
        z = z_abs**2 + c

        # Check if we've escaped
        if abs(z) > escape_radius:
            return iteration

    return max_iterations


def generate_fractal(params: FractalParams) -> np.ndarray:
    """
    Generate the Burning Ship fractal with progress tracking.

    Args:
        params: Fractal generation parameters

    Returns:
        2D array where each element is the iteration count for that point
    """
    # Create coordinate arrays
    x = np.linspace(params.xmin, params.xmax, params.width)
    y = np.linspace(params.ymin, params.ymax, params.height)

    # Initialize the result array
    fractal = np.zeros((params.height, params.width), dtype=np.int32)

    # Compute for each row with progress bar
    total_pixels = params.width * params.height
    print(f"Generating {params.width}x{params.height} fractal ({total_pixels:,} pixels)...")

    start_time = time.time()

    for i in tqdm(range(params.height), desc="Processing rows", unit="row"):
        for j in range(params.width):
            # Create complex number c = x + iy
            c = x[j] + 1j * y[i]

            # Compute iterations for this point
            fractal[i, j] = burning_ship_iteration(
                c,
                params.max_iterations,
                params.escape_radius
            )

    elapsed_time = time.time() - start_time

    # Print timing statistics
    print(f"\nCalculation completed in {elapsed_time:.2f} seconds")
    print(f"  Total pixels: {total_pixels:,}")
    print(f"  Pixels/second: {total_pixels/elapsed_time:,.0f}")
    print(f"  Time per pixel: {elapsed_time/total_pixels*1000:.3f} ms")

    return fractal


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
    # figsize is in inches, so we divide pixels by DPI
    fig_width = params.width / params.dpi
    fig_height = params.height / params.dpi

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=params.dpi)

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

    # Apply logarithmic scaling if requested
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
        f'Burning Ship Fractal\n'
        f'Region: [{params.xmin:.4f}, {params.xmax:.4f}] × [{params.ymin:.4f}, {params.ymax:.4f}]i\n'
        f'Resolution: {params.width}×{params.height}, Max iterations: {params.max_iterations}',
        fontsize=11,
        pad=15
    )

    # Add colorbar
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
    # Don't use bbox_inches='tight' to preserve exact pixel dimensions
    fig.savefig(output_path, dpi=dpi, pad_inches=0)
    print(f"Saved successfully!")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate the Burning Ship fractal',
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
    parser.add_argument('--preset', type=str, choices=['full', 'ship', 'antenna', 'ship-hires'],
                       help='Use a preset region (overrides x/y min/max)')

    return parser.parse_args()


def apply_preset(args: argparse.Namespace):
    """Apply preset region parameters."""
    presets = {
        'full': {
            'xmin': -2.0, 'xmax': 1.0,
            'ymin': -2.0, 'ymax': 1.0,
        },
        'ship': {
            'xmin': -1.8, 'xmax': -1.6,
            'ymin': -0.1, 'ymax': 0.1,
        },
        'antenna': {
            'xmin': -1.755, 'xmax': -1.745,
            'ymin': 0.02, 'ymax': 0.03,
        },
        'ship-hires': {
            'xmin': -1.8, 'xmax': -1.6,
            'ymin': -0.1, 'ymax': 0.1,
            'width': 3840,
            'height': 2160,
            'max_iterations': 300,
            'save_annotated': True,
        }
    }

    if args.preset and args.preset in presets:
        preset = presets[args.preset]
        args.xmin = preset['xmin']
        args.xmax = preset['xmax']
        args.ymin = preset['ymin']
        args.ymax = preset['ymax']

        # Apply optional preset parameters
        if 'width' in preset:
            args.width = preset['width']
        if 'height' in preset:
            args.height = preset['height']
        if 'max_iterations' in preset:
            args.max_iterations = preset['max_iterations']
        if 'save_annotated' in preset:
            args.save_annotated = preset['save_annotated']

        print(f"Using preset: {args.preset}")


def main():
    """Main entry point."""
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
        output_path=Path(args.output) if args.output else None,
        show_plot=not args.no_show,
        dpi=args.dpi,
        save_annotated=args.save_annotated,
        annotated_width=args.annotated_width,
        annotated_height=args.annotated_height
    )

    # Generate fractal
    fractal = generate_fractal(params)

    # Visualize
    print("Creating visualization...")
    fig = visualize_fractal(fractal, params)

    # Save if output path specified
    if params.output_path:
        save_fractal(fig, params.output_path, params.dpi)

        # Save annotated version if requested
        if params.save_annotated:
            # Create annotated filename
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
