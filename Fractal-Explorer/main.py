#!/usr/bin/env python3
"""
Custom Fractal Generator - Main Entry Point

A flexible, GPU-accelerated fractal generation system supporting custom
iteration logic with automatic memory optimization and flexible colormaps.

Usage:
    python main.py --output fractal.png --iterator mandelbrot
    python main.py --output custom.png --iterator burning_ship --config my_config.py
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from fractal_engine import FractalConfig, FractalEngine
from fractal_engine.colormap import get_available_colormaps
from presets import get_iterator, list_iterators, ITERATOR_REGISTRY


def create_config_from_args(args) -> FractalConfig:
    """
    Create a FractalConfig from command-line arguments.

    Args:
        args: Parsed command-line arguments

    Returns:
        FractalConfig instance
    """
    # Start with preset if specified
    if args.preset:
        config = FractalConfig.from_preset(args.preset)
    else:
        config = FractalConfig()

    # Apply bounds if specified
    if args.xmin is not None:
        config.xmin = args.xmin
    if args.xmax is not None:
        config.xmax = args.xmax
    if args.ymin is not None:
        config.ymin = args.ymin
    if args.ymax is not None:
        config.ymax = args.ymax

    # Apply resolution if specified
    if args.width:
        config.width = args.width
    if args.height:
        config.height = args.height

    # Apply iteration settings
    if args.max_iterations:
        config.max_iterations = args.max_iterations
    if args.escape_radius:
        config.escape_radius = args.escape_radius

    # Apply colormap settings
    if args.colormap:
        config.colormap = args.colormap
    if args.custom_colormap:
        config.custom_colormap = args.custom_colormap
    if args.no_log_scale:
        config.use_log_scale = False

    # Apply GPU settings
    if args.device:
        config.device = args.device
    if args.tile_size:
        config.tile_size = args.tile_size
    if args.disable_tiling:
        config.disable_tiling = True

    # Apply output settings
    config.output_path = Path(args.output)
    config.show_plot = args.show
    if args.dpi:
        config.dpi = args.dpi
    if args.no_annotated:
        config.save_annotated = False

    # Apply aspect ratio adjustment
    if args.adjust_aspect:
        config.adjust_aspect_ratio = args.adjust_aspect

    return config


def main():
    """Main entry point for the fractal generator."""
    parser = argparse.ArgumentParser(
        description='Generate fractals with custom iteration logic',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate Mandelbrot set with default settings
  python main.py --output mandelbrot.png --iterator mandelbrot

  # Generate Burning Ship at 4K resolution
  python main.py --output ship_4k.png --iterator burning_ship --preset 4k

  # Custom region with specific colormap
  python main.py --output custom.png --iterator mandelbrot \\
    --xmin -0.75 --xmax -0.73 --ymin 0.095 --ymax 0.105 \\
    --colormap viridis --max-iterations 500

  # Use custom colormap with specific parameters
  python main.py --output custom_color.png --iterator burning_ship \\
    --custom-colormap smooth

Available iterators: """ + ', '.join(list_iterators())
    )

    # Required arguments
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output file path (e.g., fractal.png)'
    )

    parser.add_argument(
        '--iterator', '-i',
        type=str,
        required=True,
        help=f'Fractal iterator to use ({", ".join(list_iterators())})'
    )

    # Configuration preset
    parser.add_argument(
        '--preset', '-p',
        type=str,
        help='Configuration preset (default, 4k, 8k, 16k, fast, ultra)'
    )

    # Complex plane bounds
    bounds_group = parser.add_argument_group('complex plane bounds')
    bounds_group.add_argument('--xmin', type=float, help='Minimum real value')
    bounds_group.add_argument('--xmax', type=float, help='Maximum real value')
    bounds_group.add_argument('--ymin', type=float, help='Minimum imaginary value')
    bounds_group.add_argument('--ymax', type=float, help='Maximum imaginary value')

    # Resolution
    resolution_group = parser.add_argument_group('resolution')
    resolution_group.add_argument('--width', '-w', type=int, help='Image width in pixels')
    resolution_group.add_argument('--height', type=int, help='Image height in pixels')
    resolution_group.add_argument('--dpi', type=int, help='DPI for output image')

    # Iteration parameters
    iteration_group = parser.add_argument_group('iteration parameters')
    iteration_group.add_argument(
        '--max-iterations', '-m',
        type=int,
        help='Maximum number of iterations'
    )
    iteration_group.add_argument(
        '--escape-radius', '-e',
        type=float,
        help='Escape radius for iteration'
    )

    # Colormap settings
    colormap_group = parser.add_argument_group('colormap settings')
    colormap_group.add_argument(
        '--colormap', '-c',
        type=str,
        help='Matplotlib colormap name (hot, viridis, plasma, etc.)'
    )
    colormap_group.add_argument(
        '--custom-colormap',
        type=str,
        choices=['log', 'power', 'smooth'],
        help='Custom colormap function'
    )
    colormap_group.add_argument(
        '--no-log-scale',
        action='store_true',
        help='Disable logarithmic color scaling'
    )

    # GPU and memory settings
    gpu_group = parser.add_argument_group('GPU and memory settings')
    gpu_group.add_argument(
        '--device', '-d',
        type=str,
        choices=['cuda', 'mps', 'cpu'],
        help='Compute device to use'
    )
    gpu_group.add_argument(
        '--tile-size',
        type=int,
        help='Tile size for memory optimization'
    )
    gpu_group.add_argument(
        '--disable-tiling',
        action='store_true',
        help='Disable automatic tiling'
    )

    # Output settings
    output_group = parser.add_argument_group('output settings')
    output_group.add_argument(
        '--show', '-s',
        action='store_true',
        help='Display the image after generation'
    )
    output_group.add_argument(
        '--no-annotated',
        action='store_true',
        help='Do not save annotated version with axes and labels'
    )
    output_group.add_argument(
        '--adjust-aspect',
        type=str,
        choices=['x', 'y'],
        help='Adjust aspect ratio by expanding x or y axis'
    )

    # Utility options
    parser.add_argument(
        '--list-iterators',
        action='store_true',
        help='List all available iterators and exit'
    )
    parser.add_argument(
        '--list-colormaps',
        action='store_true',
        help='List all available colormaps and exit'
    )

    args = parser.parse_args()

    # Handle utility options
    if args.list_iterators:
        print("Available iterators:")
        for name in list_iterators():
            iterator_class = get_iterator(name)
            iterator = iterator_class()
            bounds = iterator.default_bounds
            print(f"  {name:20s} - {iterator.name}")
            print(f"    Default bounds: [{bounds[0]}, {bounds[1]}] × [{bounds[2]}, {bounds[3]}]")
        return 0

    if args.list_colormaps:
        colormaps = get_available_colormaps()
        print("Custom colormaps:")
        for name in colormaps['custom']:
            print(f"  {name}")
        print("\nMatplotlib colormaps:")
        for name in colormaps['matplotlib']:
            print(f"  {name}")
        return 0

    # Create configuration
    try:
        config = create_config_from_args(args)
    except Exception as e:
        print(f"Error creating configuration: {e}", file=sys.stderr)
        return 1

    # Get iterator
    try:
        iterator_class = get_iterator(args.iterator)
        iterator = iterator_class()

        # Use iterator's default bounds if not specified
        if args.xmin is None and args.xmax is None and args.ymin is None and args.ymax is None and not args.preset:
            bounds = iterator.default_bounds
            config.xmin, config.xmax, config.ymin, config.ymax = bounds
            print(f"Using default bounds for {iterator.name}: [{bounds[0]}, {bounds[1]}] × [{bounds[2]}, {bounds[3]}]")

        # Use iterator's default escape radius if not specified
        if args.escape_radius is None:
            config.escape_radius = iterator.default_escape_radius

    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Create output directory if it doesn't exist
    config.output_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate fractal
    try:
        engine = FractalEngine(config)
        engine.generate(iterator)
        engine.visualize(save=True, show=args.show)

        print(f"\nFractal generation complete!")
        return 0

    except Exception as e:
        print(f"Error during fractal generation: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
