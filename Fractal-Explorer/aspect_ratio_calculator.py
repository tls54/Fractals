#!/usr/bin/env python3
"""
Aspect Ratio Calculator for Fractal Bounds

This utility helps calculate complex plane bounds that match a specific
aspect ratio (e.g., 16:9, 4:3, 1:1) for fractal generation.

Usage:
    python aspect_ratio_calculator.py --xmin -2.5 --xmax 1.0 --ymin -1.25 --ymax 1.25 --aspect 16:9
    python aspect_ratio_calculator.py --xmin -2.5 --xmax 1.0 --ymin -1.25 --ymax 1.25 --width 1920 --height 1080
"""

import argparse
import sys
from typing import Tuple


def parse_aspect_ratio(aspect_str: str) -> float:
    """
    Parse aspect ratio string like '16:9' or '1.78' into a float.

    Args:
        aspect_str: Aspect ratio as 'W:H' or decimal string

    Returns:
        Aspect ratio as float
    """
    if ':' in aspect_str:
        width, height = aspect_str.split(':')
        return float(width) / float(height)
    else:
        return float(aspect_str)


def calculate_aspect_ratio(xmin: float, xmax: float, ymin: float, ymax: float) -> float:
    """Calculate the aspect ratio of given bounds."""
    x_range = xmax - xmin
    y_range = ymax - ymin
    return x_range / y_range


def adjust_bounds_expand_x(
    xmin: float, xmax: float, ymin: float, ymax: float, target_aspect: float
) -> Tuple[float, float, float, float]:
    """
    Adjust bounds by expanding the X axis to match target aspect ratio.

    Args:
        xmin, xmax, ymin, ymax: Current bounds
        target_aspect: Target aspect ratio (width/height)

    Returns:
        Tuple of (new_xmin, new_xmax, ymin, ymax)
    """
    y_range = ymax - ymin
    new_x_range = y_range * target_aspect
    x_center = (xmin + xmax) / 2

    new_xmin = x_center - new_x_range / 2
    new_xmax = x_center + new_x_range / 2

    return (new_xmin, new_xmax, ymin, ymax)


def adjust_bounds_shrink_x(
    xmin: float, xmax: float, ymin: float, ymax: float, target_aspect: float
) -> Tuple[float, float, float, float]:
    """
    Adjust bounds by shrinking the X axis to match target aspect ratio.

    Args:
        xmin, xmax, ymin, ymax: Current bounds
        target_aspect: Target aspect ratio (width/height)

    Returns:
        Tuple of (new_xmin, new_xmax, ymin, ymax)
    """
    y_range = ymax - ymin
    new_x_range = y_range * target_aspect
    x_center = (xmin + xmax) / 2

    new_xmin = x_center - new_x_range / 2
    new_xmax = x_center + new_x_range / 2

    return (new_xmin, new_xmax, ymin, ymax)


def adjust_bounds_expand_y(
    xmin: float, xmax: float, ymin: float, ymax: float, target_aspect: float
) -> Tuple[float, float, float, float]:
    """
    Adjust bounds by expanding the Y axis to match target aspect ratio.

    Args:
        xmin, xmax, ymin, ymax: Current bounds
        target_aspect: Target aspect ratio (width/height)

    Returns:
        Tuple of (xmin, xmax, new_ymin, new_ymax)
    """
    x_range = xmax - xmin
    new_y_range = x_range / target_aspect
    y_center = (ymin + ymax) / 2

    new_ymin = y_center - new_y_range / 2
    new_ymax = y_center + new_y_range / 2

    return (xmin, xmax, new_ymin, new_ymax)


def adjust_bounds_shrink_y(
    xmin: float, xmax: float, ymin: float, ymax: float, target_aspect: float
) -> Tuple[float, float, float, float]:
    """
    Adjust bounds by shrinking the Y axis to match target aspect ratio.

    Args:
        xmin, xmax, ymin, ymax: Current bounds
        target_aspect: Target aspect ratio (width/height)

    Returns:
        Tuple of (xmin, xmax, new_ymin, new_ymax)
    """
    x_range = xmax - xmin
    new_y_range = x_range / target_aspect
    y_center = (ymin + ymax) / 2

    new_ymin = y_center - new_y_range / 2
    new_ymax = y_center + new_y_range / 2

    return (xmin, xmax, new_ymin, new_ymax)


def print_bounds(label: str, xmin: float, xmax: float, ymin: float, ymax: float):
    """Print bounds in a formatted way."""
    x_range = xmax - xmin
    y_range = ymax - ymin
    aspect = x_range / y_range

    print(f"\n{label}")
    print(f"  X: [{xmin:.6f}, {xmax:.6f}]  (range: {x_range:.6f})")
    print(f"  Y: [{ymin:.6f}, {ymax:.6f}]  (range: {y_range:.6f})")
    print(f"  Aspect ratio: {aspect:.6f} ({aspect:.2f}:1)")
    print(f"  Python tuple: ({xmin}, {xmax}, {ymin}, {ymax})")


def main():
    parser = argparse.ArgumentParser(
        description='Calculate bounds for fractals with specific aspect ratios',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Adjust bounds to 16:9 aspect ratio
  python aspect_ratio_calculator.py --xmin -2.5 --xmax 1.0 --ymin -1.25 --ymax 1.25 --aspect 16:9

  # Adjust bounds to match specific resolution
  python aspect_ratio_calculator.py --xmin -2.5 --xmax 1.0 --ymin -1.25 --ymax 1.25 --width 1920 --height 1080

  # Use 4:3 aspect ratio
  python aspect_ratio_calculator.py --xmin -2.0 --xmax 1.0 --ymin -2.0 --ymax 1.0 --aspect 4:3

  # Use square (1:1) aspect ratio
  python aspect_ratio_calculator.py --xmin -2.0 --xmax 1.0 --ymin -1.5 --ymax 1.5 --aspect 1:1
        """
    )

    # Current bounds
    parser.add_argument('--xmin', type=float, required=True, help='Minimum real value')
    parser.add_argument('--xmax', type=float, required=True, help='Maximum real value')
    parser.add_argument('--ymin', type=float, required=True, help='Minimum imaginary value')
    parser.add_argument('--ymax', type=float, required=True, help='Maximum imaginary value')

    # Target aspect ratio (either as ratio or resolution)
    aspect_group = parser.add_mutually_exclusive_group(required=True)
    aspect_group.add_argument(
        '--aspect', '-a',
        type=str,
        help='Target aspect ratio (e.g., "16:9", "4:3", "1.78")'
    )
    aspect_group.add_argument(
        '--width',
        type=int,
        help='Target width (must be used with --height)'
    )

    parser.add_argument(
        '--height',
        type=int,
        help='Target height (must be used with --width)'
    )

    args = parser.parse_args()

    # Validate width/height combination
    if args.width and not args.height:
        print("Error: --width requires --height", file=sys.stderr)
        return 1
    if args.height and not args.width:
        print("Error: --height requires --width", file=sys.stderr)
        return 1

    # Determine target aspect ratio
    if args.aspect:
        try:
            target_aspect = parse_aspect_ratio(args.aspect)
            aspect_display = args.aspect
        except ValueError as e:
            print(f"Error parsing aspect ratio: {e}", file=sys.stderr)
            return 1
    else:
        target_aspect = args.width / args.height
        aspect_display = f"{args.width}×{args.height}"

    # Print current bounds
    print("=" * 70)
    print("ASPECT RATIO CALCULATOR FOR FRACTAL BOUNDS")
    print("=" * 70)

    current_aspect = calculate_aspect_ratio(args.xmin, args.xmax, args.ymin, args.ymax)

    print_bounds(
        "Current Bounds:",
        args.xmin, args.xmax, args.ymin, args.ymax
    )

    print(f"\nTarget aspect ratio: {aspect_display} = {target_aspect:.6f} ({target_aspect:.2f}:1)")

    # Calculate all possible adjustments
    print("\n" + "=" * 70)
    print("SUGGESTED ADJUSTMENTS")
    print("=" * 70)

    # Option 1: Expand X
    new_xmin, new_xmax, new_ymin, new_ymax = adjust_bounds_expand_x(
        args.xmin, args.xmax, args.ymin, args.ymax, target_aspect
    )
    if current_aspect < target_aspect:
        print_bounds(
            "Option 1: Expand X axis (wider view, same height)",
            new_xmin, new_xmax, new_ymin, new_ymax
        )

    # Option 2: Shrink X
    if current_aspect > target_aspect:
        new_xmin, new_xmax, new_ymin, new_ymax = adjust_bounds_shrink_x(
            args.xmin, args.xmax, args.ymin, args.ymax, target_aspect
        )
        print_bounds(
            "Option 2: Shrink X axis (narrower view, same height)",
            new_xmin, new_xmax, new_ymin, new_ymax
        )

    # Option 3: Expand Y
    new_xmin, new_xmax, new_ymin, new_ymax = adjust_bounds_expand_y(
        args.xmin, args.xmax, args.ymin, args.ymax, target_aspect
    )
    if current_aspect > target_aspect:
        print_bounds(
            "Option 3: Expand Y axis (same width, taller view)",
            new_xmin, new_xmax, new_ymin, new_ymax
        )

    # Option 4: Shrink Y
    if current_aspect < target_aspect:
        new_xmin, new_xmax, new_ymin, new_ymax = adjust_bounds_shrink_y(
            args.xmin, args.xmax, args.ymin, args.ymax, target_aspect
        )
        print_bounds(
            "Option 4: Shrink Y axis (same width, shorter view)",
            new_xmin, new_xmax, new_ymin, new_ymax
        )

    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)

    if current_aspect < target_aspect:
        # Current is too narrow (or too tall)
        print("\nYour current bounds are narrower than the target aspect ratio.")
        print("Recommended: Expand X axis (Option 1) to show more horizontal detail")
        print("Alternative: Shrink Y axis (Option 4) to crop vertical range")
    else:
        # Current is too wide (or too short)
        print("\nYour current bounds are wider than the target aspect ratio.")
        print("Recommended: Expand Y axis (Option 3) to show more vertical detail")
        print("Alternative: Shrink X axis (Option 2) to crop horizontal range")

    print("\nTo use these bounds in your fractal generator:")
    print("  python main.py --output fractal.png --iterator <name> \\")
    print("    --xmin <value> --xmax <value> --ymin <value> --ymax <value>")

    return 0


if __name__ == '__main__':
    sys.exit(main())
