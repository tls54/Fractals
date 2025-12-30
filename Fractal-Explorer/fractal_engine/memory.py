"""
Memory estimation and device detection utilities.

This module provides functions to estimate GPU memory usage and determine
whether tiling is needed for fractal generation.
"""

import torch
import psutil
from typing import Tuple, Optional


def get_device(preferred_device: Optional[str] = None) -> str:
    """
    Determine the best available compute device.

    Args:
        preferred_device: Optional preferred device ('cuda', 'mps', or 'cpu')

    Returns:
        Device string ('cuda', 'mps', or 'cpu')
    """
    if preferred_device:
        if preferred_device == 'cuda' and torch.cuda.is_available():
            return 'cuda'
        elif preferred_device == 'mps' and torch.backends.mps.is_available():
            return 'mps'
        elif preferred_device == 'cpu':
            return 'cpu'
        else:
            print(f"Warning: Requested device '{preferred_device}' not available, falling back to auto-detection")

    if torch.backends.mps.is_available():
        return 'mps'
    elif torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'


def get_available_gpu_memory(device: str) -> int:
    """
    Get available GPU memory in bytes.

    Args:
        device: Device string ('cuda', 'mps', or 'cpu')

    Returns:
        Available memory in bytes
    """
    if device == 'cuda':
        return torch.cuda.get_device_properties(0).total_memory
    elif device == 'mps':
        # MPS doesn't expose memory directly, use conservative estimate
        # Assume we can use 50% of system RAM
        system_memory = psutil.virtual_memory().total
        return int(system_memory * 0.5)
    else:
        # CPU: assume large amount (effectively disable tiling)
        return 100 * 1024 * 1024 * 1024  # 100 GB


def estimate_memory_usage(width: int, height: int) -> int:
    """
    Estimate GPU memory required for fractal generation.

    This accounts for all persistent and temporary tensors used during
    iteration, including real/imaginary components, iteration counts,
    and temporary calculation buffers.

    Args:
        width: Image width in pixels
        height: Image height in pixels

    Returns:
        Estimated memory usage in bytes
    """
    total_pixels = width * height
    memory_bytes = 0

    # Persistent tensors (exist throughout computation)
    memory_bytes += total_pixels * 4  # x_grid (float32)
    memory_bytes += total_pixels * 4  # y_grid (float32)
    memory_bytes += total_pixels * 4  # c_real (float32)
    memory_bytes += total_pixels * 4  # c_imag (float32)
    memory_bytes += total_pixels * 4  # z_real (float32)
    memory_bytes += total_pixels * 4  # z_imag (float32)
    memory_bytes += total_pixels * 4  # iterations (int32)
    memory_bytes += total_pixels * 1  # mask (bool)

    # Temporary tensors (created during iteration loop)
    memory_bytes += total_pixels * 4  # z_real_new
    memory_bytes += total_pixels * 4  # z_imag_new
    memory_bytes += total_pixels * 4  # z_magnitude_sq
    memory_bytes += total_pixels * 1  # escaped (bool)

    # Additional temporary tensors for some fractals (e.g., Burning Ship)
    memory_bytes += total_pixels * 4  # z_real_abs
    memory_bytes += total_pixels * 4  # z_imag_abs

    # Add 20% safety margin for PyTorch overhead
    memory_bytes = int(memory_bytes * 1.2)

    return memory_bytes


def should_use_tiling(
    width: int,
    height: int,
    device: str,
    disable_tiling: bool = False
) -> Tuple[bool, int]:
    """
    Determine if tiling is needed based on memory constraints.

    Args:
        width: Image width in pixels
        height: Image height in pixels
        device: Device string ('cuda', 'mps', or 'cpu')
        disable_tiling: If True, force disable tiling

    Returns:
        Tuple of (should_tile, recommended_tile_size)
    """
    if disable_tiling:
        return False, max(width, height)

    estimated_memory = estimate_memory_usage(width, height)
    available_memory = get_available_gpu_memory(device)

    # Use tiling if estimated usage exceeds 70% of available memory
    memory_threshold = 0.7 * available_memory
    needs_tiling = estimated_memory > memory_threshold

    if needs_tiling:
        # Calculate tile size that fits in memory
        # Target 60% of available memory per tile for safety
        target_memory_per_tile = 0.6 * available_memory

        # Estimate pixels per tile (accounting for all tensors)
        bytes_per_pixel = estimated_memory / (width * height)
        pixels_per_tile = int(target_memory_per_tile / bytes_per_pixel)

        # Calculate square tile dimension
        tile_size = int(pixels_per_tile ** 0.5)

        # Round down to nearest power of 2 for efficiency
        tile_size = 2 ** (tile_size.bit_length() - 1)

        # Ensure minimum tile size
        tile_size = max(tile_size, 512)

        return True, tile_size
    else:
        return False, max(width, height)


def print_memory_info(width: int, height: int, device: str, use_tiling: bool, tile_size: int) -> None:
    """
    Print memory usage information.

    Args:
        width: Image width in pixels
        height: Image height in pixels
        device: Device string
        use_tiling: Whether tiling is being used
        tile_size: Tile size if tiling is used
    """
    estimated = estimate_memory_usage(width, height)
    available = get_available_gpu_memory(device)

    print(f"\nMemory Information:")
    print(f"  Device: {device.upper()}")
    print(f"  Available memory: {available / (1024**3):.2f} GB")
    print(f"  Estimated usage: {estimated / (1024**3):.2f} GB")
    print(f"  Memory utilization: {(estimated / available) * 100:.1f}%")

    if use_tiling:
        num_tiles_x = (width + tile_size - 1) // tile_size
        num_tiles_y = (height + tile_size - 1) // tile_size
        total_tiles = num_tiles_x * num_tiles_y
        print(f"  Tiling: ENABLED ({tile_size}×{tile_size} tiles)")
        print(f"  Number of tiles: {total_tiles} ({num_tiles_x}×{num_tiles_y})")
    else:
        print(f"  Tiling: DISABLED")


def clear_gpu_cache(device: str) -> None:
    """
    Clear GPU memory cache.

    Args:
        device: Device string ('cuda', 'mps', or 'cpu')
    """
    if device == 'cuda':
        torch.cuda.empty_cache()
    elif device == 'mps':
        torch.mps.empty_cache()
    # No cache clearing needed for CPU
