#!/usr/bin/env python3
"""
Simple test script to verify the fractal generator implementation.

This script runs a quick test to ensure all modules are working correctly
without requiring a full fractal generation.
"""

import sys
from pathlib import Path

print("Testing Custom Fractal Generator...")
print("=" * 60)

# Test 1: Import all modules
print("\n1. Testing module imports...")
try:
    from fractal_engine import FractalConfig, FractalEngine, FractalIterator
    from fractal_engine.memory import get_device, should_use_tiling
    from fractal_engine.colormap import get_available_colormaps, CUSTOM_COLOR_FUNCTIONS
    from presets import get_iterator, list_iterators, ITERATOR_REGISTRY
    print("   ✓ All modules imported successfully")
except ImportError as e:
    print(f"   ✗ Import error: {e}")
    sys.exit(1)

# Test 2: Check available iterators
print("\n2. Testing iterator registry...")
try:
    iterators = list_iterators()
    print(f"   ✓ Found {len(iterators)} iterators: {', '.join(iterators)}")

    for name in iterators:
        iterator_class = get_iterator(name)
        iterator = iterator_class()
        bounds = iterator.default_bounds
        print(f"     - {name}: {iterator.name}")
        print(f"       Bounds: [{bounds[0]}, {bounds[1]}] × [{bounds[2]}, {bounds[3]}]")
except Exception as e:
    print(f"   ✗ Iterator test failed: {e}")
    sys.exit(1)

# Test 3: Check device detection
print("\n3. Testing device detection...")
try:
    device = get_device()
    print(f"   ✓ Device detected: {device.upper()}")
except Exception as e:
    print(f"   ✗ Device detection failed: {e}")
    sys.exit(1)

# Test 4: Test configuration
print("\n4. Testing configuration...")
try:
    config = FractalConfig()
    print(f"   ✓ Default config created")
    print(f"     Resolution: {config.width}×{config.height}")
    print(f"     Max iterations: {config.max_iterations}")

    presets = FractalConfig.get_presets()
    print(f"   ✓ Found {len(presets)} presets: {', '.join(presets.keys())}")

    config_4k = FractalConfig.from_preset('4k')
    print(f"   ✓ 4K preset: {config_4k.width}×{config_4k.height} @ {config_4k.max_iterations} iterations")
except Exception as e:
    print(f"   ✗ Configuration test failed: {e}")
    sys.exit(1)

# Test 5: Check colormaps
print("\n5. Testing colormaps...")
try:
    colormaps = get_available_colormaps()
    print(f"   ✓ Custom colormaps: {', '.join(colormaps['custom'])}")
    print(f"   ✓ Matplotlib colormaps available: {len(colormaps['matplotlib'])}")
except Exception as e:
    print(f"   ✗ Colormap test failed: {e}")
    sys.exit(1)

# Test 6: Test memory estimation
print("\n6. Testing memory estimation...")
try:
    use_tiling, tile_size = should_use_tiling(1920, 1080, device, disable_tiling=False)
    print(f"   ✓ Memory estimation complete")
    print(f"     For 1920×1080: tiling={'enabled' if use_tiling else 'disabled'}")
    if use_tiling:
        print(f"     Recommended tile size: {tile_size}")
except Exception as e:
    print(f"   ✗ Memory estimation failed: {e}")
    sys.exit(1)

# Test 7: Test engine initialization
print("\n7. Testing engine initialization...")
try:
    config = FractalConfig(
        width=100,
        height=100,
        max_iterations=10,
        output_path=Path("test_output.png")
    )
    engine = FractalEngine(config)
    print(f"   ✓ Engine initialized successfully")
    print(f"     Device: {engine.device}")
except Exception as e:
    print(f"   ✗ Engine initialization failed: {e}")
    sys.exit(1)

# Test 8: Verify iterator interface
print("\n8. Testing iterator interface...")
try:
    mandelbrot = get_iterator('mandelbrot')()
    burning_ship = get_iterator('burning_ship')()

    print(f"   ✓ Mandelbrot iterator:")
    print(f"     Name: {mandelbrot.name}")
    print(f"     Default bounds: {mandelbrot.default_bounds}")
    print(f"     Default escape radius: {mandelbrot.default_escape_radius}")

    print(f"   ✓ Burning Ship iterator:")
    print(f"     Name: {burning_ship.name}")
    print(f"     Default bounds: {burning_ship.default_bounds}")
    print(f"     Default escape radius: {burning_ship.default_escape_radius}")
except Exception as e:
    print(f"   ✗ Iterator interface test failed: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("All tests passed! ✓")
print("\nThe fractal generator is ready to use.")
print("\nTo generate a fractal, run:")
print("  python main.py --output fractal.png --iterator mandelbrot")
print("\nFor more options, run:")
print("  python main.py --help")
