# Mandelbrot Set Generator

High-performance Mandelbrot set generator with GPU acceleration using PyTorch and Metal Performance Shaders for M-series Macs.

## Features

- **GPU Acceleration**: Uses PyTorch with Metal (MPS) for M-series Macs, CUDA for NVIDIA GPUs, or CPU fallback
- **Memory Optimization**: Automatic tiled rendering for high-resolution images to reduce memory pressure
- **Custom Colour Rules**: Optional legacy colour functions (powerColor, logColor) or modern matplotlib colormaps
- **Command-Line Interface**: Comprehensive CLI with argparse for easy configuration
- **Presets**: Built-in presets for interesting regions of the Mandelbrot set
- **Annotated Output**: Optional annotated versions with axes, labels, and colorbars
- **Progress Tracking**: Real-time progress bars with active pixel counts

## Quick Start

### Basic Usage

```bash
# Generate default Mandelbrot set (800x800, displayed on screen)
python Accelerated_mandlebrot.py

# Generate and save to file
python Accelerated_mandlebrot.py -o output.png

# High resolution 4K render
python Accelerated_mandlebrot.py --preset 4k -o mandelbrot_4k.png

# Use custom colour scheme
python Accelerated_mandlebrot.py --custom-colour powerColor -o colorful.png
```

### Common Options

```bash
# Resolution and region
python Accelerated_mandlebrot.py --width 1920 --height 1080 \
    --xmin -2.5 --xmax 1.0 --ymin -1.25 --ymax 1.25

# Iteration control
python Accelerated_mandlebrot.py --max-iterations 500

# Visualization options
python Accelerated_mandlebrot.py --colormap viridis --no-log-scale

# Save annotated version with axes and labels
python Accelerated_mandlebrot.py --save-annotated -o output.png

# Don't display plot window (useful for batch processing)
python Accelerated_mandlebrot.py --no-show -o output.png
```

## Presets

Built-in presets for interesting regions:

- `full`: Complete Mandelbrot set (default view)
- `seahorse`: Seahorse valley region (zoomed detail)
- `elephant`: Elephant valley region
- `spiral`: Deep zoom into spiral structure
- `4k`: Full set at 4K resolution (3840×2160)
- `8k`: Full set at 8K resolution (7680×4320)

```bash
# Use a preset
python Accelerated_mandlebrot.py --preset seahorse -o seahorse.png
```

## Custom Colour Rules

The script supports both modern matplotlib colormaps and legacy custom colour functions:

### Matplotlib Colormaps (Default)

```bash
# Use built-in matplotlib colormaps
python Accelerated_mandlebrot.py --colormap hot       # (default)
python Accelerated_mandlebrot.py --colormap viridis
python Accelerated_mandlebrot.py --colormap plasma
python Accelerated_mandlebrot.py --colormap twilight
```

### Legacy Custom Colours

If `colour_rules.py` is available, you can use the original custom colour schemes:

```bash
# Power-based colour scaling (original default)
python Accelerated_mandlebrot.py --custom-colour powerColor

# Logarithmic colour scaling
python Accelerated_mandlebrot.py --custom-colour logColor
```

## Memory Management

The script automatically enables tiled rendering when generating large images that would exceed available GPU memory:

```bash
# Force disable tiling (may fail for very large images)
python Accelerated_mandlebrot.py --width 15360 --height 8640 --disable-tiling

# Tiling is automatic - it will split large renders into manageable tiles
python Accelerated_mandlebrot.py --width 15360 --height 8640  # Auto-tiles if needed
```

The memory estimation uses 70% of available GPU memory as a threshold. When exceeded, the image is automatically split into 8192×8192 tiles.

## Complete Option Reference

### Region Parameters
- `--xmin FLOAT`: Minimum real value (default: -2.5)
- `--xmax FLOAT`: Maximum real value (default: 1.0)
- `--ymin FLOAT`: Minimum imaginary value (default: -1.25)
- `--ymax FLOAT`: Maximum imaginary value (default: 1.25)

### Resolution
- `--width INT`: Image width in pixels (default: 800)
- `--height INT`: Image height in pixels (default: 800)

### Iteration Parameters
- `--max-iterations INT`: Maximum iterations per point (default: 100)
- `--escape-radius FLOAT`: Escape radius threshold (default: 2.0)

### Visualization
- `--colormap STR`: Matplotlib colormap name (default: 'hot')
- `--no-log-scale`: Disable logarithmic color scaling
- `--custom-colour STR`: Use custom colour rule ('powerColor' or 'logColor')

### Output
- `-o, --output PATH`: Output file path (e.g., fractal.png)
- `--no-show`: Don't display the plot window
- `--dpi INT`: Output image DPI (default: 150)
- `--save-annotated`: Also save an annotated version with axes and colorbar
- `--annotated-width INT`: Width of annotated version (default: 1200)
- `--annotated-height INT`: Height of annotated version (default: 1000)

### Presets
- `--preset STR`: Use a preset configuration (full, seahorse, elephant, spiral, 4k, 8k)

### Performance
- `--disable-tiling`: Disable automatic tiling (use single-pass rendering)

## Examples

### Example 1: High-Quality Seahorse Valley

```bash
python Accelerated_mandlebrot.py --preset seahorse \
    --max-iterations 1000 \
    --colormap twilight \
    --save-annotated \
    -o seahorse_valley.png
```

### Example 2: Ultra-High Resolution Full Set

```bash
python Accelerated_mandlebrot.py \
    --width 15360 --height 15360 \
    --max-iterations 500 \
    --colormap plasma \
    --no-show \
    -o mandelbrot_ultra_hires.png
```

### Example 3: Custom Region with Legacy Colours

```bash
python Accelerated_mandlebrot.py \
    --xmin -0.8 --xmax -0.7 \
    --ymin 0.05 --ymax 0.15 \
    --width 3840 --height 2160 \
    --max-iterations 750 \
    --custom-colour powerColor \
    --save-annotated \
    -o custom_region.png
```

## Performance Tips

1. **GPU Selection**: The script automatically selects the best available device (Metal > CUDA > CPU)
2. **Tiling**: Let the automatic tiling handle large images - it's optimized for your system
3. **Iterations**: Higher iteration counts reveal more detail but take longer
4. **Progress**: Watch the "active_pixels" count in the progress bar - when it drops to zero, all points have escaped

## Legacy Scripts

This directory also contains older implementations:

- `mandlebrot.py`: Original CPU-only implementation with basic custom colours
- `colour_rules.py`: Legacy colour functions (powerColor, logColor)

The accelerated version supersedes these with better performance and more features.

## Requirements

```bash
pip install torch numpy matplotlib tqdm
```

For M-series Macs, ensure you have PyTorch with MPS support installed.

## Technical Details

### Mandelbrot Formula

The script iterates the formula: `z = z² + c` where `z` starts at 0 and `c` is the complex coordinate.

Points are colored based on how many iterations it takes before `|z| > escape_radius`.

### GPU Optimization

- Processes all pixels in parallel on the GPU
- Uses masking to skip already-escaped points
- Early exit when all points have escaped
- Automatic memory management with tiled rendering

### Coordinate System

- Real axis (x): horizontal
- Imaginary axis (y): vertical
- Origin typically around (-0.5, 0) for centered view
