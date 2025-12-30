# Burning Ship Fractal Generator

A Python implementation of the Burning Ship fractal with progress tracking and command-line interface. Includes both CPU and GPU-accelerated versions.

## Installation

```bash
pip install -r requirements.txt
```

**Note:** The GPU version requires PyTorch with Metal Performance Shaders (Apple Silicon) or CUDA (NVIDIA GPU) support. The CPU version works on any system.

## GPU Acceleration 🚀

For significantly faster rendering on compatible hardware, use the GPU-accelerated version:

```bash
# Automatic device detection (MPS for Apple Silicon, CUDA for NVIDIA, CPU fallback)
python burning_ship_gpu.py --preset 16k -o output.png

# GPU version supports all the same options as CPU version
python burning_ship_gpu.py --width 7680 --height 4320 --max-iterations 300 -o 8k_fractal.png
```

**GPU Performance:**
- **Apple M-series**: 10-50x faster than CPU (depending on resolution)
- **NVIDIA GPUs**: 20-100x faster than CPU
- **Automatic tiling**: Handles ultra-high resolutions (32k+) by automatically splitting into tiles based on available memory
- **Memory-aware**: Automatically detects available GPU memory and switches to tiled rendering when needed

**GPU-Specific Presets:**
```bash
# 4K resolution optimized for GPU
python burning_ship_gpu.py --preset 4k -o 4k.png

# 8K resolution
python burning_ship_gpu.py --preset 8k -o 8k.png

# 16K resolution (uses tiling automatically)
python burning_ship_gpu.py --preset 16k -o 16k.png

# Ship detail at 32K resolution
python burning_ship_gpu.py --preset ship -o ship_32k.png

# Interesting lower regions
python burning_ship_gpu.py --preset lower -o lower.png
python burning_ship_gpu.py --preset lower_zoom -o lower_zoom.png
python burning_ship_gpu.py --preset lower_zoom_ship -o lower_zoom_ship.png
```

**Disable automatic tiling** (for smaller renders or testing):
```bash
python burning_ship_gpu.py --preset 4k --disable-tiling -o single_pass.png
```

**Aspect Ratio Adjustment** (GPU version only):
```bash
# Automatically adjust X axis to match image aspect ratio
python burning_ship_gpu.py --adjust-aspect x --width 1920 --height 1080 -o output.png

# Automatically adjust Y axis to match image aspect ratio
python burning_ship_gpu.py --adjust-aspect y --width 1920 --height 1080 -o output.png
```

The GPU version will warn you if there's an aspect ratio mismatch between your domain bounds and image dimensions.

## CPU Version Usage

### Basic Usage

Generate and display the full fractal:

```bash
python burning_ship.py
```

### Save to File

```bash
python burning_ship.py -o output/fractal.png
```

### Using Presets

Three presets are available for interesting regions:

```bash
# Full view (default region)
python burning_ship.py --preset full -o full.png

# The "ship" detail
python burning_ship.py --preset ship -o ship.png --max-iterations 200

# Antenna detail (requires high iterations)
python burning_ship.py --preset antenna -o antenna.png --max-iterations 500
```

### Custom Regions

Specify your own region of the complex plane:

```bash
python burning_ship.py \
  --xmin -1.8 --xmax -1.6 \
  --ymin -0.1 --ymax 0.1 \
  --max-iterations 200 \
  -o custom_region.png
```

### Resolution and Quality

```bash
# High resolution
python burning_ship.py --width 1920 --height 1080 --dpi 300 -o hires.png

# Low resolution for quick testing
python burning_ship.py --width 400 --height 400 --max-iterations 50
```

### Color Schemes

Try different colormaps:

```bash
python burning_ship.py --colormap viridis -o viridis.png
python burning_ship.py --colormap twilight -o twilight.png
python burning_ship.py --colormap plasma -o plasma.png
```

### All Options

```bash
python burning_ship.py --help
```

## Examples

**Full fractal with default settings:**
```bash
python burning_ship.py -o examples/full_view.png
```

**High-quality ship detail:**
```bash
python burning_ship.py \
  --preset ship \
  --width 1920 \
  --height 1080 \
  --max-iterations 500 \
  --colormap twilight \
  --dpi 300 \
  -o examples/ship_detail_hq.png
```

**Quick preview without saving:**
```bash
python burning_ship.py --width 400 --height 400 --max-iterations 50
```

## The Burning Ship Formula

The fractal is generated using the iteration:

```
z_{n+1} = (|Re(z_n)| + i|Im(z_n)|)^2 + c
```

Where `z_0 = 0` and `c` is the point being tested in the complex plane.

The key difference from the Mandelbrot set is the absolute value operation on both the real and imaginary components before squaring, which creates the distinctive "flipped" ship-like appearance.

## Project Structure

```
.
├── burning_ship.py      # CPU version with CLI
├── burning_ship_gpu.py  # GPU-accelerated version with automatic tiling
├── starter.ipynb        # Jupyter notebook for exploration
├── requirements.txt     # Python dependencies (includes PyTorch)
└── README.md           # This file
```

## Performance Notes

### CPU Version
- Progress is tracked per row using `tqdm`
- Generation time depends on resolution and max_iterations
- Typical times on a modern CPU:
  - 800×800, 100 iterations: ~30 seconds
  - 1920×1080, 200 iterations: ~3 minutes
  - 4K resolution, 500 iterations: ~15-20 minutes

### GPU Version
- Progress tracked per iteration with active pixel count
- Dramatically faster than CPU for high-resolution renders
- Typical times on Apple M1/M2 (MPS):
  - 4K (3840×2160), 200 iterations: ~10-20 seconds
  - 8K (7680×4320), 300 iterations: ~45-90 seconds
  - 16K (15360×8640), 500 iterations: ~4-8 minutes
  - 32K (30720×17280), 500 iterations: ~15-25 minutes (with automatic tiling)
- Memory usage scales with resolution; tiling automatically enabled when needed
- Early exit optimization: stops iterating when all points have escaped
