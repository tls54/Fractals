# Custom Fractal Generator

A flexible, GPU-accelerated fractal generation system supporting custom iteration logic with automatic memory optimization and flexible colormaps.

## Features

- **Modular Design**: Define custom fractals by implementing a simple iterator interface
- **GPU Acceleration**: Automatic device detection (CUDA, MPS for Apple Silicon, or CPU)
- **Memory Optimization**: Automatic tiling for high-resolution images
- **Flexible Colormaps**: Support for both matplotlib colormaps and custom HSV-based color functions
- **Preset Iterators**: Built-in support for Mandelbrot and Burning Ship fractals
- **Configuration System**: All parameters configured via classes, not CLI args (except output name)

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Generate Mandelbrot Set

```bash
python main.py --output mandelbrot.png --iterator mandelbrot
```

### Generate Burning Ship Fractal

```bash
python main.py --output burning_ship.png --iterator burning_ship
```

### High-Resolution 4K Output

```bash
python main.py --output fractal_4k.png --iterator mandelbrot --preset 4k
```

### Custom Region

```bash
python main.py --output seahorse.png --iterator mandelbrot \
  --xmin -0.75 --xmax -0.73 --ymin 0.095 --ymax 0.105 \
  --max-iterations 500 --colormap viridis
```

### Custom Colormap

```bash
python main.py --output custom_color.png --iterator burning_ship \
  --custom-colormap smooth
```

## Usage

```
python main.py --output OUTPUT --iterator ITERATOR [OPTIONS]

Required arguments:
  --output, -o OUTPUT          Output file path (e.g., fractal.png)
  --iterator, -i ITERATOR      Fractal iterator to use (mandelbrot, burning_ship)

Optional arguments:
  --preset, -p PRESET          Configuration preset (default, 4k, 8k, 16k, fast, ultra)

  Complex plane bounds:
    --xmin XMIN                Minimum real value
    --xmax XMAX                Maximum real value
    --ymin YMIN                Minimum imaginary value
    --ymax YMAX                Maximum imaginary value

  Resolution:
    --width, -w WIDTH          Image width in pixels
    --height, -h HEIGHT        Image height in pixels
    --dpi DPI                  DPI for output image

  Iteration parameters:
    --max-iterations, -m N     Maximum number of iterations
    --escape-radius, -e R      Escape radius for iteration

  Colormap settings:
    --colormap, -c NAME        Matplotlib colormap (hot, viridis, plasma, etc.)
    --custom-colormap NAME     Custom colormap (log, power, smooth)
    --no-log-scale             Disable logarithmic color scaling

  GPU and memory:
    --device, -d DEVICE        Compute device (cuda, mps, cpu)
    --tile-size SIZE           Tile size for memory optimization
    --disable-tiling           Disable automatic tiling

  Output settings:
    --show, -s                 Display the image after generation
    --no-annotated             Do not save annotated version
    --adjust-aspect {x,y}      Adjust aspect ratio by expanding x or y axis

Utility options:
  --list-iterators             List all available iterators
  --list-colormaps             List all available colormaps
```

## Available Presets

- `default`: 1920×1080, 256 iterations
- `fast`: 800×600, 100 iterations (for quick testing)
- `4k`: 3840×2160, 300 iterations
- `8k`: 7680×4320, 400 iterations
- `16k`: 15360×8640, 500 iterations
- `ultra`: 30720×17280, 1000 iterations

## Available Iterators

List all available iterators:
```bash
python main.py --list-iterators
```

Built-in iterators:
- `mandelbrot`: Classic Mandelbrot set
- `burning_ship`: Burning Ship fractal

## Creating Custom Iterators

To create your own fractal, implement the `FractalIterator` interface:

```python
from fractal_engine.base_iterator import FractalIterator
import torch
from typing import Tuple

class MyCustomIterator(FractalIterator):
    @property
    def name(self) -> str:
        return "My Custom Fractal"

    @property
    def default_bounds(self) -> Tuple[float, float, float, float]:
        return (-2.0, 2.0, -2.0, 2.0)  # xmin, xmax, ymin, ymax

    def iterate_gpu(
        self,
        z_real: torch.Tensor,
        z_imag: torch.Tensor,
        c_real: torch.Tensor,
        c_imag: torch.Tensor,
        iteration: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Implement your iteration formula here
        # Example: z = z^3 + c
        z_real_sq = z_real * z_real
        z_imag_sq = z_imag * z_imag

        z_real_new = z_real * (z_real_sq - 3 * z_imag_sq) + c_real
        z_imag_new = z_imag * (3 * z_real_sq - z_imag_sq) + c_imag

        return z_real_new, z_imag_new
```

Register your iterator in `presets/__init__.py`:

```python
from .my_custom_iterator import MyCustomIterator

ITERATOR_REGISTRY = {
    'mandelbrot': MandelbrotIterator,
    'burning_ship': BurningShipIterator,
    'my_custom': MyCustomIterator,  # Add your iterator
}
```

## Architecture

```
Fractals/
├── main.py                      # CLI entry point
├── fractal_engine/              # Core engine
│   ├── config.py                # Configuration dataclass
│   ├── base_iterator.py         # Abstract iterator interface
│   ├── engine.py                # Main generation engine
│   ├── colormap.py              # Colormap utilities
│   └── memory.py                # Memory management
├── presets/                     # Preset iterators
│   ├── mandelbrot_iterator.py   # Mandelbrot implementation
│   └── burning_ship_iterator.py # Burning Ship implementation
└── outputs/                     # Generated images
```

## Memory Optimization

The engine automatically detects available GPU memory and uses tiling when needed:

- Estimates memory usage based on image dimensions
- Enables tiling if estimated usage > 70% of available memory
- Divides image into tiles that fit in memory
- Processes tiles sequentially with GPU cache clearing between tiles

You can control tiling behavior:
- `--disable-tiling`: Force single-pass generation
- `--tile-size SIZE`: Override automatic tile size calculation

## Colormaps

### Matplotlib Colormaps
Use any matplotlib colormap: `hot`, `viridis`, `plasma`, `inferno`, `magma`, etc.

```bash
python main.py --output fractal.png --iterator mandelbrot --colormap plasma
```

### Custom Colormaps
HSV-based custom colormaps for enhanced visual control:

- `log`: Logarithmic color scale
- `power`: Power-based color scale
- `smooth`: Smooth linear color scale

```bash
python main.py --output fractal.png --iterator mandelbrot --custom-colormap smooth
```

## Examples

### Seahorse Valley (Mandelbrot)
```bash
python main.py --output seahorse.png --iterator mandelbrot \
  --xmin -0.75 --xmax -0.735 --ymin 0.095 --ymax 0.105 \
  --max-iterations 500 --colormap twilight
```

### Burning Ship Antenna Detail
```bash
python main.py --output antenna.png --iterator burning_ship \
  --xmin -1.755 --xmax -1.745 --ymin 0.02 --ymax 0.03 \
  --max-iterations 400 --custom-colormap log
```

### 8K Ultra High Resolution
```bash
python main.py --output fractal_8k.png --iterator mandelbrot \
  --preset 8k --colormap inferno
```

## Performance

The engine provides detailed performance statistics:
- Total calculation time
- Pixels processed per second
- Time per pixel
- Memory usage and tiling information

Performance varies based on:
- Image resolution
- Maximum iterations
- GPU hardware
- Complexity of the iteration formula

## License

This project builds upon the existing Mandelbrot and Burning Ship implementations
in this repository, extending them into a flexible, modular fractal generation system.
