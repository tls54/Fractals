# Burning Ship Fractal Generator

A well-engineered Python implementation of the Burning Ship fractal with progress tracking and command-line interface.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

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
├── burning_ship.py      # Main script with CLI
├── starter.ipynb        # Jupyter notebook for exploration
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Performance Notes

- Progress is tracked per row using `tqdm`
- Generation time depends on resolution and max_iterations
- Typical times on a modern CPU:
  - 800×800, 100 iterations: ~30 seconds
  - 1920×1080, 200 iterations: ~3 minutes
  - 4K resolution, 500 iterations: ~15-20 minutes
