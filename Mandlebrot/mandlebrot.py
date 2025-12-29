### Inspiration for Mandelbrot set generation from: 
### https://medium.com/swlh/visualizing-the-mandelbrot-set-using-python-50-lines-f6aa5a05cf0f

from PIL import Image
import os
from datetime import datetime as dt
from time import time
from colour_rules import powerColor, logColor, colour_functions

# TODO: add y offset param.
def generate_mandelbrot(width=1000, precision=500, colour_rule="powerColor", zoom=1.0, offset=0.65):
    """Generate a Mandelbrot set image.

    Args:
        width (int): Image width in pixels.
        precision (int): Maximum iterations for divergence.
        colour_rule (str): Coloring method, either 'powerColor' or 'logColor'.
        zoom (float): Zoom level, where 1.0 is the default view, >1 zooms in, and <1 zooms out.

    Returns:
        image: callable image object.
        str: Filename of the saved image.
    """

    # Ensure output directory exists
    os.makedirs("images", exist_ok=True)

    # Define aspect ratio and calculate height
    aspect_ratio = 4 / 3
    height = round(width / aspect_ratio)

    # Frame parameters
    x, y = -offset, 0  # Center of the Mandelbrot set
    base_x_range = 3.4  # Default full range

    x_range = base_x_range / zoom  # Adjust x range based on zoom
    y_range = x_range / aspect_ratio

    min_x, max_x = x - x_range / 2, x + x_range / 2
    min_y, max_y = y - y_range / 2, y + y_range / 2


    if colour_rule not in colour_functions:
        raise ValueError(f"Invalid colour_rule '{colour_rule}'. Choose from {list(colour_functions.keys())}.")

    color_func = colour_functions[colour_rule]

    # Create image
    img = Image.new("RGB", (width, height), color="black")
    pixels = img.load()

    # Compute Mandelbrot set
    for row in range(height):
        for col in range(width):
            x = min_x + col * x_range / width
            y = max_y - row * y_range / height
            old_x, old_y = x, y

            for i in range(precision + 1):
                a, b = x * x - y * y, 2 * x * y  # z^2 real and imaginary components
                x, y = a + old_x, b + old_y
                if x * x + y * y > 4:
                    break

            if i < precision:
                distance = (i + 1) / (precision + 1)
                pixels[col, row] = color_func(distance, 0.2, 0.27, 1.0)

        # Progress update
        index = (row + 1) * width
        print(f"{index:,} / {width * height:,}, {round(index / (width * height) * 100, 1)}%", end="\r")

    # Save image
    timestamp = dt.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"images/output_{timestamp}_{width}x{height}px_zoom{zoom:.2f}.png"
    img.save(filename)

    print(f"\nGeneration completed successfully: {width * height:,} pixels")
    return img, filename


# Run with default values if executed directly
if __name__ == "__main__":
    generate_mandelbrot(colour_rule="powerColor", zoom=1.0)