import math
import colorsys


# logarithmic colour scale
def logColor(distance, base, const, scale):
    color = -1 * math.log(distance, base)
    rgb = colorsys.hsv_to_rgb(const + scale * color,0.8,0.9)
    return tuple(round(i * 255) for i in rgb)


# power based colour scale
def powerColor(distance, exp, const, scale):
    color = distance**exp
    rgb = colorsys.hsv_to_rgb(const + scale * color,1 - 0.6 * color,0.9)
    return tuple(round(i * 255) for i in rgb)



def generate_colourbar(rule, data):
    # code to generate colour bar for data range
    return 


# Define dict of all colour functions
colour_functions = {
    "powerColor": powerColor,
    "logColor": logColor,
}