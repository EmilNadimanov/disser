import numpy as np
from PIL import Image

INTERVAL_FOR_STRIPE = 10

def take_central_stripe_bin(binary: np.array, interval: int = INTERVAL_FOR_STRIPE):
    """ take a center line of a 2D np-array representation of a picture"""
    h, w = binary.shape
    axis = h // 2
    upper, lower = axis - interval, axis + interval
    return binary[upper:lower, :]

def take_central_stripe(image: Image, interval: int = INTERVAL_FOR_STRIPE):
    """ take a center line of a Pillow image"""
    w, h = image.size
    axis = h // 2

    upper, lower = (axis - interval, axis + interval)
    return image.crop((0, upper, w, lower))

def binarize(pic):
    """
    get a representation of this picture as an numpy matrix
    numbers are floating-point non-integers (8-bit pixels, black and white)
    """
    wd, ht = pic.size
    pixels = np.array(pic.convert('L').getdata(), np.uint8)
    bin_img = 1 - (pixels.reshape((ht, wd)) / 255.0)
    return bin_img


def avg(iterable):
    return sum(iterable) / len(iterable)


def median(iterable):
    iterable = list(iterable)
    if len(iterable) % 2:
        return iterable(1 + len(iterable) // 2)
    else:
        mid = len(iterable) // 2
        return avg(iterable[mid - 1:mid])
