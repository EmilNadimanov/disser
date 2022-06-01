import numpy as np
from PIL import Image

STRIPE_INTERVAL_UP = 15
STRIPE_INTERVAL_DOWN = 10

def take_central_stripe_bin(binary: np.array,
                            down: int = STRIPE_INTERVAL_DOWN,
                            up: int = STRIPE_INTERVAL_UP):
    """ take a center line of a 2D np-array representation of a picture"""
    h, w = binary.shape
    axis = int(h * 0.55)
    upper, lower = axis - up, axis + down
    return binary[upper:lower, :]

def take_central_stripe(image: Image,
                        down: int = STRIPE_INTERVAL_DOWN,
                        up: int = STRIPE_INTERVAL_UP,
                        axis = None):
    """ take a center line of a Pillow image"""
    w, h = image.size
    if axis is None:
        axis = int(h * 0.55)

    upper, lower = (axis - up, axis + down)
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

def put_in_a_box(char: 'Image'):
    """
    cut an image with white background, making it a "box" that borders the first encountered non-white
    parts of the image, e.g. bordering each side of a letter
    """
    bin_img = binarize(char)
    height, width = bin_img.shape
    def find_border(bin_img):
        for (idx, row) in enumerate(bin_img):
            if np.sum(row) > 0:
                return idx
    
    left = find_border(bin_img.T)
    top = find_border(bin_img)
    right = width - find_border(bin_img.T[::-1, :])
    bottom = height - find_border(bin_img[::-1, :])
    
    return char.crop((left, top, right, bottom))


def expand2square(pil_img, color=(255,255,255)):
    """
    An image is a rectangle, and this function expands either its height or its width
    to make it a square.
    """
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

def add_margin(pil_img, top=0, right=0, bottom=0, left=0, color=(255,255,255)):
    """
    add margin to an image. Any side you want!
    """
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result