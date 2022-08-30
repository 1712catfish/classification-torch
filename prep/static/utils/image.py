import gc
import os

import rasterio
from fast_map import fast_map
from funcy import curry
from rasterio.enums import Resampling
from rasterio.transform import Affine
import seam_carving
from toolz import pipe


def aspect_ratio_shrink(a, b, c):
    return int(a / min(a, b) * c), int(b / min(a, b) * c)


def load_image(image_path):
    return rasterio.open(image_path).read(resampling=Resampling.bilinear)


def seam_carving_resize(image):
    h, w, _ = image.shape
    image = image.transpose((1, 2, 0))
    image = seam_carving.resize(image, aspect_ratio_shrink(h, w, 512))
    image = image.transpose((2, 0, 1))
    return image


def save_image(image, output_path='image.png'):
    with rasterio.open(image, 'w',
                       driver='png', height=image.shape[1], width=image.shape[2],
                       dtype=image.dtype, count=3) as f:
        f.write(image)



