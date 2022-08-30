import dataclasses
import gc
import os
from functools import partial

import seam_carving
import torch
from fast_map import fast_map
from torch.utils import data

from static.utils.image import load_image, seam_carving_resize, save_image

seam_carving.carve.MAX_MEAN_ENERGY = 10.0


# Everything here takes a dict an return a dict

def preprocess(d):
    image = load_image(os.path.join(d['image_path'], d['image_id'] + '.tiff'))
    image = seam_carving_resize(image)
    d['image'] = image
    return d


def save(d):
    save_image(d['image'], output_path=os.path.join(d['output_path'], d['image_id'] + 'png'))


def parallel_preprocess_data(image_ids, image_path='.', output_path='.', on_result=None):
    ds = [dict(image_id=image_id,
               image_path=image_path,
               output_path=output_path
               ) for image_id in image_ids]

    for d in fast_map(preprocess, ds, threads_limit=4):
        if on_result is not None:
            on_result(d)
        gc.collect()


parallel_preprocess_data(image_ids=[],
                         image_path='.',
                         output_path='.',
                         on_result=save)
