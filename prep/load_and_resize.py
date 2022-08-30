import rasterio
from rasterio.enums import Resampling
import pandas as pd
import os
from fast_map import fast_map
from tqdm.notebook import tqdm
import gc


def aspect_ratio_shrink(*xs, MIN=None, MAX=None, dtype=int):
    if MIN is None and MAX is None:
        print('Provided neither MIN nor MAX: will do nothing.')
        return xs

    if MIN is not None and MAX is not None:
        print('Provided both MIN and MAX : will shrink to MIN.')

    if MIN is not None:
        m = min(xs)
        return (dtype(v / m * MAX) for v in xs)

    if MAX is not None:
        m = max(xs)
        return (dtype(v / m * MAX) for v in xs)


def rasterio_load(path, MIN_LENGTH=512):
    with rasterio.open(path) as data:
        return data.read(
            out_shape=(data.count, *aspect_ratio_shrink(data.height, data.width, MIN=MIN_LENGTH)),
            resampling=Resampling.bilinear
        )


def rasterio_save(path, image):
    with rasterio.open(path, 'w', driver='png',
                       height=image.shape[1], width=image.shape[2],
                       dtype=image.dtype, count=3) as f:
        f.write(image)


def process(lst):
    def custom_load(image_id):
        return image_id, rasterio_load(os.path.join(DATA_PATH, image_id + '.tif'), MIN_LENGTH=1024)

    with tqdm(total=len(lst)) as pbar:
        for image_id, image in map(custom_load, lst):
            print(image.shape)
            rasterio_save(os.path.join(OUTPUT_PATH, image_id + 'png'), image)
            gc.collect()
            pbar.update()


INPUT_PATH = '../input/mayo-clinic-strip-ai/'
OUTPUT_PATH = 'data'
DATA_PATH = '../input/mayo-clinic-strip-ai/train'
N_SHARDS = 4
SHARD_ID = 0

df = pd.read_csv(os.path.join(INPUT_PATH, 'train.csv'))
ids = df['image_id']
i, j = len(ids) * SHARD_ID, len(ids) * (SHARD_ID + 1)

print(f'In shard {SHARD_ID + 1}/{N_SHARDS} ({i}:{j}):')
process(ids[i:j])
