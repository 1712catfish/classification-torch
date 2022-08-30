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
        return (dtype(v / m * MIN) for v in xs)

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


def process(d):
    def custom_load(idd):
        return idd, rasterio_load(os.path.join(d['INPUT_PATH'], idd + '.tif'), MIN_LENGTH=1024)

    with tqdm(total=len(d['DATA'])) as pbar:
        for iid, image in map(custom_load, d['DATA']):
            print(image.shape)
            rasterio_save(os.path.join(d['OUTPUT_PATH'], iid + 'png'), image)
            gc.collect()
            pbar.update()


def sys_load_and_resize(CONFIG):
    df = pd.read_csv(os.path.join(CONFIG['CSV_PATH'], 'train.csv'))
    ids = df['image_id']
    i = len(ids) // CONFIG['NUM_SHARD'] * CONFIG['SHARD_ID']
    j = len(ids) // CONFIG['NUM_SHARD'] * (CONFIG['SHARD_ID'] + 1)

    print(f"In shard {CONFIG['SHARD_ID'] + 1}/{CONFIG['N_SHARDS']} ({i}:{j}):")

    d = CONFIG.copy()
    d['DATA'] = ids[i:j]
    process(d)


BASE_KAGGLE_CONFIG = dict(
    CSV_PATH='../input/mayo-clinic-strip-ai/',
    INPUT_PATH='../input/mayo-clinic-strip-ai/train',
    OUTPUT_PATH='data',
    N_SHARDS=7,
)
