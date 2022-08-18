try:
    INTERACTIVE
except Exception:
    from setup_libraries import *

CSV_PATH = '/kaggle/input/mayo-clinic-strip-ai'
IMAGE_PATH = '/kaggle/input/jpg-images-strip-ai'

TRAIN_CSV = os.path.join(CSV_PATH, 'train.csv')
TEST_CSV = os.path.join(CSV_PATH, 'test.csv')

TRAIN_DIR = os.path.join(IMAGE_PATH, 'train')
TEST_DIR = os.path.join(IMAGE_PATH, 'test')

IMAGE_SHAPE = (512, 512, 3)
S2I_LBL_MAP = {'CE': 0, 'LAA': 1}
I2S_LBL_MAP = {v: k for k, v in S2I_LBL_MAP.items()}
N_CLASSES = len(S2I_LBL_MAP)

BATCH_SIZE = 4
