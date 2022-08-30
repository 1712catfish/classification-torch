import os

# ENV CONFIGS
CSV_PATH = '/kaggle/input/mayo-clinic-strip-ai'
IMAGE_PATH = '/kaggle/input/jpg-images-strip-ai'

TRAIN_CSV = os.path.join(CSV_PATH, 'train.csv')
TEST_CSV = os.path.join(CSV_PATH, 'test.csv')

TRAIN_DIR = os.path.join(IMAGE_PATH, 'train')
TEST_DIR = os.path.join(IMAGE_PATH, 'test')

# DATA CONFIGS
IMAGE_SHAPE = (512, 512, 3)
S2I_LBL_MAP = {'CE': 0, 'LAA': 1}
I2S_LBL_MAP = {v: k for k, v in S2I_LBL_MAP.items()}
N_CLASSES = len(S2I_LBL_MAP)

TEST_SIZE = 0.2
BATCH_SIZE = 4

# TRAINING CONFIGS
BACKBONE = 'efficientnet-b4'
STATE_DICT_SAVE_PATH = 'efficientnet_model.pth'
CHECKPOINT = '/kaggle/input/efficientnet-pytorch/efficientnet-b4-e116e8b3.pth'
INITIAL_LEARNING_RATE = 1e-4
