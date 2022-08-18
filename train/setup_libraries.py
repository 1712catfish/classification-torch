import os
import gc
import cv2
import copy
import time
import torch
import random
import string
import joblib
import tifffile
import numpy as np
import pandas as pd
import torch.nn as nn
import seaborn as sns
from random import randint
from einops import rearrange
from torchvision import models
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
from einops.layers.torch import Rearrange
from efficientnet_pytorch import EfficientNet
from PIL import Image
from torchvision import datasets, models, transforms
import torch.utils.data
import torch
from sklearn.model_selection import train_test_split
import tensorflow as tf
import warnings; warnings.filterwarnings("ignore")
gc.enable()


def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

seed = 42
seed_everything(seed)