import os
import gc
import cv2
import math
import copy
import time
import random
import glob
from PIL import Image
from matplotlib import pyplot as plt

# For data manipulation
import numpy as np
import pandas as pd

# Pytorch Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.cuda import amp
import torchvision

# Utils
import joblib
from tqdm import tqdm
from collections import defaultdict

# Sklearn Imports
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

# For Image Models
import timm

# Albumentations for augmentations
import albumentations as A
from albumentations.pytorch import ToTensorV2

# For colored terminal text
from colorama import Fore, Back, Style
b_ = Fore.BLUE
sr_ = Style.RESET_ALL

import warnings
warnings.filterwarnings("ignore")

# For descriptive error messages
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

CONFIG = {
    "seed": 42,
    "img_size": 2048,
    "model_name": "tf_efficientnet_b0_ns",
    "num_classes": 5,
    "valid_batch_size": 1,
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
}

def set_seed(seed=42):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    
set_seed(CONFIG['seed'])

ROOT_DIR = 'kaggle/input/UBC-OCEAN'
TEST_DIR = 'kaggle/input/UBC-OCEAN/test_thumbnails'
TRAIN_DIR = 'kaggle/input/UBC-OCEAN/train_thumbnails'
ALT_TRAIN_DIR = 'kaggle/input/UBC-OCEAN/train_images'

LABEL_ENCODER_BIN = "kaggle/input/ubc-efficienetnetb0-fold1of10-2048pix-thumbnails/label_encoder.pkl"
BEST_WEIGHT = "kaggle/input/ubc-efficienetnetb0-fold1of10-2048pix-thumbnails/Recall0.9178_Acc0.9437_Loss0.1685_epoch9.bin"

def get_train_file_path(image_id):
    if os.path.exists(f"{TRAIN_DIR}/{image_id}_thumbnail.png"):
        return f"{TRAIN_DIR}/{image_id}_thumbnail.png"
    else:
        return f"{ALT_TRAIN_DIR}/{image_id}.png"
    
    
def get_test_file_path(image_id):
    return f"{TEST_DIR}/{image_id}_thumbnail.png"


d = {'EC':1,'HGSC':2,'LGSC':3,'MC':4,'CC':0}
def get_labels(label_name):
    return d[label_name]

df = pd.read_csv(f"{ROOT_DIR}/test.csv")
df['file_path'] = df['image_id'].apply(get_test_file_path)
df['label'] = 0 # dummy

df_t = pd.read_csv(f"{ROOT_DIR}/train.csv")
index = [i for i,a in enumerate(df_t["is_tma"].values) if a == True]
df_t = df_t.drop(index)
df_t['file_path'] = df_t['image_id'].apply(get_train_file_path)
df_t['label'] = df_t['label'].apply(get_labels)