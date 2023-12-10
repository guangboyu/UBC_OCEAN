from config import Config
import os, glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from joblib import Parallel, delayed
from tqdm.auto import tqdm
import os
import torch
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
import multiprocessing as mproc
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import timm
import torch
import torchvision
from adan_pytorch import Adan
from lion_pytorch import Lion
from torch_optimizer import AdaBound, RAdam, Yogi

from torch import nn
from torch.nn import functional as F
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score
import seaborn as sn
import multiprocessing

from data_processing import CancerTilesDataset, CancerSubtypeDM, CancerTilesDataset_No_Mask, CancerSubtypeDM_No_Mask
from utils import _color_means
from config import Config
from model import LitCancerSubtype


def train():
    # print("config: ", )
    df_train = pd.read_csv(os.path.join(Config.DATASET_FOLDER, "train.csv"))
    # labels = list(df_train["label"].unique())
    print(f"Dataset/train size: {len(df_train)}")
    # display(df_train.head())

    # color normalization
    # os.path.join(DATASET_SMALL_FOLDER, "train_images")
    print("Config.DATASET_IMAGES", Config.DATASET_IMAGES)
    ls_images = glob.glob(os.path.join(Config.DATASET_IMAGES, "*", "*.png"))
    print(f"len of ls_images: {len(ls_images)}")
    clr_mean_std = Parallel(n_jobs=os.cpu_count())(delayed(_color_means)(fn) for fn in tqdm(ls_images[:9000]))

    img_color_mean = pd.DataFrame([c[0] for c in clr_mean_std]).describe()
    # display(img_color_mean.T)
    img_color_std = pd.DataFrame([c[1] for c in clr_mean_std]).describe()
    # display(img_color_std.T)

    img_color_mean = list(img_color_mean.T["mean"])
    img_color_std = list(img_color_std.T["mean"])
    print(f"{img_color_mean=}\n{img_color_std=}")

    # Dataset
    dataset = CancerTilesDataset_No_Mask(df_train, Config.DATASET_IMAGES)
    
    print(f"length of dataset: {len(dataset)}")

    TRAIN_TRANSFORM = T.Compose([
        T.CenterCrop(512),
        #T.RandomResizedCrop(512, interpolation=InterpolationMode.BICUBIC, antialias=True),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.ToTensor(),
        #T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        T.Normalize(img_color_mean, img_color_std),  # custom
    ])

    VALID_TRANSFORM = T.Compose([
        T.CenterCrop(512),
        T.ToTensor(),
        #T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        T.Normalize(img_color_mean, img_color_std),  # custom
    ])

    dm = CancerSubtypeDM_No_Mask(df_train, Config.DATASET_IMAGES, batch_size=Config.batch_size, train_transforms = TRAIN_TRANSFORM,
            valid_transforms = VALID_TRANSFORM)
    dm.setup()
    print(dm.num_classes)

    # Modeling
    
    model_names = ["tf_efficientnetv2_s.in21k", "tf_efficientnetv2_s.in21k_ft_in1k", "tf_efficientnet_b0_ns"]
    for model_name in model_names:
        net = timm.create_model(model_name, pretrained=True, num_classes=dm.num_classes)
        model = LitCancerSubtype(net=net, lr=1e-4)

        logger = pl.loggers.CSVLogger(save_dir='logs/', name=model.arch)
        nb_epochs = 30 if torch.cuda.is_available() else 2

        # ==============================

        trainer = pl.Trainer(
            accelerator="auto",
        #     devices=2,
            # fast_dev_run=True,
            # callbacks=[swa],
            logger=logger,
            max_epochs=nb_epochs,
            precision="16-mixed",
            accumulate_grad_batches=14,
            #val_check_interval=0.5,
        )

        # ==============================

        # trainer.tune(model, datamodule=dm)
        trainer.fit(model=model, datamodule=dm)


        # training visualization
        metrics = pd.read_csv(f'{trainer.logger.log_dir}/metrics.csv')
        del metrics["step"]
        metrics.set_index("epoch", inplace=True)
        # display(metrics.dropna(axis=1, how="all").head())
        g = sn.relplot(data=metrics, kind="line")
        plt.gcf().set_size_inches(12, 4)
        # plt.gca().set_yscale('log')
        plt.grid()

        # save
        trainer.save_checkpoint(f"saved_models/{model_name}.pt")
    
    
    
if __name__ == '__main__':
    # On Windows calling this function is necessary.
    multiprocessing.freeze_support()
    train()