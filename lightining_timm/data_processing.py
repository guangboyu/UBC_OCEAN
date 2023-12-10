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


# class CancerTilesDataset(Dataset):
#     """V1

#     Args:
#         Dataset (_type_): _description_

#     Returns:
#         _type_: _description_
#     """

#     split: float = 0.90

#     def __init__(
#         self,
#         df_data,
#         path_img_dir: str =  '',
#         transforms = None,
#         mode: str = 'train',
#         labels_lut = None,
#         white_thr: int = 225,
#         thr_max_bg: float = 0.2,
#     ):
#         assert os.path.isdir(path_img_dir)
#         self.path_img_dir = path_img_dir
#         self.transforms = transforms
#         self.mode = mode
#         self.white_thr = white_thr
#         self.thr_max_bg = thr_max_bg

#         self.data = df_data
#         self.labels_unique = sorted(self.data["label"].unique())
#         self.labels_lut = labels_lut or {lb: i for i, lb in enumerate(self.labels_unique)}
#         # shuffle data
#         self.data = self.data.sample(frac=1, random_state=42).reset_index(drop=True)

#         # split dataset
#         assert 0.0 <= self.split <= 1.0
#         frac = int(self.split * len(self.data))
#         self.data = self.data[:frac] if mode == 'train' else self.data[frac:]
#         self.img_dirs = [glob.glob(os.path.join(path_img_dir, str(idx), "*.png")) for idx in self.data["image_id"]]
#         #print(f"missing: {sum([not os.path.isfile(os.path.join(self.path_img_dir, im))
#         #                       for im in self.img_names])}")
#         self.labels = list(self.data['label'])

#     @property
#     def num_classes(self) -> int:
#         return len(self.labels_lut)

#     def to_one_hot(self, label: str) -> tuple:
#         one_hot = [0] * self.num_classes
#         one_hot[self.labels_lut[label]] = 1
#         return tuple(one_hot)

#     def __getitem__(self, idx: int) -> tuple:
#         random.shuffle(self.img_dirs[idx])
#         for img_path in self.img_dirs[idx]:
#             assert os.path.isfile(img_path), f"missing: {img_path}"
#             tile = np.array(Image.open(img_path))[..., :3]
#             black_bg = np.sum(tile, axis=2) == 0
#             tile[black_bg, :] = 255
#             mask_bg = np.mean(tile, axis=2) > self.white_thr
#             if np.sum(mask_bg) < (np.prod(mask_bg.shape) * self.thr_max_bg):
#                 break
#         labels = self.to_one_hot(self.labels[idx])

#         # augmentation
#         if self.transforms:
#             tile = self.transforms(Image.fromarray(tile))
#         #print(f"img dim: {img.shape}")
#         return tile, torch.tensor(labels).to(int)

#     def __len__(self) -> int:
#         return len(self.data)

class CancerTilesDataset_No_Mask(Dataset):
    """V2 w/o mask

    Args:
        Dataset (_type_): _description_

    Returns:
        _type_: _description_
    """
    split: float = 0.90

    def __init__(
        self,
        df_data,
        path_img_dir: str,
        transforms = None,
        mode: str = 'train',
        labels_lut = None,
        #white_thr: int = 225,
        #thr_max_bg: float = 0.2,
    ):
        assert os.path.isdir(path_img_dir)
        self.path_img_dir = path_img_dir
        self.transforms = transforms
        self.mode = mode
        #self.white_thr = white_thr
        #self.thr_max_bg = thr_max_bg

        self.data = df_data
        # self.labels_unique = sorted(self.data["label"].unique()) + ["Other"]
        self.labels_unique = sorted(self.data["label"].unique())

        self.labels_lut = labels_lut or {lb: i for i, lb in enumerate(self.labels_unique)}
        # shuffle data
        ls_img = sorted(glob.glob(os.path.join(self.path_img_dir, "*", "*.png")))
        random.Random(42).shuffle(ls_img)
        self.imgs = [(os.path.basename(os.path.dirname(p)), os.path.basename(p))
                     for p in ls_img]

        # split dataset
        assert 0.0 <= self.split <= 1.0
        frac = int(self.split * len(self.imgs))
        self.imgs = self.imgs[:frac] if mode == 'train' else self.imgs[frac:]
        #self.labels = list(self.data['label'])

    @property
    def num_classes(self) -> int:
        return len(self.labels_lut)

    def to_one_hot(self, label: str) -> tuple:
        one_hot = [0] * self.num_classes
        one_hot[self.labels_lut[label]] = 1
        return tuple(one_hot)

    def __getitem__(self, idx: int) -> tuple:
        image_id, tile = self.imgs[idx]
        img_path = os.path.join(self.path_img_dir, image_id, tile)
        
        img = np.array(Image.open(img_path))[..., :3]
        black_bg = np.sum(img, axis=2) == 0
        img[black_bg, :] = 255
        
        tumor_type = self.data.loc[self.data['image_id'] == int(image_id), "label"]
        #print(tumor_mask, tumor_type)
        # lb = tumor_type.item() if tumor_mask > self.tumor_thr else "Other"
        lb = tumor_type.item()
        # TODO: assume as multilabel problem
        labels = self.to_one_hot(lb)

        # augmentation
        if self.transforms:
            img = self.transforms(Image.fromarray(img))
        #print(f"img dim: {img.shape}")
        return img, torch.tensor(labels).to(int)

    def __len__(self) -> int:
        return len(self.imgs)

class CancerTilesDataset(Dataset):
    """V2 with mask

    Args:
        Dataset (_type_): _description_

    Returns:
        _type_: _description_
    """
    split: float = 0.90

    def __init__(
        self,
        df_data,
        path_img_dir: str,
        path_mask_dir: str,
        transforms = None,
        mode: str = 'train',
        labels_lut = None,
        tumor_thr: float = 0.3,
        #white_thr: int = 225,
        #thr_max_bg: float = 0.2,
    ):
        assert os.path.isdir(path_img_dir)
        self.path_img_dir = path_img_dir
        self.path_mask_dir = path_mask_dir
        self.transforms = transforms
        self.mode = mode
        self.tumor_thr = tumor_thr
        #self.white_thr = white_thr
        #self.thr_max_bg = thr_max_bg

        self.data = df_data
        # self.labels_unique = sorted(self.data["label"].unique()) + ["Other"]
        self.labels_unique = sorted(self.data["label"].unique())

        self.labels_lut = labels_lut or {lb: i for i, lb in enumerate(self.labels_unique)}
        # shuffle data
        ls_img = sorted(glob.glob(os.path.join(self.path_img_dir, "*", "*.png")))
        random.Random(42).shuffle(ls_img)
        self.imgs = [(os.path.basename(os.path.dirname(p)), os.path.basename(p))
                     for p in ls_img]

        # split dataset
        assert 0.0 <= self.split <= 1.0
        frac = int(self.split * len(self.imgs))
        self.imgs = self.imgs[:frac] if mode == 'train' else self.imgs[frac:]
        #self.labels = list(self.data['label'])

    @property
    def num_classes(self) -> int:
        return len(self.labels_lut)

    def to_one_hot(self, label: str) -> tuple:
        one_hot = [0] * self.num_classes
        one_hot[self.labels_lut[label]] = 1
        return tuple(one_hot)

    def __getitem__(self, idx: int) -> tuple:
        image_id, tile = self.imgs[idx]
        img_path = os.path.join(self.path_img_dir, image_id, tile)
        mask_path = os.path.join(self.path_mask_dir, image_id, tile)
        
        img = np.array(Image.open(img_path))[..., :3]
        black_bg = np.sum(img, axis=2) == 0
        img[black_bg, :] = 255
        
        mask = np.array(Image.open(mask_path))[..., :3]
        tumor_mask = float(np.sum(mask == 1)) / np.prod(mask.shape)
        tumor_type = self.data.loc[self.data['image_id'] == int(image_id), "label"]
        #print(tumor_mask, tumor_type)
        # lb = tumor_type.item() if tumor_mask > self.tumor_thr else "Other"
        lb = tumor_type.item()
        # TODO: assume as multilabel problem
        labels = self.to_one_hot(lb)

        # augmentation
        if self.transforms:
            img = self.transforms(Image.fromarray(img))
        #print(f"img dim: {img.shape}")
        return img, torch.tensor(labels).to(int)

    def __len__(self) -> int:
        return len(self.imgs)
    

# class CancerSubtypeDM(pl.LightningDataModule):
#     """without mask

#     Args:
#         pl (_type_): _description_
#     """
#     def __init__(
#         self,
#         df_data,
#         path_img_dir: str = '',
#         batch_size: int = 16,
#         num_workers: int = None,
#         train_transforms = None,
#         valid_transforms = None
#     ):
#         super().__init__()
#         self.df_data = df_data
#         self.path_img_dir = path_img_dir
#         self.batch_size = batch_size
#         self.num_workers = num_workers or mproc.cpu_count()
#         self.train_dataset = None
#         self.valid_dataset = None
#         self.train_transforms = train_transforms
#         self.valid_transforms = valid_transforms

#     def prepare_data(self):
#         pass

#     @property
#     def num_classes(self) -> int:
#         assert self.train_dataset and self.valid_dataset
#         return len(set(self.train_dataset.labels_unique + self.valid_dataset.labels_unique))

#     def setup(self, stage=None):
#         self.train_dataset = CancerTilesDataset(
#             self.df_data, self.path_img_dir, mode='train', transforms=self.train_transforms)
#         print(f"training dataset: {len(self.train_dataset)}")
#         self.valid_dataset = CancerTilesDataset(
#             self.df_data, self.path_img_dir, mode='valid', transforms=self.valid_transforms,
#             # as validation is subsampled it may happen that some labels are missing
#             # and so created one-hot-encoding vector will be sorter
#             labels_lut=self.train_dataset.labels_lut)
#         print(f"validation dataset: {len(self.valid_dataset)}")

#     def train_dataloader(self):
#         return DataLoader(
#             self.train_dataset,
#             batch_size=self.batch_size,
#             num_workers=self.num_workers,
#             shuffle=True,
#         )

#     def val_dataloader(self):
#         return DataLoader(
#             self.valid_dataset,
#             batch_size=self.batch_size,
#             num_workers=self.num_workers,
#             shuffle=False,
#         )

#     def test_dataloader(self):
#         pass

class CancerSubtypeDM_No_Mask(pl.LightningDataModule):
    """w/o mask, all data

    Args:
        pl (_type_): _description_
    """

    def __init__(
        self,
        df_data,
        path_img_dir: str,
        batch_size: int = 32,
        num_workers: int = None,
        train_transforms = None,
        valid_transforms = None
    ):
        super().__init__()
        self.df_data = df_data
        self.path_img_dir = path_img_dir
        self.batch_size = batch_size
        self.num_workers = num_workers or mproc.cpu_count()
        self.train_dataset = None
        self.valid_dataset = None
        self.train_transforms = train_transforms
        self.valid_transforms = valid_transforms

    def prepare_data(self):
        pass

    @property
    def num_classes(self) -> int:
        assert self.train_dataset and self.valid_dataset
        return len(set(self.train_dataset.labels_unique + self.valid_dataset.labels_unique))

    def setup(self, stage=None):
        self.train_dataset = CancerTilesDataset_No_Mask(
            self.df_data, self.path_img_dir,
            mode='train', transforms=self.train_transforms)
        print(f"training dataset: {len(self.train_dataset)}")
        self.valid_dataset = CancerTilesDataset_No_Mask(
            self.df_data, self.path_img_dir,
            mode='valid', transforms=self.valid_transforms,
            # as validation is subsampled it may happen that some labels are missing
            # and so created one-hot-encoding vector will be sorter
            labels_lut=self.train_dataset.labels_lut)
        print(f"validation dataset: {len(self.valid_dataset)}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True, pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        pass
        
class CancerSubtypeDM(pl.LightningDataModule):
    """with mask, all data

    Args:
        pl (_type_): _description_
    """

    def __init__(
        self,
        df_data,
        path_img_dir: str,
        path_mask_dir: str,
        batch_size: int = 32,
        num_workers: int = None,
        train_transforms = None,
        valid_transforms = None
    ):
        super().__init__()
        self.df_data = df_data
        self.path_img_dir = path_img_dir
        self.path_mask_dir = path_mask_dir
        self.batch_size = batch_size
        self.num_workers = num_workers or mproc.cpu_count()
        self.train_dataset = None
        self.valid_dataset = None
        self.train_transforms = train_transforms
        self.valid_transforms = valid_transforms

    def prepare_data(self):
        pass

    @property
    def num_classes(self) -> int:
        assert self.train_dataset and self.valid_dataset
        return len(set(self.train_dataset.labels_unique + self.valid_dataset.labels_unique))

    def setup(self, stage=None):
        self.train_dataset = CancerTilesDataset(
            self.df_data, self.path_img_dir, self.path_mask_dir,
            mode='train', transforms=self.train_transforms)
        print(f"training dataset: {len(self.train_dataset)}")
        self.valid_dataset = CancerTilesDataset(
            self.df_data, self.path_img_dir, self.path_mask_dir,
            mode='valid', transforms=self.valid_transforms,
            # as validation is subsampled it may happen that some labels are missing
            # and so created one-hot-encoding vector will be sorter
            labels_lut=self.train_dataset.labels_lut)
        print(f"validation dataset: {len(self.valid_dataset)}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        pass