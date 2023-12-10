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


class LitCancerSubtype(pl.LightningModule):

    def __init__(self, net, lr: float = 1e-4):
        super().__init__()
        self.net = net
        self.arch = net.pretrained_cfg.get('architecture')
        self.num_classes = net.num_classes
        self.train_accuracy = MulticlassAccuracy(num_classes=self.num_classes)
        self.val_accuracy = MulticlassAccuracy(num_classes=self.num_classes)
        self.val_f1_score = MulticlassF1Score(num_classes=self.num_classes)
        self.learn_rate = lr

    def forward(self, x):
        y = F.softmax(self.net(x))
        if y.isnan().any():
            y = torch.ones(self.num_classes) / self.num_classes
        return y

    def compute_loss(self, y_hat, y):
        return F.cross_entropy(y_hat, y.to(y_hat.dtype))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        lbs = torch.argmax(y, axis=1)
        #print(f"{lbs=} ?= {y_hat=}")
        loss = self.compute_loss(y_hat, y)
        #print(f"{y=} ?= {y_hat=} -> {loss=}")
        self.log("train_loss", loss, logger=True, prog_bar=True)
        #print(f"{lb=} ?= {y_hat=} -> {self.train_accuracy(y_hat, lbs)}")
        self.log("train_acc", self.train_accuracy(y_hat, lbs), logger=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        lbs = torch.argmax(y, axis=1)
        loss = self.compute_loss(y_hat, y)
        self.log("valid_loss", loss, logger=True, prog_bar=False)
        self.log("valid_acc", self.val_accuracy(y_hat, lbs), logger=True, prog_bar=False)
        self.log("valid_f1", self.val_f1_score(y_hat, lbs), logger=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = AdaBound(self.parameters(), lr=self.learn_rate)
        #optimizer = RAdam(self.parameters(), lr=self.learn_rate)
        #optimizer = torch.optim.AdamW(self.parameters(), lr=self.learn_rate)
        #optimizer = Lion(self.parameters(), lr=self.learn_rate, weight_decay=1e-2)
        #optimizer = Adan(self.parameters(), lr=self.learn_rate * 10, betas=(0.02, 0.08, 0.01), weight_decay=0.02)
        
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #    optimizer, T_max=self.trainer.max_epochs, eta_min=1e-6, verbose=True)
        scheduler = torch.optim.lr_scheduler.CyclicLR(
          optimizer, base_lr=self.learn_rate, max_lr=self.learn_rate * 10,
          step_size_up=5, cycle_momentum=False, mode="triangular2", verbose=True)
        #scheduler = torch.optim.lr_scheduler.OneCycleLR(
        #    optimizer, max_lr=self.learn_rate * 5, steps_per_epoch=1, epochs=self.trainer.max_epochs)
        return [optimizer], [scheduler]
