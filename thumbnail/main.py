import os
import gc
import cv2
import math
import copy
import time
import random
import glob
from matplotlib import pyplot as plt

# For data manipulation
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

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

from config import CONFIG
from data_process import data_transforms, UBCDataset
from model import UBCModel

import warnings
warnings.filterwarnings("ignore")

# For descriptive error messages
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

ROOT_DIR = 'kaggle/input/UBC-OCEAN'
TRAIN_DIR = 'kaggle/input/UBC-OCEAN/train_thumbnails'
ALT_TEST_DIR = 'kaggle/input/UBC-OCEAN/test_images'
TEST_DIR = 'kaggle/input/UBC-OCEAN/test_thumbnails'
ALT_TRAIN_DIR = 'kaggle/input/UBC-OCEAN/train_images'

def get_train_file_path(image_id):
    if os.path.exists(f"{TRAIN_DIR}/{image_id}_thumbnail.png"):
        return f"{TRAIN_DIR}/{image_id}_thumbnail.png"
    else:
        return f"{ALT_TRAIN_DIR}/{image_id}.png"
    
def compute_class_weights(df, label_column):
    """
    Compute class weights based on the inverse of class frequencies.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing the data.
        label_column (str): Name of the column containing class labels.
    
    Returns:
        class_weights (dict): Dictionary containing weights for each class.
    """
    # Get the total number of samples
    total_samples = len(df)
    
    # Get the number of classes
    num_classes = df[label_column].nunique()
    
    # Get the count of each class
    class_counts = df[label_column].value_counts().to_dict()
    
    # Compute class weights
    class_weights = {class_label: total_samples / (num_classes * count) 
                     for class_label, count in class_counts.items()}
    
    return class_weights

# def criterion(outputs, labels, loss):
#     return loss(outputs, labels)

def train_one_epoch(model, optimizer, scheduler, dataloader, device, epoch, criterion):
    model.train()
    
    dataset_size = 0
    running_loss = 0.0
    running_acc  = 0.0
    all_preds=[]
    all_labels=[]
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:
        images = data['image'].to(device, dtype=torch.float)
        labels = data['label'].to(device, dtype=torch.long)
        
        batch_size = images.size(0)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss = loss / CONFIG['n_accumulate']
            
        loss.backward()
    
        if (step + 1) % CONFIG['n_accumulate'] == 0:
            optimizer.step()

            # zero the parameter gradients
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()
                
        _, predicted = torch.max(torch.nn.Softmax(dim=1)(outputs), 1)
        acc = torch.sum( predicted == labels )
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        running_loss += (loss.item() * batch_size)
        running_acc  += acc.item()
        dataset_size += batch_size
        
        epoch_loss = running_loss / dataset_size
        epoch_acc = running_acc / dataset_size
        
        bar.set_postfix(Epoch=epoch, Train_Loss=epoch_loss, Train_Acc=epoch_acc,
                        LR=optimizer.param_groups[0]['lr'])
    gc.collect()
    bl_accuracy_score = balanced_accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc,bl_accuracy_score

@torch.inference_mode()
def valid_one_epoch(model, dataloader, device, epoch, criterion):
    model.eval()
    
    dataset_size = 0
    running_loss = 0.0
    running_acc = 0.0
    all_preds=[]
    all_labels=[]
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:        
        images = data['image'].to(device, dtype=torch.float)
        labels = data['label'].to(device, dtype=torch.long)
        
        batch_size = images.size(0)

        outputs = model(images)
        loss = criterion(outputs, labels)

        _, predicted = torch.max(torch.nn.Softmax(dim=1)(outputs), 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        acc = torch.sum( predicted == labels )

        running_loss += (loss.item() * batch_size)
        running_acc  += acc.item()
        dataset_size += batch_size
        
        epoch_loss = running_loss / dataset_size
        epoch_acc = running_acc / dataset_size
        
        bar.set_postfix(Epoch=epoch, Valid_Loss=epoch_loss, Valid_Acc=epoch_acc,
                        LR=optimizer.param_groups[0]['lr'])   
    
    gc.collect()
    bl_accuracy_score = balanced_accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc,bl_accuracy_score

def run_training(model, train_loader, valid_loader, optimizer, scheduler, device, num_epochs, criterion):
    if torch.cuda.is_available():
        print("[INFO] Using GPU: {}\n".format(torch.cuda.get_device_name()))
    
    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch_acc = -np.inf
    history = defaultdict(list)
    
    for epoch in range(1, num_epochs + 1): 
        gc.collect()
        train_epoch_loss, train_epoch_acc, train_bl_accuracy_score = train_one_epoch(model, optimizer, scheduler, 
                                           dataloader=train_loader, 
                                           device=CONFIG['device'], epoch=epoch, criterion=criterion)
        
        val_epoch_loss, val_epoch_acc, val_bl_accuracy_score = valid_one_epoch(model, valid_loader, device=CONFIG['device'], 
                                         epoch=epoch)
    
        history['Train Loss'].append(train_epoch_loss)
        history['Valid Loss'].append(val_epoch_loss)
        history['Train Accuracy'].append(train_epoch_acc)
        history['Valid Accuracy'].append(val_epoch_acc)
        history['Train balanced Accuracy'].append(train_bl_accuracy_score)
        history['Valid balanced Accuracy'].append(val_bl_accuracy_score)
        history['lr'].append( scheduler.get_lr()[0] )
        
        # deep copy the model
        if best_epoch_acc <= val_bl_accuracy_score:
            print(f"{b_}Validation Balanced Accuracy Improved ({best_epoch_acc} ---> {val_bl_accuracy_score})")
            best_epoch_acc = val_bl_accuracy_score
            best_model_wts = copy.deepcopy(model.state_dict())
            PATH = "Acc{:.2f}_Loss{:.4f}_epoch{:.0f}.bin".format(best_epoch_acc, val_epoch_loss, epoch)
            torch.save(model.state_dict(), PATH)
            # Save a model file from the current directory
            print(f"Model Saved{sr_}")
            
        print()
    
    end = time.time()
    time_elapsed = end - start
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    print("Best Accuracy: {:.4f}".format(best_epoch_acc))
    
    # load best model weights
    model.load_state_dict(best_model_wts)
    
    return model, history

def fetch_scheduler(optimizer):
    if CONFIG['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,T_max=CONFIG['T_max'], 
                                                   eta_min=CONFIG['min_lr'])
    elif CONFIG['scheduler'] == 'CosineAnnealingWarmRestarts':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=CONFIG['T_0'], 
                                                             eta_min=CONFIG['min_lr'])
    elif CONFIG['scheduler'] == None:
        return None
        
    return scheduler

def prepare_loaders(df, fold):
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    
    train_dataset = UBCDataset(df_train, transforms=data_transforms["train"])
    valid_dataset = UBCDataset(df_valid, transforms=data_transforms["valid"])

    train_loader = DataLoader(train_dataset, batch_size=16, 
                              num_workers=2, shuffle=True, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=CONFIG['valid_batch_size'], 
                              num_workers=2, shuffle=False, pin_memory=True)
    
    return train_loader, valid_loader

    
def main():
    # load data
    train_images = sorted(glob.glob(f"{TRAIN_DIR}/*.png"))
    df = pd.read_csv(f"{ROOT_DIR}/train.csv")
    df['file_path'] = df['image_id'].apply(get_train_file_path)
    print(df.describe())
    
    # class weight
    class_weights = compute_class_weights(df, 'label')
    class_weights = np.array([1.0868686868686868,0.867741935483871,0.4846846846846847,2.2893617021276595,2.3391304347826085])
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
  
    
    # create folds
    skf = StratifiedKFold(n_splits=CONFIG['n_fold'])

    for fold, ( _, val_) in enumerate(skf.split(X=df, y=df.label)):
        df.loc[val_ , "kfold"] = int(fold)

    model = UBCModel(CONFIG['model_name'], CONFIG['num_classes'], checkpoint_path=CONFIG['checkpoint_path'])
    model.to(CONFIG['device'])
    
    loss = nn.CrossEntropyLoss(weight = class_weights_tensor.to(CONFIG['device']))
    train_loader, valid_loader = prepare_loaders(df, fold=CONFIG["fold"])
    print(len(train_loader))
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'], 
                       weight_decay=CONFIG['weight_decay'])
    scheduler = fetch_scheduler(optimizer)
    model, history = run_training(model, train_loader, valid_loader, optimizer, scheduler,
                              device=CONFIG['device'],
                              num_epochs=30, criterion=loss)

    
    
if __name__ == '__main__':
    main()