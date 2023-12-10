import torch

CONFIG = {
    "seed": 42,
    "epochs": 20,
    "img_size": 224,
    "model_name": "coat_tiny",
    "checkpoint_path" : "/kaggle/input/ver-21-10/Acc0.69_Loss1.1052_epoch7.bin",
    "num_classes": 5,
    "train_batch_size": 8,
    "valid_batch_size": 64,
    "learning_rate": 1e-4,
    "scheduler": 'CosineAnnealingLR',
    "min_lr": 1e-6,
    "T_max": 500,
    "weight_decay": 1e-6,
    "fold" : 0,
    "n_fold": 5,
    "n_accumulate": 1,
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
}