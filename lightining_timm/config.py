import os

class Config:
    DATASET_FOLDER = "kaggle/input/UBC-OCEAN"
    DATASET_TILES = "tiles-of-cancer-2048px-scale-0-25"
    # DATASET_IMAGES = os.path.join(DATASET_TILES, "train_images")
    DATASET_IMAGES = DATASET_TILES
    DATASET_MASKS = os.path.join(DATASET_TILES, "train_masks")
    
    batch_size = 32
    
    

# print(Config.DATASET_IMAGES)