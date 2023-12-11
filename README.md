# Topic Overview
https://www.kaggle.com/competitions/UBC-OCEAN

# Data & Models
Dataset: Download the data here: https://www.kaggle.com/competitions/UBC-OCEAN/data

Model: Using `timm` to load model
# Project Structure
`lightning_timm`: original WSIs-based model training 

`thumbnail`: thumbnails-based model training

# How to run
1. Put data to `kaggle\input\ubc-ocean-tiles-w-masks-2048px-scale-0-25`, then run `lightning_timm/main.py` for original WSIs dataset

2. Put data to `kaggle\input\UBC-OCEAN`, then run `thumbnail/main.py` for thumbnail-based dataset training

# Notes
checkpoints, data are not uploaded
