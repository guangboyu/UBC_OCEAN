from PIL import Image
import numpy as np


def _color_means(img_path):
    img = np.array(Image.open(img_path))
    mask = np.sum(img[..., :3], axis=2) == 0
    img[mask, :] = 255
    if np.max(img) > 1.5:
        img = img / 255.0
    clr_mean = {i: np.mean(img[..., i]) for i in range(3)}
    clr_std = {i: np.std(img[..., i]) for i in range(3)}
    return clr_mean, clr_std