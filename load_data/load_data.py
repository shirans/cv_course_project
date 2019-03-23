from os import listdir
import numpy as np
import os

from PIL import Image


def _load_folder(path, is_train):
    if is_train:
        path = os.path.join(path, "training")
    else:
        path = os.path.join(path, "test")
    images_path = os.path.join(path, "images")
    images = _load_images_in_path(images_path)
    masks = os.path.join(path, "mask")
    masks = _load_images_in_path(masks)
    if is_train:
        manual_path = os.path.join(path, "1st_manual")
        manual = _load_images_in_path(manual_path)
        return images, masks, manual
    return images, masks


def _load_images_in_path(path):
    x = []
    for f in listdir(path):
        if f.endswith("tif"):
            im = Image.open(os.path.join(path, f))
            # im.show()
            x.append(np.array(im))
    return x


def load(path):
    x_train, masks, manual = _load_folder(path, True)
    x_test, masks = _load_folder(path, False)
    print(len(x_train))
    print(len(x_test))
