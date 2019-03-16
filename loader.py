import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
from pathlib import Path
import re
from torchvision.datasets import DatasetFolder
from torchvision.transforms import ToTensor
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def make_dataset(folder):
    masks = Path(folder).glob('mask/**/*.gif')
    raw = Path(folder).glob('images/**/*.tif')
    segmentation = Path(folder).glob('1st_manual/**/*.gif')

    d = {}
    for m in masks:
        idx = re.findall(r'\d+', str(m))[0]
        d[idx] = {'mask': str(m)}

    for m in raw:
        idx = re.findall(r'\d+', str(m))[0]
        d[idx]['raw'] = str(m)

    for m in segmentation:
        idx = re.findall(r'\d+', str(m))[1]
        d[idx]['segmentation'] = str(m)

    samples = [v for k, v in d.items()]

    return samples


class EyeLoader(Dataset):
    def __init__(self, folder, augment=False):
        self.samples = make_dataset(folder)
        self.augment = augment

    def __getitem__(self, idx):
        mask_path = self.samples[idx]['mask']
        segmentation_path = self.samples[idx]['segmentation']
        image_path = self.samples[idx]['raw']

        image = pil_loader(image_path)
        mask = TF.to_grayscale(pil_loader(mask_path))
        segmentation = TF.to_grayscale(pil_loader(segmentation_path))

        i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(128, 128))
        image = TF.crop(image, i, j, h, w)
        mask = TF.crop(mask, i, j, h, w)
        segmentation = TF.crop(segmentation, i, j, h, w)

        if self.augment:
            if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)
                segmentation = TF.vflip(segmentation)

            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
                segmentation = TF.hflip(segmentation)

        return TF.to_tensor(image), TF.to_tensor(mask), TF.to_tensor(segmentation)

    def __len__(self):
        return len(self.samples)


if __name__ == '__main__':
    a = EyeLoader('training')
    b = DataLoader(a)
    i, (image, mask, segmentation) = next(enumerate(b))
    print(segmentation.size())
