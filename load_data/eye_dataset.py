import numbers
from abc import ABC
import torch

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
from pathlib import Path
import re
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random
import logging

logging.basicConfig(level=logging.DEBUG)
logging.getLogger().setLevel(logging.INFO)

logger = logging.getLogger(__name__)


def pil_loader(path, resize=True):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        if resize:  # issue during training
            img = img.resize((512, 512), Image.ANTIALIAS)
        return img.convert('RGB')


def normalize(tensor):
    tmean = tensor.mean()
    tstd = tensor.std()
    norm = transforms.Normalize(mean=[tmean], std=[tstd])
    return norm(tensor)


def make_dataset(folder):
    folder_abs = os.path.abspath(folder)
    logger.info("path images:{}".format(folder_abs))

    masks = Path(folder_abs).glob('mask/**/*.gif')
    raw = Path(folder_abs).glob('images/**/*.tif')
    segmentation = Path(folder_abs).glob('1st_manual/**/*.gif')

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


class DriveDataset(Dataset, ABC):
    def __init__(self, folder, augment=False, normalization=True, is_crop=True):
        self.is_crop = is_crop
        self.normalization = normalization
        self.samples = make_dataset(folder)
        self.augment = augment


def normalize_if_defined(normalization, image):
    tensor = TF.to_tensor(image)
    if normalization:
        return normalize(tensor)
    return tensor


class EyeDataset(DriveDataset):
    def __getitem__(self, idx):
        mask_path = self.samples[idx]['mask']
        segmentation_path = self.samples[idx]['segmentation']
        image_path = self.samples[idx]['raw']

        image = pil_loader(image_path)
        mask = TF.to_grayscale(pil_loader(mask_path))
        segmentation = TF.to_grayscale(pil_loader(segmentation_path))

        if self.is_crop:
            i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(128, 128))
            image = TF.crop(image, i, j, h, w)
            mask = TF.crop(mask, i, j, h, w)
            segmentation = TF.crop(segmentation, i, j, h, w)

        image, mask, segmentation = augment(self.augment, image, mask, segmentation)

        tensor = normalize_if_defined(self.normalization, image)
        segmentation =  TF.to_tensor(segmentation)
        ## Train as the segmentation is binary (0-no vessel, 1-vessel)
        # x = torch.ones(segmentation.size(1), segmentation.size(2))
        # y = torch.zeros(segmentation.size(1), segmentation.size(2))
        # segmentation = torch.where(segmentation > 0.5, x, y)
        return tensor, TF.to_tensor(mask), segmentation

    def __len__(self):
        return len(self.samples)


class EyeDatasetOverfitCorners(DriveDataset):
    def __init__(self, folder, augment=False, normalization=True, is_crop=True):
        self.crop = transforms.FiveCrop(128)
        super().__init__(folder, augment, normalization, is_crop)

    def __getitem__(self, idx):
        mask_path = self.samples[idx]['mask']
        segmentation_path = self.samples[idx]['segmentation']
        image_path = self.samples[idx]['raw']

        image = pil_loader(image_path)
        mask = TF.to_grayscale(pil_loader(mask_path))
        segmentation = TF.to_grayscale(pil_loader(segmentation_path))

        itl, itr, ibl, ibr, ic = self.crop(image)
        image_crops = [itl, itr, ibl, ibr, ic]
        mtl, mtr, mbl, mbr, mc = self.crop(mask)
        mask_crops = [mtl, mtr, mbl, mbr, mc]
        stl, str, sbl, sbr, sc = self.crop(segmentation)
        segmentation_crops = [stl, str, sbl, sbr, sc]

        rand_idx = random.randint(1, 5)
        image = image_crops[rand_idx - 1]
        mask = mask_crops[rand_idx - 1]
        segmentation = segmentation_crops[rand_idx - 1]

        image, mask, segmentation = augment(self.augment, image, mask, segmentation)
        tensor = normalize_if_defined(self.normalization, image)
        return tensor, TF.to_tensor(mask), TF.to_tensor(segmentation)

    def __len__(self):
        return len(self.samples)


def augment(augment, image, mask, segmentation):
    if augment:
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
            segmentation = TF.vflip(segmentation)

        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
            segmentation = TF.hflip(segmentation)
    return image, mask, segmentation


class EyeDatasetOverfitCenter(DriveDataset):

    def __init__(self, folder, augment=False, normalization=True, is_crop=True):
        super().__init__(folder, augment, normalization, is_crop)
        self.crop = transforms.CenterCrop(128)

    def __getitem__(self, idx):
        mask_path = self.samples[idx]['mask']
        segmentation_path = self.samples[idx]['segmentation']
        image_path = self.samples[idx]['raw']

        image = pil_loader(image_path)
        mask = TF.to_grayscale(pil_loader(mask_path))
        segmentation = TF.to_grayscale(pil_loader(segmentation_path))

        image = plus_crop(image, 128)
        mask = plus_crop(mask, 128)
        segmentation = plus_crop(segmentation, 128)

        image, mask, segmentation = augment(self.augment, image, mask, segmentation)

        tensor = normalize_if_defined(self.normalization, image)
        return tensor, TF.to_tensor(mask), TF.to_tensor(segmentation)

    def __len__(self):
        return len(self.samples)


def plus_crop(img, output_size):
    if isinstance(output_size, numbers.Number):
        output_size = (int(output_size), int(output_size))
    w, h = img.size
    th, tw = output_size
    i = int(round((h - th) / 4.))
    j = int(round((w - tw) / 4.))
    return TF.crop(img, i, j, th, tw)


if __name__ == '__main__':
    a = EyeDataset('training')
    b = DataLoader(a)
    i, (image, mask, segmentation) = next(enumerate(b))
    print(segmentation.size())
