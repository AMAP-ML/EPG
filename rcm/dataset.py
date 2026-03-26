"""
Train a diffusion model on images.
"""
import os
import glob
import random
import blobfile as bf
from absl import logging
import numpy as np
from PIL import Image, ImageOps, ImageFilter

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import shutil


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif", "tif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class Solarize(object):
    """Solarize augmentation from BYOL: https://arxiv.org/abs/2006.07733"""

    def __call__(self, x):
        return ImageOps.solarize(x)


class GaussianBlur(object):
    """Gaussian blur augmentation from SimCLR: https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    if pil_image.size[0] == image_size and pil_image.size[1] == image_size:
        return np.array(pil_image)

    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


class TwoCropsTransform:
    """Take two random crops of one image"""

    def __init__(self, base_transform1, base_transform2):
        self.base_transform1 = base_transform1
        self.base_transform2 = base_transform2

    def __call__(self, x):
        im1 = self.base_transform1(x)
        im2 = self.base_transform2(x)
        return [im1, im2]


class ImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        load_from_memory=False,
        std="simple",
    ):
        super().__init__()
        self.name = "imagenet"
        self.resolution = resolution
        self.local_images = image_paths
        self.load_from_memory = load_from_memory
        self.loaded_images = {}
        assert std in ["simple", "imagenet"], f"Unsupported mean/std type: {std}"
        if std == "simple":
            self.mean = [0.5, 0.5, 0.5]
            self.std = [0.5, 0.5, 0.5]
        else:
            # std == "imagenet"
            self.mean = [0.485, 0.456, 0.406]
            self.std = [0.229*2, 0.224*2, 0.225*2]


        augmentation1 = transforms.Compose([
            transforms.RandomResizedCrop(self.resolution, scale=(0.08, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # jc: lower the strength for CIFAR10
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2), # jc: detrimental for performance on CIFAR10
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=1.0), # jc: remove blur for CIFAR10
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

        augmentation2 = transforms.Compose([
            transforms.RandomResizedCrop(self.resolution, scale=(0.08, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # jc: lower the strength for CIFAR10
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2), # jc: detrimental for performance on CIFAR10
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.1), # jc: remove blur for CIFAR10
            transforms.RandomApply([Solarize()], p=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        self.augmentation = TwoCropsTransform(augmentation1, augmentation2)

        self.simple_augmentation = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])

    def __len__(self):
        return len(self.local_images)

    def read_image(self, path):
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")
        return pil_image

    def __getitem__(self, idx):
        path = self.local_images[idx]
        if self.load_from_memory:
            if path in self.loaded_images:
                pil_image = self.loaded_images[path]
            else:
                pil_image = self.read_image(path)
                self.loaded_images[path] = pil_image
        else:
            pil_image = self.read_image(path)

        cropped_pil= Image.fromarray(center_crop_arr(pil_image, self.resolution))
        x = self.simple_augmentation(cropped_pil)
        x_aug, x_aug2 = self.augmentation(pil_image)
        return x, x_aug, x_aug2


def load_data(
    *,
    data_dir,
    batch_size,
    image_size,
    num_workers=8,
    load_from_memory = False,
    std="simple",
):
    if not data_dir:
        raise ValueError("unspecified data directory")

    print("reading imagenet images...")
    all_files = glob.glob(os.path.join(data_dir, "*/*.png")) # for imagenet in png format

    logging.info(f"total training samples: {len(all_files)}")   
    dataset = ImageDataset(image_size, all_files, load_from_memory, std=std)

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True,
        pin_memory=True, persistent_workers=True,
    )

    return loader