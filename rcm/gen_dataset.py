import os
import glob
import random
import numpy as np
from PIL import Image
import blobfile as bf
from absl import logging

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


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


class ImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        classes=None,
        cfg_drop_rate = 0.0,
        mask_generator=None,
        load_from_local=False,
        std = "simple",
        horizontal_flip=True,
        **kwargs
    ):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths
        self.local_classes = classes
            
        if std == "simple":
            self.mean = [0.5, 0.5, 0.5]
            self.std = [0.5, 0.5, 0.5]
        elif std == "imagenet":
            self.mean = [0.485, 0.456, 0.406]
            self.std = [0.229*2, 0.224*2, 0.225*2]

        if horizontal_flip:
            self.simple_augmentation = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ])
        else:
            self.simple_augmentation = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ])

        self.cfg_drop_rate = cfg_drop_rate
        self.num_classes = len(set(self.local_classes))

        self.K = max(classes) + 1
        cnt = dict(zip(*np.unique(classes, return_counts=True)))
        self.cnt = torch.tensor([cnt[k] for k in range(self.K)]).float()

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]

        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()

        pil_image = pil_image.convert("RGB")
        cropped_pil= Image.fromarray(center_crop_arr(pil_image, self.resolution))
        x = self.simple_augmentation(cropped_pil)

        model_kwargs = {}
        if self.local_classes is not None:
            if random.random() > self.cfg_drop_rate:
                model_kwargs = dict(y = self.local_classes[idx])
            else:
                model_kwargs = dict(y = self.num_classes)

        return x, model_kwargs

    def data_shape(self):
        return [3, self.resolution, self.resolution]

    def unpreprocess(self, x):
        _mean = torch.as_tensor(self.mean)[None, :, None, None].to(x.device)
        _std = torch.as_tensor(self.std)[None, :, None, None].to(x.device)
        return x * _std + _mean

    def sample_label(self, n_samples, device):
        return torch.multinomial(self.cnt, n_samples, replacement=True).to(device) if self.cnt is not None else None

    def get_uncond_token(self, bs, device):
        return torch.full((bs, ), self.num_classes, device=device)


def load_data(
    *,
    data_dir,
    batch_size,
    image_size,
    num_workers=8,
    cfg_drop_rate = 0.0,
    std = "simple",
    horizontal_flip=True,
    **kwargs,
):
    if not data_dir:
        raise ValueError("unspecified data directory")

    all_files = glob.glob(os.path.join(data_dir, "*/*.JPEG")) # for imagenet
    if len(all_files) == 0:
        all_files = glob.glob(os.path.join(data_dir, "*/*.png"))

    logging.info(f"total training samples: {len(all_files)}")

    class_names = [bf.basename(path).split("_")[0] for path in all_files]
    sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
    classes = [sorted_classes[x] for x in class_names]
    dataset = ImageDataset(image_size, all_files, classes, cfg_drop_rate=cfg_drop_rate, std=std, horizontal_flip=horizontal_flip)

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True,
        pin_memory=True, persistent_workers=True,
    )

    return dataset, loader