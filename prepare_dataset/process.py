import os
import glob
import numpy as np
from PIL import Image
from functools import partial
from concurrent.futures import ProcessPoolExecutor


def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
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

def train_worker_fn(fpath, dest, resolution):
    postfix = fpath.split("/")[7:]
    dest_class_path = os.path.join(dest, "/".join(postfix[:-1]))
    dest_fpath = os.path.join(dest, "/".join(postfix))
    os.makedirs(dest_class_path, exist_ok=True)

    if os.path.exists(dest_fpath):
        return

    img = Image.open(fpath).convert("RGB")
    img_arr = center_crop_arr(img, resolution)
    Image.fromarray(img_arr).save(dest_fpath.split(".")[0] + ".png", format='png', compress_level=0, optimize=False)


def eval_worker_fn(fpath, dest, resolution):
    dest_fpath = os.path.join(dest, fpath.split("/")[-1])
    if os.path.exists(dest_fpath):
        return
    img = Image.open(fpath).convert("RGB")
    img_arr = center_crop_arr(img, resolution)
    Image.fromarray(img_arr).save(dest_fpath.split(".")[0] + ".png", format='png', compress_level=0, optimize=False)


TARGET_RESOLUTION = 256
SPLIT = "train"
SOUCE_TO_RAW_DATASET = "" # e.g., /mnt/workspace/imagenet-1k
DEST_FOLDER = "" # e.g. /mnt/workspace/ImageNet512/

def main():

    split = SPLIT
    path = f"{SOUCE_TO_RAW_DATASET}/{SPLIT}/"
    dest = f"{DEST_FOLDER}/{SPLIT}"
    resolution = TARGET_RESOLUTION

    worker_fn = train_worker_fn if split == "train" else eval_worker_fn
    worker_fn = partial(worker_fn, dest=dest, resolution=resolution)


    all_files = glob.glob(f"{path}/*/*.JPEG")
    with ProcessPoolExecutor(max_workers=12) as executor:
        future = executor.map(worker_fn, all_files)

if __name__ == "__main__":
    main()
