import glob
import numpy as np
from PIL import Image
from tqdm import tqdm
import os
import sys

path = sys.argv[1]

sample_batch = []
for img in tqdm(glob.glob(path+"/*.png")):
    npz = np.array(Image.open(img).convert("RGB")) # H W C
    sample_batch.append(npz)

sample_batch = np.stack(sample_batch, axis=0)
if path.endswith("/"):
    path = path.strip("/")
np.savez(path+".npz", arr_0=sample_batch)