import os
import tempfile
from torchgeo.datasets import OSCD100

root = os.path.join(tempfile.gettempdir(), "oscd100")
print(f"Downloading OSCD100 to: {root}")

dataset = OSCD100(root=root, split="train", bands=OSCD100.rgb_bands, download=True)
print(f"Train samples: {len(dataset)}")

val = OSCD100(root=root, split="val", bands=OSCD100.rgb_bands, download=True)
print(f"Val samples: {len(val)}")

sample = dataset[0]
print(f"Image shape: {sample['image'].shape}")
print(f"Mask shape: {sample['mask'].shape}")

import numpy as np
mask = sample["mask"].numpy()
total = mask.size
changed = (mask > 0).sum()
print(f"Changed pixels: {changed}/{total} ({changed/total*100:.2f}%)")
print("Done!")
