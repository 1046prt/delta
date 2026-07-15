from torchgeo.datasets import OSCD100

ds = OSCD100(root="data/oscd100", split="train", bands=OSCD100.rgb_bands, download=False)
print(f"Loaded from data/oscd100: {len(ds)} samples")
s = ds[0]
print(f"Image: {s['image'].shape}, Mask: {s['mask'].shape}")
