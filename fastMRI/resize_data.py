#!/usr/bin/env python
"""
Resize 3-D volumes with TorchIO.

Assumptions
-----------
* Each .npy file contains a single volume shaped (X, Y, Z) = (320, 320, 83).
* Images and masks live in separate directories but share identical
  file names (e.g. img_001.npy ↔ mask_001.npy).
* You want every resized volume saved with the **same file name** into
  an output directory that already exists (or will be created by the script).
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torchio as tio
from tqdm import tqdm


def load_npy_as_scalar_image(npy_path: Path, is_mask=False) -> tio.ScalarImage:
    arr = np.load(npy_path)

    print(arr.shape)

    if is_mask:
        arr = arr.astype(np.uint8)       # or np.int16
    else:
        arr = arr.astype(np.float32)     # <-  changed from float16

    tensor = torch.from_numpy(arr).unsqueeze(0)  # 1×D×H×W
    affine = torch.eye(4)
    return tio.ScalarImage(tensor=tensor, affine=affine)



def save_scalar_image_as_npy(img: tio.ScalarImage, out_path: Path):
    """Save TorchIO ScalarImage back to .npy (drops affine)."""
    arr = img.data.squeeze(0).numpy()    # remove channel dim → (X, Y, Z)
    np.save(out_path, arr)


def resize_directory(in_dir: Path, out_dir: Path, resize_tf: tio.Resize):
    """
    Resize all .npy volumes in one directory using the supplied transform.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted([f for f in in_dir.iterdir() if f.suffix == '.npy'])

    for f in tqdm(files, desc=f"Resizing {in_dir.name}"):
        img = load_npy_as_scalar_image(f)
        img_resized = resize_tf(img)
        save_scalar_image_as_npy(img_resized, out_dir / f.name)


def main():
    parser = argparse.ArgumentParser(description="Batch-resize 3-D .npy volumes")
    parser.add_argument("--image_dir", required=True, type=Path,
                        help="Directory with original images (.npy)")
    parser.add_argument("--mask_dir", required=True, type=Path,
                        help="Directory with original masks (.npy)")
    parser.add_argument("--out_image_dir", required=True, type=Path,
                        help="Destination for resized images")
    parser.add_argument("--out_mask_dir", required=True, type=Path,
                        help="Destination for resized masks")
    parser.add_argument("--target_size", default="144,144,96",
                        help="Output size as comma-separated 'X,Y,Z'")
    args = parser.parse_args()


    target_size = tuple(int(x) for x in args.target_size.split(','))
    assert len(target_size) == 3, "target_size must be three integers"

    # TorchIO transform (trilinear for images, nearest for masks)
    resize_img = tio.Resize(target_size, image_interpolation='linear')
    resize_mask = tio.Resize(target_size, image_interpolation='nearest')

    resize_directory(args.image_dir, args.out_image_dir, resize_img)
    resize_directory(args.mask_dir,  args.out_mask_dir,  resize_mask)


if __name__ == "__main__":
    main()
