import os
import nibabel as nib
import numpy as np
from pathlib import Path
import re
from preprocessing import normalize_image, zscore_image


def load_volume_from_slices(slice_dir):
    """
    Load a 3D MRI volume from a directory of per-slice .nii files.
    Assumes each slice is a (2, H, W) complex image → magnitude is computed.
    
    Args:
        slice_dir (str or Path): Directory containing per-slice .nii files

    Returns:
        np.ndarray: 3D volume of shape (H, W, D)
    """
    slice_dir = Path(slice_dir)
    slice_files = sorted(slice_dir.glob("slice_*.nii"), key=lambda p: int(re.search(r'slice_(\d+)', p.name).group(1)))

    slices = []
    for nii_path in slice_files:
        img = nib.load(str(nii_path))
        data = img.get_fdata()  # shape: (2, H, W) or (H, W)

        if data.shape[0] == 2:  # complex image: compute magnitude
            mag = np.abs(data[0] + 1j * data[1])
        else:  # already magnitude
            mag = data

        slices.append(mag)

    # Stack into 3D volume: (H, W, num_slices)
    volume = np.stack(slices, axis=-1)
    return volume

# --------------------------- Main Loop ---------------------------
base_dir = Path("/ess/scratch/scratch1/rachelgordon/complex_fully_sampled/")
out_dir = Path("/ess/scratch/scratch1/rachelgordon/3dseg_preprocessed/")
out_dir.mkdir(parents=True, exist_ok=True)

for case_dir in base_dir.iterdir():
    if not case_dir.is_dir():
        continue

    try:
        print(f"Processing: {case_dir.name}")
        volume = load_volume_from_slices(case_dir)
        norm_volume = zscore_image(normalize_image(volume))

        out_path = out_dir / f"{case_dir.name}.npy"
        np.save(out_path, norm_volume)
        print(f"Saved: {out_path}")

    except Exception as e:
        print(f"Failed to process {case_dir.name}: {e}")