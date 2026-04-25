"""
whole_preprocess_pipeline.py
============================

ANTs + NiBabel pipeline: read each ``.nii.gz`` from ``--input_dir``, reorient,
Gaussian smooth, resample, pad, crop, set origin to (0,0,0), min–max intensity
normalization to [0,1], and save with identity affine via NiBabel (header spacing
in outputs may be simplified; see project README). Special case: ``--sigma
100 100 100`` skips intensity smoothing and is intended for label/mask
volumes (uses nearest resampling in ``smooth_then_resample``).

**Functions**

- ``crop_img``: ANTs ``crop_indices`` to a target 3D shape.
- ``smooth_then_resample``: ANTs smooth + resample, or resample-only for
  the mask branch when ``sigma==[100,100,100]`` (implementation detail preserved).
- ``normalize_01``: min–max on image numpy array; returns array (not ANTs image).
- ``save_nii``: write array with given affine.
- ``main(args)``: CLI driver over files in ``input_dir``.

**CLI (``argparse``)**

- ``--input_dir`` (str, required): folder of input ``.nii.gz`` files.
- ``--output_dir`` (str, required): output folder.
- ``--spacing`` (3 floats, default 4 4 4): target spacing (mm) for resampling.
- ``--output_shape`` (3 ints, default 64 64 64): final crop size after padding.
- ``--sigma`` (3 floats, optional): Gaussian sigma in mm; if omitted, a
  spacing-derived anti-aliasing sigma is used.
- ``--orientation`` (str, default "LPI"): target orientation for
  ``ants.reorient_image2``.

**Assumptions**

- ANTsPy (``ants``), numpy, natsort, nibabel installed.

This file is behaviorally aligned with a sibling project script that used
``ants.save``; here outputs are written through NiBabel for consistent grids.
"""
import ants
import numpy as np
import os
import argparse
from natsort import natsorted
import nibabel as nib

# Sibling-pipeline note: a related script also implemented this file; the
# present version does not use ants.save in physical space, but uses nib to save
# on a common grid.

# Stages: 1) read, 2) reorient to LPS, 3) Gaussian smoothing, 4) resample
# (isotropic target spacing in ``args``), 5) pad, 6) crop, 7) origin = (0,0,0),
# 8) min–max normalize to [0,1], 9) write.


# =========================
# Crop function
# =========================
def crop_img(image, target_size):
    current_size = image.shape

    start_indices = [(cs - ts) // 2 for cs, ts in zip(current_size, target_size)]
    end_indices = [si + ts for si, ts in zip(start_indices, target_size)]

    # print(f'current size: {current_size}')
    # print(f'start indices: {start_indices}')
    # print(f'end indices: {end_indices}')

    cropped_image = ants.crop_indices(image, start_indices, end_indices)
    return cropped_image


# =========================
# Gaussian + Resample
# =========================
def smooth_then_resample(vol, target_spacing, sigma=None):

    orig_spacing = vol.spacing

    if sigma is None:
        # Auto anti-aliasing sigma from spacing change
        factor = [target_spacing[i] / orig_spacing[i] for i in range(3)]
        sigma = [f / 2.0 for f in factor]

    elif sigma == [100,100,100]:
        smooth_vol = vol
        resample_vol = ants.resample_image(
        smooth_vol,
        target_spacing,
        False,
        1 # for mask
        )
    else:
        smooth_vol = ants.smooth_image(vol, sigma)
        resample_vol = ants.resample_image(
        smooth_vol,
        target_spacing,
        False,
        0
        )

    print(f'original spacing: {orig_spacing}')
    print(f'target spacing: {target_spacing}')
    print(f'smoothing sigma: {sigma}')

    return resample_vol


# =========================
# Normalize
# =========================
def normalize_01(image):
    arr = image.numpy()

    # p1, p99 = np.percentile(arr, (1, 99))
    # arr = np.clip(arr, p1, p99)

    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)

    # norm_img = ants.from_numpy(
    #     arr,
    #     origin=image.origin,
    #     spacing=image.spacing,
    #     direction=image.direction
    # )

    return arr


def save_nii(data, affine, output_path):
    """Save a numpy volume as a NIfTI on disk."""
    nii = nib.Nifti1Image(data, affine)
    nib.save(nii, output_path)


# =========================
# Main
# =========================
def main(args):

    os.makedirs(args.output_dir, exist_ok=True)

    for filename in natsorted(os.listdir(args.input_dir)):

        if not filename.endswith(".nii.gz"):
            continue

        niipath = os.path.join(args.input_dir, filename)

        print("\n==============================")
        print(f"Processing: {filename}")

        vol = ants.image_read(niipath)

        # reorient
        vol = ants.reorient_image2(vol, args.orientation)

        # smooth + resample
        resample_vol = smooth_then_resample(
            vol,
            target_spacing=tuple(args.spacing),
            sigma=args.sigma
        )

        # pad
        pad_vol = ants.pad_image(resample_vol, shape=tuple(args.output_shape))

        # crop
        crop_vol = crop_img(pad_vol, args.output_shape)

        # set origin
        crop_vol.set_origin((0, 0, 0))

        # normalize
        if args.sigma!=[100.0, 100.0, 100.0]:
            norm_vol = normalize_01(crop_vol)
        else:
            norm_vol = crop_vol.numpy()

        # save
        save_path = os.path.join(args.output_dir, filename)
        # ants.image_write(norm_vol, save_path)
        idt_affine = np.eye(4)
        save_nii(norm_vol, idt_affine, save_path)



        print(f"Saved to: {save_path}")


# =========================
# Argparse
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument(
        "--spacing",
        type=float,
        nargs=3,
        # default=[1, 1, 1],
        default=[4, 4, 4],
        help="target spacing (e.g. 1 1 1)"
    )

    parser.add_argument(
        "--output_shape",
        type=int,
        nargs=3,
        # default=[256, 256, 256],
        default=[64, 64, 64],
        help="final crop size"
    )

    parser.add_argument(
        "--sigma",
        type=float,
        nargs=3,
        default=None,
        help="Gaussian sigma (mm), default auto"
    )

    parser.add_argument(
        "--orientation",
        type=str,
        default="LPI",
        help="target orientation (e.g. LPI)"
    )

    args = parser.parse_args()

    main(args)


# Example (64^3, spacing 4, Gaussian sigma 2): run from repo root or set paths.
# python preprocess/nii_process/whole_preprocess_pipeline.py \
#   --input_dir VFM/data/raw_nifti \
#   --output_dir VFM/data/preprocessed_64 \
#   --sigma 2 2 2
