"""
mask_generation_erosion.py
============================

For each ``*.nii.gz`` in ``input_folder``, estimate a binary foreground mask
(soft threshold, 3D hole fill, then distance-based shrink), and save
``mask_<name>.nii.gz`` to ``output_folder``.

**Functions**

- ``shrink_mask``: Erosion via distance transform (keep voxels farther than
  ``shrink_voxel`` from the mask boundary).
- ``filehole_in_3axis``: 3D binary hole filling on axis-parallel 2D slices.
- ``gen_mask(input_folder, output_folder)``: main batch loop over inputs.

**Arguments (runtime)**

- ``shrink_voxel=10`` inside ``gen_mask`` call chain controls how much the mask
  is eroded after hole filling (see ``shrink_mask``).

**Dependencies**

- nibabel, numpy, scipy.ndimage, natsort, ``distance_transform_edt``.
"""
import nibabel as nib
import numpy as np
import os
import torch
import torch.nn.functional as F
import math
import sys
# from pytorch_ssim import *
from natsort import natsorted
from scipy import ndimage
# Legacy note: this module was also used in mask generation for original datasets via
# thresholding and hole filling.

from scipy.ndimage import distance_transform_edt

def shrink_mask(mask, shrink_voxel=2):
    dist = distance_transform_edt(mask)
    return (dist > shrink_voxel).astype(np.uint8)

def filehole_in_3axis(img):
    for i in range(img.shape[0]):
        img[i,:,:] = ndimage.binary_fill_holes(img[i,:,:]).astype(int)
    for j in range(img.shape[1]):
        img[:,j,:] = ndimage.binary_fill_holes(img[:,j,:]).astype(int)
    for k in range(img.shape[2]):
        img[:,:,k] = ndimage.binary_fill_holes(img[:,:,k]).astype(int)
    return img

def gen_mask(input_folder,output_folder):
    nccs, ssims, mses, psnrs = [], [], [], []

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    img_list = os.listdir(input_folder)
    img_list_sorted = natsorted(img_list)


    # for file in os.listdir(gt_folder):
    for file in img_list_sorted:
        if file.endswith('.nii.gz'):
            # print(f'file1 is {os.path.join(gt_folder, file)}')
            # print(f'file2 is {os.path.join(pred_folder, file)}')
            # Load volume
            gt_nii = nib.load(os.path.join(input_folder, file))
            gt_img = gt_nii.get_fdata()

            # additional step: get the non-zero mask first from the vo1 and vol2
            # gt_binary = (gt_img > 0.001*np.max(gt_img))
            gt_binary = (gt_img > 0.001 * np.mean(gt_img))
            # gt_binary = (gt_img > 0.001)

            # non_zero_mask = gt_binary
            non_zero_mask =filehole_in_3axis(gt_binary).astype(int)

            # non_zero_mask = filehole_in_3axis(gt_binary).astype(np.uint8)

            # Optional shrink
            non_zero_mask = shrink_mask(non_zero_mask, shrink_voxel=10)
            # print(non_zero_mask.shape)
            # print(np.max(non_zero_mask))
            # save the non zero mask, after the first round we can comment this line
            mask_save_path = os.path.join(output_folder)
            mask_name = 'mask_'+ file

            if not os.path.exists(mask_save_path):
                os.makedirs(mask_save_path)

            mask_save_name = os.path.join(mask_save_path, mask_name)
            # print(non_zero_mask.dtype)
            mask_nii = nib.Nifti1Image(non_zero_mask.astype(np.uint8), gt_nii.affine)
            nib.save(mask_nii, mask_save_name)




# compare_folders: unchanged from the original project stub.

if __name__ == "__main__":
    # Example (uncomment): per-volume foreground mask + ``mask_*.nii.gz`` output
    # input_folder = 'VFM/path/to/your/preprocessed_nifti'   # ``*.nii.gz`` in
    # output_folder = 'VFM/path/to/your/mask_erosion'        # ``mask_*.nii.gz`` out
    # gen_mask(input_folder, output_folder)
    pass
