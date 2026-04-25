"""
apply_mask.py
=============

Multiply each 3D NIfTI volume in ``input_folder`` by a same-named mask from
``mask_folder`` (mask filenames are ``mask_`` + image basename) and write
masked volumes to ``output_folder``.

**Functions**

- ``filehole_in_3axis``: (unused here) 3D hole filling via binary fill on axis planes.
- ``apply_mask(input_folder, mask_folder, output_folder)``: For each
  ``mask_*.nii.gz``, load mask and image ``*.nii.gz`` (``file[5:]`` strips the
  ``mask_`` prefix), then save the product with the original image affine.

**Assumptions**

- NiBabel; matching filenames between mask and image folders; ``natsort`` for ordering.

Dependencies: nibabel, numpy, scipy (imported; torch imports are unused in this
script as checked against original), natsort.
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
# Legacy note: this script was used to apply masks to original datasets via
# thresholding and hole filling in related pipelines.

def filehole_in_3axis(img):
    for i in range(img.shape[0]):
        img[i,:,:] = ndimage.binary_fill_holes(img[i,:,:]).astype(int)
    for j in range(img.shape[1]):
        img[:,j,:] = ndimage.binary_fill_holes(img[:,j,:]).astype(int)
    for k in range(img.shape[2]):
        img[:,:,k] = ndimage.binary_fill_holes(img[:,:,k]).astype(int)
    return img

def apply_mask(input_folder,mask_folder, output_folder):
    nccs, ssims, mses, psnrs = [], [], [], []

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    img_list = os.listdir(input_folder)
    img_list_sorted = natsorted(img_list)

    mask_list = os.listdir(mask_folder)
    mask_list_sorted = natsorted(mask_list)


    # for file in os.listdir(gt_folder):
    for file in mask_list_sorted:
        if file.endswith('.nii.gz'):
            # print(f'file1 is {os.path.join(gt_folder, file)}')
            # print(f'file2 is {os.path.join(pred_folder, file)}')
            # Load each volume
            mask_nii = nib.load(os.path.join(mask_folder, file))
            mask = mask_nii.get_fdata()

            img_nii = nib.load(os.path.join(input_folder, file[5:]))
            img = img_nii.get_fdata()

            img = img * mask

            img_save_path = os.path.join(output_folder)
            img_name = file[5:]

            if not os.path.exists(img_save_path):
                os.makedirs(img_save_path)

            img_save_name = os.path.join(img_save_path, img_name)
            # print(non_zero_mask.dtype)
            mask_img_nii = nib.Nifti1Image(img, img_nii.affine)
            nib.save(mask_img_nii, img_save_name)




# compare_folders: unchanged from the original project stub.

if __name__ == "__main__":
    # Example (uncomment, set ``VFM/...`` to your tree; masks are ``mask_``+ image basename):
    # input_folder = 'VFM/path/to/your/images'    # directory of image ``*.nii.gz``
    # mask_folder = 'VFM/path/to/your/masks'     # ``mask_*.nii.gz`` for each case
    # output_folder = 'VFM/path/to/your/masked'  # min–max masked volumes
    # apply_mask(input_folder, mask_folder, output_folder)
    pass
