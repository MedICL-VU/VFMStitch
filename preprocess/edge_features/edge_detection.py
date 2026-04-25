"""
edge_detection.py
=================

Batch script: for each ``*.nii.gz`` in ``input_folder``, compute 3D Sobel
magnitude, slice-wise 2D Canny, a custom 3D Canny, and 3D LoG, writing
subfolders under ``output_folder`` (``img64_sobel``, ``img64_canny3d``,
``img64_canny2d``, ``img64_log``) as in the original layout.

**Module-level variables**

- ``input_folder`` / ``output_folder``: set before running (no CLI).

**Dependencies**

- nibabel, numpy, scipy.ndimage, scikit-image (``feature``, ``filters``);
  duplicate import blocks are preserved.
"""
import os
import numpy as np
import scipy.ndimage as ndi
import nibabel as nib
import skimage.feature as feature
import skimage.filters as filters
import skimage.io as io

import numpy as np
import scipy.ndimage as ndi
import nibabel as nib

# from code.preprocess.feature_analysis.wavelet.wavelet_analysis import output_folder


def gaussian_smooth_3d(volume, sigma=1.0):
    """3D Gaussian smoothing of a numpy volume."""
    return ndi.gaussian_filter(volume, sigma=sigma)


def compute_gradient_3d(volume):
    """3D Sobel gradient magnitude and a derived direction (see original)."""
    gx = ndi.sobel(volume, axis=0)
    gy = ndi.sobel(volume, axis=1)
    gz = ndi.sobel(volume, axis=2)
    gradient_magnitude = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)
    gradient_direction = np.arctan2(np.sqrt(gx ** 2 + gy ** 2), gz)  # gradient direction
    return gradient_magnitude, gradient_direction


def non_maximum_suppression_3d(grad_mag, grad_dir):
    """3D non-maximum suppression (discretized sector tests)."""
    suppressed = np.zeros_like(grad_mag)
    z, y, x = grad_mag.shape

    for i in range(1, z - 1):
        for j in range(1, y - 1):
            for k in range(1, x - 1):
                angle = grad_dir[i, j, k] * 180.0 / np.pi
                angle = (angle + 180) % 180

                q, r = 255, 255
                if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                    q = grad_mag[i, j, k + 1]
                    r = grad_mag[i, j, k - 1]
                elif (22.5 <= angle < 67.5):
                    q = grad_mag[i + 1, j - 1, k]
                    r = grad_mag[i - 1, j + 1, k]
                elif (67.5 <= angle < 112.5):
                    q = grad_mag[i + 1, j, k]
                    r = grad_mag[i - 1, j, k]
                elif (112.5 <= angle < 157.5):
                    q = grad_mag[i - 1, j - 1, k]
                    r = grad_mag[i + 1, j + 1, k]

                if (grad_mag[i, j, k] >= q) and (grad_mag[i, j, k] >= r):
                    suppressed[i, j, k] = grad_mag[i, j, k]
                else:
                    suppressed[i, j, k] = 0
    return suppressed


def double_threshold_hysteresis(volume, low_thresh, high_thresh):
    """Double-threshold map (255 = strong, 100 = weak in this encoding)."""
    strong_edges = (volume >= high_thresh)
    weak_edges = ((volume < high_thresh) & (volume >= low_thresh))

    output = np.zeros_like(volume, dtype=np.uint8)
    # output[strong_edges] = 1  # strong
    # output[weak_edges] = 0.5  # weak
    output[strong_edges] = 255  # strong
    output[weak_edges] = 100  # weak

    return output


def canny_3d(volume, sigma=1.0, low_thresh=0.1, high_thresh=0.3):
    """3D Canny-style pipeline (Gaussian, gradient, NMS, hysteresis)."""
    smoothed = gaussian_smooth_3d(volume, sigma)
    grad_mag, grad_dir = compute_gradient_3d(smoothed)
    suppressed = non_maximum_suppression_3d(grad_mag, grad_dir)

    # Normalize magnitudes to [0,1]
    suppressed = suppressed / np.max(suppressed)

    # Hysteresis
    edges = double_threshold_hysteresis(suppressed, low_thresh, high_thresh)

    edges = edges / np.max(edges)
    return edges

# I/O: set before running
input_folder = 'VFM/data/preprocessed_nifti'  # input directory of ``*.nii.gz``
output_folder = 'VFM/data/edge_features_bundle'


output_sobel = output_folder+"/img64_sobel"
output_canny3d = output_folder+"/img64_canny3d"
output_canny2d = output_folder+"/img64_canny2d"
output_log = output_folder+"/img64_log"

# Ensure subfolders exist
os.makedirs(output_sobel, exist_ok=True)
os.makedirs(output_canny3d, exist_ok=True)
os.makedirs(output_canny2d, exist_ok=True)
os.makedirs(output_log, exist_ok=True)

# All input volumes
nii_files = [f for f in os.listdir(input_folder) if f.endswith(".nii.gz")]

# Per-volume multi-filter export
for nii_file in nii_files:
    file_path = os.path.join(input_folder, nii_file)
    print(f"Processing: {nii_file}")

    # Read 3DUS
    nii_img = nib.load(file_path)
    us_data = nii_img.get_fdata()

    # 3D Sobel magnitude
    sobel_x = ndi.sobel(us_data, axis=0)
    sobel_y = ndi.sobel(us_data, axis=1)
    sobel_z = ndi.sobel(us_data, axis=2)
    sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2 + sobel_z**2)

    # Save Sobel
    sobel_nifti = nib.Nifti1Image(sobel_magnitude, nii_img.affine)
    sobel_output_path = os.path.join(output_sobel, f"Sobel_{nii_file}")
    nib.save(sobel_nifti, sobel_output_path)

    # 2D Canny: axis-2 stack (``skimage.feature.canny`` is 2D only)
    canny_edges = np.zeros_like(us_data, dtype=np.uint8)

    for i in range(us_data.shape[2]):  # along Z
        canny_edges[:, :, i] = feature.canny(us_data[:, :, i], sigma=1).astype(np.uint8)
    # Save 2D Canny stack
    canny_nifti = nib.Nifti1Image(canny_edges, nii_img.affine)
    canny_output_path = os.path.join(output_canny2d, f"2DCanny_{nii_file}")
    nib.save(canny_nifti, canny_output_path)

    # 3D Canny
    canny_edges = canny_3d(us_data, sigma=1.0, low_thresh=0.1, high_thresh=0.3)

    canny_nifti = nib.Nifti1Image(canny_edges, nii_img.affine)
    canny_output_path = os.path.join(output_canny3d, f"3DCanny_{nii_file}")
    nib.save(canny_nifti, canny_output_path)

    # 3D LoG
    log_filtered = filters.laplace(ndi.gaussian_filter(us_data, sigma=1))

    log_nifti = nib.Nifti1Image(log_filtered, nii_img.affine)
    log_output_path = os.path.join(output_log, f"LoG_{nii_file}")
    nib.save(log_nifti, log_output_path)

    print(f"Saved Sobel, Canny, and LoG edges for {nii_file}")

print("Processing complete!")

# Example: set ``input_folder`` and ``output_folder`` to ``VFM/path/to/...``, then
#   python edge_detection.py
