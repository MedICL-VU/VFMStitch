"""
merge_edge_features_womask.py
=============================

Mask-free variant of ``merge_edge_features.py``: the same edge-feature
extraction and fusion, but no multiplication by a ``mask_*.nii.gz`` volume
(mask loading lines are commented in-place as in the original). Writes one fused
NIfTI per input under ``output_folder``.

**I/O (set before run)**

- ``input_folder`` / ``output_folder`` — use ``VFM/...`` paths (see project convention).

**Dependencies**

- As in ``merge_edge_features.py`` (nibabel, numpy, scipy, PyWavelets, open3d,
  matplotlib, scikit-image).
"""

# Per-folder: each ``*.nii.gz``; wavelet denoising, edge maps, no mask; fused NIfTI out.


import os
import numpy as np
import nibabel as nib
import scipy.ndimage as ndi
import pywt
import open3d as o3d
import matplotlib.pyplot as plt
from skimage.feature import canny, structure_tensor, structure_tensor_eigenvalues

def gaussian_smooth_3d(volume, sigma=1.0):
    """3D Gaussian smoothing."""
    return ndi.gaussian_filter(volume, sigma=sigma)


def compute_gradient_3d(volume):
    """3D Sobel gradient (magnitude and direction, original convention)."""
    gx = ndi.sobel(volume, axis=0)
    gy = ndi.sobel(volume, axis=1)
    gz = ndi.sobel(volume, axis=2)
    gradient_magnitude = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)
    gradient_direction = np.arctan2(np.sqrt(gx ** 2 + gy ** 2), gz)  # direction
    return gradient_magnitude, gradient_direction


def non_maximum_suppression_3d(grad_mag, grad_dir):
    """3D NMS (sector discretization)."""
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
    """Hysteresis threshold map."""
    strong_edges = (volume >= high_thresh)
    weak_edges = ((volume < high_thresh) & (volume >= low_thresh))

    output = np.zeros_like(volume, dtype=np.uint8)
    # output[strong_edges] = 1
    # output[weak_edges] = 0.5
    output[strong_edges] = 255
    output[weak_edges] = 100

    return output


def canny_3d(volume, sigma=1.0, low_thresh=0.1, high_thresh=0.3):
    """3D Canny (custom, original)."""
    smoothed = gaussian_smooth_3d(volume, sigma)
    grad_mag, grad_dir = compute_gradient_3d(smoothed)
    suppressed = non_maximum_suppression_3d(grad_mag, grad_dir)

    # Normalize mags
    suppressed = suppressed / np.max(suppressed)

    edges = double_threshold_hysteresis(suppressed, low_thresh, high_thresh)

    edges = edges / np.max(edges)
    return edges


# Load 3DUS
def load_nifti(filepath):
    nii = nib.load(filepath)
    return nii.get_fdata(), nii.affine

# 3D wavelet
def wavelet_transform_3d(volume, wavelet='haar', level=1):
    coeffs = pywt.wavedecn(volume, wavelet=wavelet, level=level)
    return coeffs

# inverse wavelet
def inverse_wavelet_transform_3d(coeffs):
    return pywt.waverecn(coeffs, wavelet='haar')

# denoise
def wavelet_denoise_3d(volume, wavelet='haar', level=1, threshold=0.05):
    coeffs = wavelet_transform_3d(volume, wavelet=wavelet, level=level)
    for i in range(1, len(coeffs)):
        for key in coeffs[i]:
            coeffs[i][key] = pywt.threshold(coeffs[i][key], threshold, mode='soft')
    return inverse_wavelet_transform_3d(coeffs)

# sobel
def compute_sobel(us_data):
    sobel_x = ndi.sobel(us_data, axis=0)
    sobel_y = ndi.sobel(us_data, axis=1)
    sobel_z = ndi.sobel(us_data, axis=2)
    return np.sqrt(sobel_x**2 + sobel_y**2 + sobel_z**2)

# 2d canny per slice
def compute_canny(us_data):
    canny_edges = np.zeros_like(us_data)
    for i in range(us_data.shape[2]):
        canny_edges[:, :, i] = canny(us_data[:, :, i], sigma=1).astype(np.uint8)
    return canny_edges

# LoG
def compute_log(us_data):
    return ndi.gaussian_laplace(us_data, sigma=1.0)

# structure tensor
def compute_structure_tensor(us_data):
    A = structure_tensor(us_data, sigma=1.0)
    eigvals = structure_tensor_eigenvalues(A)
    return np.max(eigvals, axis=0)

# norm
def normalize_feature(feature):
    return (feature - feature.min()) / (feature.max() - feature.min() + 1e-8)

# fuse
def fuse_features(sobel, canny, log, structure, alpha=[0.3, 0.3, 0.2, 0.2]):
    return alpha[0] * sobel + alpha[1] * canny + alpha[2] * log + alpha[3] * structure

# indices
def extract_point_cloud(fused_feature, threshold=0.5):
    indices = np.argwhere(fused_feature > threshold)
    return indices  # (N,3)

# outliers
def remove_outliers(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    return np.asarray(pcd.points)

input_folder = 'VFM/data/3dus_no_mask'   # 3D US ``*.nii.gz``
output_folder = 'VFM/data/edge_features/merge_womask_canny3d'

os.makedirs(output_folder, exist_ok=True)

for nii_file in os.listdir(input_folder):
    if nii_file.endswith(".nii.gz"):
        file_path = os.path.join(input_folder, nii_file)
        print(f"Processing: {nii_file}")

        # 3DUS; optional mask path (disabled in this variant)
        us_data, affine = load_nifti(file_path)
        # mask_path = os.path.join(input_mask_folder, 'mask_' + nii_file)
        # mask_data, mask_affine = load_nifti(mask_path)

        us_data = wavelet_denoise_3d(us_data, wavelet='haar', level=2, threshold=0.05)

        # unmasked feature maps
        sobel_feature = normalize_feature(compute_sobel(us_data))
        canny_feature = normalize_feature(compute_canny(us_data))
        canny_feature3d = normalize_feature(canny_3d(us_data, sigma=1.0, low_thresh=0.1, high_thresh=0.3))
        log_feature = normalize_feature(compute_log(us_data))
        structure_feature = normalize_feature(compute_structure_tensor(us_data))

        # # Alternative 2D-Canny fusion — reference only
        # fused_feature = fuse_features(sobel_feature, canny_feature, log_feature, structure_feature)
        # fused_feature_nifti = nib.Nifti1Image(fused_feature, affine)
        # fused_feature_output_path = os.path.join(output_folder, f"2dcanny_{nii_file}")
        # nib.save(fused_feature_nifti, fused_feature_output_path)

        fused_feature = fuse_features(sobel_feature, canny_feature3d, log_feature, structure_feature)

        fused_feature_nifti = nib.Nifti1Image(fused_feature, affine)
        fused_feature_output_path = os.path.join(output_folder, nii_file)
        nib.save(fused_feature_nifti, fused_feature_output_path)

print("Processing complete!")

# Example: set ``input_folder`` and ``output_folder`` under ``VFM/...``, then
#   python merge_edge_features_womask.py
