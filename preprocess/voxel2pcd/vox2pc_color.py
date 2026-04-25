"""
vox2pc_color.py
================

For each ``*.nii.gz`` + ``mask_*.nii.gz`` pair: wavelet-denoise the volume, fuse
Sobel / 2D Canny / LoG / structure-tensor features (mask-multiplied), threshold
a point set from the fused map, color points in several styles (``white``,
``feature_based``, etc.), and write PLYs under subfolders of ``output_base_folder``.

**Folder variables (module scope)**

- ``input_folder`` — preprocessed ``*.nii.gz``
- ``input_mask_folder`` — ``mask_*.nii.gz``
- ``output_base_folder`` — one subfolder per ``color_modes`` key

**Dependencies**
- pywt, nibabel, numpy, scipy, open3d, matplotlib, scikit-image, natsort; duplicate
  ``open3d`` import kept as in the original.
"""

# Per-folder: wavelet, edge fusion, then colored point cloud export; choice of
# 2D/3D Canny variants lives in the merged edge path used here; figure-friendly coloring.


import os
import numpy as np
import nibabel as nib
import scipy.ndimage as ndi
import pywt
import open3d as o3d
from skimage.feature import canny, structure_tensor, structure_tensor_eigenvalues
import matplotlib.pyplot as plt

import open3d as o3d
from natsort import natsorted

def visualize_grayscale_point_cloud(pcd):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)

    # black background, explicit point color mode
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])  # black
    opt.point_color_option = o3d.visualization.PointColorOption.Color  # per-point

    vis.run()
    vis.destroy_window()



# NIfTI reader
def load_nifti(filepath):
    nii = nib.load(filepath)
    return nii.get_fdata(), nii.affine

# 3D wavelet
def wavelet_transform_3d(volume, wavelet='haar', level=1):
    coeffs = pywt.wavedecn(volume, wavelet=wavelet, level=level)
    return coeffs

# inverse
def inverse_wavelet_transform_3d(coeffs):
    return pywt.waverecn(coeffs, wavelet='haar')

# soft denoise
def wavelet_denoise_3d(volume, wavelet='haar', level=1, threshold=0.05):
    coeffs = wavelet_transform_3d(volume, wavelet=wavelet, level=level)
    for i in range(1, len(coeffs)):  # high-frequency only
        for key in coeffs[i]:
            coeffs[i][key] = pywt.threshold(coeffs[i][key], threshold, mode='soft')
    return inverse_wavelet_transform_3d(coeffs)

# sobel
def compute_sobel(us_data):
    sobel_x = ndi.sobel(us_data, axis=0)
    sobel_y = ndi.sobel(us_data, axis=1)
    sobel_z = ndi.sobel(us_data, axis=2)
    return np.sqrt(sobel_x**2 + sobel_y**2 + sobel_z**2)

# 2d canny
def compute_canny(us_data):
    canny_edges = np.zeros_like(us_data)
    for i in range(us_data.shape[2]):
        canny_edges[:, :, i] = canny(us_data[:, :, i], sigma=1).astype(np.uint8)
    return canny_edges

# LoG
def compute_log(us_data):
    return ndi.gaussian_laplace(us_data, sigma=1.0)

# max eig structure tensor
def compute_structure_tensor(us_data):
    A = structure_tensor(us_data, sigma=1.0)
    eigvals = structure_tensor_eigenvalues(A)
    return np.max(eigvals, axis=0)

# [0,1] norm (original omits +eps in min denom one place — preserved)
def normalize_feature(feature):
    return (feature - feature.min()) / (feature.max() - feature.min())

# four-way fusion
def fuse_features(sobel, canny, log, structure, alpha=[0.3, 0.3, 0.2, 0.2]):
    return alpha[0] * sobel + alpha[1] * canny + alpha[2] * log + alpha[3] * structure

# dense indices
def extract_point_cloud(fused_feature, threshold=0.5):
    indices = np.argwhere(fused_feature > threshold)
    return indices  # (N, 3)

# statistical outlier removal
def remove_outliers(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    # pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=100, std_ratio=0.5)
    return np.asarray(pcd.points)

# I/O
input_folder = 'VFM/data/preprocessed_nifti'   # 3D US ``*.nii.gz``
input_mask_folder = 'VFM/data/preprocessed_nifti_mask_erosion'   # ``mask_*.nii.gz``
output_base_folder = 'VFM/data/point_clouds_by_color_mode'

# Subfolders for each color layout
color_modes = {
    "white": "white",
    "feature_based": "feature_based",
    "height_based": "height_based",
    "edge_intensity": "edge_intensity",
    "grayscale": "grayscale"
}

output_folders = {mode: os.path.join(output_base_folder, folder) for mode, folder in color_modes.items()}
for folder in output_folders.values():
    os.makedirs(folder, exist_ok=True)

for nii_file in natsorted(os.listdir(input_folder)):
    if nii_file.endswith(".nii.gz"):
        file_path = os.path.join(input_folder, nii_file)
        print(f"Processing: {nii_file}")

        us_data, affine = load_nifti(file_path)
        mask_path = os.path.join(input_mask_folder, 'mask_' + nii_file)
        mask_data, mask_affine = load_nifti(mask_path)

        us_data = wavelet_denoise_3d(us_data, wavelet='haar', level=2, threshold=0.05)

        sobel_feature = normalize_feature(compute_sobel(us_data)) * mask_data
        canny_feature = normalize_feature(compute_canny(us_data)) * mask_data
        log_feature = normalize_feature(compute_log(us_data)) * mask_data
        structure_feature = normalize_feature(compute_structure_tensor(us_data)) * mask_data

        fused_feature = fuse_features(sobel_feature, canny_feature, log_feature, structure_feature)

        raw_point_cloud = extract_point_cloud(fused_feature, threshold=0.5)
        transformed_points = np.dot(affine[:3, :3], raw_point_cloud.T).T + affine[:3, 3]

        filtered_points = remove_outliers(transformed_points)

        # intensity norms for color modes
        eps = 1e-8
        us_data_norm = (us_data - np.min(us_data)) / (np.max(us_data) - np.min(us_data) + eps)
        fused_feature_norm = (fused_feature - np.min(fused_feature)) / (
                    np.max(fused_feature) - np.min(fused_feature) + eps)

        colors = {
            "white": np.ones_like(filtered_points) * 255,  # solid white
            "feature_based": np.stack([sobel_feature.flatten(), canny_feature.flatten(), structure_feature.flatten()],
                                      axis=-1),  # R/G/B channels
            "height_based": np.stack([
                np.zeros_like(filtered_points[:, 2]),
                np.zeros_like(filtered_points[:, 2]),
                filtered_points[:, 2] / (np.max(filtered_points[:, 2]) + eps)
            ], axis=-1),  # height → blue
            "edge_intensity": plt.cm.jet(fused_feature_norm.flatten())[:, :3],  # JET
            "grayscale": np.stack([us_data_norm.flatten()] * 3, axis=-1)  # R=G=B
        }

        # # optional clip
        # for mode in colors:
        #     colors[mode] = np.clip(colors[mode], 0, 1)
        for mode in colors:
            min_val = np.min(colors[mode])
            max_val = np.max(colors[mode])
            colors[mode] = (colors[mode] - min_val) / (max_val - min_val + eps)
        # ---------------------------------------------------------------------------------------------
        for mode, folder in output_folders.items():
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(filtered_points)
            pcd.colors = o3d.utility.Vector3dVector(colors[mode])
            if mode == 'white':
                output_ply = os.path.join(folder, f"{nii_file.replace('.nii.gz', f'.ply')}")
            else:
                output_ply = os.path.join(folder, f"{nii_file.replace('.nii.gz', f'_{mode}.ply')}")
            o3d.io.write_point_cloud(output_ply, pcd)

        # o3d.visualization.draw_geometries([pcd])

            # optional interactive view
            # visualize_grayscale_point_cloud(pcd)

        print(f"Saved point clouds with different color modes for {nii_file}")

print("Processing complete!")

# Example: set ``input_folder``, ``input_mask_folder``, and ``output_base_folder`` to ``VFM/...``, then
#   python vox2pc_color.py
