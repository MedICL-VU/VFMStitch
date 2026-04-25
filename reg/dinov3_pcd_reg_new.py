"""
dinov3_pcd_reg_new.py
======================

Point-cloud registration (Open3D) for 3D ultrasound: FPFH-only, DINOv3
feature-based, sequential (FPFH → DINOv3 RANSAC → ICP), or weighted fusion
(FPFH + DINO) modes.  Applies the final rigid transform to the moving NIfTI via
``scipy.ndimage.affine_transform`` and ``nibabel`` (origin preserved w.r.t.
reference, see ``apply_affine_to_voxel``).

**Main routines**

- ``load_nifti`` / ``load_point_cloud`` / ``save_nifti`` / ``save_affine_matrix`` — I/O.
- ``apply_affine_to_voxel`` — resample moving volume to fixed geometry without
  the affine-compounding quirk of naive matrix multiplication (long derivation
  in docstring).
- ``load_dinov3_feature`` … ``numpy_to_o3d_feature`` — DINOv3 NIfTI ↔ Open3D features.
- ``fuse_features`` / ``l2norm`` — weighted, normalized fusion of FPFH + DINO
  descriptors.
- ``preprocess_point_cloud_*`` — FPFH, DINO, or fused descriptors after voxel downsample.
- ``global_registration`` / ``refine_registration`` — RANSAC and point-to-point ICP.
- ``register_and_save`` / ``process_registration`` — one pair and batch of A/B
  NIfTI+PLY+optional DINO feature NIfTIs.

**Batching**

``process_registration`` expects ``nii_files`` / ``pc_files`` sorted; pairs are
``(0,1), (2,3), …``; runs both moving→fix and fix→moving directions (see
``prefix="M2F"`` / ``"F2M"`` in source).

**Run-time paths (``if __name__``)**

Set ``input_nii_folder``, ``input_pc_folder``, ``dino_feature_folder`` (DINO
required for non-FPFH modes), and ``output_folder`` under a root ``VFM/…``.

**Assumptions**

- Open3D, nibabel, numpy, scipy, natsort.
"""
import os
import open3d as o3d
import numpy as np
import nibabel as nib
from natsort import natsorted
from scipy import ndimage

# --- Basic I/O ---

# Load 3DUS NIfTI
def load_nifti(filepath):
    nii = nib.load(filepath)
    return nii.get_fdata(), nii.affine


# Load Open3D point cloud
def load_point_cloud(filepath):
    return o3d.io.read_point_cloud(filepath)


# Write NIfTI
def save_nifti(data, affine, output_path):
    nii = nib.Nifti1Image(data, affine)
    nib.save(nii, output_path)
    print(f"✅ NIfTI saved at: {output_path}")


# Save 4x4 to text
def save_affine_matrix(matrix, output_path):
    np.savetxt(output_path, matrix)
    print(f"✅ Affine matrix saved at: {output_path}")


# Resample a moving NIfTI with a point-cloud 4x4; keeps reference geometry (ANTs-style).
def apply_affine_to_voxel(nifti_file, affine_matrix, reference_file=None):
    """
    Map a point-cloud rigid transform (moving → fixed) to voxel resampling of the
    moving NIfTI so that the output grid uses the *reference* affine and shape.

    *Why resampling, not* ``new_affine = R @ T`` *only*: composing translation
    inside the NIfTI affine changes effective origin. We instead build the pull-back
    from each output (reference) voxel to a location in the moving array and use
    ``scipy.ndimage.affine_transform``.

    **Sketch (same as inline comments in the first public version of this file)**

    1) Naive ``new_affine = moving_affine @ affine`` maps voxel indices to world
    but shifts origin when ``affine`` encodes translation.

    2) Desired: output voxels on the **fixed** grid, ``w = target_affine @ v_out``.

    3) World consistency: ``(moving_affine @ T) @ v_in = target_affine @ v_out`` ⇒
    build the inverse map into moving indices for resampling.

    4) ``ndimage.affine_transform`` uses ``output[\\mathbf i] = input( R \\mathbf i + t )``;
    we pass the derived ``R,t`` (see code).

    **Parameters**
    nifti_file : str
        Path to the **moving** ``.nii.gz`` to be warped.
    affine_matrix : array_like (4,4)
        Rigid/affine from the Open3D pipeline (maps moving → fixed in scanner space as used in this script).
    reference_file : str, optional
        If given, the output shape and ``affine`` are taken from this **fixed** volume.

    **Returns**
    ``(nib.Nifti1Image, ndarray 4x4)`` for the resampled array and the applied output affine.
    """
    # moving
    moving_nii = nib.load(nifti_file)
    moving_volume = moving_nii.get_fdata()
    moving_affine = moving_nii.affine
    
    # target grid
    if reference_file is not None:
        fixed_nii = nib.load(reference_file)
        target_affine = fixed_nii.affine
        target_shape = fixed_nii.shape
    else:
        target_affine = moving_affine.copy()
        target_shape = moving_volume.shape
    
    # "composed" world matrix (as in a naive-affine write-up; used only to derive the pull-back)
    transformed_affine = np.dot(moving_affine, affine_matrix)
    
    # Output voxel → input voxel
    # v_src = inv(transformed_affine) @ target_affine @ v_out  (homogeneous)
    voxel_transform = np.dot(np.linalg.inv(transformed_affine), target_affine)
    
    rotation_scale = voxel_transform[:3, :3]
    translation = voxel_transform[:3, 3]
    
    resampled_volume = ndimage.affine_transform(
        moving_volume,
        rotation_scale,
        offset=translation,
        output_shape=target_shape,
        order=1,  # linear
        mode='constant',
        cval=0.0
    )
    
    new_affine = target_affine
    
    print(f'Using scipy.ndimage resampling (preserving origin)')
    print(f'Transformed affine (like dinov3_pcd_reg.py):\n{transformed_affine}')
    print(f'Target affine (origin preserved):\n{new_affine}')
    print(f'Resampled array shape: {resampled_volume.shape}, min: {np.min(resampled_volume):.4f}, max: {np.max(resampled_volume):.4f}')
    
    return nib.Nifti1Image(resampled_volume, new_affine), new_affine


# --- DINOv3 feature I/O and sampling ---

def load_dinov3_feature(feature_path):
    """
    Load a DINOv3 feature NIfTI (4D, e.g. 64×64×64×C).
    """
    nii = nib.load(feature_path)
    feature_volume = nii.get_fdata()  # shape: (64, 64, 64, 16)
    affine = nii.affine
    return feature_volume, affine


def world_to_voxel_coords(world_coords, affine):
    """
    World (scanner) to voxel indices. ``world_coords`` is (N,3) or (3,);
    ``affine`` is 4x4. Returns (N,3) or (3,).
    """
    if world_coords.ndim == 1:
        world_coords = world_coords.reshape(1, -1)
    
    # homogeneous
    world_homo = np.column_stack([world_coords, np.ones(len(world_coords))])
    
    inv_affine = np.linalg.inv(affine)
    
    voxel_homo = (inv_affine @ world_homo.T).T
    voxel_coords = voxel_homo[:, :3]
    
    return voxel_coords.squeeze() if voxel_coords.shape[0] == 1 else voxel_coords


def extract_dinov3_features_for_pointcloud(pcd, feature_volume, affine, interpolation='trilinear'):
    """
    Sample DINO features at each 3D point of ``pcd`` in feature-volume voxel space.
    ``feature_volume`` shape (H, W, D, C). ``interpolation`` in ``{'trilinear','nearest'}``.
    Returns (N, C) float array.
    """
    points = np.asarray(pcd.points)  # (N, 3)
    
    voxel_coords = world_to_voxel_coords(points, affine)  # (N, 3)
    
    if voxel_coords.ndim == 1:
        voxel_coords = voxel_coords.reshape(1, -1)
    
    H, W, D, C = feature_volume.shape
    
    features = []
    for i, vc in enumerate(voxel_coords):
        x, y, z = vc
        
        x = np.clip(x, 0, H - 1)
        y = np.clip(y, 0, W - 1)
        z = np.clip(z, 0, D - 1)
        
        if interpolation == 'trilinear':
            # trilinear
            x0, y0, z0 = int(np.floor(x)), int(np.floor(y)), int(np.floor(z))
            x1, y1, z1 = min(x0 + 1, H - 1), min(y0 + 1, W - 1), min(z0 + 1, D - 1)
            
            wx = x - x0
            wy = y - y0
            wz = z - z0
            
            # eight corners
            c000 = feature_volume[x0, y0, z0, :]
            c001 = feature_volume[x0, y0, z1, :]
            c010 = feature_volume[x0, y1, z0, :]
            c011 = feature_volume[x0, y1, z1, :]
            c100 = feature_volume[x1, y0, z0, :]
            c101 = feature_volume[x1, y0, z1, :]
            c110 = feature_volume[x1, y1, z0, :]
            c111 = feature_volume[x1, y1, z1, :]
            
            c00 = c000 * (1 - wz) + c001 * wz
            c01 = c010 * (1 - wz) + c011 * wz
            c10 = c100 * (1 - wz) + c101 * wz
            c11 = c110 * (1 - wz) + c111 * wz
            
            c0 = c00 * (1 - wy) + c01 * wy
            c1 = c10 * (1 - wy) + c11 * wy
            
            feature = c0 * (1 - wx) + c1 * wx
        else:
            # nearest
            x, y, z = int(np.round(x)), int(np.round(y)), int(np.round(z))
            feature = feature_volume[x, y, z, :]
        
        features.append(feature)
    
    return np.array(features)  # (N, C)


def numpy_to_o3d_feature(features):
    """
    Wrap (N, C) rows as an Open3D ``Feature`` (data layout (C, N)).
    """
    feature_o3d = o3d.pipelines.registration.Feature()
    feature_o3d.data = features.T.astype(np.float64)  # (C, N) for Open3D
    return feature_o3d


# --- FPFH / DINO fusion ---

def l2norm(x, axis=-1, eps=1e-8):
    """L2 row normalize."""
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / (norm + eps)


def fuse_features(fpfh_features, dino_features, alpha=1.0, beta=1.0):
    """
    Concatenate L2-normalized FPFH and DINO blocks with ``alpha, beta`` scaling,
    then L2 again on the concatenation (per-row).
    Shapes: ``(N,F)``, ``(N,D)`` → ``(N, F+D)``.
    """
    # per-branch norm
    fpfh_n = l2norm(fpfh_features)
    dino_n = l2norm(dino_features)
    
    fused = np.concatenate([alpha * fpfh_n, beta * dino_n], axis=1)
    
    fused = l2norm(fused)
    
    return fused


# --- Point cloud pre-features ---

def preprocess_point_cloud_fpfh(pcd, voxel_size):
    """Voxel downsample, normals, FPFH."""
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100)
    )
    return pcd_down, fpfh


def preprocess_point_cloud_dino(pcd, feature_volume, affine, voxel_size, interpolation='trilinear'):
    """
    Voxel downsample, then DINO trilinear (or nearest) features at downsampled points.
    """
    pcd_down = pcd.voxel_down_sample(voxel_size)
    dino_features = extract_dinov3_features_for_pointcloud(pcd_down, feature_volume, affine, interpolation)
    dino_feature_o3d = numpy_to_o3d_feature(dino_features)
    return pcd_down, dino_feature_o3d, dino_features


def preprocess_point_cloud_fused(pcd, feature_volume, affine, voxel_size, alpha=1.0, beta=1.0, interpolation='trilinear'):
    """
    FPFH + DINO on the *same* downsampled point set; weighted fusion.
    """
    pcd_down, fpfh = preprocess_point_cloud_fpfh(pcd, voxel_size)
    fpfh_data = fpfh.data.T  # (N, F)
    
    dino_features = extract_dinov3_features_for_pointcloud(pcd_down, feature_volume, affine, interpolation)
    
    fused_features = fuse_features(fpfh_data, dino_features, alpha, beta)
    fused_feature_o3d = numpy_to_o3d_feature(fused_features)
    
    return pcd_down, fused_feature_o3d


# --- Registration steps ---

def global_registration(source, target, source_feature, target_feature, distance_threshold):
    """RANSAC on feature matches."""
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source, target, source_feature, target_feature, True, distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        4,  # inlier subset size
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)
    )
    return result


def refine_registration(source, target, init_transformation, distance_threshold):
    """Point-to-point ICP."""
    return o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, init_transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )


# --- Single case (one moving / one fixed) ---

def register_and_save(fix_nii_path, mov_nii_path, fix_pc_path, mov_pc_path,
                      fix_dino_path, mov_dino_path,
                      output_pc_folder, output_nii_folder, output_txt_folder, 
                      prefix, mode='fpfh_only', voxel_size=5.0, alpha=1.0, beta=1.0):
    """
    RANSAC + ICP; save PLY, warped NIfTI, 4x4 text. ``mode`` in
    ``{fpfh_only, dino_only, sequential, fused}``; ``alpha,beta`` only for
    ``fused``.
    """
    print(f"🔹 Registration: Fix {os.path.basename(fix_nii_path)} ↔ Mov {os.path.basename(mov_nii_path)}")
    print(f"   Mode: {mode}")
    
    target_pcd = load_point_cloud(fix_pc_path)
    source_pcd = load_point_cloud(mov_pc_path)
    
    distance_threshold = voxel_size * 2.0
    transformation = np.eye(4)
    
    if mode == 'fpfh_only':
        print("  Using FPFH features only (baseline)...")
        source_down, source_fpfh = preprocess_point_cloud_fpfh(source_pcd, voxel_size)
        target_down, target_fpfh = preprocess_point_cloud_fpfh(target_pcd, voxel_size)
        
        ransac_result = global_registration(
            source_down, target_down, source_fpfh, target_fpfh, distance_threshold
        )
        print(f"{prefix} FPFH RANSAC Transformation:\n", ransac_result.transformation)
        
        icp_result = refine_registration(source_pcd, target_pcd, ransac_result.transformation, distance_threshold)
        print(f"{prefix} ICP Transformation:\n", icp_result.transformation)
        transformation = icp_result.transformation
        
    elif mode == 'dino_only':
        print("  Using DINOv3 features only...")
        fix_feature_volume, fix_affine = load_dinov3_feature(fix_dino_path)
        mov_feature_volume, mov_affine = load_dinov3_feature(mov_dino_path)
        
        source_down, source_dino_feature, _ = preprocess_point_cloud_dino(
            source_pcd, mov_feature_volume, mov_affine, voxel_size
        )
        target_down, target_dino_feature, _ = preprocess_point_cloud_dino(
            target_pcd, fix_feature_volume, fix_affine, voxel_size
        )
        
        ransac_result = global_registration(
            source_down, target_down, source_dino_feature, target_dino_feature, distance_threshold
        )
        print(f"{prefix} DINOv3 RANSAC Transformation:\n", ransac_result.transformation)
        
        icp_result = refine_registration(source_pcd, target_pcd, ransac_result.transformation, distance_threshold)
        print(f"{prefix} ICP Transformation:\n", icp_result.transformation)
        transformation = icp_result.transformation
        
    elif mode == 'sequential':
        print("  Sequential: FPFH -> DINOv3 -> ICP...")
        fix_feature_volume, fix_affine = load_dinov3_feature(fix_dino_path)
        mov_feature_volume, mov_affine = load_dinov3_feature(mov_dino_path)
        
        # Step 1: FPFH
        source_down_fpfh, source_fpfh = preprocess_point_cloud_fpfh(source_pcd, voxel_size)
        target_down_fpfh, target_fpfh = preprocess_point_cloud_fpfh(target_pcd, voxel_size)
        
        ransac_fpfh = global_registration(
            source_down_fpfh, target_down_fpfh, source_fpfh, target_fpfh, distance_threshold
        )
        print(f"{prefix} FPFH RANSAC Transformation:\n", ransac_fpfh.transformation)
        
        source_pcd_transformed = source_pcd.transform(ransac_fpfh.transformation)
        
        # Step 2: DINO on FPFH-warped moving cloud
        source_down_dino, source_dino_feature, _ = preprocess_point_cloud_dino(
            source_pcd_transformed, mov_feature_volume, mov_affine, voxel_size
        )
        target_down_dino, target_dino_feature, _ = preprocess_point_cloud_dino(
            target_pcd, fix_feature_volume, fix_affine, voxel_size
        )
        
        ransac_dino = global_registration(
            source_down_dino, target_down_dino, source_dino_feature, target_dino_feature, distance_threshold
        )
        print(f"{prefix} DINOv3 RANSAC Transformation:\n", ransac_dino.transformation)
        
        combined_transform = np.dot(ransac_dino.transformation, ransac_fpfh.transformation)
        
        # Step 3: ICP
        icp_result = refine_registration(source_pcd, target_pcd, combined_transform, distance_threshold)
        print(f"{prefix} ICP Transformation:\n", icp_result.transformation)
        transformation = icp_result.transformation
        
    elif mode == 'fused':
        print(f"  Fusing FPFH and DINOv3 features (alpha={alpha}, beta={beta})...")
        fix_feature_volume, fix_affine = load_dinov3_feature(fix_dino_path)
        mov_feature_volume, mov_affine = load_dinov3_feature(mov_dino_path)
        
        source_down, source_fused_feature = preprocess_point_cloud_fused(
            source_pcd, mov_feature_volume, mov_affine, voxel_size, alpha, beta
        )
        target_down, target_fused_feature = preprocess_point_cloud_fused(
            target_pcd, fix_feature_volume, fix_affine, voxel_size, alpha, beta
        )
        
        ransac_result = global_registration(
            source_down, target_down, source_fused_feature, target_fused_feature, distance_threshold
        )
        print(f"{prefix} Fused RANSAC Transformation:\n", ransac_result.transformation)
        
        icp_result = refine_registration(source_pcd, target_pcd, ransac_result.transformation, distance_threshold)
        print(f"{prefix} ICP Transformation:\n", icp_result.transformation)
        transformation = icp_result.transformation
    
    else:
        raise ValueError(f"Unknown mode: {mode}. Choose from 'fpfh_only', 'dino_only', 'sequential', 'fused'")
    
    reg_mov_pcd = source_pcd.transform(transformation)
    registered_mov_nii, new_affine = apply_affine_to_voxel(mov_nii_path, transformation, reference_file=fix_nii_path)
    
    o3d.io.write_point_cloud(os.path.join(output_pc_folder, f"{os.path.basename(mov_pc_path)}"), reg_mov_pcd)
    save_nifti(registered_mov_nii.get_fdata(), registered_mov_nii.affine,
               os.path.join(output_nii_folder, f"{os.path.basename(mov_nii_path)}"))
    save_affine_matrix(new_affine,
                       os.path.join(output_txt_folder, f"{os.path.basename(mov_nii_path).replace('.nii.gz', '.txt')}"))


# --- Batch: paired files in two folders ---

def process_registration(input_nii_folder, input_pc_folder, dino_feature_folder, output_folder,
                        mode='fpfh_only', voxel_size=5.0, alpha=1.0, beta=1.0):
    """
    Pair A/B NIfTIs and PLYs, optionally DINO4D NIfTIs; run two directions per pair.
    ``voxel_size`` — Open3D downsample voxel; ``alpha,beta`` for ``fused`` only.
    """
    nii_files = natsorted([f for f in os.listdir(input_nii_folder) if f.endswith(".nii.gz")])
    pc_files = natsorted([f for f in os.listdir(input_pc_folder) if f.endswith(".ply") or f.endswith(".pcd")])
    
    if len(nii_files) < 2 or len(pc_files) < 2:
        print("⚠️ At least 2 NIfTI files and 2 point clouds are required for registration.")
        return
    
    output_pc_folder = os.path.join(output_folder, "reg_pc")
    output_nii_folder = os.path.join(output_folder, "reg_nii")
    output_txt_folder = os.path.join(output_folder, "reg_txt")
    os.makedirs(output_pc_folder, exist_ok=True)
    os.makedirs(output_nii_folder, exist_ok=True)
    os.makedirs(output_txt_folder, exist_ok=True)
    
    reg_datadict = [
        {
            "fix_img": nii_files[i * 2],
            "mov_img": nii_files[i * 2 + 1],
            "fix_pc": pc_files[i * 2],
            "mov_pc": pc_files[i * 2 + 1],
        }
        for i in range(len(nii_files) // 2)
    ]
    
    for i, item in enumerate(reg_datadict):
        fix_dino_path = None
        mov_dino_path = None
        if mode in ['dino_only', 'sequential', 'fused']:
            fix_dino_path = os.path.join(dino_feature_folder, item['fix_img'])
            mov_dino_path = os.path.join(dino_feature_folder, item['mov_img'])
            
            if not os.path.exists(fix_dino_path) or not os.path.exists(mov_dino_path):
                print(f"⚠️ Skipping {item['fix_img']} ↔ {item['mov_img']}: DINOv3 features not found")
                continue
        
        # M2F: fix image fixed, mov moving
        register_and_save(
            os.path.join(input_nii_folder, item['fix_img']),
            os.path.join(input_nii_folder, item['mov_img']),
            os.path.join(input_pc_folder, item['fix_pc']),
            os.path.join(input_pc_folder, item['mov_pc']),
            fix_dino_path, mov_dino_path,
            output_pc_folder, output_nii_folder, output_txt_folder, 
            prefix="M2F", mode=mode, voxel_size=voxel_size, alpha=alpha, beta=beta
        )
        
        # F2M: swap
        register_and_save(
            os.path.join(input_nii_folder, item['mov_img']),
            os.path.join(input_nii_folder, item['fix_img']),
            os.path.join(input_pc_folder, item['mov_pc']),
            os.path.join(input_pc_folder, item['fix_pc']),
            mov_dino_path, fix_dino_path,
            output_pc_folder, output_nii_folder, output_txt_folder, 
            prefix="F2M", mode=mode, voxel_size=voxel_size, alpha=alpha, beta=beta
        )


if __name__ == '__main__':
    # Paths under root ``VFM/``; align NIfTIs, PLY/PCD white point clouds, and (if needed) 4D DINO NIfTIs
    input_nii_folder = 'VFM/data/preprocessed_nifti'
    input_pc_folder = 'VFM/data/point_clouds/mask_erosion/white'
    dino_feature_folder = 'VFM/data/dino_feature_nifti'
    # Modes: 'fpfh_only' | 'dino_only' | 'sequential' | 'fused'
    mode = 'fpfh_only'
    # Output root (subfolders reg_pc, reg_nii, reg_txt are created)
    output_folder = 'VFM/output/registration/V2_LPS_Norm_S1_256_sig1' + mode + '_vs10'

    # fusion-only scalars
    alpha = 1.0
    beta = 1.0
    
    # Open3D voxel size for FPFH / DINO downsampling
    voxel_size = 10.0  # 5.0 used as default in some runs
    
    process_registration(
        input_nii_folder, input_pc_folder, dino_feature_folder, output_folder,
        mode=mode, voxel_size=voxel_size, alpha=alpha, beta=beta
    )

# Example: set ``VFM/...`` folders, choose ``mode``, adjust ``voxel_size``, then
#   python reg/dinov3_pcd_reg_new.py

