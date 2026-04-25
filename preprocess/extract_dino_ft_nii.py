"""
extract_dino_ft_nii.py
======================

End-to-end DINOv3 feature extraction for paired 3D NIfTIs: slice 3D volumes
along user-selected axes, run a ViT backbone, optionally PCA-reduce
channelwise, upsample per-slice feature maps to the input cube size, and save
4D DIfTI volumes plus optional RGB center-slice PNGs.

**Major pieces**

- ``ModelManager`` + ``MODEL_CONFIGS``: ``torch.hub`` load of local DINOv3; CUDA inference.
- ``resize_3d_volume`` / ``normalize_3d_volume`` / ``normalize_2d_slice`` — I/O and scaling.
- ``find_subject_pairs`` — filename regex ``<base>{a|b}`` to pair A/B NIfTIs.
- ``extract_features`` + ``process_subject_pair`` — per-axis slice loop, optional PCA, stack to 4D.
- ``main`` — CLI, Loops all subject pairs in ``--input_dir``.

**CLI (see ``--help``):** ``--input_dir``, ``--output_dir``, ``--model``,
``--custom_weights``, ``--axes`` (0/1/2), ``--target_size`` (patch multiple of 16),
``--k_components`` (PCA), ``--pca_mode`` (``all_pair`` / ``plane_pair``; same
implementation in current code for both, preserved), ``--interpolation``,
2D slice resize group, ``--norm_mode`` (``volume`` / ``slice``).

**Paths**

- Set ``DINOV3_LOCATION`` and default checkpoint paths under a root ``VFM/...``
  (or pass ``--custom_weights``).

**Dependencies**
- torch, torchvision, PIL, nibabel, sklearn, scipy, tqdm, numpy; local DINOv3
  import via ``sys.path`` as in the original.
"""
import os
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
import nibabel as nib
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import argparse
import time
from tqdm import tqdm
import sys
import re
from scipy import ndimage
from sklearn.decomposition import PCA
import warnings
from natsort import natsorted
warnings.filterwarnings('ignore')

# DINOv3 source tree (local clone) — set to your ``VFM/dinov3`` checkout
DINOV3_LOCATION = 'VFM/dinov3'
sys.path.append(DINOV3_LOCATION)

# --- model registry ---
@dataclass
class ModelConfig:
    """Name, local checkpoint, depth, feature dim, short help text."""
    name: str
    weights_path: str
    num_layers: int
    feature_dim: int
    description: str = ""

MODEL_CONFIGS = {
    'dinov3_vits16': ModelConfig(
        name='dinov3_vits16',
        weights_path='VFM/dinov3/ckpt/dinov3_vits16_pretrain_lvd1689m-08c60483.pth',
        num_layers=12,
        feature_dim=384,
        description='ViT-S/16 (22M params, fastest)'
    ),
    'dinov3_vitb16': ModelConfig(
        name='dinov3_vitb16',
        weights_path='VFM/dinov3/ckpt/dinov3_vitb16_pretrain_lvd1689m.pth',
        num_layers=12,
        feature_dim=768,
        description='ViT-B/16 (86M params, balanced)'
    ),
    'dinov3_vitl16': ModelConfig(
        name='dinov3_vitl16',
        weights_path='VFM/dinov3/ckpt/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth',
        num_layers=24,
        feature_dim=1024,
        description='ViT-L/16 (303M params, higher quality)'
    ),
}

# IMAGENET mean/std, ViT-P16
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
PATCH_SIZE = 16

# --- model loading ---
class ModelManager:
    """Loads one DINOv3 checkpoint."""
    
    def __init__(self):
        self.model = None
        self.config = None
    
    def load_model(self, model_key: str, custom_weights_path: str = None):
        """``torch.hub`` load."""
        if model_key not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model key: {model_key}. Valid: {list(MODEL_CONFIGS.keys())}")
        
        config = MODEL_CONFIGS[model_key]
        weights_path = custom_weights_path or config.weights_path
        
        print(f"Loading model: {config.name}")
        print(f"  description: {config.description}")
        print(f"  weights: {weights_path}")
        
        if not os.path.exists(weights_path):
            print(f"  warn: weight file missing; will try torch hub / download")
            model = torch.hub.load(
                repo_or_dir=DINOV3_LOCATION,
                model=config.name,
                source="local",
                pretrained=True,
            )
        else:
            model = torch.hub.load(
                repo_or_dir=DINOV3_LOCATION,
                model=config.name,
                source="local",
                weights=weights_path,
            )
        
        model.cuda()
        model.eval()
        
        self.model = model
        self.config = config
        
        print(f"ok: model loaded")
        return model, config

# --- 3D resize (legacy helper) ---
def resize_3d_volume(volume, target_shape, interpolation='linear'):
    """
    Resize a 3D array to ``target_shape`` (``scipy.ndimage.zoom``).
    
    Args:
        volume: 3D numpy array [D, H, W]
        target_shape: target shape (D', H', W')
        interpolation: interpolation ('linear', 'cubic', 'nearest')
    
    Returns:
        resized: resized ndarray
    """
    original_shape = volume.shape
    
    # zoom factors
    zoom_factors = (
        target_shape[0] / original_shape[0],
        target_shape[1] / original_shape[1],
        target_shape[2] / original_shape[2]
    )
    
    # zoom order
    if interpolation == 'linear':
        order = 1
    elif interpolation == 'cubic':
        order = 3
    elif interpolation == 'nearest':
        order = 0
    else:
        order = 1
    
    # float32 (scipy)
    if volume.dtype != np.float32:
        volume = volume.astype(np.float32)
    
    # ndimage.zoom
    resized = ndimage.zoom(volume, zoom_factors, order=order, mode='constant', cval=0.0)
    
    print(f"  3D Resize: {original_shape} -> {resized.shape} (zoom factors: {zoom_factors})")
    
    return resized

# ==================== 3D Normalization ====================
def normalize_3d_volume(volume):
    """
    Min–max 3D volume; scale to [0,255] as in the original.
    
    Args:
        volume: 3D numpy array
    
    Returns:
        float32 array
    """
    # nan_to_num
    volume = np.nan_to_num(volume, nan=0.0, posinf=0.0, neginf=0.0)
    
    # then global min–max
    vmin = volume.min()
    vmax = volume.max()
    
    if vmax - vmin > 0:
        normalized = (volume - vmin) / (vmax - vmin)*255
    else:
        normalized = np.zeros_like(volume, dtype=np.float32)
    
    return normalized.astype(np.float32)

# ==================== 2D Slice Normalization ====================
def normalize_2d_slice(slice_data):
    """
    Min–max one slice to uint8 in [0,255]
    """
    slice_data = np.nan_to_num(slice_data, nan=0.0, posinf=0.0, neginf=0.0)
    vmin = slice_data.min()
    vmax = slice_data.max()
    if vmax - vmin > 0:
        out = (slice_data - vmin) / (vmax - vmin) * 255.0
    else:
        out = np.zeros_like(slice_data, dtype=np.float32)
    return out.astype(np.uint8)

# --- PIL / tensor image prep ---
def numpy_to_pil_image(slice_data, convert_rgb=True):
    """
    ``numpy`` 2D slice to ``PIL.Image`` (optionally RGB)
    
    Args:
        slice_data: 2D numpy array
        convert_rgb: if True, convert Luma to RGB
    
    Returns:
        ``PIL.Image``
    """
    # clip to 0–255
    if slice_data.dtype != np.uint8:
        if slice_data.max() <= 1.0:
            slice_data = (slice_data * 255).astype(np.uint8)
        else:
            slice_data = np.clip(slice_data, 0, 255).astype(np.uint8)
    
    # ``Image.fromarray``
    if convert_rgb:
        # L → RGB
        img = Image.fromarray(slice_data, mode='L')
        img = img.convert('RGB')
    else:
        img = Image.fromarray(slice_data, mode='L')
    
    return img

def resize_to_patch_size(image, target_size, patch_size=PATCH_SIZE):
    """Resize so H,W are multiples of ``patch_size``."""
    h_patches = max(1, int(target_size / patch_size))
    w_patches = max(1, int(target_size / patch_size))
    new_h = h_patches * patch_size
    new_w = w_patches * patch_size
    resized = TF.resize(image, (new_h, new_w), antialias=True)
    return TF.to_tensor(resized)

def prepare_image_for_model(image, target_size, patch_size=PATCH_SIZE):
    """CHW tensor, ImageNet norm."""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image_tensor = resize_to_patch_size(image, target_size, patch_size)
    image_tensor = TF.normalize(image_tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)
    return image_tensor.unsqueeze(0)

def extract_features(model, image, target_size, num_layers):
    """Last layer grid features from DINOv3."""
    image_tensor = prepare_image_for_model(image, target_size).cuda()
    
    start_time_extract_features = time.time()
    with torch.inference_mode():
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            feats = model.get_intermediate_layers(
                image_tensor, 
                n=range(num_layers), 
                reshape=True, 
                norm=True
            )
    end_time_extract_features = time.time()

    # ------ memory check
    # allocated VRAM
    allocated = torch.cuda.memory_allocated(device=0)  # first CUDA device
    print(f"cuda allocated: {allocated / 1024**2:.2f} MB")

    # reserved
    reserved = torch.cuda.memory_reserved(device=0)
    print(f"cuda reserved: {reserved / 1024**2:.2f} MB")

    # max allocated
    max_allocated = torch.cuda.max_memory_allocated(device=0)
    print(f"cuda max allocated: {max_allocated / 1024**2:.2f} MB")

    # reset peak
    torch.cuda.reset_peak_memory_stats(device=0)

    # --------

    print(f"        dino one-slice time (s): {end_time_extract_features - start_time_extract_features}")
    # last layer, CHW
    feat = feats[-1].squeeze().detach().cpu().numpy()
    # if CHW lost C, add axis
    if len(feat.shape) == 2:
        feat = feat[np.newaxis, :, :]  # [1, H, W]
    return feat

# --- A/B file pairing ---
def find_subject_pairs(input_dir):
    """
    List ``<stem>{a|b}`` NIfTIs; keep pairs with both a and b.
    
    Args:
        input_dir: folder path
    
    Returns:
        subject_pairs: {(base_name): {'a': path, 'b': path}}
    """
    input_path = Path(input_dir)
    nifti_files = list(input_path.glob('*.nii.gz')) + list(input_path.glob('*.nii'))
    # nifti_files = natsorted(nifti_files)[:2]
    nifti_files = natsorted(nifti_files)
    
    subject_pairs = {}
    
    for nifti_file in nifti_files:
        filename = nifti_file.stem
        if filename.endswith('.nii'):
            filename = filename[:-4]
        
        # ``(.+)([ab])$`` on stem
        match = re.match(r'(.+?)([ab])$', filename)
        if match:
            base_name = match.group(1)
            suffix = match.group(2)
            
            if base_name not in subject_pairs:
                subject_pairs[base_name] = {}
            
            subject_pairs[base_name][suffix] = nifti_file
    
    # need both
    valid_pairs = {k: v for k, v in subject_pairs.items() if 'a' in v and 'b' in v}
    
    print(f"  pairs found: {len(valid_pairs)}")
    return valid_pairs

def extract_slices_along_axis(volume, axis):
    """
    2D slices without reorientation
    
    Args:
        volume: 3D numpy array
        axis: 0|1|2
    
    Returns:
        list of 2D arrays
    """
    num_slices = volume.shape[axis]
    slices = []
    
    for slice_idx in range(num_slices):
        if axis == 0:
            slice_data = volume[slice_idx, :, :]
        elif axis == 1:
            slice_data = volume[:, slice_idx, :]
        else:  # axis == 2
            slice_data = volume[:, :, slice_idx]
        
        slices.append(slice_data)
    
    return slices

# --- PCA on patch stacks ---
def apply_pca_to_features(features_list, k_components):
    """
    PCA across flattened patches, then reshape
    
    Args:
        list of [C,H,W] tensors
        k_components: output channels
    
    Returns:
        list of [k,H,W]
        fitted ``sklearn`` PCA
    """
    if not features_list:
        return [], None
    
    # first tensor shape
    first_feat = features_list[0]
    if len(first_feat.shape) == 3:
        # [C,H,W] CHW
        C, H, W = first_feat.shape
        M = H  # assume square H=W
        assert H == W, f"non-square feature map: {H} != {W}"
    else:
        raise ValueError(f"bad feature map rank/shape: {first_feat.shape}")
    
    print(f"    feat: {first_feat.shape}, C={C}, M={M}")
    
    # vstack all patches
    all_features_flat = []
    for feat in features_list:
        # [C, M, M] -> [M*M, C]
        feat_flat = feat.reshape(C, -1).T
        all_features_flat.append(feat_flat)
    
    combined_features = np.vstack(all_features_flat)  # [N*M*M, C]
    
    # sklearn PCA
    pca = PCA(n_components=k_components)
    pca_result = pca.fit_transform(combined_features)  # [N*M*M, k]
    
    print(f"    PCA: {combined_features.shape} -> {pca_result.shape}")
    print(f"    explained var (first 3): {pca.explained_variance_ratio_[:3]}")
    
    # reshape to [k,M,M] per slice
    pca_features = []
    start_idx = 0
    for feat in features_list:
        num_patches = M * M
        end_idx = start_idx + num_patches
        
        pca_feat_flat = pca_result[start_idx:end_idx]  # [M*M, k]
        pca_feat = pca_feat_flat.reshape(M, M, k_components)  # [M, M, k]
        pca_feat = np.transpose(pca_feat, (2, 0, 1))  # [k, M, M]
        
        pca_features.append(pca_feat)
        start_idx = end_idx
    
    return pca_features, pca

# --- per-channel 2D zoom, stack 4D ---
def upsample_feature(feature, target_size, interpolation='linear'):
    """
    Zoom each PCA channel to ``target_size``×``target_size``
    
    Args:
        feature: [k,M,M]
        target_size: N (square)
        interpolation: interpolation
    
    Returns:
        upsampled: [k,N,N]
    """
    k, M, M_feat = feature.shape
    assert M == M_feat, f"non-square: {M} != {M_feat}"
    
    # float32 for ndimage
    if feature.dtype == np.float16:
        feature = feature.astype(np.float32)
    
    # zoom factors
    zoom_factor = target_size / M
    
    # zoom order
    if interpolation == 'linear':
        order = 1
    elif interpolation == 'cubic':
        order = 3
    elif interpolation == 'nearest':
        order = 0
    else:
        order = 1
    
    # per-channel
    upsampled_channels = []
    for c in range(k):
        channel_2d = feature[c, :, :]  # [M, M]
        upsampled_channel = ndimage.zoom(channel_2d, zoom_factor, order=order, mode='constant', cval=0.0)
        upsampled_channels.append(upsampled_channel)
    
    # stack channels
    upsampled = np.stack(upsampled_channels, axis=0)
    
    return upsampled

def stack_slices_to_volume(slices_features, axis, original_shape):
    """
    Stack [k,N,N] slices to [D,H,W,k] (axis-dependent)
    
    Args:
        per-slice [k,N,N] list
        axis: slice index axis (0, 1, or 2)
        3D shape for allocation
    
    Returns:
        4D feature volume
    """
    k = slices_features[0].shape[0]
    N = slices_features[0].shape[1]
    
    num_slices = len(slices_features)
    
    # axis-slice stacking
    if axis == 0:  # x-normal
        # (slices, N, N, k)
        volume_4d = np.zeros((num_slices, N, N, k), dtype=np.float32)
        for slice_idx, feat in enumerate(slices_features):
            # feat: [k, N, N] -> [N, N, k]
            volume_4d[slice_idx, :, :, :] = np.transpose(feat, (1, 2, 0))
    
    elif axis == 1:  # y-normal
        # (N, slices, N, k)
        volume_4d = np.zeros((original_shape[0], num_slices, N, k), dtype=np.float32)
        for slice_idx, feat in enumerate(slices_features):
            # feat: [k, N, N] -> [N, N, k]
            volume_4d[:, slice_idx, :, :] = np.transpose(feat, (1, 2, 0))
    
    else:  # z-normal
        # (N, N, slices, k)
        volume_4d = np.zeros((original_shape[0], original_shape[1], num_slices, k), dtype=np.float32)
        for slice_idx, feat in enumerate(slices_features):
            # feat: [k, N, N] -> [N, N, k]
            volume_4d[:, :, slice_idx, :] = np.transpose(feat, (1, 2, 0))
    
    return volume_4d

# --- write 4D NIfTI ---
def save_4d_nifti(volume_4d, output_path, reference_nifti):
    """
    Write 4D D,H,W,k under reference header/affine
    
    Args:
        float array D,H,W,k
        file path
        ref ``Nifti1Image``
    """
    D, H, W, k = volume_4d.shape
    
    # copy ref header
    header = reference_nifti.header.copy()
    affine = reference_nifti.affine.copy()
    
    # set 4D shape
    header.set_data_shape((W, H, D, k))  # NIfTI (x,y,z) + 4th dim
    header.set_xyzt_units(xyz='mm', t='unknown')
    
    # as stacked
    nifti_data = volume_4d
    
    # Nifti1Image
    nifti_img = nib.Nifti1Image(nifti_data, affine, header)
    
    # save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nifti_img, str(output_path))
    
print(f"  saved: {output_path} shape {volume_4d.shape}")


def save_center_rgb_pngs(volume_4d, png_dir, filename_prefix):
    """
    Center-slice RGB (first 3 components) for axes 0,1,2; PNGs for QC.
    files ``{prefix}_axis{0|1|2}_center.png``
    """
    D, H, W, k = volume_4d.shape
    if k < 3:
        return
    # first 3 comp
    comp0 = volume_4d[:, :, :, 0]
    comp1 = volume_4d[:, :, :, 1]
    comp2 = volume_4d[:, :, :, 2]

    os.makedirs(png_dir, exist_ok=True)

    # center indices
    centers = {0: D // 2, 1: H // 2, 2: W // 2}
    for axis in [0, 1, 2]:
        idx = centers[axis]
        if axis == 0:
            r = comp0[idx, :, :]
            g = comp1[idx, :, :]
            b = comp2[idx, :, :]
        elif axis == 1:
            r = comp0[:, idx, :]
            g = comp1[:, idx, :]
            b = comp2[:, idx, :]
        else:
            r = comp0[:, :, idx]
            g = comp1[:, :, idx]
            b = comp2[:, :, idx]

        # per-channel 2D norm
        r8 = normalize_2d_slice(r)
        g8 = normalize_2d_slice(g)
        b8 = normalize_2d_slice(b)
        rgb = np.stack([r8, g8, b8], axis=-1)
        img = Image.fromarray(rgb, mode='RGB')
        out_path = Path(png_dir) / f"{filename_prefix}_axis{axis}_center.png"
        img.save(out_path)

# --- one (a,b) subject, per axis ---
def process_subject_pair(subject_pair, model, config, axes, target_size, 
                        k_components, pca_mode, interpolation, output_dir,
                        slice_resize_mode='none', slice_resize_size=None,
                        norm_mode='volume'):
    """
    Run DINO, PCA, per-channel 2D upsample, 4D stack, and NIfTI/PNG export
    for one A/B subject along each entry in ``axes``.

    **Arguments (same as original)**
    ``subject_pair`` — ``{name: {a: Path, b: Path}}``; ``model`` / ``config`` —
    loaded ViT; ``target_size`` — input side; ``k_components`` — PCA out dim;
    ``pca_mode`` in ``{all_pair, plane_pair}``; ``interpolation`` — 2D zoom
    order for feature maps; ``output_dir`` — base for ``nii/`` and ``png/``;
    optional 2D slice resize args (mostly disabled in the loop, see comments);
    ``norm_mode`` in ``{volume, slice}``.
    """
    base_name = list(subject_pair.keys())[0] if isinstance(subject_pair, dict) else None
    pair_data = subject_pair if base_name is None else subject_pair[base_name]
    
    # NIfTI a,b
    nii_a = nib.load(pair_data['a'])
    nii_b = nib.load(pair_data['b'])
    
    volume_a = nii_a.get_fdata()
    volume_b = nii_b.get_fdata()
    
    original_shape_raw = volume_a.shape  # for upsample target
    original_shape = volume_a.shape
    print(f"\n  subject: {base_name}")
    print(f"  shape: {original_shape}")
    
    # 3D resize not used; optional 2D in-loop (see comments)
    
    # vol vs slice
    if norm_mode == 'volume':
        volume_a = normalize_3d_volume(volume_a)
        volume_b = normalize_3d_volume(volume_b)
    
    # for each axis
    axis_names = {0: 'sagittal', 1: 'coronal', 2: 'axial'}
    
    for axis in axes:
        axis_name = axis_names[axis]
        print(f"\n  axis: {axis_name} (axis={axis})")
        
        # 2D stacks
        slices_a = extract_slices_along_axis(volume_a, axis)
        slices_b = extract_slices_along_axis(volume_b, axis)
        
        print(f"    num slices: a={len(slices_a)} b={len(slices_b)}")
        
        # forward
        print(f"    DINO forward…")

        # start_time_dino = time.time()

        features_a = []
        features_b = []
        

        for slice_data in tqdm(slices_a, desc=f"      {base_name}a", leave=False):
            # to PIL
            if norm_mode == 'slice':
                slice_proc = normalize_2d_slice(slice_data)
            else:
                slice_proc = slice_data
            img = numpy_to_pil_image(slice_proc, convert_rgb=True)

            # # optional PIL resize (usually disabled; see TODO in original: slice resize can desync a vs b)
            # if slice_resize_mode != 'none' and slice_resize_size is not None:
            #     original_size = img.size
            #     img = img.resize(slice_resize_size, Image.Resampling.LANCZOS)
            #     print(f"        2D {slice_resize_mode}: {original_size} -> {slice_resize_size}")
            
            # forward
            # start_time_extract_features = time.time()
            feat = extract_features(model, img, target_size, config.num_layers)
            # end_time_extract_features = time.time()
            # print(f"        one-slice dino time (s): {end_time_extract_features - start_time_extract_features}")
            features_a.append(feat)
        
        for slice_data in tqdm(slices_b, desc=f"      {base_name}b", leave=False):
            if norm_mode == 'slice':
                slice_proc = normalize_2d_slice(slice_data)
            else:
                slice_proc = slice_data
            img = numpy_to_pil_image(slice_proc, convert_rgb=True)

            # here should be comment out but in the former version I forget
            # if slice_resize_mode != 'none' and slice_resize_size is not None:
            #     original_size = img.size
            #     img = img.resize(slice_resize_size, Image.Resampling.LANCZOS)
            #     print(f"        2D {slice_resize_mode}: {original_size} -> {slice_resize_size}")
            feat = extract_features(model, img, target_size, config.num_layers)
            features_b.append(feat)

        # end_time_dino = time.time()
        # print(f"    dino total time (s): {end_time_dino - start_time_dino}")
        
        # first tensor shape
        first_feat = features_a[0]
        print(f"    first feat: {first_feat.shape}")
        
        # PCA
        print(f"    PCA: mode={pca_mode} k={k_components}")
        start_time_pca = time.time()
        if pca_mode == 'all_pair':
            # concat a+b
            all_features = features_a + features_b
            pca_features_all, pca_model = apply_pca_to_features(all_features, k_components)
            pca_features_a = pca_features_all[:len(features_a)]
            pca_features_b = pca_features_all[len(features_a):]
        
        elif pca_mode == 'plane_pair':
            # (same as all_pair in current code)
            all_features = features_a + features_b
            pca_features_all, pca_model = apply_pca_to_features(all_features, k_components)
            pca_features_a = pca_features_all[:len(features_a)]
            pca_features_b = pca_features_all[len(features_a):]
        
        else:
            raise ValueError(f"unknown pca_mode: {pca_mode}")
        end_time_pca = time.time()
        print(f"    PCA time (s): {end_time_pca - start_time_pca}")
        
        N = original_shape_raw[0]
        assert original_shape_raw[0] == original_shape_raw[1] == original_shape_raw[2], \
            f"expected isotropic cube, got {original_shape_raw}"
        
        print(f"    upsample features to {N} and stack to 4D")
        
        upsampled_features_a = []
        for feat in tqdm(pca_features_a, desc=f"      up {base_name}a", leave=False):
            upsampled = upsample_feature(feat, N, interpolation)
            upsampled_features_a.append(upsampled)
        
        if axis == 0:
            target_shape_for_stack = (len(upsampled_features_a), N, N)
        elif axis == 1:
            target_shape_for_stack = (N, len(upsampled_features_a), N)
        else:
            target_shape_for_stack = (N, N, len(upsampled_features_a))
        volume_4d_a = stack_slices_to_volume(upsampled_features_a, axis, target_shape_for_stack)
        
        upsampled_features_b = []
        for feat in tqdm(pca_features_b, desc=f"      up {base_name}b", leave=False):
            upsampled = upsample_feature(feat, N, interpolation)
            upsampled_features_b.append(upsampled)
        
        if axis == 0:
            target_shape_for_stack_b = (len(upsampled_features_b), N, N)
        elif axis == 1:
            target_shape_for_stack_b = (N, len(upsampled_features_b), N)
        else:
            target_shape_for_stack_b = (N, N, len(upsampled_features_b))
        volume_4d_b = stack_slices_to_volume(upsampled_features_b, axis, target_shape_for_stack_b)
        
        # output_dir/nii and output_dir/png
        nii_dir = Path(output_dir) / "nii"
        png_dir = Path(output_dir) / "png"
        nii_dir.mkdir(parents=True, exist_ok=True)
        png_dir.mkdir(parents=True, exist_ok=True)

        # output_path_a = nii_dir / f"{base_name}a_{axis_name}_features.nii.gz"
        # output_path_b = nii_dir / f"{base_name}b_{axis_name}_features.nii.gz"
        output_path_a = nii_dir / f"{base_name}a.nii.gz"
        output_path_b = nii_dir / f"{base_name}b.nii.gz"
        
        save_4d_nifti(volume_4d_a, output_path_a, nii_a)
        save_4d_nifti(volume_4d_b, output_path_b, nii_b)

        save_center_rgb_pngs(volume_4d_a, png_dir, f"{base_name}a_{axis_name}")
        save_center_rgb_pngs(volume_4d_b, png_dir, f"{base_name}b_{axis_name}")

# --- CLI ---
def main():
    """argparse entrypoint (see module docstring)."""
    parser = argparse.ArgumentParser(
        description='DINOv3 feature extraction: NIfTI pairs in, 4D feature NIfTI (+ PNG) out',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example (all three slice axes, ``all_pair`` PCA, relative ``VFM`` paths)::

  python preprocess/extract_dino_ft_nii.py \\
    --input_dir VFM/data/paired_nifti \\
    --output_dir VFM/data/dino_4d_features \\
    --model dinov3_vits16 \\
    --axes 0 1 2 \\
    --pca_mode all_pair \\
    --k_components 16
        """
    )
    
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Directory of paired `*a` / `*b` NIfTIs (see filename regex in code).')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output base (creates `nii/` and `png/` under it).')
    parser.add_argument('--model', type=str, default='dinov3_vits16',
                       choices=list(MODEL_CONFIGS.keys()),
                       help='Which entry in `MODEL_CONFIGS` / torch.hub name.')
    parser.add_argument('--custom_weights', type=str, default=None,
                       help='Override checkpoint path (else use config default).')
    parser.add_argument('--axes', type=int, nargs='+', default=[0, 1, 2],
                       choices=[0, 1, 2],
                       help='Slice normal axes to process: 0=sagittal, 1=coronal, 2=axial; repeat allowed.')
    parser.add_argument('--target_size', type=int, default=64,
                       help='2D size fed to the ViT (divisible by 16, default 64).')
    parser.add_argument('--k_components', type=int, default=16,
                       help='PCA output channel count (default 16).')
    parser.add_argument('--pca_mode', type=str, default='all_pair',
                       choices=['all_pair', 'plane_pair'],
                       help='`all_pair` or `plane_pair` (see note in module doc: paths currently identical).')
    parser.add_argument('--interpolation', type=str, default='nearest',
                       choices=['linear', 'cubic', 'nearest'],
                       help='`scipy.ndimage.zoom` order for upsampling 2D feature maps (default: `nearest` in args; help text in original mentioned linear).')
    
    slice_resize_group = parser.add_argument_group('2D slice resize (optional)')
    slice_resize_group.add_argument('--slice_resize_mode', type=str, default='none',
                                   choices=['none', 'upsample', 'downsample'],
                                   help='Resize each 2D slice in PIL before the network (default `none`).')
    slice_resize_group.add_argument('--slice_resize_width', type=int, default=None,
                                   help='Target width (pixels) for slice resize.')
    slice_resize_group.add_argument('--slice_resize_height', type=int, default=None,
                                   help='Target height (pixels) for slice resize.')

    parser.add_argument('--norm_mode', type=str, default='volume',
                       choices=['volume', 'slice'],
                       help='`volume`: 3D min–max; `slice`: per-slice min–max to uint8 before DINO.')
    
    args = parser.parse_args()
    
    print("DINOv3 NIfTI feature export")
    print("="*60)
    print(f"input_dir:  {args.input_dir}")
    print(f"output_dir: {args.output_dir}")
    print(f"model: {args.model}")
    print(f"axes: {args.axes}")
    print(f"target_size: {args.target_size}")
    print(f"pca_mode: {args.pca_mode}")
    print(f"k_components: {args.k_components}")
    if args.slice_resize_mode != 'none':
        print(f"2D slice resize: {args.slice_resize_mode} -> {(args.slice_resize_width, args.slice_resize_height)}")
    else:
        print("2D slice resize: off")
    print("="*60)
    
    try:
        print(f"\n[1/4] load model")
        model_manager = ModelManager()
        model, config = model_manager.load_model(args.model, args.custom_weights)
        
        print(f"\n[2/4] find A/B pairs")
        subject_pairs = find_subject_pairs(args.input_dir)
        
        if len(subject_pairs) == 0:
            print("no (a,b) pairs found")
            return
        
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n[3/4] per-subject")
        for base_name, pair_data in sorted(subject_pairs.items()):
            try:
                process_subject_pair(
                    {base_name: pair_data},
                    model, config,
                    args.axes,
                    args.target_size,
                    args.k_components,
                    args.pca_mode,
                    args.interpolation,
                    args.output_dir,
                    args.slice_resize_mode,
                    (args.slice_resize_width, args.slice_resize_height) if args.slice_resize_width and args.slice_resize_height else None,
                    args.norm_mode
                )
            except Exception as e:
                print(f"  error on {base_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"\ndone. outputs under: {args.output_dir}")
    
    except Exception as e:
        print(f"fatal: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()

# Example: pair A/B NIfTIs in ``VFM/.../input``; run from repo root (DINOv3 on ``sys.path``):
# python preprocess/extract_dino_ft_nii.py \
#   --input_dir VFM/data/paired_nifti \
#   --output_dir VFM/data/dino_4d_features \
#   --model dinov3_vits16 \
#   --axes 0 1 2 \
#   --pca_mode all_pair \
#   --k_components 16
