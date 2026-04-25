"""
structure_tensor.py
=====================

For each ``*.nii.gz`` in ``input_folder``, compute a per-voxel max eigenvalue
of the 3D structure-tensor of the volume (scikit-image) and save
``StructureTensor_<name>.nii.gz`` to ``output_folder`` with the same affine.

**Parameters in script body**

- ``input_folder`` / ``output_folder``: set below before running (batch script
  style, no ``argparse``). Use paths under a root label ``VFM/`` (see project
  conventions).

**Dependencies**

- nibabel, numpy, scipy.ndimage (imported in duplicate blocks as in the original
  file), scikit-image ``structure_tensor`` / ``structure_tensor_eigenvalues``.

Note: duplicate import blocks are preserved to match the original file exactly.
"""
import os
import numpy as np
import nibabel as nib
import scipy.ndimage as ndi
from skimage.feature import structure_tensor, structure_tensor_eigenvalues

import os
import numpy as np
import nibabel as nib
import scipy.ndimage as ndi
from skimage.feature import structure_tensor, structure_tensor_eigenvalues

# I/O directory layout (set before running; use ``VFM/...`` in public configs)
# Set input and output folder paths
input_folder = 'VFM/data/example_3dus'   # folder of ``*.nii.gz`` 3D ultrasound volumes
output_folder = "VFM/data/example_3dus/structure_tensor"
os.makedirs(output_folder, exist_ok=True)

# List all NIfTI files
nii_files = [f for f in os.listdir(input_folder) if f.endswith(".nii.gz")]

# One volume at a time
for nii_file in nii_files:
    file_path = os.path.join(input_folder, nii_file)
    print(f"Processing: {nii_file}")

    # Read 3DUS volume
    nii_img = nib.load(file_path)
    us_data = nii_img.get_fdata()

    # 3D structure tensor (single ``sigma`` as in skimage)
    structure_tensor_matrix = structure_tensor(us_data, sigma=1.0)  # pass one sigma

    # Eigenvalues at each voxel
    eigvals = structure_tensor_eigenvalues(structure_tensor_matrix)  # pass tensor in

    # Use maximum eigenvalue as edge-strength map
    eigvals_max = np.max(eigvals, axis=0)

    # Save NIfTI
    output_path = os.path.join(output_folder, f"StructureTensor_{nii_file}")
    nib.save(nib.Nifti1Image(eigvals_max, nii_img.affine), output_path)

    print(f"Saved Structure Tensor for {nii_file}")

print("Processing complete!")


# Example: after setting ``input_folder`` / ``output_folder`` to your ``VFM/...`` paths, run
#   python structure_tensor.py
