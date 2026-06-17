"""
Utilities for extracting 3D patches from CT volumes.
"""

import numpy as np


def extract_patch(
    volume: np.ndarray,
    voxel_x: int,
    voxel_y: int,
    voxel_z: int,
    patch_size: int = 64,
    fill_value: int = -1000,
) -> np.ndarray:
    """
    Extract a fixed-size cubic 3D patch centered at the specified voxel.

    Always returns shape: (patch_size, patch_size, patch_size).
    """

    half = patch_size // 2

    patch = np.full(
        (patch_size, patch_size, patch_size),
        fill_value,
        dtype=volume.dtype,
    )

    src_z_min = max(voxel_z - half, 0)
    src_z_max = min(voxel_z + half, volume.shape[0])

    src_y_min = max(voxel_y - half, 0)
    src_y_max = min(voxel_y + half, volume.shape[1])

    src_x_min = max(voxel_x - half, 0)
    src_x_max = min(voxel_x + half, volume.shape[2])

    dst_z_min = max(half - voxel_z, 0)
    dst_y_min = max(half - voxel_y, 0)
    dst_x_min = max(half - voxel_x, 0)

    dst_z_max = dst_z_min + max(src_z_max - src_z_min, 0)
    dst_y_max = dst_y_min + max(src_y_max - src_y_min, 0)
    dst_x_max = dst_x_min + max(src_x_max - src_x_min, 0)

    if src_z_max <= src_z_min or src_y_max <= src_y_min or src_x_max <= src_x_min:
        return patch

    patch[
        dst_z_min:dst_z_max,
        dst_y_min:dst_y_max,
        dst_x_min:dst_x_max,
    ] = volume[
        src_z_min:src_z_max,
        src_y_min:src_y_max,
        src_x_min:src_x_max,
    ]

    return patch








# """
# Utilities for extracting 3D patches from CT volumes.
# """

# import numpy as np


# def extract_patch(
#     volume: np.ndarray,
#     voxel_x: int,
#     voxel_y: int,
#     voxel_z: int,
#     patch_size: int = 64,
# ) -> np.ndarray:
#     """
#     Extract a cubic 3D patch centered at the specified voxel.

#     If the patch crosses the volume boundary, padding is added.
#     """

#     half = patch_size // 2

#     z_min = voxel_z - half
#     z_max = voxel_z + half

#     y_min = voxel_y - half
#     y_max = voxel_y + half

#     x_min = voxel_x - half
#     x_max = voxel_x + half

#     pad_z_before = max(0, -z_min)
#     pad_y_before = max(0, -y_min)
#     pad_x_before = max(0, -x_min)

#     pad_z_after = max(0, z_max - volume.shape[0])
#     pad_y_after = max(0, y_max - volume.shape[1])
#     pad_x_after = max(0, x_max - volume.shape[2])

#     z_min = max(z_min, 0)
#     y_min = max(y_min, 0)
#     x_min = max(x_min, 0)

#     z_max = min(z_max, volume.shape[0])
#     y_max = min(y_max, volume.shape[1])
#     x_max = min(x_max, volume.shape[2])

#     patch = volume[
#         z_min:z_max,
#         y_min:y_max,
#         x_min:x_max,
#     ]

#     patch = np.pad(
#         patch,
#         pad_width=(
#             (pad_z_before, pad_z_after),
#             (pad_y_before, pad_y_after),
#             (pad_x_before, pad_x_after),
#         ),
#         mode="constant",
#         constant_values=-1000,
#     )

#     return patch
