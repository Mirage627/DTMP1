"""Shape and numeric checks shared by rollout wrappers and preflight."""
import numpy as np


def validate_point_cloud(raw_obs, key, expected_shape):
    if key not in raw_obs:
        raise ValueError(
            f"Live environment is missing {key!r}; observed keys: {sorted(raw_obs)}. "
            "Install the same XYZRGB observation generator used to create the HDF5. "
            "Registering a point_cloud modality alone does not generate observations.")
    cloud = np.asarray(raw_obs[key])
    if cloud.shape != tuple(expected_shape) or not np.issubdtype(cloud.dtype, np.number):
        raise ValueError(f"{key}: expected {tuple(expected_shape)}, got {cloud.shape}, {cloud.dtype}")
    if not np.isfinite(cloud).all():
        raise ValueError(f"{key}: live point cloud contains non-finite values")
    if cloud.ndim != 2 or cloud.shape[1] not in (3, 6):
        raise ValueError(f"{key}: expected an XYZ or XYZRGB point cloud")
    # Coordinate frames, padding and color scale belong to the dataset's
    # generator contract. Do not guess them from one frame's extrema.
    return cloud
