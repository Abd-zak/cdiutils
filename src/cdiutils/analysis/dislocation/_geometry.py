import numpy as np

from cdiutils.io.vtk import save_as_vti
from cdiutils.utils import nan_to_zero


def extract_structure(volume, threshold=0.5):
    """Extract points from the volume where the intensity exceeds a threshold."""
    indices = np.argwhere(volume > threshold)
    return indices


def fit_line_3d(points):
    """Fit a 3D line to the given points using SVD."""
    centroid = np.mean(points, axis=0)
    centered_points = points - centroid
    _, _, vh = np.linalg.svd(centered_points)
    direction = -vh[0]
    return centroid, direction


def generate_filled_cylinder(
    shape, centroid, direction, radius, height, step=1
):
    """Generate a 3D volume with a filled cylinder using disks along the fitted line."""
    direction = direction / np.linalg.norm(direction)
    volume = np.zeros(shape)

    # Generate points along the line within the specified height
    t_values = np.arange(-height / 2, height / 2, step)
    for t in t_values:
        # Compute the center of the current disk
        disk_center = centroid + t * direction

        # Create grid coordinates for the volume
        x, y, z = np.indices(shape)

        # Compute the distance of each grid point to the disk center
        distances = np.sqrt(
            (x - disk_center[0]) ** 2
            + (y - disk_center[1]) ** 2
            + (z - disk_center[2]) ** 2
        )

        # Set points within the disk radius to 1
        volume[distances <= radius] = 1

    return volume


def create_circular_mask(
    data_shape,
    centroid,
    direction,
    selected_point_index,
    r,
    dr,
    slice_thickness=2,
):
    """
    Create a cylindrical ring mask around a dislocation line and compute
    associated cylindrical coordinates in the local frame.

    Parameters
    ----------
    data_shape : tuple
        Shape of the 3D volume (nx, ny, nz).
    centroid : np.ndarray
        Reference point on the dislocation line.
    direction : np.ndarray
        Direction vector of the dislocation line (will be normalized).
    selected_point_index : float
        Position along the dislocation line (relative to centroid).
    r : float
        Inner radius of the cylindrical shell.
    dr : float
        Radial thickness of the shell.
    slice_thickness : float, optional
        Half-thickness along the dislocation line (local z-axis).

    Returns
    -------
    circular_mask : np.ndarray
        Binary mask defining the cylindrical shell region.
    polar_angles_masked : np.ndarray
        Polar angle (θ) in the local transverse plane, defined only inside the mask.
    displacement_vectors : np.ndarray
        Absolute voxel coordinates of masked points (shape: [nx, ny, nz, 3]).
    radial_distance_masked : np.ndarray
        Radial distance from the dislocation line (only inside the mask).
    direction : np.ndarray
        Normalized direction vector of the dislocation line.
    """
    selected_point_index = selected_point_index / 2

    direction = direction / np.linalg.norm(direction)
    disk_center = centroid + selected_point_index * direction

    z_axis = direction

    random_vector = (
        np.array([1, 0, 0]) if np.abs(z_axis[0]) < 0.9 else np.array([0, 1, 0])
    )
    x_axis = np.cross(z_axis, random_vector)
    x_axis = x_axis / np.linalg.norm(x_axis)

    y_axis = np.cross(z_axis, x_axis)

    grid_x, grid_y, grid_z = np.meshgrid(
        np.arange(data_shape[0]),
        np.arange(data_shape[1]),
        np.arange(data_shape[2]),
        indexing="ij",
    )
    grid_points = np.vstack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()]).T

    shifted_points = grid_points - disk_center

    local_x = np.dot(shifted_points, x_axis)
    local_y = np.dot(shifted_points, y_axis)
    local_z = np.dot(shifted_points, z_axis)

    radial_distances = np.sqrt(local_x**2 + local_y**2)
    polar_angles = np.arctan2(local_y, local_x)

    circular_mask = np.zeros(data_shape, dtype=np.uint8)
    circular_mask_flat = (
        (radial_distances >= r)
        & (radial_distances <= r + dr)
        & (np.abs(local_z) <= slice_thickness)
    )
    circular_mask.flat[circular_mask_flat] = 1

    polar_angles_masked = np.zeros(data_shape, dtype=np.float32)
    polar_angles_masked.flat[circular_mask_flat] = polar_angles[
        circular_mask_flat
    ]

    displacement_vectors = np.zeros((*data_shape, 3), dtype=np.float32)
    displacement_vectors_flat = grid_points[circular_mask_flat]
    displacement_vectors.reshape(-1, 3)[circular_mask_flat] = (
        displacement_vectors_flat
    )

    radial_distance_masked = np.zeros(data_shape, dtype=np.float32)
    radial_distance_masked.flat[circular_mask_flat] = radial_distances[
        circular_mask_flat
    ]

    return (
        circular_mask,
        polar_angles_masked,
        displacement_vectors,
        radial_distance_masked,
        direction,
    )


def plot_phase_around_dislo(
    amp,
    phase,
    selected_dislocation_data,
    r,
    dr,
    centroid,
    direction,
    slice_thickness=1,
    selected_point_index=0,
    save_vti=False,
    save_path=None,
    voxel_sizes=(1, 1, 1),
):
    """
    Extract and analyze the phase distribution around a dislocation
    using a cylindrical shell sampling.

    Parameters
    ----------
    amp : np.ndarray
        Amplitude volume.
    phase : np.ndarray
        Phase volume.
    selected_dislocation_data : np.ndarray
        Binary or labeled dislocation volume.
    r : float
        Inner radius of the cylindrical shell.
    dr : float
        Shell thickness.
    centroid : np.ndarray
        Dislocation line centroid.
    direction : np.ndarray
        Dislocation line direction.
    slice_thickness : float, optional
        Thickness along the dislocation line.
    selected_point_index : float, optional
        Position along the dislocation line.
    save_vti : bool, optional
        Whether to export results as VTI.
    save_path : str or Path, optional
        Output path for VTI file.
    voxel_sizes : tuple, optional
        Voxel size for VTI export.

    Returns
    -------
    masked_region_phase : np.ndarray
        Phase restricted to the cylindrical shell.
    polar_angles : np.ndarray
        Polar angles in the shell.
    circular_mask : np.ndarray
        Binary shell mask.
    displacement_vectors : np.ndarray
        Coordinates of masked voxels.
    radial_distance_masked : np.ndarray
        Radial distances inside the shell.
    direction : np.ndarray
        Normalized dislocation direction.
    """
    # create the circular mask and polar angle map
    (
        circular_mask,
        polar_angles,
        displacement_vectors,
        radial_distance_masked,
        direction,
    ) = create_circular_mask(
        selected_dislocation_data.shape,
        centroid,
        direction,
        selected_point_index,
        r,
        dr,
        slice_thickness=slice_thickness,
    )
    masked_region_phase = phase * circular_mask

    if save_vti:
        vect_x = displacement_vectors[..., 0]
        vect_y = displacement_vectors[..., 1]
        vect_z = displacement_vectors[..., 2]

        # Save or visualize the circular mask and polar angles#
        dict_to_vti = {
            "density": nan_to_zero(amp),
            "phase": nan_to_zero(phase),
            "dislo": selected_dislocation_data,
            "circular_mask": circular_mask,
            "polar_angles": polar_angles,
            "vect_x": vect_x,
            "vect_y": vect_y,
            "vect_z": vect_z,
            "radial_distance": radial_distance_masked,
        }
        save_as_vti(
            output_path=save_path, voxel_size=tuple(voxel_sizes), **dict_to_vti
        )
    return (
        masked_region_phase,
        polar_angles,
        circular_mask,
        displacement_vectors,
        radial_distance_masked,
        direction,
    )
