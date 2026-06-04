import numpy as np

## Compute the theoretical phase due to a dislocation.


def dislo_phase_model(
    theta,
    t,
    G,
    b,
    x_ref,
    nu=0.3,
    r=None,
    radial=False,
    print_debug=False,
    print_debug_u=False,
):
    """
    Compute the theoretical BCDI phase induced by a mixed dislocation.

    The model builds a local dislocation frame where z is aligned with the
    dislocation line direction, x is aligned with the edge component of the
    Burgers vector, and y completes the orthonormal frame. The angular coordinate
    is anchored to a fixed experimental in-plane reference direction x_ref.

    Parameters
    ----------
    theta : np.ndarray or float
        Polar angle values in radians.
    t : array-like, shape (3,)
        Dislocation line direction.
    G : array-like, shape (3,)
        Reciprocal-space vector used for BCDI phase projection.
    b : array-like, shape (3,)
        Burgers vector.
    x_ref : array-like, shape (3,)
        Experimental in-plane reference direction used to anchor theta.
    nu : float, optional
        Poisson ratio.
    r : np.ndarray or float, optional
        Radial distance from the dislocation line. Required when radial=True.
    radial : bool, optional
        If True, include the logarithmic radial term in the edge displacement.
    print_debug : bool, optional
        If True, print frame and projection diagnostics.
    print_debug_u : bool, optional
        If True, print displacement and final phase values.

    Returns
    -------
    u_final : np.ndarray
        Theoretical phase projected along G.
    """

    theta = np.asarray(theta, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    G = np.asarray(G, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    x_ref = np.asarray(x_ref, dtype=np.float64).reshape(-1)

    if r is not None:
        r = np.asarray(r, dtype=np.float64)

    # Fixed line direction
    z_hat = normalize_vector(t)

    # Candidate-dependent edge direction
    b_perp = project_vector(b, z_hat)
    b_perp_norm = np.linalg.norm(b_perp)

    if b_perp_norm < 1e-12:
        # pure screw fallback
        x_hat = normalize_vector(project_vector(x_ref, z_hat))
    else:
        x_hat = b_perp / b_perp_norm

    y_hat = normalize_vector(np.cross(z_hat, x_hat))

    # Experimental anchoring:
    # shift theta so that the candidate x-axis is compared to the same x_ref
    x_ref_proj = normalize_vector(project_vector(x_ref, z_hat))
    theta_shift_deg = signed_angle_3d(x_hat, x_ref_proj, z_hat)
    theta_shift = np.deg2rad(theta_shift_deg)

    theta_used = theta - theta_shift

    # Signed components in candidate frame
    Gx = np.dot(G, x_hat)
    Gy = np.dot(G, y_hat)
    Gz = np.dot(G, z_hat)

    bx = np.dot(b, x_hat)
    bz = np.dot(b, z_hat)

    if print_debug:
        print("=== Candidate anchored frame ===")
        print(f"x_hat: {x_hat}")
        print(f"y_hat: {y_hat}")
        print(f"z_hat: {z_hat}")
        print(f"theta_shift (deg): {theta_shift_deg}")
        print(f"Gx, Gy, Gz = {Gx}, {Gy}, {Gz}")
        print(f"bx, bz     = {bx}, {bz}")
        print("x_ref_proj:", x_ref_proj)
        print("x_hat:", x_hat)

        print(
            "angle x_hat -> x_ref :", signed_angle_3d(x_hat, x_ref_proj, z_hat)
        )
        print(
            "angle x_ref -> x_hat :", signed_angle_3d(x_ref_proj, x_hat, z_hat)
        )
        edge_x_scale = Gx * bx
        edge_y_scale = Gy * bx
        screw_scale = Gz * bz

        print("edge_x_scale:", edge_x_scale)
        print("edge_y_scale:", edge_y_scale)
        print("screw_scale :", screw_scale)
        slope = (edge_x_scale + screw_scale) / (2 * np.pi)
        print("slope:", slope)

    u_x_theo = (bx / (2 * np.pi)) * (
        theta_used + np.sin(2 * theta_used) / (4 * (1 - nu))
    )

    u_z_theo = (bz / (2 * np.pi)) * theta_used

    if radial:
        if r is None:
            raise ValueError("radial=True but r is not provided.")
        if np.any(r <= 0):
            raise ValueError("r must be > 0 for log(r).")

        u_y_theo = -(bx / (8 * np.pi * (1 - nu))) * (
            2 * (1 - 2 * nu) * np.log(r) + np.cos(2 * theta_used)
        )
    else:
        u_y_theo = -(bx / (8 * np.pi * (1 - nu))) * np.cos(2 * theta_used)

    if print_debug_u:
        print(f"u_x_theo: {u_x_theo}")
        print(f"u_y_theo: {u_y_theo}")
        print(f"u_z_theo: {u_z_theo}")

    u_final = Gx * u_x_theo + Gy * u_y_theo + Gz * u_z_theo

    if print_debug_u:
        print(f"Final Phase Shift: {u_final}")

    return u_final


## utils for dislo_phase_model
def dislo_rotation_matrix_real_to_theo(t, b):
    """
    Compute the rotation matrix from the real (laboratory or crystal) frame
    to the dislocation (theoretical) frame.

    The dislocation frame is defined as:
    - ẑ aligned with the dislocation line direction `t`,
    - x̂ aligned with the edge component of the Burgers vector, i.e. the
      component of `b` perpendicular to `t`,
    - ŷ completing a right-handed orthonormal basis (ŷ = ẑ × x̂).

    Parameters
    ----------
    t : array_like, shape (3,)
        Dislocation line direction vector in real space. Must be non-zero.
    b : array_like, shape (3,)
        Burgers vector in real space.

    Returns
    -------
    numpy.ndarray, shape (3, 3)
        Rotation matrix `R` whose rows correspond to the unit vectors
        (x̂, ŷ, ẑ) of the dislocation frame expressed in the real-space
        coordinate system. A vector `v_real` can be transformed to the
        dislocation frame via:
            v_theo = R @ v_real

    Notes
    -----
    - If the Burgers vector is parallel to the dislocation line
      (pure screw dislocation), the perpendicular component vanishes.
      In this case, an arbitrary direction perpendicular to `t` is chosen
      to define x̂.
    - The resulting basis is orthonormal and right-handed.
    - The accuracy of the rotation depends on the numerical stability of
      the normalization and projection operations.

    Examples
    --------
    >>> t = [0, 0, 1]
    >>> b = [1, 0, 0]
    >>> R = dislo_rotation_matrix_real_to_theo(t, b)
    >>> R.shape
    (3, 3)
    """
    # 1) ẑ = t̂ = t / ||t||
    t_hat = normalize_vector(t)  # new z-axis

    # 2) b_perp = b - (b·t̂) t̂  (the component of b perpendicular to t)
    b_perp = project_vector(b, t)
    b_perp_norm = np.linalg.norm(b_perp)

    # 3) x̂ = b_perp / ||b_perp||  (edge direction) unless b_perp=0 => pick any perpendicular
    if b_perp_norm < 1e-10:
        # Choose an arbitrary x-axis perpendicular to t
        temp = np.array([1.0, 0.0, 0.0])
        x_prime = temp - np.dot(temp, t_hat) * t_hat
        x_prime = normalize_vector(x_prime)
    else:
        x_prime = b_perp / b_perp_norm

    # 4) ŷ = ẑ × x̂  (right-hand rule)
    y_prime = normalize_vector(np.cross(t_hat, x_prime))

    # 5) R has rows = [x̂, ŷ, ẑ]
    R = np.array([x_prime, y_prime, t_hat])
    return R


def normalize_vector(v, eps=1e-12):
    """
    Normalize a vector to unit length.

    Parameters
    ----------
    v : array_like
        Input vector. Must have non-zero magnitude.

    Returns
    -------
    numpy.ndarray
        Unit vector in the direction of `v`.

    Raises
    ------
    ValueError
        If the input vector has zero magnitude.

    Notes
    -----
    This function performs an ℓ2 (Euclidean) normalization using
    ``np.linalg.norm``. The direction of the vector is preserved.

    Examples
    --------
    >>> normalize_vector([3, 0, 4])
    array([0.6, 0. , 0.8])
    """
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if not np.isfinite(n) or n < eps:
        raise ValueError(f"Cannot normalize zero or non-finite vector: {v}")
    return v / n


def project_vector(v, t):
    """
    Compute the component of vector `v` perpendicular to vector `t`.

    This function removes the projection of `v` along `t`:
        v_perp = v - (v · t / ||t||²) t

    Parameters
    ----------
    v : array_like
        Input vector to be projected.
    t : array_like
        Reference vector defining the direction to be removed.
        Must be non-zero.

    Returns
    -------
    numpy.ndarray
        Component of `v` perpendicular to `t`.

    Raises
    ------
    ValueError
        If `t` has zero magnitude.

    Notes
    -----
    The function does not normalize the output. If a unit vector is required,
    apply `normalize_vector` to the result.

    Examples
    --------
    >>> project_vector([1, 1, 0], [1, 0, 0])
    array([0., 1., 0.])
    """
    v = np.array(v, dtype=np.float64)  # Ensure `v` is a NumPy array
    t = np.array(t, dtype=np.float64)  # Ensure `t` is a NumPy array

    t_norm = np.linalg.norm(t)
    if not np.isfinite(t_norm) or t_norm < 1e-12:
        raise ValueError(
            f"Cannot project using zero or invalid direction: {t}"
        )

    return v - (np.dot(v, t) / t_norm**2) * t


def signed_angle_3d(u, v, normal, eps=1e-12):
    """
    Compute the signed angle (in degrees) between two 3D vectors `u` and `v`,
    measured around a specified `normal` axis direction.

    The sign of the angle is determined by the direction of the cross product
    of `u` and `v` relative to `normal`.
    - Positive if the rotation from `u` to `v` is counterclockwise around `normal`.
    - Negative if the rotation is clockwise.

    Args:
        u (array-like): First 3D vector (starting vector).
        v (array-like): Second 3D vector (ending vector).
        normal (array-like): 3D vector defining the rotation axis (normal to the rotation plane).

    Returns:
        float: Signed angle in degrees.

    Example:
        >>> u = np.array([1, 0, 0])
        >>> v = np.array([0, 1, 0])
        >>> normal = np.array([0, 0, 1])
        >>> signed_angle_3d(u, v, normal)
        90.0

        >>> signed_angle_3d(v, u, normal)
        -90.0
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    normal = np.asarray(normal, dtype=np.float64)

    normal_norm = np.linalg.norm(normal)
    if not np.isfinite(normal_norm) or normal_norm < eps:
        raise ValueError(f"Normal vector is zero or invalid: {normal}")

    angle = angle_between_vectors(u, v, eps=eps)
    cross = np.cross(u, v)
    s = np.dot(cross, normal)

    if np.isclose(s, 0.0, atol=eps):
        return 0.0 if np.isclose(angle, 0.0, atol=eps) else angle

    return angle * np.sign(s)


def angle_between_vectors(u, v, eps=1e-12):
    """
    Compute the angle between two vectors in Euclidean space.

    The angle is calculated using the dot product formula:
        cos(θ) = (u · v) / (||u|| ||v||)
    and returned in degrees.

    Parameters
    ----------
    u : sequence of float
        First input vector. Must be a non-zero vector.
    v : sequence of float
        Second input vector. Must be a non-zero vector.

    Returns
    -------
    float
        Angle between vectors `u` and `v` in degrees, in the range [0, 180].

    Raises
    ------
    ValueError
        If either vector has zero magnitude.

    Notes
    -----
    The function assumes that `u` and `v` have the same dimensionality.
    Numerical errors may occur if the dot product divided by the product
    of magnitudes is slightly outside the interval [-1, 1].

    Examples
    --------
    >>> angle_between_vectors([1, 0, 0], [0, 1, 0])
    90.0
    >>> angle_between_vectors([1, 0], [1, 0])
    0.0
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)

    magnitude_u = np.linalg.norm(u)
    magnitude_v = np.linalg.norm(v)

    if not np.isfinite(magnitude_u) or not np.isfinite(magnitude_v):
        raise ValueError("Non-finite vector norm in angle_between_vectors.")
    if magnitude_u < eps or magnitude_v < eps:
        raise ValueError(
            f"Cannot compute angle with zero vector: u={u}, v={v}"
        )

    cosang = np.dot(u, v) / (magnitude_u * magnitude_v)
    cosang = np.clip(cosang, -1.0, 1.0)

    return np.degrees(np.arccos(cosang))


def transform_known_vector_to_crystallographic(vx, vy, vz, R):
    """
    Transforms a given vector (vx, vy, vz) from the original frame to the crystallographic basis.

    Args:
        vx: X-component of the vector in the original frame (can be scalar or array)
        vy: Y-component of the vector in the original frame (can be scalar or array)
        vz: Z-component of the vector in the original frame (can be scalar or array)
        R: 3x3 rotation matrix that maps the original frame to the crystallographic basis.

    Returns:
        - Transformed vector components (vx_cryst, vy_cryst, vz_cryst) in the crystallographic basis.
    """
    # Stack vector components into a matrix form
    original_vector = np.array([vx, vy, vz]).reshape(3, -1)

    # Apply the rotation matrix (no translation)
    transformed_vector = R @ original_vector

    # Extract transformed components
    vx_cryst = transformed_vector[0].squeeze()
    vy_cryst = transformed_vector[1].squeeze()
    vz_cryst = transformed_vector[2].squeeze()

    return vx_cryst, vy_cryst, vz_cryst


def normalize_vectors_3d(vx, vy, vz):
    """
    Normalizes a set of vectors given their X, Y, and Z components.

    Args:
        vx: X-component of vectors (array or scalar)
        vy: Y-component of vectors (array or scalar)
        vz: Z-component of vectors (array or scalar)

    Returns:
        - Normalized vector components (vx_norm, vy_norm, vz_norm)
    """
    # Convert to numpy arrays if inputs are scalars
    vx, vy, vz = np.asarray(vx), np.asarray(vy), np.asarray(vz)

    # Compute vector magnitudes
    magnitudes = np.sqrt(vx**2 + vy**2 + vz**2)

    # Avoid division by zero (if magnitude is 0, set to 1 to prevent NaN)
    magnitudes = np.where(magnitudes == 0, 1, magnitudes)

    # Normalize each component
    vx_norm = vx / magnitudes
    vy_norm = vy / magnitudes
    vz_norm = vz / magnitudes

    return vx_norm, vy_norm, vz_norm


def closest_to_zero_in_array(vec):
    vec = np.asarray(vec)  # Ensure it's a NumPy array
    idx = np.argmin(np.abs(vec))  # Index of the value closest to zero
    return vec[idx], idx
