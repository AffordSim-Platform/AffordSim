"""
Pose math utility functions

Unified quaternion operations, EE orientation normalization and other low-level math tools.
All functions are pure functions depending only on numpy.
"""

from typing import Optional, Union

import numpy as np


_FACING_AXIS_STR_TO_VEC = {
    "+x": np.array([1.0, 0.0, 0.0]),
    "-x": np.array([-1.0, 0.0, 0.0]),
    "+y": np.array([0.0, 1.0, 0.0]),
    "-y": np.array([0.0, -1.0, 0.0]),
    "+z": np.array([0.0, 0.0, 1.0]),
    "-z": np.array([0.0, 0.0, -1.0]),
    "x": np.array([1.0, 0.0, 0.0]),
    "y": np.array([0.0, 1.0, 0.0]),
    "z": np.array([0.0, 0.0, 1.0]),
}


def parse_facing_axis_local(
    value: Union[str, list, tuple, np.ndarray, None],
) -> Optional[np.ndarray]:
    """Parse the facing_axis_local field from containers.yaml into a unit 3D vector.

    Accepted values:
    - None  to  returns None (not declared)
    - "+x" / "-x" / "+y" / "-y" / "+z" / "-z" (case insensitive)
    - Length-3 sequence [x, y, z] (will be normalized)
    """
    if value is None:
        return None
    if isinstance(value, str):
        key = value.strip().lower()
        vec = _FACING_AXIS_STR_TO_VEC.get(key)
        if vec is None:
            raise ValueError(
                f"Invalid facing_axis_local string '{value}'. "
                "Expected one of '+x','-x','+y','-y','+z','-z' or a 3-vector."
            )
        return vec.copy()
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.shape != (3,):
        raise ValueError(
            f"Invalid facing_axis_local shape {arr.shape}; expected length-3 vector."
        )
    norm = float(np.linalg.norm(arr))
    if norm < 1e-8:
        raise ValueError("facing_axis_local vector has zero length.")
    return arr / norm


def _quat_multiply_wxyz(q1, q2):
    """Quaternion multiplication q1 * q2, format [w, x, y, z]"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])


def _quat_wxyz_to_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
    """Convert quaternion [w, x, y, z] to 3x3 rotation matrix."""
    w, x, y, z = [float(v) for v in quat_wxyz]
    n = np.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        return np.eye(3, dtype=float)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _quat_wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    """Convert quaternion from [w, x, y, z] to [x, y, z, w]."""
    return np.array([q[1], q[2], q[3], q[0]], dtype=np.float64)


def _quat_xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    """Convert quaternion from [x, y, z, w] to [w, x, y, z]."""
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)


def normalize_ee_orientation_wxyz(
    quat_wxyz: np.ndarray,
    reference_quat_wxyz: np.ndarray = None,
    mode: str = "pick",
) -> np.ndarray:
    """
    Eliminate the 180-degree ambiguity of the gripper around its own z-axis.

    The two-finger gripper is symmetric; rotating 180 degrees around its z-axis produces
    an equivalent grasp, but causes the wrist to rotate an extra half turn.

    Two modes:
    - pick: choose the candidate with smaller rotation distance to the reference (quaternion dot product)
    - place: choose the candidate whose EE x-axis world z-component has the same sign as the reference,
             ensuring the object does not flip; falls back to dot-product selection if both match

    Args:
        quat_wxyz: EE target orientation quaternion [w, x, y, z]
        reference_quat_wxyz: Current EE orientation quaternion [w, x, y, z]
        mode: "pick" or "place"

    Returns:
        Corrected quaternion [w, x, y, z]
    """
    quat_wxyz = np.array(quat_wxyz, dtype=np.float64)
    if reference_quat_wxyz is None:
        return quat_wxyz

    # Rotate 180 degrees around EE's own z-axis = right-multiply by q_flip
    q_flip = np.array([0.0, 0.0, 0.0, 1.0])  # wxyz: 180 deg around z
    quat_flipped = _quat_multiply_wxyz(quat_wxyz, q_flip)
    ref = np.array(reference_quat_wxyz, dtype=np.float64)

    if mode == "place":
        # Compute the component of EE x-axis in world z direction
        def _x_axis_z(q):
            # rot.apply([1,0,0])[2] == (R @ [1,0,0])[2] == R[2,0]
            return _quat_wxyz_to_matrix(q)[2, 0]

        ref_z = _x_axis_z(ref)
        orig_z = _x_axis_z(quat_wxyz)
        flip_z = _x_axis_z(quat_flipped)

        orig_same = (ref_z * orig_z) >= 0
        flip_same = (ref_z * flip_z) >= 0

        if orig_same and not flip_same:
            return quat_wxyz
        elif flip_same and not orig_same:
            return quat_flipped
        # Both have same sign  to  fall back to dot product to pick the closer one

    # pick mode, or place mode fallback when both have same sign
    dot_orig = abs(np.dot(quat_wxyz, ref))
    dot_flip = abs(np.dot(quat_flipped, ref))
    return quat_flipped if dot_flip > dot_orig else quat_wxyz


def compute_face_target_place_quat(
    target_quat_wxyz: np.ndarray,
    facing_axis_local: np.ndarray,
    grasp_quat_wxyz: Optional[np.ndarray] = None,
    q_base_wxyz: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Rotate the placement pose around world z-axis so the EE horizontal approach direction faces the target interior.

    Two mutually exclusive call forms:

    1. Affordance path (`grasp_quat_wxyz` given, `q_base_wxyz=None`):
       Preserves grasp pitch/roll (angle between EE z-axis and horizontal plane unchanged),
       only replaces yaw so the EE horizontal approach direction points into the container.
       Returns `qz(delta) * grasp_quat`, delta = target_yaw - current_grasp_yaw.

    2. Semi-enclosed path (`q_base_wxyz` given, `grasp_quat_wxyz` may be None):
       `q_base` defines "the base pose when EE faces +x direction (including pitch)",
       returns `qz(target_yaw) * q_base`, rotating the base pose around world z to target_yaw.
       This branch is equivalent to `a = facing_angle + pi; qz * q_base` in the original
       `compute_semi_enclosed_placement_poses`.

    Args:
        target_quat_wxyz: Target container world orientation [w, x, y, z]
        facing_axis_local: Local unit vector pointing to the target's "opening/front" when init_rot=I
        grasp_quat_wxyz: Current grasp orientation [w, x, y, z] (used by affordance path)
        q_base_wxyz: Base orientation [w, x, y, z] (used by semi-enclosed path)

    Returns:
        Placement orientation [w, x, y, z] (not yet normalized for 180-degree ambiguity;
        caller should apply `normalize_ee_orientation_wxyz(..., mode="place")` as needed).
    """
    if (grasp_quat_wxyz is None) == (q_base_wxyz is None):
        raise ValueError(
            "compute_face_target_place_quat: must provide exactly one of "
            "`grasp_quat_wxyz` or `q_base_wxyz`."
        )

    target_quat_wxyz = np.asarray(target_quat_wxyz, dtype=np.float64)
    facing_axis_local = np.asarray(facing_axis_local, dtype=np.float64).reshape(3)

    # Transform facing_axis from target local frame to world frame
    R_tgt = _quat_wxyz_to_matrix(target_quat_wxyz)
    facing_world = R_tgt @ facing_axis_local

    # Project onto world xy plane and extract yaw; pure +/-z facing cannot define a horizontal direction
    facing_xy = facing_world.copy()
    facing_xy[2] = 0.0
    xy_norm = float(np.linalg.norm(facing_xy))
    if xy_norm < 1e-6:
        raise ValueError(
            "compute_face_target_place_quat: facing_axis projected onto world XY "
            "is nearly zero (facing is +/-z). A horizontal target direction is required."
        )
    facing_xy /= xy_norm
    facing_yaw = float(np.arctan2(facing_xy[1], facing_xy[0]))

    # EE needs to face the container interior (opposite direction of facing)
    target_yaw = facing_yaw + np.pi

    if q_base_wxyz is not None:
        # Semi-enclosed path: qz(target_yaw) * q_base
        q_base_wxyz = np.asarray(q_base_wxyz, dtype=np.float64)
        qz_rot = np.array(
            [np.cos(target_yaw / 2.0), 0.0, 0.0, np.sin(target_yaw / 2.0)],
            dtype=np.float64,
        )
        return _quat_multiply_wxyz(qz_rot, q_base_wxyz)

    # Affordance path: we want the horizontal projection of the EE-z axis (approach direction) to point at target_yaw.
    # Rotating around world z does not change the z-component of EE-z, only its horizontal direction.
    # Therefore current_yaw should be the "horizontal angle of the EE-z axis world projection under the grasp pose",
    # not the yaw from ZYX Euler decomposition (which differs when pitch > 90 degrees).
    grasp_quat_wxyz = np.asarray(grasp_quat_wxyz, dtype=np.float64)
    R_grasp = _quat_wxyz_to_matrix(grasp_quat_wxyz)
    ee_z_world = R_grasp @ np.array([0.0, 0.0, 1.0])
    ee_z_xy_norm = float(np.linalg.norm(ee_z_world[:2]))
    if ee_z_xy_norm < 1e-6:
        # EE-z is nearly vertical (top-down grasp); rotating around world z won't make it point in any horizontal direction.
        # Fall back to using the "horizontal projection of the EE-x axis as wrist direction", equivalent to rotating the wrist.
        ee_x_world = R_grasp @ np.array([1.0, 0.0, 0.0])
        ee_x_xy_norm = float(np.linalg.norm(ee_x_world[:2]))
        if ee_x_xy_norm < 1e-6:
            # Edge case: EE-x is also vertical, no rotation applied
            return grasp_quat_wxyz.copy()
        current_yaw = float(np.arctan2(ee_x_world[1], ee_x_world[0]))
    else:
        current_yaw = float(np.arctan2(ee_z_world[1], ee_z_world[0]))

    delta = target_yaw - current_yaw
    qz_rot = np.array(
        [np.cos(delta / 2.0), 0.0, 0.0, np.sin(delta / 2.0)],
        dtype=np.float64,
    )
    return _quat_multiply_wxyz(qz_rot, grasp_quat_wxyz)
