"""Load a DA3 background image together with its matching camera calibration.

The image-path mapping in ``background_cameras.json`` contains the calibration
of the actual rendered images. Its ``pose_user`` is already in the canonical,
metric robot frame: do not apply ``M_user2da3`` or ``origin_offset_applied`` again. Rotations
are camera-to-world quaternions in wxyz order, with OpenCV/ROS camera axes.

This module deliberately has no Isaac Sim imports, so exported backgrounds can
be checked before starting a simulator.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


# Four distinct views available in all three bundled scenes: -90, -60, -30, 0 deg.
DEFAULT_VIEW_INDICES = (2, 4, 6, 8)


def _unique_json_object(pairs):
    """Reject duplicate keys instead of silently choosing the last calibration."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def load_da3_background_image(
    image_path: str | Path,
    calibration_path: str | Path | None = None,
    output_size: tuple[int, int] = (640, 480),
) -> dict:
    """Select one image and its calibration by its full path.

    ``calibration_path`` defaults to this checkout's
    ``backgrounds/background_cameras.json``. Relative calibration paths are
    resolved from the repository root. Image paths, including mapping keys,
    are resolved from ``image_path_base`` relative to the JSON's directory.
    The bundled JSON sets this to ``..``, so YAML paths such as
    ``backgrounds/coffee_shop/orbit_renders/coffee_shop_view_06.png`` are
    repository-relative. Absolute image paths are also accepted.

    Matching never falls back to a filename, camera index, or another image.
    Returned camera fields have the same units and format as
    :func:`load_da3_background`.
    """
    project_root = Path(__file__).resolve().parents[5]
    metadata_path = Path(calibration_path or "backgrounds/background_cameras.json").expanduser()
    if not metadata_path.is_absolute():
        metadata_path = project_root / metadata_path
    metadata_path = metadata_path.resolve()
    if not metadata_path.is_file():
        raise FileNotFoundError(f"DA3 calibration mapping not found: {metadata_path}")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"), object_pairs_hook=_unique_json_object)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid DA3 calibration JSON: {metadata_path}") from exc
    if not isinstance(metadata, dict) or metadata.get("schema_version") != 2:
        raise ValueError(f"DA3 calibration mapping requires schema_version 2: {metadata_path}")
    base = metadata.get("image_path_base")
    if not isinstance(base, str) or not base or Path(base).is_absolute():
        raise ValueError(f"image_path_base must be a path relative to the JSON directory: {metadata_path}")
    image_root = (metadata_path.parent / base).resolve()
    cameras = metadata.get("cameras")
    if not isinstance(cameras, dict) or not cameras:
        raise ValueError(f"DA3 calibration 'cameras' must map image paths to calibrations: {metadata_path}")
    resolved = {}
    for key, camera in cameras.items():
        if not isinstance(key, str) or not key or Path(key).is_absolute():
            raise ValueError(f"DA3 camera keys must be relative image paths: {metadata_path}")
        candidate = (image_root / key).resolve()
        if not candidate.is_relative_to(image_root):
            raise ValueError(f"DA3 image path leaves image_path_base: {key}")
        if candidate in resolved:
            raise ValueError(f"Duplicate DA3 image path after normalization: {key}")
        if not isinstance(camera, dict):
            raise ValueError(f"DA3 image calibration must be an object: {key}")
        resolved[candidate] = camera
    if not isinstance(image_path, (str, Path)) or not str(image_path):
        raise ValueError("image_path must specify a background image")
    selected_path = Path(image_path).expanduser()
    if not selected_path.is_absolute():
        selected_path = image_root / selected_path
    selected_path = selected_path.resolve()
    if selected_path not in resolved:
        raise ValueError(f"No DA3 calibration for image path {selected_path} in {metadata_path}")
    camera = resolved[selected_path]
    context = f"DA3 image {selected_path} in {metadata_path}"
    _size(camera.get("image_size"), f"{context}: image_size (width, height)")
    return _validate_calibration(camera, selected_path, output_size, context, camera.get("view_index"))


def _matrix(value, shape: tuple[int, ...], name: str) -> np.ndarray:
    """Read a finite numeric array with an explicit shape."""
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric array with shape {shape}") from exc
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite and have shape {shape}; got {result.shape}")
    return result


def _size(value, name: str) -> tuple[int, int]:
    """Validate a width/height pair without silently truncating floats."""
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"{name} must be a (width, height) pair")
    if any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v <= 0 for v in value):
        raise ValueError(f"{name} must contain two positive integers")
    return int(value[0]), int(value[1])


def _quaternion_matrix(q: np.ndarray) -> np.ndarray:
    """Convert a unit wxyz quaternion to a rotation matrix using NumPy only."""
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def load_da3_background(
    scene_dir: str | Path,
    view_index: int,
    output_size: tuple[int, int] = (640, 480),
) -> dict:
    """Return a background path and matching simulator camera configuration.

    Args:
        scene_dir: Directory containing ``orbit_renders_meta.json`` and renders.
        view_index: The metadata entry's ``index`` field, not its list position.
        output_size: Simulator image size as ``(width, height)``. The intrinsics
            are scaled by the same horizontal/vertical factors as the image.

    Returns:
        A dictionary containing ``image_path`` (absolute path), ``intrinsic_matrix``
        (scaled, row-major list of nine values), ``pos`` (xyz), ``rot`` (wxyz),
        ``convention`` (``ros``), ``source_size``, ``output_size``, ``view_index``,
        ``angle_deg``, and the original ``extrinsics_w2c_user`` matrix.

    Raises:
        FileNotFoundError: The scene, metadata, or selected rendered image is absent.
        ValueError: The requested index or calibration is malformed or inconsistent.
    """
    if isinstance(view_index, bool) or not isinstance(view_index, (int, np.integer)):
        raise ValueError("view_index must be an integer metadata index")
    view_index = int(view_index)
    _size(output_size, "output_size")
    scene_path = Path(scene_dir).expanduser().resolve()
    metadata_path = scene_path / "orbit_renders_meta.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"DA3 orbit metadata not found: {metadata_path}")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid DA3 metadata JSON: {metadata_path}") from exc
    cameras = metadata.get("cameras") if isinstance(metadata, dict) else None
    if not isinstance(cameras, list):
        raise ValueError(f"DA3 metadata must contain a 'cameras' list: {metadata_path}")
    indices = []
    for camera in cameras:
        index = camera.get("index") if isinstance(camera, dict) else None
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValueError(f"Each DA3 camera must have an integer 'index': {metadata_path}")
        indices.append(index)
    if len(indices) != len(set(indices)):
        raise ValueError(f"Duplicate DA3 camera indices in {metadata_path}")
    if view_index not in indices:
        raise ValueError(f"DA3 view {view_index} is absent from {metadata_path}; available indices: {indices}")
    camera = cameras[indices.index(view_index)]
    context = f"DA3 view {view_index} in {metadata_path}"

    render = camera.get("render")
    if not isinstance(render, str) or not render:
        raise ValueError(f"{context}: 'render' must be a relative image path")
    image_path = (scene_path / render).resolve()
    if Path(render).is_absolute() or not image_path.is_relative_to(scene_path):
        raise ValueError(f"{context}: render path must stay inside the scene directory")
    return _validate_calibration(camera, image_path, output_size, context, view_index)


def _validate_calibration(camera, image_path: Path, output_size, context: str, view_index=None) -> dict:
    """Validate exported geometry and resize intrinsics for either metadata format."""
    target_width, target_height = _size(output_size, "output_size")
    if not image_path.is_file():
        raise FileNotFoundError(f"{context}: rendered background image not found: {image_path}")
    with Image.open(image_path) as image:
        source_width, source_height = image.size
        image.verify()
    image_size = camera.get("image_size")
    if image_size is not None:
        expected_size = _size(image_size, f"{context}: image_size (width, height)")
        if expected_size != (source_width, source_height):
            raise ValueError(
                f"{context}: image_size {image_size} does not match image size "
                f"(width={source_width}, height={source_height})"
            )
    render_hw = camera.get("render_hw")
    if render_hw is not None:
        height, width = _size(render_hw, f"{context}: render_hw (height, width)")
        if (width, height) != (source_width, source_height):
            raise ValueError(
                f"{context}: render_hw {render_hw} does not match image size "
                f"(height={source_height}, width={source_width})"
            )

    intrinsic = _matrix(camera.get("intrinsics"), (3, 3), f"{context}: intrinsics")
    if intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0 or not np.allclose(intrinsic[2], [0, 0, 1]):
        raise ValueError(f"{context}: invalid pinhole intrinsics")
    extrinsic = _matrix(camera.get("extrinsics_w2c_user"), (4, 4), f"{context}: extrinsics_w2c_user")
    rotation = extrinsic[:3, :3]
    if (
        not np.allclose(extrinsic[3], [0, 0, 0, 1], atol=1e-6, rtol=0)
        or not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-5, rtol=0)
        or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5, rtol=0)
    ):
        raise ValueError(f"{context}: extrinsics_w2c_user must be a rigid world-to-camera transform")
    pose = camera.get("pose_user")
    if not isinstance(pose, dict):
        raise ValueError(f"{context}: pose_user must contain 'pos' and 'rot_wxyz'")
    position = _matrix(pose.get("pos"), (3,), f"{context}: pose_user.pos")
    quaternion = _matrix(pose.get("rot_wxyz"), (4,), f"{context}: pose_user.rot_wxyz")
    if not np.isclose(np.linalg.norm(quaternion), 1.0, atol=1e-5, rtol=0):
        raise ValueError(f"{context}: pose_user.rot_wxyz must be a unit quaternion")
    quaternion = quaternion / np.linalg.norm(quaternion)
    if (
        not np.allclose(position, -rotation.T @ extrinsic[:3, 3], atol=1e-5, rtol=0)
        or not np.allclose(_quaternion_matrix(quaternion), rotation.T, atol=1e-5, rtol=0)
    ):
        raise ValueError(f"{context}: pose_user is inconsistent with extrinsics_w2c_user")
    try:
        angle_deg = float(camera["angle_deg"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{context}: angle_deg must be a finite number") from exc
    if not np.isfinite(angle_deg):
        raise ValueError(f"{context}: angle_deg must be a finite number")

    scaled_intrinsic = intrinsic.copy()
    scaled_intrinsic[0] *= target_width / source_width
    scaled_intrinsic[1] *= target_height / source_height
    return {
        "image_path": str(image_path),
        "intrinsic_matrix": scaled_intrinsic.reshape(-1).tolist(),
        "pos": tuple(float(v) for v in position),
        "rot": tuple(float(v) for v in quaternion),
        "convention": "ros",
        "source_size": (source_width, source_height),
        "output_size": (target_width, target_height),
        "view_index": view_index,
        "angle_deg": angle_deg,
        "extrinsics_w2c_user": extrinsic.tolist(),
    }


def load_da3_background_views(
    scene_dir: str | Path,
    view_indices: Iterable[int] = DEFAULT_VIEW_INDICES,
    output_size: tuple[int, int] = (640, 480),
) -> list[dict]:
    """Load the four bundled orbit views, or an explicitly chosen index sequence."""
    return [load_da3_background(scene_dir, index, output_size) for index in view_indices]
