"""Repair pre-fix formal trajectory action alignment; dry-run unless --apply.

Only pass trusted local NPY files: numpy loads the repository's pickled dict
format. The original RGB images are never modified. Each migrated NPY keeps an
independent, byte-identical sibling backup, and replacement is atomic.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np

CONVENTION = "command_applied_to_reach_next_frame"
BACKUP_SUFFIX = ".pre_transition_migration"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def migrated_data(data: dict) -> tuple[dict, int]:
    """Construct aligned data without mutating the supplied original dict."""
    if not isinstance(data, dict):
        raise ValueError("NPY must contain the recorder's dictionary")
    previous = data.get("transition_action_convention")
    if previous is not None:
        if previous == CONVENTION:
            raise ValueError("Trajectory is already on the corrected convention")
        raise ValueError(f"Unsupported existing action convention: {previous!r}")
    required = ("source_frame_index", "next_frame_index", "obs_policy_actions",
                "obs_policy_joint_actions", "obs_policy_lastactions", "obs_policy_arm_ee_pose")
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"Missing formal trajectory fields: {missing}")
    actions = np.asarray(data["obs_policy_actions"])
    joint = np.asarray(data["obs_policy_joint_actions"])
    source = np.asarray(data["source_frame_index"])
    next_frame = np.asarray(data["next_frame_index"])
    if actions.ndim != 2 or actions.shape[1] < 2 or len(actions) < 2:
        raise ValueError("At least two transition actions are required")
    length = len(actions)
    if joint.ndim != 2 or len(joint) != length or joint.shape[1] < 2:
        raise ValueError("Joint actions do not align with the old transitions")
    if source.ndim != 1 or next_frame.ndim != 1 or len(source) != length or len(next_frame) != length:
        raise ValueError("Frame index arrays do not align with the old transitions")
    if not np.issubdtype(source.dtype, np.integer) or not np.issubdtype(next_frame.dtype, np.integer):
        raise ValueError("Frame indices must be integers")
    if not np.all(source < next_frame) or not np.array_equal(next_frame[:-1], source[1:]):
        raise ValueError("Frame indices do not describe a connected sequence of transitions")
    if not np.isfinite(actions).all() or not np.isfinite(joint).all():
        raise ValueError("Action arrays contain non-finite values")
    if not np.allclose(actions[:, -1], joint[:, -1], atol=1e-7, rtol=0):
        raise ValueError("Old gripper actions do not match the corresponding old joint commands")

    migrated = {}
    for key, value in data.items():
        if key.startswith("obs_") or key in ("source_frame_index", "next_frame_index"):
            value = np.asarray(value)
            if value.ndim == 0 or len(value) != length:
                raise ValueError(f"Unaligned per-transition field {key}: shape {value.shape}")
            migrated[key] = value[:-1].copy()
        else:
            # Do not guess whether an unknown equally long array is time-indexed.
            if isinstance(value, np.ndarray) and value.ndim > 0 and len(value) == length:
                raise ValueError(f"Unknown possibly time-indexed field: {key}")
            migrated[key] = value
    # The final next command was never saved in the old NPY. The remaining
    # command for transition i is old command i+1.
    migrated["obs_policy_joint_actions"] = joint[1:].copy()
    migrated["obs_policy_actions"][:, -1] = joint[1:, -1]
    migrated["obs_policy_lastactions"] = np.zeros_like(migrated["obs_policy_actions"])
    migrated["obs_policy_lastactions"][1:] = migrated["obs_policy_actions"][:-1]
    migrated["transition_action_convention"] = CONVENTION
    return migrated, length


def migrate(path: Path, apply: bool = False) -> dict:
    path = Path(path).resolve()
    if not path.is_file() or path.suffix != ".npy":
        raise FileNotFoundError(f"Not a trajectory NPY file: {path}")
    original_hash = sha256(path)
    original = np.load(path, allow_pickle=True).item()
    if isinstance(original, dict) and original.get("transition_action_convention") == CONVENTION:
        return {"path": str(path), "status": "already_correct", "sha256": original_hash}
    fixed, old_length = migrated_data(original)
    backup = path.with_name(path.name + BACKUP_SUFFIX)
    result = {"path": str(path), "status": "would_migrate", "backup": str(backup),
              "sha256_before": original_hash, "old_transitions": old_length,
              "new_transitions": old_length - 1, "discarded_final_transition": True}
    if not apply:
        return result

    if backup.exists():
        if not backup.is_file() or sha256(backup) != original_hash:
            raise FileExistsError(f"Existing backup differs from the current source: {backup}")
    else:
        # Exclusive creation means an existing backup can never be replaced.
        with path.open("rb") as source_file, backup.open("xb") as backup_file:
            shutil.copyfileobj(source_file, backup_file)
            backup_file.flush()
            os.fsync(backup_file.fileno())
        if sha256(backup) != original_hash:
            raise OSError(f"Backup verification failed: {backup}")

    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", prefix="." + path.name + ".", suffix=".tmp",
                                         dir=path.parent, delete=False) as file:
            temporary = Path(file.name)
            np.save(file, fixed, allow_pickle=True)
            file.flush()
            os.fsync(file.fileno())
        # Refuse an unexpected concurrent modification between backup and swap.
        if sha256(path) != original_hash:
            raise RuntimeError(f"Trajectory changed during migration: {path}")
        os.replace(temporary, path)
        temporary = None
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    result.update(status="migrated", sha256_after=sha256(path))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="Explicit trusted trajectory.npy file paths")
    parser.add_argument("--apply", action="store_true", help="Write verified backups and replace NPY files")
    args = parser.parse_args()
    failed = False
    for path in args.paths:
        try:
            result = migrate(path, apply=args.apply)
        except Exception as exc:
            failed = True
            result = {"path": str(path), "status": "error", "error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(result, ensure_ascii=False), flush=True)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
