"""Migration must repair temporal alignment while retaining the source bytes."""
import importlib.util
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("migrate_recorded_transitions", ROOT / "scripts/migrate_recorded_transitions.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def old_episode():
    # Diverse and changing gripper commands make the off-by-one observable.
    joint = np.array([[10, -1], [20, 1], [30, 1], [40, -1]], dtype=np.float32)
    actions = np.array([[.1, -.1, -1], [.2, -.2, 1], [.3, -.3, 1], [.4, -.4, -1]])
    return {"source_frame_index": np.array([2, 3, 5, 6]),
            "next_frame_index": np.array([3, 5, 6, 7]),
            "obs_policy_actions": actions,
            "obs_policy_lastactions": np.vstack([np.zeros((1, 3)), actions[:-1]]),
            "obs_policy_joint_actions": joint,
            "obs_policy_arm_ee_pose": np.arange(28, dtype=np.float32).reshape(4, 7),
            "obs_policy_q_pos": np.arange(8, dtype=np.float32).reshape(4, 2),
            "metadata_scalar": "keep"}


def test_dry_run_is_read_only_and_apply_shifts_commands_preserves_backup(tmp_path):
    image = tmp_path / "original_images/external_cam_side/step_0000.webp"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"unchanged original image")
    path = tmp_path / "trajectory.npy"
    old = old_episode()
    np.save(path, old)
    original_bytes = path.read_bytes()
    check = module.migrate(path)
    assert check["status"] == "would_migrate"
    assert check["old_transitions"] == 4 and check["new_transitions"] == 3
    assert path.read_bytes() == original_bytes
    assert not Path(check["backup"]).exists()

    result = module.migrate(path, apply=True)
    assert result["status"] == "migrated"
    assert Path(result["backup"]).read_bytes() == original_bytes
    assert image.read_bytes() == b"unchanged original image"
    data = np.load(path, allow_pickle=True).item()
    assert data["transition_action_convention"] == module.CONVENTION
    np.testing.assert_array_equal(data["source_frame_index"], [2, 3, 5])
    np.testing.assert_array_equal(data["next_frame_index"], [3, 5, 6])
    np.testing.assert_array_equal(data["obs_policy_joint_actions"], old["obs_policy_joint_actions"][1:])
    np.testing.assert_array_equal(data["obs_policy_actions"][:, :2], old["obs_policy_actions"][:-1, :2])
    np.testing.assert_array_equal(data["obs_policy_actions"][:, -1], old["obs_policy_joint_actions"][1:, -1])
    np.testing.assert_array_equal(data["obs_policy_lastactions"][0], [0, 0, 0])
    np.testing.assert_array_equal(data["obs_policy_lastactions"][1:], data["obs_policy_actions"][:-1])
    np.testing.assert_array_equal(data["obs_policy_arm_ee_pose"], old["obs_policy_arm_ee_pose"][:-1])
    np.testing.assert_array_equal(data["obs_policy_q_pos"], old["obs_policy_q_pos"][:-1])
    assert data["metadata_scalar"] == "keep"
    corrected_bytes = path.read_bytes()
    assert module.migrate(path, apply=True)["status"] == "already_correct"
    assert path.read_bytes() == corrected_bytes
    assert Path(result["backup"]).read_bytes() == original_bytes


def test_failed_atomic_replace_keeps_original_and_reusable_backup(tmp_path):
    path = tmp_path / "trajectory.npy"
    np.save(path, old_episode())
    original_bytes = path.read_bytes()
    with patch.object(module.os, "replace", side_effect=OSError("simulated failure")):
        with pytest.raises(OSError, match="simulated failure"):
            module.migrate(path, apply=True)
    assert path.read_bytes() == original_bytes
    assert path.with_name(path.name + module.BACKUP_SUFFIX).read_bytes() == original_bytes
    assert not list(tmp_path.glob("*.tmp"))
    assert module.migrate(path, apply=True)["status"] == "migrated"


def test_existing_conflicting_backup_prevents_overwrite(tmp_path):
    path = tmp_path / "trajectory.npy"
    np.save(path, old_episode())
    original_bytes = path.read_bytes()
    backup = path.with_name(path.name + module.BACKUP_SUFFIX)
    backup.write_bytes(b"existing unrelated backup")
    with pytest.raises(FileExistsError):
        module.migrate(path, apply=True)
    assert path.read_bytes() == original_bytes
    assert backup.read_bytes() == b"existing unrelated backup"


@pytest.mark.parametrize("error_case", ["disconnected_frames", "wrong_gripper", "unknown_array", "already_marked"])
def test_incompatible_files_are_rejected_without_mutation(tmp_path, error_case):
    data = old_episode()
    if error_case == "disconnected_frames":
        data["next_frame_index"][0] = 4
    elif error_case == "wrong_gripper":
        data["obs_policy_actions"][0, -1] = 0
    elif error_case == "unknown_array":
        data["unknown_timed_array"] = np.arange(4)
    else:
        data["transition_action_convention"] = "unsupported"
    path = tmp_path / "trajectory.npy"
    np.save(path, data)
    original_bytes = path.read_bytes()
    with pytest.raises(ValueError):
        module.migrate(path, apply=True)
    assert path.read_bytes() == original_bytes
    assert not path.with_name(path.name + module.BACKUP_SUFFIX).exists()
