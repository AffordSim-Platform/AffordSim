"""Formal collection retains hold steps and maps states to lossless frames."""
import importlib.util
from pathlib import Path

import numpy as np


spec = importlib.util.spec_from_file_location(
    "episode_buffer_formal_test",
    Path(__file__).resolve().parents[1] / "source/isaaclab_tasks/isaaclab_tasks/real2sim/utils/EpisodeBuffer.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_formal_recording_preserves_stationary_steps_and_frame_alignment(tmp_path):
    recorder = module.EpisodeDataRecorder(
        save_dir=tmp_path, save_npy=True, save_camera_images=False,
        save_original_images=True, sim_camera_resolution=(80, 60), image_noise_std=0,
    )
    recorder.preserve_idle_steps = True
    recorder.separate_image_observations = True
    recorder.start_episode(1, {})
    positions = [0., 0., 0., .1, .1, .1]
    for index, x in enumerate(positions):
        obs = {"policy": {
            "arm_ee_pose": np.array([[x, 0., 0., 1., 0., 0., 0.]], dtype=np.float32),
            "external_cam_side": np.full((1, 3 * 80 * 60), index / 10., dtype=np.float32),
        }}
        command = np.zeros((1, 8), dtype=np.float32)
        command[0, -1] = -1. if index < 2 else 1.
        recorder.add_step(obs, command, False, False)
    paths = recorder.save_episode(success=True)
    assert len(paths) == 1
    data = np.load(paths[0], allow_pickle=True).item()
    assert len(data["obs_policy_actions"]) == 5
    assert np.count_nonzero(np.linalg.norm(data["obs_policy_actions"][:, :6], axis=1) == 0) == 4
    np.testing.assert_array_equal(data["source_frame_index"], np.arange(5))
    np.testing.assert_array_equal(data["next_frame_index"], np.arange(1, 6))
    np.testing.assert_array_equal(data["obs_policy_joint_actions"][:, -1], [-1., -1., 1., 1., 1.])
    np.testing.assert_array_equal(data["obs_policy_actions"][:, -1], [-1., -1., 1., 1., 1.])
    np.testing.assert_array_equal(data["observed_next_joint_command"][:, -1], [-1., 1., 1., 1., 1.])
    assert data["action_convention"] == 'legacy_ee_delta_source_gripper'
    assert "obs_policy_external_cam_side" not in data
    assert len(list((tmp_path / "original_images/external_cam_side/env0_ep1").glob("*.png"))) == 6
