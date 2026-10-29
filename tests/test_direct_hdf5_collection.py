"""Direct HDF5 recording preserves legacy actions, images and held steps."""
import importlib.util
from pathlib import Path
import sys

import cv2
import h5py
import numpy as np
import pytest


spec = importlib.util.spec_from_file_location(
    'episode_buffer_hdf5_test',
    Path(__file__).resolve().parents[1] / 'source/isaaclab_tasks/isaaclab_tasks/real2sim/utils/EpisodeBuffer.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_direct_hdf5_records_aligned_actions_and_observations(tmp_path):
    recorder = module.EpisodeDataRecorder(save_dir=tmp_path, save_npy=False, save_hdf5=True,
        save_camera_images=False, sim_camera_resolution=(8, 6), image_noise_std=0)
    recorder.preserve_idle_steps = True
    recorder._formal_arm_joint_names = ['j1', 'j2']
    recorder.start_episode(1, {})
    for i, x in enumerate([0., 0., .1, .1]):
        image = np.zeros((1, 3, 6, 8), dtype=np.float32)
        image[:, :, :, 2:6] = i / 4
        obs = {'policy': {
            'arm_ee_pose': np.array([[x, 0, 0, 1, 0, 0, 0]], dtype=np.float32),
            'q_pos': np.array([[i, i + 1]], dtype=np.float32),
            'q_vel': np.zeros((1, 2), dtype=np.float32),
            'external_cam_side': image.reshape(1, -1), 'wrist_cam': image.reshape(1, -1)}}
        action = np.array([[0, 0, -1 if i < 2 else 1]], dtype=np.float32)
        recorder.add_step(obs, action, False, False)
    [path] = recorder.save_episode(True)
    assert path.suffix == '.hdf5'
    assert not list(tmp_path.glob('*.npy'))
    with h5py.File(path, 'r') as data:
        np.testing.assert_array_equal(data['action'][:, -1], [-1, -1, 1])
        np.testing.assert_array_equal(data['observations/qpos'][:, 0], [0, 1, 2])
        np.testing.assert_array_equal(data['recordings/next_frame_index'][:], [1, 2, 3])
        assert data.attrs['action_convention'] == 'legacy_ee_delta_source_gripper'
        original = np.zeros((6, 8, 3), dtype=np.uint8)
        original[:, 2:6] = 63
        expected = cv2.resize(original[:, 1:7], (256, 256), interpolation=cv2.INTER_AREA)
        np.testing.assert_array_equal(data['observations/images/front'][1], expected)


def test_no_remove_idle_statistics_include_held_states(tmp_path):
    spec = importlib.util.spec_from_file_location('robo_convert', Path(__file__).resolve().parents[1] /
                                                   'tools/convert_hdf5_for_robomimic.py')
    converter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(converter)
    path = tmp_path / 'source.hdf5'
    with h5py.File(path, 'w') as f:
        group = f.create_group('traj_0')
        group.create_dataset('action', data=[[1., 0, 0, 0, 0, 0, -1], [0., 0, 0, 0, 0, 0, 1]])
        group.create_dataset('observations/qpos', data=[[1.], [7.]])
        low, high = converter.compute_minmax(f, ['traj_0'], 'observations/qpos', remove_idle=False)
        np.testing.assert_array_equal(low, [1.])
        np.testing.assert_array_equal(high, [7.])
        low, high = converter.compute_minmax(f, ['traj_0'], 'observations/qpos', remove_idle=True)
        np.testing.assert_array_equal(high, [1.])


def test_direct_recording_runs_through_benchmark_staging_and_validation(tmp_path):
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / 'scripts'))
    from run_benchmark_50 import configure_formal_recorder, stage_structured_trajectory
    from collect_50_trajectories import classify, finalize_trajectory_hdf5, validate_final_hdf5

    recording = tmp_path / 'recording'
    recorder = module.EpisodeDataRecorder(save_dir=recording, save_hdf5=True, save_npy=False,
        save_camera_images=False, sim_camera_resolution=(8, 6), image_noise_std=0)
    recorder._formal_arm_joint_names = ['j1', 'j2']
    configure_formal_recorder(recorder, True)
    assert recorder.preserve_idle_steps and not recorder.separate_image_observations
    recorder.start_episode(1, {})
    for i in range(4):
        image = np.zeros((1, 3, 6, 8), dtype=np.float32)
        image[:, :, :, 2:6] = i / 4
        obs = {'policy': {'arm_ee_pose': np.array([[i / 10, 0, 0, 1, 0, 0, 0]], dtype=np.float32),
            'q_pos': np.array([[i, i + 1]], dtype=np.float32), 'q_vel': np.zeros((1, 2), dtype=np.float32),
            'external_cam_side': image.reshape(1, -1), 'wrist_cam': image.reshape(1, -1)}}
        recorder.add_step(obs, np.array([[i, i + 1, -1 if i < 2 else 1]], dtype=np.float32), False, False)
    [source] = recorder.save_episode(True)
    original_bytes = source.read_bytes()
    directory = tmp_path / 'collection' / 'task' / 'attempt_0001' / 'success' / 'trial_001'
    with pytest.raises(ValueError, match='joint names'):
        stage_structured_trajectory(recording, directory, 1, 0, 1, ['j2', 'j1'])
    assert source.read_bytes() == original_bytes
    result = Path(stage_structured_trajectory(recording, directory, 1, 0, 1, ['j1', 'j2']))
    assert result.suffix == '.hdf5' and result.read_bytes() == original_bytes
    assert not source.exists() and not list(tmp_path.rglob('*.npy'))
    source.write_bytes(original_bytes)
    with pytest.raises(FileExistsError):
        stage_structured_trajectory(recording, directory, 1, 0, 1, ['j1', 'j2'])
    assert source.read_bytes() == original_bytes and result.read_bytes() == original_bytes
    with h5py.File(result, 'r') as data:
        np.testing.assert_array_equal(data['recordings/source_frame_index'][:], [0, 1, 2])
        np.testing.assert_array_equal(data['recordings/next_frame_index'][:], [1, 2, 3])
        np.testing.assert_array_equal(data['action'][:, -1], [-1, -1, 1])
        for i in range(3):
            original = np.zeros((6, 8, 3), dtype=np.uint8)
            original[:, 2:6] = int(i / 4 * 255)
            expected = cv2.resize(original[:, 1:7], (256, 256), interpolation=cv2.INTER_AREA)
            for camera in ('front', 'wrist'):
                np.testing.assert_array_equal(data[f'observations/images/{camera}'][i], expected)
    trial = {'success': True, 'directory': str(directory), 'trajectory_path': str(result),
        'collision_flyaway': {'status': 'complete', 'excluded': False, 'events': [],
                              'observed_substeps': 10, 'thresholds': {'speed': 2}},
        'side_camera_visibility': {'all_objects_in_frame': True, 'checked_control_steps': 4}}
    report = {'completed': True, 'trials': [trial]}
    status, check = classify(report)
    assert status == 'accepted' and check['storage_format'] == 'direct_hdf5'
    finalize_trajectory_hdf5(trial, tmp_path / 'collection')
    trial['data_integrity'] = check
    assert validate_final_hdf5(trial)['valid']
    with h5py.File(result, 'r+') as data:
        data['recordings/next_frame_index'][1] = 99
    assert not validate_final_hdf5(trial)['valid']
    assert classify({'completed': True, 'trials': [dict(trial, hdf5_path=None)]})[0] == 'data_error'
