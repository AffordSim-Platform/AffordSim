"""Check exact transition/frame alignment and a lossless self-contained archive."""
import importlib.util
from pathlib import Path
import sys

import cv2
import h5py
import numpy as np
from PIL import Image
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
spec = importlib.util.spec_from_file_location('convert_separate_trajectory_to_hdf5',
                                               ROOT / 'tools/convert_separate_trajectory_to_hdf5.py')
converter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(converter)


def make_trial(root):
    root.mkdir()
    data = {
        'source_frame_index': np.array([0, 2, 4], dtype=np.int64),
        'next_frame_index': np.array([1, 3, 5], dtype=np.int64),
        'action_convention': 'legacy_ee_delta_source_gripper',
        'obs_policy_actions': np.array([[.1234567890123, 0., 1.], [.2, .3, -1.], [.4, .5, 1.]], dtype=np.float64),
        'obs_policy_q_pos': np.arange(21, dtype=np.float32).reshape(3, 7),
        'obs_policy_q_vel': -np.arange(21, dtype=np.float32).reshape(3, 7),
        'observed_next_joint_command': np.arange(24, dtype=np.float32).reshape(3, 8),
    }
    trajectory = root / 'trajectory.npy'
    np.save(trajectory, data, allow_pickle=True)
    yy, xx = np.mgrid[:480, :640]
    for camera_number, camera in enumerate(['external_cam_side', 'wrist_cam']):
        directory = root / 'original_images' / camera
        directory.mkdir(parents=True)
        for index in range(6):
            rgb = np.stack(((xx + index * 13) % 256, (yy + camera_number * 67) % 256,
                            (xx // 3 + yy // 5 + index) % 256), axis=-1).astype(np.uint8)
            Image.fromarray(rgb).save(directory / f'step_{index:04d}.webp', format='WEBP', lossless=True)
    return trajectory, data


def test_exact_alignment_lossless_archive_and_no_overwrite(tmp_path):
    trajectory, source = make_trial(tmp_path / 'trial')
    source_bytes = trajectory.read_bytes()
    output = tmp_path / 'episode_0.hdf5'
    result = converter.convert_trajectory(trajectory, output)
    assert result['verification']['valid']
    assert converter.verify_trajectory_hdf5(trajectory, output)['valid']
    with h5py.File(output, 'r') as h5:
        for output_path, source_key in [('action', 'obs_policy_actions'),
                                        ('observations/qpos', 'obs_policy_q_pos'),
                                        ('observations/qvel', 'obs_policy_q_vel'),
                                        ('recordings/trajectory_fields/observed_next_joint_command', 'observed_next_joint_command')]:
            actual = h5[output_path][...]
            assert actual.dtype == source[source_key].dtype
            np.testing.assert_array_equal(actual, source[source_key])
        assert h5['action'].dtype == np.dtype('float64')
        assert h5.attrs['action_convention'] == source['action_convention']
        assert h5['recordings/trajectory_npy'][...].tobytes() == source_bytes
        np.testing.assert_array_equal(h5['recordings/source_frame_index'][...], source['source_frame_index'])
        np.testing.assert_array_equal(h5['recordings/next_frame_index'][...], source['next_frame_index'])
        for output_name, camera in [('front', 'external_cam_side'), ('wrist', 'wrist_cam')]:
            directory = trajectory.parent / 'original_images' / camera
            stored = h5[f'recordings/cameras/{camera}']
            np.testing.assert_array_equal(stored['frame_index'][...], range(6))
            for index in range(6):
                file = directory / f'step_{index:04d}.webp'
                assert stored['encoded_bytes'][index].tobytes() == file.read_bytes()
            for row, index in enumerate(source['source_frame_index']):
                with Image.open(directory / f'step_{index:04d}.webp') as image:
                    rgb = np.array(image.convert('RGB'))
                expected = cv2.resize(rgb[:, 80:560], (256, 256), interpolation=cv2.INTER_AREA)
                np.testing.assert_array_equal(h5[f'observations/images/{output_name}'][row], expected)
    original_output = output.read_bytes()
    with pytest.raises(FileExistsError):
        converter.convert_trajectory(trajectory, output)
    assert output.read_bytes() == original_output
    with h5py.File(output, 'r+') as h5:
        h5['action'][0, 0] = 0
    with pytest.raises(ValueError, match='action values'):
        converter.verify_trajectory_hdf5(trajectory, output)
    with h5py.File(output, 'r+') as h5:
        h5['action'][0, 0] = source['obs_policy_actions'][0, 0]
        frame = h5['recordings/cameras/wrist_cam/encoded_bytes'][3]
        frame[0] ^= 1
        h5['recordings/cameras/wrist_cam/encoded_bytes'][3] = frame
    with pytest.raises(ValueError, match='original frame bytes differ'):
        converter.verify_trajectory_hdf5(trajectory, output)


def test_refuses_missing_state_or_referenced_images_without_writing_output(tmp_path):
    trajectory, data = make_trial(tmp_path / 'trial')
    output = tmp_path / 'bad.hdf5'
    del data['obs_policy_q_pos']
    np.save(trajectory, data)
    with pytest.raises(KeyError, match='obs_policy_q_pos'):
        converter.convert_trajectory(trajectory, output)
    assert not output.exists()
    data['obs_policy_q_pos'] = np.zeros((3, 7), dtype=np.float32)
    np.save(trajectory, data)
    (trajectory.parent / 'original_images/wrist_cam/step_0003.webp').unlink()
    with pytest.raises(FileNotFoundError, match='referenced frame'):
        converter.convert_trajectory(trajectory, output)
    assert not output.exists()
