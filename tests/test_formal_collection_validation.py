"""A collected episode must have aligned, intact data before acceptance."""
import sys
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
from PIL import Image
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from collect_50_trajectories import (classify, finalize_trajectory_hdf5,
                                     cleanup_verified_temporary_npy, validate_final_hdf5,
                                     cleanup_verified_original_frames, update_report)


def formal_data(arm_dof=7):
    return {'obs_policy_actions': np.zeros((2, 7)), 'obs_policy_joint_actions': np.zeros((2, arm_dof + 1)),
            'observed_next_joint_command': np.zeros((2, arm_dof + 1)),
            'obs_policy_arm_ee_pose': np.zeros((2, 7)),
            'obs_policy_q_pos': np.zeros((2, arm_dof)), 'obs_policy_q_vel': np.zeros((2, arm_dof)),
            'source_frame_index': np.array([0, 1]), 'next_frame_index': np.array([1, 2]),
            'action_convention': 'legacy_ee_delta_source_gripper'}


def test_rejects_missing_camera_frame_and_out_of_view(tmp_path):
    path = tmp_path / 'trajectory.npy'
    np.save(path, formal_data())
    for camera in ['external_cam_side', 'wrist_cam']:
        directory = tmp_path / 'original_images' / camera
        directory.mkdir(parents=True)
        for number in range(3):
            (directory / f'step_{number:04d}.webp').write_bytes(b'frame')
    trial = {'success': True, 'directory': str(tmp_path), 'trajectory_path': str(path),
             'collision_flyaway': {'status': 'complete', 'excluded': False, 'events': [],
                                    'observed_substeps': 10, 'thresholds': {'speed': 2}},
             'side_camera_visibility': {'all_objects_in_frame': True, 'checked_control_steps': 3}}
    report = {'completed': True, 'trials': [trial]}
    assert classify(report)[0] == 'data_error'
    assert classify(report, allow_legacy_npy=True)[0] == 'accepted'
    (tmp_path / 'original_images/wrist_cam/step_0002.webp').unlink()
    assert classify(report)[0] == 'data_error'
    trial['side_camera_visibility']['all_objects_in_frame'] = False
    assert classify(report)[0] == 'out_of_view'
    trial['collision_flyaway']['status'] = 'unavailable'
    assert classify(report)[0] == 'collision_unknown'


def make_trial(root, arm_dof=7):
    directory = root / 'task/attempt_0001/success/trial_001'
    directory.mkdir(parents=True)
    path = directory / 'trajectory.npy'
    np.save(path, formal_data(arm_dof))
    for camera in ['external_cam_side', 'wrist_cam']:
        frames = directory / 'original_images' / camera
        frames.mkdir(parents=True)
        for index in range(3):
            Image.new('RGB', (640, 480), (20 + index * 30, 50, 70)).save(
                frames / f'step_{index:04d}.webp', lossless=True)
    return {'directory': str(directory), 'trajectory_path': str(path), 'success': True}


@pytest.mark.parametrize('arm_dof', [6, 7])
def test_finalization_verifies_before_removing_only_current_npy(tmp_path, arm_dof):
    root = tmp_path / 'collection'
    trial = make_trial(root, arm_dof)
    source = Path(trial['trajectory_path'])
    old = source.read_bytes()
    unrelated = tmp_path / 'historical.npy'
    unrelated.write_bytes(b'prior data')
    verification = finalize_trajectory_hdf5(trial, root, allow_legacy_npy=True)
    assert verification['valid'] is True
    assert source.read_bytes() == old  # Conversion alone never deletes it.
    trial['data_integrity'] = {'valid': True, 'hdf5': verification}
    assert trial['hdf5_integrity']['valid'] is True
    assert trial['hdf5_integrity']['steps'] == 2 and trial['hdf5_integrity']['qpos_dim'] == arm_dof
    assert validate_final_hdf5(trial)['valid'] is True
    destination = Path(trial['hdf5_path'])
    with h5py.File(destination, 'r') as data:
        assert data['recordings/trajectory_npy'][...].tobytes() == old
        np.testing.assert_array_equal(data['action'][...], formal_data(arm_dof)['obs_policy_actions'])
        np.testing.assert_array_equal(data['recordings/trajectory_fields/observed_next_joint_command'][...],
                                      formal_data(arm_dof)['observed_next_joint_command'])
    assert cleanup_verified_temporary_npy(trial, root)
    assert not source.exists()
    assert destination.is_file() and unrelated.read_bytes() == b'prior data'
    assert validate_final_hdf5(trial)['valid'] is True  # Safe resume without NPY.
    assert not cleanup_verified_temporary_npy(trial, root)
    assert (destination.parent / 'original_images/external_cam_side/step_0000.webp').is_file()


def test_failed_verification_or_bad_cleanup_never_removes_npy(tmp_path):
    root = tmp_path / 'collection'
    trial = make_trial(root)
    source = Path(trial['trajectory_path'])
    sys.path.insert(0, str(ROOT / 'tools'))
    import convert_separate_trajectory_to_hdf5 as converter
    with patch.object(converter, 'verify_trajectory_hdf5', side_effect=ValueError('bad image')):
        with pytest.raises(ValueError, match='bad image'):
            finalize_trajectory_hdf5(trial, root, allow_legacy_npy=True)
    assert source.is_file()
    assert not (source.parent / 'trajectory.hdf5').exists()
    with pytest.raises(ValueError, match='outside this collection'):
        finalize_trajectory_hdf5(trial, tmp_path / 'other-collection', allow_legacy_npy=True)
    assert source.is_file()


def test_modified_hdf5_prevents_temporary_data_cleanup(tmp_path):
    root = tmp_path / 'collection'
    trial = make_trial(root)
    source = Path(trial['trajectory_path'])
    verification = finalize_trajectory_hdf5(trial, root, allow_legacy_npy=True)
    trial['data_integrity'] = {'valid': True, 'hdf5': verification}
    with h5py.File(trial['hdf5_path'], 'r+') as data:
        data['action'][0, 0] = .125
    with pytest.raises(ValueError, match='without verified HDF5'):
        cleanup_verified_temporary_npy(trial, root)
    assert source.is_file()


def prepared_frame_cleanup(tmp_path):
    root = tmp_path / 'collection'
    trial = make_trial(root)
    check = finalize_trajectory_hdf5(trial, root, allow_legacy_npy=True)
    trial['trial'] = 1
    trial['data_integrity'] = {'valid': True, 'hdf5': check}
    report_path = root / 'task/attempt_0001/report.json'
    report = {'task_id': 'task', 'completed': True, 'trials': [trial]}
    update_report(report_path, report, 'accepted', trial['data_integrity'])
    gif = root / 'review/media/task/attempt_0001/trial_001/trajectory.gif'
    gif.parent.mkdir(parents=True)
    first = Image.new('RGB', (20, 20), (20, 50, 70))
    second = Image.new('RGB', (20, 20), (70, 50, 20))
    first.save(gif, save_all=True, append_images=[second], duration=[5000, 5000], loop=0)
    return root, trial, report_path, report, gif


def test_raw_frame_cleanup_requires_verified_hdf5_report_and_ten_second_gif(tmp_path):
    root, trial, report_path, report, gif = prepared_frame_cleanup(tmp_path)
    raw = Path(trial['directory']) / 'original_images'
    source = Path(trial['temporary_npy_path'])
    unrelated = tmp_path / 'historical.webp'
    unrelated.write_bytes(b'historic')
    result = cleanup_verified_original_frames(trial, report_path, root)
    assert result['status'] == 'cleaned' and result['frames_removed'] == 6
    assert not raw.exists()
    assert Path(trial['hdf5_path']).is_file() and source.is_file() and gif.is_file()
    assert unrelated.read_bytes() == b'historic'
    trial['raw_frame_cleanup'] = result
    update_report(report_path, report, 'accepted', trial['data_integrity'])
    assert cleanup_verified_original_frames(trial, report_path, root) == result


@pytest.mark.parametrize('failure', ['missing', 'unknown', 'different_bytes', 'missing_preview',
                                    'bad_duration', 'hdf5_changed', 'not_persisted'])
def test_frame_cleanup_preflight_never_deletes_on_any_invalid_input(tmp_path, failure):
    root, trial, report_path, report, gif = prepared_frame_cleanup(tmp_path)
    raw = Path(trial['directory']) / 'original_images'
    victim = raw / 'external_cam_side/step_0000.webp'
    if failure == 'missing':
        victim.unlink()
    elif failure == 'unknown':
        (raw / 'wrist_cam/notes.txt').write_text('preserve me')
    elif failure == 'different_bytes':
        victim.write_bytes(b'changed')
    elif failure == 'missing_preview':
        gif.unlink()
    elif failure == 'bad_duration':
        Image.new('RGB', (20, 20)).save(gif, duration=100)
    elif failure == 'hdf5_changed':
        with h5py.File(trial['hdf5_path'], 'r+') as data:
            data['action'][0, 0] = .125
    else:
        report['collection_classification'] = 'not_accepted'
        report_path.write_text(__import__('json').dumps(report))
    before = {p.relative_to(raw): p.read_bytes() for p in raw.rglob('*') if p.is_file()}
    with pytest.raises((OSError, ValueError)):
        cleanup_verified_original_frames(trial, report_path, root)
    after = {p.relative_to(raw): p.read_bytes() for p in raw.rglob('*') if p.is_file()}
    assert before == after
def test_preview_classification_does_not_require_training_data():
    from collect_50_trajectories import classify
    trial = {'success': True,
             'collision_flyaway': {'status': 'complete', 'excluded': False, 'events': [],
                                  'observed_substeps': 12, 'thresholds': {'min_speed_m_s': 2.0}},
             'side_camera_visibility': {'all_objects_in_frame': True, 'checked_control_steps': 3}}
    report = {'completed': True, 'trials': [trial]}
    assert classify(report, preview_only=True) == ('accepted', None)
    assert classify(report)[0] == 'data_error'


def test_preview_resume_refreshes_once_before_and_after_reusing_reports(tmp_path, monkeypatch):
    import json
    import collect_50_trajectories as collection

    output = tmp_path / 'output'
    tasks = [{'task_id': f'task{i:02}', 'robot': 'ur5e'} for i in range(50)]
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'output_format': 'preview', 'tasks': tasks}))
    for task in tasks:
        name = task['task_id']
        attempt = output / name / 'attempt_0001'
        trial_dir = attempt / 'success/trial_001'
        trial_dir.mkdir(parents=True)
        trial = {'trial': 1, 'success': True, 'directory': str(trial_dir),
                 'collision_flyaway': {'status': 'complete', 'excluded': False, 'events': [],
                                       'observed_substeps': 12, 'thresholds': {'min_speed_m_s': 2.0}},
                 'side_camera_visibility': {'all_objects_in_frame': name != 'task00',
                                            'checked_control_steps': 3}}
        if name == 'task00':
            trial['manual_visibility_review'] = {'source': 'user', 'all_objects_in_frame': True,
                                                 'task_id': name, 'attempt': 'attempt_0001', 'trial': 1}
        report = {'completed': True, 'task_id': name, 'config_path': str(attempt / 'task.yaml'),
                  'save_trajectories': False, 'trials': [trial]}
        (attempt / 'report.json').write_text(json.dumps(report))
    refreshes = []
    monkeypatch.setattr(collection, 'review', lambda *args: refreshes.append(args) or True)
    monkeypatch.setattr(collection, 'valid_preview', lambda _: True)
    monkeypatch.setattr(sys, 'argv', ['collector', '--manifest', str(manifest), '--output-dir', str(output),
                                     '--preview-only', '--max-attempts', '1', '--minimum-free-gb', '0'])
    collection.main()
    state = json.loads((output / 'preview_status.json').read_text())
    assert state['completed'] == 50
    assert state['history']['task00'][0]['status'] == 'accepted'
    assert len(refreshes) == 2


@pytest.mark.parametrize('options,expected', [
    ({'preview_only': True, 'reused_report': True, 'initial_reviewed': True, 'gif_valid': True}, False),
    ({'preview_only': True, 'reused_report': False, 'initial_reviewed': True, 'gif_valid': True}, True),
    ({'preview_only': True, 'reused_report': True, 'initial_reviewed': False, 'gif_valid': True}, True),
    ({'preview_only': True, 'reused_report': True, 'initial_reviewed': True, 'gif_valid': False}, True),
    ({'preview_only': False, 'reused_report': True, 'initial_reviewed': True, 'gif_valid': True}, True),
])
def test_review_refresh_is_skipped_only_for_valid_reused_previews(options, expected):
    from collect_50_trajectories import should_refresh_review
    assert should_refresh_review(**options) is expected


def test_preview_cleanup_requires_valid_gif_and_keeps_other_files(tmp_path):
    from PIL import Image
    import collect_50_trajectories as collection

    trial = tmp_path / 'alpha/attempt_0001/success/trial_001'
    camera = trial / 'original_images/external_cam_side'
    camera.mkdir(parents=True)
    frame = camera / 'step_0000.webp'
    Image.new('RGB', (4, 4), 'red').save(frame)
    other = trial / 'trajectory.hdf5'
    other.write_bytes(b'old data')
    gif = tmp_path / 'review/media/alpha/attempt_0001/trial_001/trajectory.gif'
    gif.parent.mkdir(parents=True)
    Image.new('RGB', (4, 4), 'red').save(gif, duration=5000)
    with pytest.raises(ValueError):
        collection.cleanup_preview_frames({'directory': str(trial)}, tmp_path)
    assert frame.exists()
    Image.new('RGB', (4, 4), 'red').save(gif, duration=10_000)
    result = collection.cleanup_preview_frames({'directory': str(trial)}, tmp_path)
    assert result['frames_removed'] == 1 and not frame.exists()
    assert other.read_bytes() == b'old data'
