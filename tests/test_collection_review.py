"""The review must not mistake failed, excluded, unframed or partial work for success."""
import importlib.util
import json
from pathlib import Path
import sys

import h5py
import numpy as np
from PIL import Image
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
spec = importlib.util.spec_from_file_location('render_collection_review', ROOT / 'scripts/render_collection_review.py')
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)


def collision():
    return {'status': 'complete', 'excluded': False, 'events': [], 'observed_substeps': 12,
            'thresholds': {'min_speed_m_s': 2.0}}


def test_success_requires_valid_collision_visibility_and_verified_hdf5(tmp_path):
    hdf5 = tmp_path / 'episode.hdf5'
    hdf5.write_bytes(b'validated artifact fixture')
    trial = {'success': True, 'collision_flyaway': collision(),
             'side_camera_visibility': {'all_objects_in_frame': True, 'checked_control_steps': 20},
             'hdf5_path': str(hdf5), 'hdf5_integrity': {'valid': True}}
    assert review.classify_trial(trial, require_hdf5=True) == 'success'
    assert review.classify_trial({**trial, 'hdf5_integrity': {'valid': False}}, require_hdf5=True) == 'provisional'
    assert review.classify_trial({**trial, 'hdf5_path': str(tmp_path / 'missing.hdf5')}, require_hdf5=True) == 'provisional'
    assert review.classify_trial({**trial, 'hdf5_path': None}, require_hdf5=False) == 'success'
    assert review.classify_trial({**trial, 'side_camera_visibility': None}) == 'unknown'
    assert review.classify_trial({**trial, 'side_camera_visibility': {'all_objects_in_frame': True, 'checked_control_steps': 0}}) == 'unknown'
    assert review.classify_trial({**trial, 'side_camera_visibility': {'all_objects_in_frame': False, 'checked_control_steps': 20}}) == 'out_of_view'
    assert review.classify_trial({**trial, 'collision_flyaway': None}) == 'unknown'
    assert review.classify_trial({**trial, 'success': False}) == 'failure'
    assert review.classify_trial({**trial, 'data_integrity': {'valid': False}}) == 'error'
    assert review.classify_trial({**trial, 'data_integrity': {'valid': True, 'last_transition_omitted_from_legacy_recording': True}}) == 'provisional'


def test_manual_visibility_review_is_bound_to_exact_preview_attempt():
    from collect_50_trajectories import classify
    trial = {'trial': 1, 'success': True, 'collision_flyaway': collision(),
             'side_camera_visibility': {'all_objects_in_frame': False, 'checked_control_steps': 20},
             'manual_visibility_review': {'source': 'user', 'all_objects_in_frame': True,
                'task_id': 'alpha', 'attempt': 'attempt_0001', 'trial': 1}}
    report = {'completed': True, 'task_id': 'alpha', 'config_path': '/tmp/attempt_0001/task.yaml', 'trials': [trial]}
    def page(t=trial, task='alpha', attempt='attempt_0001', preview=True):
        return review.classify_trial(t, task_id=task, attempt=attempt, preview_only=preview)
    assert page() == 'success'
    assert classify(report, preview_only=True)[0] == 'accepted'
    assert page(task='beta') == 'out_of_view'
    assert page(attempt='attempt_0002') == 'out_of_view'
    assert page(preview=False) == 'out_of_view'
    assert page({**trial, 'trial': 2}) == 'out_of_view'
    assert page({**trial, 'success': False}) == 'failure'
    assert page({**trial, 'collision_flyaway': None}) == 'unknown'
    assert classify({**report, 'task_id': 'beta'}, preview_only=True)[0] == 'out_of_view'
    assert classify({**report, 'config_path': '/tmp/attempt_0002/task.yaml'}, preview_only=True)[0] == 'out_of_view'
    assert classify(report, preview_only=False)[0] == 'out_of_view'


def test_user_acceptance_keeps_automatic_visibility_and_is_preview_specific():
    trial = {'trial': 5, 'success': True, 'collision_flyaway': collision(),
             'side_camera_visibility': {'all_objects_in_frame': False, 'checked_control_steps': 20},
             'manual_preview_review': {'source': 'user', 'decision': 'accepted',
                'task_id': 'can', 'attempt': 'attempt_0004', 'trial': 5}}
    kwargs = dict(task_id='can', attempt='attempt_0004', preview_only=True)
    assert review.classify_trial(trial, **kwargs) == 'success'
    assert trial['side_camera_visibility']['all_objects_in_frame'] is False
    assert review.classify_trial({**trial, 'trial': 4}, **kwargs) == 'out_of_view'
    assert review.classify_trial(trial, **{**kwargs, 'attempt': 'attempt_0005'}) == 'out_of_view'
    assert review.classify_trial(trial, **{**kwargs, 'preview_only': False}) == 'out_of_view'
    assert review.classify_trial({**trial, 'success': False}, **kwargs) == 'failure'
    assert review.classify_trial({**trial, 'collision_flyaway': None}, **kwargs) == 'unknown'


def test_user_allows_partial_view_for_only_the_bound_preview():
    trial = {'trial': 1, 'success': True, 'collision_flyaway': collision(),
             'side_camera_visibility': {'all_objects_in_frame': False, 'checked_control_steps': 20},
             'preview_visibility_exception': {'source': 'user', 'decision': 'allow_partial',
                'task_id': 'pull_cabinet', 'attempt': 'attempt_0001', 'trial': 1}}
    kwargs = dict(task_id='pull_cabinet', attempt='attempt_0001', preview_only=True)
    assert review.classify_trial(trial, **kwargs) == 'success'
    assert trial['side_camera_visibility']['all_objects_in_frame'] is False
    assert review.classify_trial(trial, **{**kwargs, 'task_id': 'other'}) == 'out_of_view'
    assert review.classify_trial(trial, **{**kwargs, 'attempt': 'attempt_0002'}) == 'out_of_view'
    assert review.classify_trial(trial, **{**kwargs, 'preview_only': False}) == 'out_of_view'
    assert review.classify_trial({**trial, 'success': False}, **kwargs) == 'failure'
    assert review.classify_trial({**trial, 'collision_flyaway': None}, **kwargs) == 'unknown'


@pytest.mark.parametrize('count', [1, 3, 87, 160, 251, 999, 1000])
def test_gif_durations_are_positive_and_sum_to_ten_seconds(count):
    durations = review.gif_frame_durations(count)
    assert len(durations) == count
    assert sum(durations) == 10_000
    assert min(durations) >= 10
    assert all(duration % 10 == 0 for duration in durations)
    assert max(durations) - min(durations) <= 10


def test_gif_frame_sampling_preserves_all_frames_when_possible():
    assert review.sample_indices(251, min(251, review.MAX_GIF_FRAMES)) == list(range(251))
    indexes = review.sample_indices(1001, min(1001, review.MAX_GIF_FRAMES))
    assert len(indexes) == 1000
    assert indexes[0] == 0 and indexes[-1] == 1000
    assert len(set(indexes)) == 1000


def test_preview_report_reuses_gif_after_frames_removed_without_linking_hdf5(tmp_path):
    results, output = tmp_path / 'results', tmp_path / 'review'
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'output_format': 'preview', 'tasks': [{'task_id': 'alpha', 'robot': 'ur5e'}]}))
    attempt = results / 'alpha' / 'attempt_0001'
    trial_dir = attempt / 'success/trial_001'
    trial_dir.mkdir(parents=True)
    hdf5 = trial_dir / 'trajectory.hdf5'
    hdf5.write_bytes(b'keep historical artifact')
    trial = {'trial': 1, 'directory': str(trial_dir), 'success': True,
             'collision_flyaway': collision(),
             'side_camera_visibility': {'all_objects_in_frame': True, 'checked_control_steps': 3},
             'hdf5_path': str(hdf5), 'hdf5_integrity': {'valid': True}}
    (attempt / 'report.json').write_text(json.dumps({'completed': True, 'returncode': 0,
                                                      'task_id': 'alpha', 'robot': 'ur5e', 'trials': [trial]}))
    media = output / 'media/alpha/attempt_0001/trial_001'
    media.mkdir(parents=True)
    gif = media / 'trajectory.gif'
    Image.new('RGB', (16, 16), 'red').save(gif, duration=10_000)
    before = gif.read_bytes()
    result = review.make_report(manifest, results, output)
    assert result['counts']['success'] == 1
    assert gif.read_bytes() == before
    assert hdf5.read_bytes() == b'keep historical artifact'
    assert '本轮不保存训练数据' in (output / 'index.html').read_text()
    assert result['tasks'][0]['attempts'][0]['hdf5'] == []


def test_recent_retest_is_displayed_ahead_of_historical_import(tmp_path):
    results, output = tmp_path / 'results', tmp_path / 'review'
    results.mkdir()
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'output_format': 'preview', 'tasks': [{'task_id': 'alpha', 'robot': 'franka_long'}]}))
    trial = {'trial': 1, 'success': True, 'collision_flyaway': collision(),
             'side_camera_visibility': {'all_objects_in_frame': True, 'checked_control_steps': 2}}
    attempt = results / 'alpha' / 'attempt_0002'
    attempt.mkdir(parents=True)
    (attempt / 'report.json').write_text(json.dumps({'completed': True, 'returncode': 0,
        'task_id': 'alpha', 'robot': 'franka_long', 'trials': [{**trial, 'directory': str(attempt)}]}))
    current = output / 'media/alpha/attempt_0002/trial_001/trajectory.gif'
    current.parent.mkdir(parents=True)
    Image.new('RGB', (16, 16), 'red').save(current, duration=10_000)
    old_report = tmp_path / 'old_report.json'
    old_report.write_text(json.dumps({'completed': True, 'task_id': 'alpha', 'robot': 'franka_long', 'trials': [trial]}))
    old_gif = tmp_path / 'old.gif'
    Image.new('RGB', (16, 16), 'blue').save(old_gif, duration=10_000)
    (results / 'imported_previews.json').write_text(json.dumps({'alpha': {'gif': str(old_gif),
        'report': str(old_report), 'robot': 'franka_long', 'trial': 1,
        'source_attempt': 'imported_old', 'source_page': str(tmp_path / 'old.html')}}))
    (results / 'review_minimum_attempts.json').write_text(json.dumps({'alpha': 2}))
    result = review.make_report(manifest, results, output)
    assert result['tasks'][0]['status'] == 'success'
    assert result['tasks'][0]['attempts'][-1]['status'] == 'imported'
    assert 'src="media/alpha/attempt_0002/trial_001/trajectory.gif"' in (output / 'index.html').read_text()

    # A specifically requested fixed-pose recording remains visible if its
    # object is cropped, while retaining the truthful out-of-view status.
    partial = {**trial, 'directory': str(attempt),
               'side_camera_visibility': {'all_objects_in_frame': False, 'checked_control_steps': 2},
               'fixed_pose_recording': {'source': 'user', 'task_id': 'alpha', 'attempt': 'attempt_0002', 'trial': 1}}
    (attempt / 'report.json').write_text(json.dumps({'completed': True, 'returncode': 0,
        'task_id': 'alpha', 'robot': 'franka_long', 'trials': [partial]}))
    result = review.make_report(manifest, results, output)
    assert result['tasks'][0]['status'] == 'out_of_view'
    assert 'src="media/alpha/attempt_0002/trial_001/trajectory.gif"' in (output / 'index.html').read_text()
    assert '物体部分出画' in (output / 'index.html').read_text()


def test_preview_encoding_produces_only_ten_second_gif(tmp_path):
    frames = []
    for index in range(3):
        frame = tmp_path / f'step_{index:04d}.webp'
        Image.new('RGB', (16, 16), (index * 40, 100, 20)).save(frame)
        frames.append(frame)
    gif, video, error = review.render_media(frames, tmp_path / 'media', '/nonexistent/ffmpeg', gif_only=True)
    assert error is None and video is None
    assert review.valid_preview_gif(gif)
    assert not (tmp_path / 'media/trajectory.mp4').exists()


def test_incremental_report_preserves_attempts_and_reuses_media(tmp_path):
    results, output = tmp_path / 'results', tmp_path / 'review'
    tasks = [{'task_id': task, 'robot': 'ur5e'} for task in ['alpha', 'beta', 'gamma']]
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'output_format': 'hdf5', 'tasks': tasks}))

    def write_attempt(task, number, *, success, visible=True, complete=True, hdf5_only=False):
        attempt = results / task / f'attempt_{number:04d}'
        trial_dir = attempt / ('success' if success else 'failure') / 'trial_001'
        frames = trial_dir / 'original_images/external_cam_side'
        frames.mkdir(parents=True)
        for index in range(3):
            Image.new('RGB', (16, 16), (100 + index, 40, 80)).save(frames / f'step_{index:04d}.webp')
        npy = trial_dir / 'trajectory.npy'
        npy.write_bytes(b'test')
        trial = {'trial': 1, 'directory': str(trial_dir), 'success': success,
                 'failure_reason': 'none' if success else 'task_failed',
                 'collision_flyaway': collision(),
                 'side_camera_visibility': {'all_objects_in_frame': visible, 'checked_control_steps': 3},
                 'trajectory_npy': str(npy)}
        if hdf5_only:
            path = trial_dir / 'trajectory.hdf5'
            with h5py.File(path, 'w') as h5:
                group = h5.create_group('recordings/cameras/external_cam_side')
                group.create_dataset('frame_index', data=np.arange(3))
                group.create_dataset('file_name', data=[f'step_{i:04d}.webp' for i in range(3)],
                                     dtype=h5py.string_dtype('utf-8'))
                encoded = group.create_dataset('encoded_bytes', (3,), dtype=h5py.vlen_dtype(np.dtype('uint8')))
                for i in range(3):
                    file = frames / f'step_{i:04d}.webp'
                    encoded[i] = np.frombuffer(file.read_bytes(), dtype=np.uint8)
                    file.unlink()
            npy.unlink()
            trial.update(hdf5_path=str(path), hdf5_integrity={'valid': True})
        report = {'completed': complete, 'returncode': 0, 'task_id': task, 'robot': 'ur5e',
                  'simulation': {'dt': .01, 'decimation': 25}, 'trials': [trial]}
        (attempt / 'report.json').write_text(json.dumps(report))

    write_attempt('alpha', 1, success=False)
    write_attempt('alpha', 2, success=True, hdf5_only=True)
    write_attempt('beta', 1, success=True, visible=False)
    write_attempt('gamma', 1, success=True, complete=False)
    first = review.make_report(manifest, results, output)
    assert [item['status'] for item in first['tasks']] == ['success', 'out_of_view', 'running']
    assert [item['status'] for item in first['tasks'][0]['attempts']] == ['failure', 'success']
    assert first['tasks'][0]['attempts'][1]['hdf5']
    assert not first['tasks'][0]['attempts'][1]['npy']
    assert not first['tasks'][2]['attempts']
    gif = output / first['tasks'][0]['attempts'][1]['gif']
    assert gif.is_file()
    animation = Image.open(gif)
    durations = []
    for index in range(animation.n_frames):
        animation.seek(index)
        durations.append(animation.info['duration'])
    assert sum(durations) == 10_000
    movie = first['tasks'][0]['attempts'][1]['video']
    if review.find_ffmpeg():
        assert movie and (output / movie).is_file()
        import imageio_ffmpeg
        frames = imageio_ffmpeg.read_frames(str(output / movie))
        metadata = next(frames)
        assert abs(metadata['duration'] - 10.0) < .02
        assert sum(1 for _ in frames) == 3
    before = gif.stat().st_mtime_ns
    review.make_report(manifest, results, output)
    assert gif.stat().st_mtime_ns == before
    assert '碰撞检测完整' in (output / 'index.html').read_text()
