"""Diagnose-only archives must preserve every source byte and never enter training."""
import importlib.util
import json
from pathlib import Path
import sys

import h5py
import numpy as np
from PIL import Image
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'scripts')]
spec = importlib.util.spec_from_file_location('archive_failed_attempt', ROOT / 'tools/archive_failed_attempt.py')
archive_tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(archive_tool)
import render_collection_review as review


def collision(excluded=False):
    events = [] if not excluded else [{'object':'cup','other_object':'rack','collision_time_s':.1,
        'detection_time_s':.2,'impulse_ns':.2,'velocity_jump_m_s':3.,'displacement_m':.3}]
    return {'status':'complete','excluded':excluded,'events':events,
            'observed_substeps':10,'thresholds':{'min_speed_m_s':2.}}


def make_attempt(tmp_path, status='failure'):
    root = tmp_path / 'collection'
    category = {'failure':'failure','collision_excluded':'collision_excluded',
                'out_of_view':'success','data_error':'success'}[status]
    trial_dir = root / 'pick_demo' / 'attempt_0001' / category / 'trial_001'
    for camera in ('external_cam_side', 'wrist_cam'):
        frames = trial_dir / 'original_images' / camera
        frames.mkdir(parents=True)
        for i in range(3):
            Image.new('RGB', (16, 16), (50 + 20*i, 20 if camera == 'wrist_cam' else 80, 120)).save(
                frames / f'step_{i:04d}.webp', format='WEBP', lossless=True)
    Image.new('RGB', (16, 16), (1, 2, 3)).save(trial_dir / 'initial.png')
    npy = trial_dir / 'trajectory.npy'
    np.save(npy, {'test': np.array([1., 2.])})
    success = status != 'failure'
    trial = {'trial':1,'directory':str(trial_dir.resolve()),'success':success,'program_error':False,
             'failure_reason':'task_failed' if not success else 'none','trajectory_path':str(npy.resolve()),
             'collision_flyaway':collision(status == 'collision_excluded'),
             'side_camera_visibility':{'all_objects_in_frame':status != 'out_of_view','checked_control_steps':3}}
    if status == 'data_error':
        trial['data_integrity'] = {'valid': False, 'reason': 'missing qpos'}
    (trial_dir / 'trial.json').write_text(json.dumps(trial))
    report = {'completed':True,'returncode':0,'task_id':'pick_demo','robot':'ur5e',
              'collection_classification':status,'trials':[trial]}
    report_path = trial_dir.parent.parent / 'report.json'
    report_path.write_text(json.dumps(report))
    (report_path.parent / 'task.yaml').write_text('task: pick_demo\n')
    (report_path.parent / 'run.log').write_text('test log\n')
    manifest = root / 'manifest.json'
    manifest.write_text(json.dumps({'output_format':'hdf5','tasks':[{'task_id':'pick_demo','robot':'ur5e'}]}))
    review.make_report(manifest, root, root / 'review')
    return root, report_path, trial_dir, manifest


def source_bytes(trial):
    return {str(p.relative_to(trial)):p.read_bytes() for p in trial.rglob('*')
            if p.is_file() and p.suffix.lower() in {'.webp','.png','.npy'}}


def test_archive_and_cleanup_are_lossless_non_training_and_previewable(tmp_path):
    root, report, trial, manifest = make_attempt(tmp_path)
    before = source_bytes(trial)
    result = archive_tool.archive_attempt(report, root)
    assert result['training_eligible'] is False and result['source_files_removed'] == 0
    assert source_bytes(trial) == before
    with h5py.File(result['archive'], 'r') as h5:
        assert not bool(h5.attrs['training_eligible'])
        assert h5.attrs['collection_classification'] == 'failure'
        assert 'action' not in h5 and 'observations' not in h5
        assert h5['recordings/trajectory_npy'][...].tobytes() == before['trajectory.npy']
        for camera in ('external_cam_side', 'wrist_cam'):
            for i in range(3):
                name = f'step_{i:04d}.webp'
                assert h5[f'recordings/cameras/{camera}/encoded_bytes'][i].tobytes() == before[f'original_images/{camera}/{name}']
    result = archive_tool.archive_attempt(report, root, cleanup=True)
    assert result['source_files_removed'] == len(before) and not source_bytes(trial)
    assert not (trial / 'original_images').exists()
    page = review.make_report(manifest, root, root / 'review')
    saved = page['tasks'][0]
    assert saved['status'] == 'failure'
    assert saved['attempts'][0]['diagnostic_hdf5'] and not saved['attempts'][0]['hdf5']
    assert saved['attempts'][0]['gif'] and saved['attempts'][0]['video']
    assert '不可训练' in (root / 'review/index.html').read_text()
    assert not (root / 'accepted_manifest.json').exists()


def test_modified_source_or_missing_preview_keeps_every_source_file(tmp_path):
    root, report, trial, _ = make_attempt(tmp_path)
    archive_tool.archive_attempt(report, root)
    frame = trial / 'original_images/wrist_cam/step_0001.webp'
    frame.write_bytes(frame.read_bytes() + b'changed')
    expected = source_bytes(trial)
    with pytest.raises(ValueError):
        archive_tool.archive_attempt(report, root, cleanup=True)
    assert source_bytes(trial) == expected
    preview = root / 'review/media/pick_demo/attempt_0001/trial_001/trajectory.mp4'
    preview.unlink()
    with pytest.raises(ValueError, match='GIF, MP4'):
        archive_tool.archive_attempt(report, root, cleanup=True)
    assert source_bytes(trial) == expected


def test_cleanup_failure_restores_already_unlinked_sources(tmp_path, monkeypatch):
    root, report, trial, _ = make_attempt(tmp_path)
    before = source_bytes(trial)
    original = archive_tool._unlink_source
    counter = {'n': 0}
    def fail_second(path):
        counter['n'] += 1
        if counter['n'] == 2:
            raise OSError('simulated failure')
        original(path)
    monkeypatch.setattr(archive_tool, '_unlink_source', fail_second)
    with pytest.raises(RuntimeError, match='all original source files were restored'):
        archive_tool.archive_attempt(report, root, cleanup=True)
    assert source_bytes(trial) == before
    assert (trial / 'diagnostic.hdf5').is_file()


def test_metadata_commit_failure_restores_sources(tmp_path, monkeypatch):
    root, report, trial, _ = make_attempt(tmp_path)
    before = source_bytes(trial)
    original = archive_tool.write_json
    def fail_final_commit(path, data):
        if isinstance(data, dict) and data.get('diagnostic_source_cleanup', {}).get('status') == 'cleaned':
            raise OSError('simulated metadata write failure')
        original(path, data)
    monkeypatch.setattr(archive_tool, 'write_json', fail_final_commit)
    with pytest.raises(RuntimeError, match='all source files were restored'):
        archive_tool.archive_attempt(report, root, cleanup=True)
    assert source_bytes(trial) == before
    assert json.loads(report.read_text())['trials'][0]['diagnostic_source_cleanup']['status'] == 'retained'


@pytest.mark.parametrize('status', ['collision_excluded','out_of_view','data_error'])
def test_excluded_and_out_of_view_are_eligible_only_for_diagnostics(tmp_path, status):
    root, report, _, _ = make_attempt(tmp_path, status)
    inspected = archive_tool.inspect_attempt(report, root)
    assert inspected[2]['collection_classification'] == status
    assert inspected[3]['success']
    result = archive_tool.archive_attempt(report, root)
    assert not result['training_eligible']


def test_accepted_attempt_is_never_reclassified_as_diagnostic(tmp_path):
    root, report, trial, _ = make_attempt(tmp_path)
    before = source_bytes(trial)
    data = json.loads(report.read_text())
    data['collection_classification'] = 'accepted'
    report.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='non-training'):
        archive_tool.archive_attempt(report, root, cleanup=True)
    assert source_bytes(trial) == before
