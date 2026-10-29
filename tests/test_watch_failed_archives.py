"""A watcher must touch only finalized, eligible diagnostic attempts."""
import importlib.util
import json
from pathlib import Path
import signal
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'tools')]
spec = importlib.util.spec_from_file_location('watch_failed_archives', ROOT / 'scripts/watch_failed_archives.py')
watcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(watcher)


def make_report(root, task, status, *, complete=True, preview=True, diagnostic=False):
    attempt = root / task / 'attempt_0001'
    directory = attempt / 'failure/trial_001'
    directory.mkdir(parents=True)
    source = directory / 'trajectory.npy'
    source.write_bytes(b'original')
    trial = {'trial': 1, 'directory': str(directory), 'success': False}
    if diagnostic:
        trial['diagnostic_hdf5_path'] = str(directory / 'diagnostic.hdf5')
    report = {'task_id': task, 'completed': complete, 'returncode': 0,
              'collection_classification': status, 'trials': [trial]}
    (attempt / 'report.json').write_text(json.dumps(report))
    if preview:
        media = root / 'review/media' / task / attempt.name / 'trial_001'
        media.mkdir(parents=True)
        for file in ['trajectory.gif', 'trajectory.mp4', 'signature.json']:
            (media / file).write_bytes(b'fixture')
    return attempt / 'report.json', source


def test_sweep_filters_accepted_incomplete_missing_preview_and_existing_archive(tmp_path, monkeypatch):
    root = tmp_path / 'collection'
    good, good_source = make_report(root, 'good', 'failure')
    excluded, excluded_source = make_report(root, 'excluded', 'collision_excluded')
    accepted, accepted_source = make_report(root, 'accepted', 'accepted')
    active, active_source = make_report(root, 'active', 'failure', complete=False)
    missing, missing_source = make_report(root, 'missing', 'out_of_view', preview=False)
    archived, archived_source = make_report(root, 'archived', 'failure', diagnostic=True)
    called = []
    def fake_archive(path, parent, cleanup):
        called.append((path, parent, cleanup))
        return {'archive': 'test.hdf5', 'training_eligible': False, 'source_files_removed': 0}
    monkeypatch.setattr(watcher, 'archive_attempt', fake_archive)
    result = watcher.scan_once(root, root / 'watch.jsonl', {}, now=100)
    assert result == {'archived': 2, 'locked': False}
    assert {path for path, _, _ in called} == {good, excluded}
    assert all(cleanup for _, _, cleanup in called)
    for source in [good_source, excluded_source, accepted_source, active_source, missing_source, archived_source]:
        assert source.read_bytes() == b'original'


def test_locked_review_defers_cleanup_without_touching_files(tmp_path, monkeypatch):
    root = tmp_path / 'collection'
    _, source = make_report(root, 'good', 'failure')
    called = []
    monkeypatch.setattr(watcher, 'archive_attempt', lambda *args, **kwargs: called.append(args))
    with watcher.media_lock(root / 'review'):
        result = watcher.scan_once(root, root / 'watch.jsonl', {}, now=1)
    assert result == {'archived': 0, 'locked': True}
    assert not called and source.read_bytes() == b'original'


def test_detects_older_review_process_for_same_output(tmp_path):
    root = tmp_path / 'collection'
    proc = tmp_path / 'proc'
    process = proc / '123'
    process.mkdir(parents=True)
    args = ['python', '/repo/scripts/render_collection_review.py', '--output-dir', str(root / 'review')]
    (process / 'cmdline').write_bytes(b'\0'.join(arg.encode() for arg in args) + b'\0')
    assert watcher.reviewer_process_active(root, proc)
    assert not watcher.reviewer_process_active(tmp_path / 'another', proc)


def test_final_sweep_waits_for_previous_review_process(tmp_path, monkeypatch):
    root = tmp_path / 'collection'
    root.mkdir()
    states = iter([True, False])
    monkeypatch.setattr(watcher, 'reviewer_process_active', lambda root: next(states))
    waits = []
    monkeypatch.setattr(watcher.time, 'sleep', lambda seconds: waits.append(seconds))
    result = watcher.scan_once(root, root / 'watch.jsonl', {}, blocking=True)
    assert result == {'archived': 0, 'locked': False} and waits == [1]


def test_refresh_waits_for_previous_review_process(tmp_path, monkeypatch):
    root = tmp_path / 'collection'
    root.mkdir()
    states = iter([True, False])
    monkeypatch.setattr(watcher, 'reviewer_process_active', lambda root: next(states))
    calls = []
    monkeypatch.setattr(watcher.time, 'sleep', lambda seconds: calls.append(('sleep', seconds)))
    monkeypatch.setattr(watcher.review, 'make_report', lambda *args: calls.append(('render',)))
    assert watcher.refresh_review(root, root / 'watch.jsonl')
    assert calls == [('sleep', 1), ('render',)]


def test_failed_archive_is_logged_and_retried_only_after_backoff(tmp_path, monkeypatch):
    root = tmp_path / 'collection'
    _, source = make_report(root, 'good', 'failure')
    failures = {}
    calls = []
    def fail(*args, **kwargs):
        calls.append(args)
        raise ValueError('simulated corruption')
    monkeypatch.setattr(watcher, 'archive_attempt', fail)
    for moment in [0, 1, 181]:
        watcher.scan_once(root, root / 'watch.jsonl', failures, now=moment)
    assert len(calls) == 2 and source.read_bytes() == b'original'
    logs = [json.loads(s) for s in (root / 'watch.jsonl').read_text().splitlines()]
    assert all(item['event'] == 'archive_failed_requires_review' for item in logs)
    assert all('simulated corruption' in item['error'] for item in logs)


def test_stop_is_not_swallowed_if_cleanup_rolls_back(tmp_path, monkeypatch):
    root = tmp_path / 'collection'
    _, source = make_report(root, 'good', 'failure')
    monkeypatch.setattr(watcher, '_STOP_SIGNAL', signal.SIGTERM)
    def interrupted(*args, **kwargs):
        raise RuntimeError('cleanup interrupted and restored')
    monkeypatch.setattr(watcher, 'archive_attempt', interrupted)
    with pytest.raises(watcher.StopRequested):
        watcher.scan_once(root, root / 'watch.jsonl', {}, now=1)
    assert source.read_bytes() == b'original'


def test_scheduler_exit_triggers_final_preview_refresh_and_blocking_sweep(tmp_path, monkeypatch):
    root = tmp_path / 'collection'
    root.mkdir()
    (root / 'manifest.json').write_text('{"tasks": []}')
    states = iter(['start_ticks', 'start_ticks', None])
    monkeypatch.setattr(watcher, 'scheduler_identity', lambda pid: next(states))
    monkeypatch.setattr(watcher.time, 'sleep', lambda _: None)
    calls = []
    def sweep(root, log, errors, *, blocking=False, now=None):
        calls.append(('sweep', blocking))
        return {'archived': 1 if blocking else 0, 'locked': False}
    monkeypatch.setattr(watcher, 'scan_once', sweep)
    monkeypatch.setattr(watcher, 'refresh_review', lambda *args: calls.append(('refresh',)) or True)
    watcher.run_watcher(root, 123, 1., root / 'watch.jsonl')
    assert calls == [('sweep', False), ('refresh',), ('sweep', True), ('refresh',)]
    logs = [json.loads(s) for s in (root / 'watch.jsonl').read_text().splitlines()]
    assert [item['event'] for item in logs] == ['started', 'scheduler_exited_final_sweep', 'finished']


def test_real_sweep_archives_a_completed_failure_and_preview_survives(tmp_path):
    from test_archive_failed_attempt import make_attempt, source_bytes
    root, report, trial, manifest = make_attempt(tmp_path)
    assert source_bytes(trial)
    result = watcher.scan_once(root, root / 'watch.jsonl', {}, now=1)
    assert result == {'archived': 1, 'locked': False}
    assert not source_bytes(trial) and (trial / 'diagnostic.hdf5').is_file()
    page = watcher.review.make_report(manifest, root, root / 'review')
    record = page['tasks'][0]
    assert record['status'] == 'failure'
    assert record['attempts'][0]['diagnostic_hdf5']
    assert record['attempts'][0]['gif'] and record['attempts'][0]['video']
    assert not (root / 'accepted_manifest.json').exists()
