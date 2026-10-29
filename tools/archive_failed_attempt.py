"""Losslessly archive one finalized, non-training collection attempt to HDF5.

Default mode only creates/verifies diagnostic.hdf5 and never removes sources.
--cleanup explicitly removes only the source files verified twice against the
durable archive, after validating the existing 10-second GIF and MP4 preview.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile

import h5py
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import render_collection_review as review
from run_benchmark_50 import collision_flyaway_status, write_json

FRAME = re.compile(r'^step_(\d+)\.(webp|png)$')
TASK = re.compile(r'^[A-Za-z0-9_-]+$')
ATTEMPT = re.compile(r'^attempt_\d+$')
CAMERAS = ('external_cam_side', 'wrist_cam')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def sync_file_and_directory(path):
    path = Path(path)
    with path.open('rb') as file:
        os.fsync(file.fileno())
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def inspect_attempt(report_path, output_root):
    """Reject active, successful, ambiguous or path-unsafe attempts."""
    root = Path(output_root).resolve()
    if Path(report_path).is_symlink():
        raise ValueError('Refusing a symlinked report')
    report_path = Path(report_path).resolve()
    if not report_path.is_relative_to(root) or report_path.name != 'report.json':
        raise ValueError('Report is outside the specified collection')
    attempt = report_path.parent
    if not ATTEMPT.fullmatch(attempt.name) or not TASK.fullmatch(attempt.parent.name):
        raise ValueError('Unexpected task/attempt layout')
    report = json.loads(report_path.read_text())
    if (report.get('task_id') != attempt.parent.name or report.get('completed') is not True
            or report.get('returncode', 0) != 0 or report.get('error') or report.get('timed_out')):
        raise ValueError('Report is incomplete, failed to finalize, or has the wrong task ID')
    trials = report.get('trials')
    if not isinstance(trials, list) or len(trials) != 1 or trials[0].get('trial') != 1:
        raise ValueError('Expected exactly one completed trial')
    trial = trials[0]
    directory = Path(trial['directory']).resolve()
    if (not directory.is_relative_to(root) or directory.name != 'trial_001'
            or directory.parent.parent != attempt or not directory.is_dir()):
        raise ValueError('Trial directory is outside the completed attempt')
    if Path(trial['directory']).is_symlink() or directory.parent.is_symlink():
        raise ValueError('Refusing symlinked trial directory')
    if trial.get('hdf5_path') or (directory / 'trajectory.hdf5').exists():
        raise ValueError('Refusing to reclassify a training HDF5 as diagnostic')
    status = report.get('collection_classification')
    collision = collision_flyaway_status(trial)
    visibility = trial.get('side_camera_visibility') or {}
    allowed = {
        'failure': collision == 'complete' and trial.get('success') is False and directory.parent.name == 'failure',
        'collision_excluded': collision == 'excluded' and directory.parent.name == 'collision_excluded',
        'collision_unknown': collision == 'unknown' and directory.parent.name == 'collision_unknown',
        'out_of_view': (collision == 'complete' and trial.get('success') is True
                        and visibility.get('checked_control_steps', 0) > 0
                        and visibility.get('all_objects_in_frame') is False and directory.parent.name == 'success'),
        'visibility_unknown': (collision == 'complete' and trial.get('success') is True
                               and visibility.get('checked_control_steps', 0) <= 0
                               and directory.parent.name == 'success'),
        'data_error': (collision == 'complete' and trial.get('success') is True
                       and visibility.get('checked_control_steps', 0) > 0
                       and visibility.get('all_objects_in_frame') is True
                       and isinstance(trial.get('data_integrity'), dict)
                       and trial['data_integrity'].get('valid') is False
                       and directory.parent.name == 'success'),
    }
    if status not in allowed or not allowed[status]:
        raise ValueError(f'Report is not a supported finalized non-training attempt: {status!r}')
    trial_json = directory / 'trial.json'
    if trial_json.is_symlink() or not trial_json.is_file():
        raise ValueError('Missing or unsafe trial.json')
    saved_trial = json.loads(trial_json.read_text())
    if (saved_trial.get('trial') != trial['trial'] or saved_trial.get('directory') != trial['directory']
            or saved_trial.get('success') != trial.get('success')):
        raise ValueError('Trial metadata does not match report')

    image_root = directory / 'original_images'
    if image_root.is_symlink() or not image_root.is_dir():
        raise ValueError('Original image root is missing or unsafe')
    if {p.name for p in image_root.iterdir()} != set(CAMERAS):
        raise ValueError('Both camera directories are required; unknown entries are retained')
    cameras = {}
    sources = []
    for camera in CAMERAS:
        frame_dir = image_root / camera
        if frame_dir.is_symlink() or not frame_dir.is_dir():
            raise ValueError(f'Unsafe camera directory: {camera}')
        entries = list(frame_dir.iterdir())
        indexed = []
        for path in entries:
            match = FRAME.fullmatch(path.name)
            if not match or path.is_symlink() or not path.is_file():
                raise ValueError(f'Unexpected or unsafe camera file: {path}')
            indexed.append((int(match.group(1)), path))
        indexed.sort()
        if not indexed or len({index for index, _ in indexed}) != len(indexed):
            raise ValueError(f'Camera {camera} has no frames or duplicate indices')
        cameras[camera] = indexed
        sources.extend(path for _, path in indexed)
    initial = directory / 'initial.png'
    if initial.exists():
        if initial.is_symlink() or not initial.is_file():
            raise ValueError('Unsafe initial image')
        sources.append(initial)
    npy = directory / 'trajectory.npy'
    if npy.exists():
        if npy.is_symlink() or not npy.is_file():
            raise ValueError('Unsafe temporary NPY')
        sources.append(npy)
    recorded_path = trial.get('trajectory_path')
    if recorded_path and Path(recorded_path).resolve() != npy:
        raise ValueError('NPY is not at the expected path in this attempt')
    extras = []
    known = {p.resolve() for p in sources}
    for path in attempt.rglob('*'):
        if path.suffix.lower() in {'.webp', '.png', '.npy'} and path.resolve() not in known:
            extras.append(str(path))
    if extras:
        raise ValueError(f'Unknown image/NPY files retained: {extras[:6]}')
    return root, report_path, report, trial, directory, cameras, sources


def preview_paths(root, report, report_path):
    task, attempt = report['task_id'], report_path.parent.name
    directory = root / 'review/media' / task / attempt / 'trial_001'
    return directory / 'trajectory.gif', directory / 'trajectory.mp4', directory / 'signature.json'


def verify_preview(root, report, trial, report_path, side_frames):
    """Require a readable 10-second GIF and full-frame, 10-second MP4."""
    gif, video, marker = preview_paths(root, report, report_path)
    if not all(p.is_file() and not p.is_symlink() for p in (gif, video, marker)):
        raise ValueError('A verified GIF, MP4 and media signature are required before cleanup')
    fps = len(side_frames) * 1000.0 / review.PREVIEW_DURATION_MS
    signature = json.loads(marker.read_text()).get('signature')
    if signature != review.media_signature(side_frames, fps):
        raise ValueError('Preview does not match the current source frames')
    with Image.open(gif) as image:
        if image.format != 'GIF' or image.n_frames < 1:
            raise ValueError('Preview is not a valid GIF')
        duration = 0
        for index in range(image.n_frames):
            image.seek(index)
            image.load()
            duration += int(image.info.get('duration', 0))
        if duration != 10_000:
            raise ValueError(f'GIF duration is {duration} ms, not 10000 ms')
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise ValueError('Cannot verify MP4 without imageio_ffmpeg') from exc
    reader = imageio_ffmpeg.read_frames(str(video))
    try:
        metadata = next(reader)
        count = sum(1 for _ in reader)
    finally:
        reader.close()
    if abs(float(metadata['duration']) - 10.0) > .05 or count != len(side_frames):
        raise ValueError(f'MP4 must contain every side frame over 10 seconds: {metadata}, frames={count}')
    return gif, video


def write_diagnostic(path, report, trial, report_path, directory, cameras):
    fd, temporary = tempfile.mkstemp(prefix='.diagnostic.', suffix='.tmp', dir=directory)
    os.close(fd)
    try:
        with h5py.File(temporary, 'w') as out:
            out.attrs['artifact_type'] = 'diagnostic'
            out.attrs['training_eligible'] = False
            out.attrs['collection_classification'] = report['collection_classification']
            out.attrs['task_id'] = report['task_id']
            out.attrs['attempt'] = report_path.parent.name
            rec = out.create_group('recordings')
            original = rec.create_group('cameras')
            for camera, entries in cameras.items():
                group = original.create_group(camera)
                group.create_dataset('frame_index', data=np.asarray([index for index, _ in entries], dtype=np.int64))
                group.create_dataset('file_name', data=[p.name for _, p in entries], dtype=h5py.string_dtype('utf-8'))
                encoded = group.create_dataset('encoded_bytes', (len(entries),), dtype=h5py.vlen_dtype(np.dtype('uint8')))
                for row, (_, source) in enumerate(entries):
                    encoded[row] = np.frombuffer(source.read_bytes(), dtype=np.uint8)
            for key, source in [('trajectory_npy', directory / 'trajectory.npy'), ('initial_png', directory / 'initial.png'),
                                ('report_json', report_path), ('trial_json', directory / 'trial.json'),
                                ('task_yaml', report_path.parent / 'task.yaml'), ('run_log', report_path.parent / 'run.log')]:
                if source.is_file() and not source.is_symlink():
                    rec.create_dataset(key, data=np.frombuffer(source.read_bytes(), dtype=np.uint8), compression='gzip')
        os.link(temporary, path)  # Do not replace any existing diagnostic archive.
    finally:
        Path(temporary).unlink(missing_ok=True)


def verify_diagnostic(path, report, directory, cameras, metadata_snapshots):
    with h5py.File(path, 'r') as data:
        if (data.attrs.get('artifact_type') != 'diagnostic' or bool(data.attrs.get('training_eligible', True))
                or data.attrs.get('collection_classification') != report['collection_classification']
                or data.attrs.get('task_id') != report['task_id']
                or data.attrs.get('attempt') != directory.parent.parent.name
                or 'action' in data or 'observations' in data):
            raise ValueError('Diagnostic archive metadata is wrong or appears usable for training')
        for camera, entries in cameras.items():
            group = data[f'recordings/cameras/{camera}']
            if list(group['frame_index'][...]) != [index for index, _ in entries]:
                raise ValueError(f'{camera} frame indices do not match')
            if list(group['file_name'].asstr()[...]) != [p.name for _, p in entries]:
                raise ValueError(f'{camera} frame names do not match')
            encoded = group['encoded_bytes']
            if len(encoded) != len(entries):
                raise ValueError(f'{camera} frame count does not match')
            for row, (_, source) in enumerate(entries):
                if encoded[row].tobytes() != source.read_bytes():
                    raise ValueError(f'Frame differs from archive: {source}')
        for key, source in [('trajectory_npy', directory / 'trajectory.npy'), ('initial_png', directory / 'initial.png')]:
            if source.is_file() != (key in data['recordings']):
                raise ValueError(f'{key} presence differs')
            if source.is_file() and data['recordings'][key][...].tobytes() != source.read_bytes():
                raise ValueError(f'{key} differs from archive')
        for key, content in metadata_snapshots.items():
            if key not in data['recordings'] or data['recordings'][key][...].tobytes() != content:
                raise ValueError(f'{key} metadata differs from archive')


def _unlink_source(path):
    path.unlink()


def restore_sources(snapshots, archive):
    restoration_errors = []
    for path, content, mode, atime, mtime in snapshots:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                with path.open('xb') as output:
                    output.write(content)
                    output.flush()
                    os.fsync(output.fileno())
                os.chmod(path, mode)
                os.utime(path, ns=(atime, mtime))
            elif path.is_symlink() or path.read_bytes() != content:
                restoration_errors.append(str(path))
        except OSError as restore_exc:
            restoration_errors.append(f'{path}: {restore_exc}')
    if restoration_errors:
        raise RuntimeError(f'Some paths could not be restored; data remains in {archive}: {restoration_errors}')


def cleanup_sources(sources, archive, expected_hashes):
    """Roll back already unlinked files if deletion fails midway."""
    snapshots = [(p, p.read_bytes(), p.stat().st_mode & 0o777, p.stat().st_atime_ns, p.stat().st_mtime_ns)
                 for p in sources]
    if any(p.is_symlink() or hashlib.sha256(content).hexdigest() != expected_hashes[str(p)]
           for p, content, _, _, _ in snapshots):
        raise ValueError('A source changed after archive verification; no files removed')
    removed = []
    try:
        for path, content, _, _, _ in snapshots:
            if path.is_symlink() or not path.is_file() or path.read_bytes() != content:
                raise ValueError(f'Source changed immediately before cleanup: {path}')
            _unlink_source(path)
            removed.append(path)
        for directory in sorted({p.parent for p in sources if p.parent.name in CAMERAS}, key=str):
            directory.rmdir()
        image_roots = {p.parent.parent for p in sources if p.parent.name in CAMERAS}
        for root in image_roots:
            root.rmdir()
    except BaseException as exc:
        restore_sources(snapshots, archive)
        raise RuntimeError('Cleanup failed; all original source files were restored') from exc
    return len(removed), snapshots


def archive_attempt(report_path, output_root, cleanup=False):
    root, report_path, report, trial, directory, cameras, sources = inspect_attempt(report_path, output_root)
    report_hash, trial_hash = sha(report_path), sha(directory / 'trial.json')
    metadata_paths = {'report_json': report_path, 'trial_json': directory / 'trial.json',
                      'task_yaml': report_path.parent / 'task.yaml', 'run_log': report_path.parent / 'run.log'}
    metadata_snapshots = {key: path.read_bytes() for key, path in metadata_paths.items()
                          if path.is_file() and not path.is_symlink()}
    expected_hashes = {str(path): sha(path) for path in sources}
    side = [path for _, path in cameras['external_cam_side']]
    gif, video = verify_preview(root, report, trial, report_path, side)
    archive = directory / 'diagnostic.hdf5'
    if archive.is_symlink():
        raise ValueError('Diagnostic archive path is a symlink')
    recorded_hash = trial.get('diagnostic_hdf5_sha256')
    if archive.exists() and recorded_hash:
        if sha(archive) != recorded_hash:
            raise ValueError('Existing diagnostic archive differs from its recorded checksum')
        # The original snapshots predate the first archival metadata update.
        with h5py.File(archive, 'r') as previous:
            for key in ('report_json', 'trial_json'):
                metadata_snapshots[key] = previous[f'recordings/{key}'][...].tobytes()
            original_report = json.loads(metadata_snapshots['report_json'])
            if (original_report.get('task_id') != report.get('task_id')
                    or original_report.get('collection_classification') != report.get('collection_classification')):
                raise ValueError('Existing archive describes a different task or classification')
    if not archive.exists():
        write_diagnostic(archive, report, trial, report_path, directory, cameras)
    verify_diagnostic(archive, report, directory, cameras, metadata_snapshots)
    sync_file_and_directory(archive)
    archive_hash = sha(archive)
    trial.update({'diagnostic_hdf5_path': str(archive), 'diagnostic_hdf5_sha256': archive_hash,
                  'diagnostic_hdf5_integrity': {'valid': True, 'training_eligible': False,
                                                'camera_frames': {k: len(v) for k, v in cameras.items()},
                                                'preview_gif': str(gif), 'preview_mp4': str(video)}})
    # Recheck classification and source consistency before changing metadata.
    _, _, latest_report, _, _, _, _ = inspect_attempt(report_path, root)
    if (sha(report_path) != report_hash or sha(directory / 'trial.json') != trial_hash
            or latest_report.get('collection_classification') != report.get('collection_classification')):
        raise ValueError('Report or trial metadata changed during archiving')
    latest_report['trials'][0].update(trial)
    write_json(directory / 'trial.json', latest_report['trials'][0])
    write_json(report_path, latest_report)
    sync_file_and_directory(directory / 'trial.json')
    sync_file_and_directory(report_path)
    if cleanup:
        # Second full verification, after archive and report are durable.
        verify_diagnostic(archive, latest_report, directory, cameras, metadata_snapshots)
        verify_preview(root, latest_report, trial, report_path, side)
        if sha(archive) != archive_hash:
            raise ValueError('Archive changed during verification')
        count, snapshots = cleanup_sources(sources, archive, expected_hashes)
        try:
            latest_report['trials'][0]['diagnostic_source_cleanup'] = {'status': 'cleaned', 'files_removed': count}
            write_json(directory / 'trial.json', latest_report['trials'][0])
            write_json(report_path, latest_report)
            sync_file_and_directory(directory / 'trial.json')
            sync_file_and_directory(report_path)
        except BaseException as exc:
            restore_sources(snapshots, archive)
            latest_report['trials'][0]['diagnostic_source_cleanup'] = {'status': 'retained', 'reason': str(exc)}
            try:
                write_json(directory / 'trial.json', latest_report['trials'][0])
                write_json(report_path, latest_report)
            except OSError:
                pass
            raise RuntimeError('Cleanup metadata could not be committed; all source files were restored') from exc
    else:
        count = 0
    return {'archive': str(archive), 'sha256': archive_hash, 'classification': report['collection_classification'],
            'training_eligible': False, 'source_files_removed': count}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', required=True, type=Path, help='A finalized attempt report.json')
    parser.add_argument('--collection-root', required=True, type=Path)
    parser.add_argument('--cleanup', action='store_true', help='Remove verified source files after the durable archive and previews pass all checks')
    args = parser.parse_args()
    with review.media_lock(args.collection_root / 'review'):
        print(json.dumps(archive_attempt(args.report, args.collection_root, args.cleanup), indent=2))


if __name__ == '__main__':
    main()
