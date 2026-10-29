"""Collect one qualified, frame-aligned trajectory for each task in a manifest.

Runs one simulator at a time. Failed, excluded, or unframed attempts remain
separate and are retried in later passes; successful files are never overwritten.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys

import numpy as np
import h5py
from PIL import Image, ImageSequence

from run_benchmark_50 import collision_flyaway_status, prepare_config, task_seed, write_json
from visibility_review import user_confirmed_visibility

ROOT = Path(__file__).resolve().parents[1]
ACTION_CONVENTION = 'legacy_ee_delta_source_gripper'


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def sync_file_and_directory(path):
    """Make the file and its directory entry durable before deleting a source."""
    path = Path(path)
    with path.open('rb') as file:
        os.fsync(file.fileno())
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def validate_trajectory(trial):
    """Check that all state/action rows reference existing, lossless RGB frames."""
    path = Path(trial.get('trajectory_path') or '')
    if not path.is_file():
        return {'valid': False, 'reason': 'structured trajectory is missing'}
    if path.suffix.lower() in ('.hdf5', '.h5'):
        return validate_direct_hdf5(path)
    try:
        data = np.load(path, allow_pickle=True).item()
        if data.get('action_convention') != ACTION_CONVENTION:
            raise ValueError('trajectory does not declare the original repository action convention')
        actions = np.asarray(data['obs_policy_actions'])
        joints = np.asarray(data['obs_policy_joint_actions'])
        next_joints = np.asarray(data['observed_next_joint_command'])
        ee = np.asarray(data['obs_policy_arm_ee_pose'])
        qpos = np.asarray(data['obs_policy_q_pos'])
        qvel = np.asarray(data['obs_policy_q_vel'])
        source = np.asarray(data['source_frame_index'])
        target = np.asarray(data['next_frame_index'])
        n = len(actions)
        if n == 0 or any(len(a) != n for a in [joints, next_joints, ee, qpos, qvel, source, target]):
            raise ValueError('state, action, and frame index lengths do not match')
        if actions.ndim != 2 or joints.ndim != 2 or not np.allclose(actions[:, -1], joints[:, -1]):
            raise ValueError('gripper commands are not aligned between action fields')
        if (next_joints.shape != joints.shape or qpos.ndim != 2 or qvel.shape != qpos.shape
                or joints.shape[1] != qpos.shape[1] + 1):
            raise ValueError('joint command and real joint state dimensions do not match')
        if len(joints) > 1 and not np.array_equal(next_joints[:-1], joints[1:]):
            raise ValueError('the separately stored next commands do not match consecutive source commands')
        if not all(np.isfinite(a).all() for a in [actions, joints, next_joints, ee, qpos, qvel, source, target]):
            raise ValueError('non-finite state, action, or frame index')
        if not all(np.equal(a, np.floor(a)).all() for a in [source, target]):
            raise ValueError('non-integral frame index')
        if (source < 0).any() or (target <= source).any():
            raise ValueError('invalid source/next frame order')
        if not np.array_equal(target[:-1], source[1:]):
            raise ValueError('frame indices do not describe consecutive recorded transitions')
        if any('cam' in key.lower() or 'image' in key.lower() for key in data):
            raise ValueError('image unexpectedly embedded in structured data')
        directory = Path(trial['directory']) / 'original_images'
        counts = {}
        for camera in ('external_cam_side', 'wrist_cam'):
            frames = directory / camera
            if not frames.is_dir():
                raise ValueError(f'missing camera frame directory: {camera}')
            available = {int(p.stem.split('_')[-1]) for p in frames.glob('step_*') if p.suffix in {'.webp', '.png'}}
            missing = (set(source.astype(int)) | set(target.astype(int))) - available
            if missing:
                raise ValueError(f'{camera} missing referenced frames: {sorted(missing)[:8]}')
            counts[camera] = len(available)
        return {'valid': True, 'steps': n, 'qpos_dim': int(qpos.shape[1]), 'camera_frames': counts,
                'npy_bytes': path.stat().st_size, 'source_frame_first': int(source[0]),
                'next_frame_last': int(target[-1])}
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        return {'valid': False, 'reason': str(exc)}


def validate_direct_hdf5(path):
    """Validate a directly recorded HDF5 without claiming source-byte verification."""
    try:
        path = Path(path)
        with h5py.File(path, 'r') as data:
            if data.attrs.get('action_convention') != ACTION_CONVENTION:
                raise ValueError('HDF5 declares a different action convention')
            if data.attrs.get('image_alignment') != 'source_frame_index':
                raise ValueError('HDF5 does not declare source-frame image alignment')
            action = data['action'][...]
            qpos, qvel = data['observations/qpos'][...], data['observations/qvel'][...]
            fields = data['recordings/trajectory_fields']
            joints = fields['obs_policy_joint_actions'][...]
            next_joints = fields['observed_next_joint_command'][...]
            ee = fields['obs_policy_arm_ee_pose'][...]
            source, target = data['recordings/source_frame_index'][...], data['recordings/next_frame_index'][...]
            names = list(fields['arm_joint_names'].asstr()[...])
            n = len(action)
            if (action.ndim != 2 or action.shape[1] != 7 or n == 0 or qpos.ndim != 2
                    or qvel.shape != qpos.shape or len(qpos) != n or joints.shape != (n, qpos.shape[1]+1)
                    or next_joints.shape != joints.shape or ee.ndim != 2 or len(ee) != n
                    or source.shape != (n,) or target.shape != (n,)
                    or len(names) != qpos.shape[1] or len(set(names)) != len(names)):
                raise ValueError('HDF5 state/action/joint/frame shapes do not match')
            if not all(np.isfinite(a).all() for a in (action, qpos, qvel, joints, next_joints, ee, source, target)):
                raise ValueError('HDF5 contains non-finite state, action, or frame indices')
            if (not np.issubdtype(source.dtype, np.integer) or not np.issubdtype(target.dtype, np.integer)
                    or (source < 0).any() or (target <= source).any()
                    or not np.array_equal(target[:-1], source[1:])):
                raise ValueError('HDF5 frame indices do not describe consecutive transitions')
            if (not np.allclose(action[:, -1], joints[:, -1])
                    or not np.array_equal(next_joints[:-1], joints[1:])):
                raise ValueError('HDF5 gripper or next joint commands are misaligned')
            for name, value in (('obs_policy_actions', action), ('obs_policy_q_pos', qpos),
                                ('obs_policy_q_vel', qvel), ('source_frame_index', source),
                                ('next_frame_index', target)):
                if not np.array_equal(fields[name][...], value):
                    raise ValueError(f'HDF5 {name} differs from its recorded source field')
            for camera in ('front', 'wrist'):
                image = data['observations/images/' + camera]
                if image.shape != (n, 256, 256, 3) or image.dtype != np.dtype('uint8'):
                    raise ValueError(f'HDF5 {camera} image shape or dtype is invalid')
        return {'valid': True, 'storage_format': 'direct_hdf5', 'steps': n,
                'qpos_dim': qpos.shape[1], 'source_frame_first': int(source[0]),
                'next_frame_last': int(target[-1])}
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        return {'valid': False, 'reason': str(exc)}


def validate_final_hdf5(trial):
    """Check a previously verified, immutable HDF5 after temporary NPY removal."""
    try:
        path = Path(trial['hdf5_path'])
        expected = trial['hdf5_sha256']
        integrity = trial['data_integrity']
        hdf5_integrity = trial['hdf5_integrity']
        if not path.is_file() or not isinstance(expected, str) or file_sha256(path) != expected:
            raise ValueError('HDF5 file is missing or differs from its verified checksum')
        if not isinstance(integrity, dict) or integrity.get('valid') is not True:
            raise ValueError('HDF5 has no successful source verification record')
        if not isinstance(hdf5_integrity, dict) or hdf5_integrity.get('valid') is not True:
            raise ValueError('HDF5 has no successful HDF5 integrity record')
        if hdf5_integrity.get('storage_format') == 'direct_hdf5':
            direct = validate_direct_hdf5(path)
            if not direct.get('valid'):
                raise ValueError(direct.get('reason', 'invalid direct HDF5'))
            if any(direct[key] != hdf5_integrity.get(key) for key in ('steps', 'qpos_dim')):
                raise ValueError('HDF5 dimensions differ from its integrity record')
            return direct
        with h5py.File(path, 'r') as data:
            if data.attrs.get('action_convention') != ACTION_CONVENTION:
                raise ValueError('HDF5 declares a different action convention')
            action, qpos, qvel = data['action'], data['observations/qpos'], data['observations/qvel']
            if (len(action) == 0 or action.ndim != 2 or qpos.ndim != 2 or qvel.shape != qpos.shape
                    or len(qpos) != len(action)):
                raise ValueError('HDF5 action and state dimensions do not match')
            if hdf5_integrity.get('steps') != len(action) or hdf5_integrity.get('qpos_dim') != qpos.shape[1]:
                raise ValueError('HDF5 dimensions differ from the recorded integrity metadata')
            for name in ('front', 'wrist'):
                if len(data['observations/images/' + name]) != len(action):
                    raise ValueError(f'HDF5 {name} training frame count does not match actions')
            for name in ('trajectory_npy', 'source_frame_index', 'next_frame_index',
                         'cameras/external_cam_side/encoded_bytes', 'cameras/wrist_cam/encoded_bytes'):
                if 'recordings/' + name not in data:
                    raise ValueError(f'HDF5 missing original recording {name}')
        return dict(integrity)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {'valid': False, 'reason': str(exc)}


def finalize_trajectory_hdf5(trial, output, *, allow_legacy_npy=False):
    """Register direct HDF5 or convert and independently verify a legacy NPY."""
    directory = Path(trial['directory']).resolve()
    root = Path(output).resolve()
    source = Path(trial['trajectory_path']).resolve()
    if directory.is_relative_to(root) and source == directory / 'trajectory.hdf5':
        check = validate_direct_hdf5(source)
        if not check.get('valid'):
            raise ValueError(f"Direct HDF5 validation failed: {check.get('reason')}")
        sync_file_and_directory(source)
        trial.update({'hdf5_path': str(source), 'hdf5_sha256': file_sha256(source),
                      'hdf5_integrity': check, 'hdf5_verification': check})
        return check
    if not allow_legacy_npy:
        raise ValueError('Legacy NPY conversion requires explicit --allow-legacy-npy')
    if not directory.is_relative_to(root) or source != directory / 'trajectory.npy':
        raise ValueError('Refusing conversion outside this collection attempt')
    if not source.is_file():
        raise FileNotFoundError(f'Temporary trajectory is missing: {source}')
    destination = directory / 'trajectory.hdf5'
    tools_path = str(ROOT / 'tools')
    if tools_path not in sys.path:
        sys.path.insert(0, tools_path)
    from convert_separate_trajectory_to_hdf5 import convert_trajectory, verify_trajectory_hdf5
    source_hash = file_sha256(source)
    if not destination.exists():
        convert_trajectory(source, destination)
    verification = verify_trajectory_hdf5(source, destination)
    if not isinstance(verification, dict) or verification.get('valid') is not True:
        raise ValueError('HDF5 verifier did not affirm successful validation')
    if file_sha256(source) != source_hash:
        raise RuntimeError('Temporary trajectory changed during HDF5 conversion')
    sync_file_and_directory(destination)
    with h5py.File(destination, 'r') as data:
        hdf5_integrity = {'valid': True, 'steps': len(data['action']),
                          'qpos_dim': data['observations/qpos'].shape[1],
                          'source_npy_sha256': source_hash}
    trial.update({'temporary_npy_path': str(source), 'temporary_npy_sha256': source_hash,
                  'hdf5_path': str(destination), 'hdf5_sha256': file_sha256(destination),
                  'trajectory_path': str(destination),
                  'hdf5_verification': verification, 'hdf5_integrity': hdf5_integrity})
    return verification


def cleanup_verified_temporary_npy(trial, output):
    """Remove only this collection's source NPY after durable HDF5 validation."""
    source_value = trial.get('temporary_npy_path')
    if not source_value:
        return False
    source = Path(source_value).resolve()
    directory = Path(trial['directory']).resolve()
    root = Path(output).resolve()
    if not directory.is_relative_to(root) or source != directory / 'trajectory.npy':
        raise ValueError('Refusing to remove an NPY outside this collection attempt')
    check = validate_final_hdf5(trial)
    if not check.get('valid'):
        raise ValueError(f'Refusing to remove NPY without verified HDF5: {check.get("reason")}')
    if not source.exists():
        return False
    if not source.is_file() or file_sha256(source) != trial.get('temporary_npy_sha256'):
        raise ValueError('Refusing to remove a modified temporary NPY')
    source.unlink()
    return True


def cleanup_verified_original_frames(trial, report_path, output):
    """Delete only byte-verified frames after the HDF5 and 10-second GIF exist.

    Preflight both cameras in full before deleting either. Missing, additional,
    changed or unsafe paths leave all current frames untouched. No other trial
    files are removed.
    """
    root = Path(output).resolve()
    report_path = Path(report_path).resolve()
    directory = Path(trial['directory']).resolve()
    if not directory.is_relative_to(root) or not report_path.is_relative_to(root):
        raise ValueError('Refusing frame cleanup outside this collection')
    if directory.parent.name != 'success' or directory.name != 'trial_001':
        raise ValueError('Refusing frame cleanup outside a successful trial')
    if report_path != directory.parent.parent / 'report.json':
        raise ValueError('Report does not belong to this trial attempt')
    check = validate_final_hdf5(trial)
    if not check.get('valid'):
        raise ValueError(f'Frame cleanup requires verified HDF5: {check.get("reason")}')
    report = load_report(report_path)
    if (not report or report.get('collection_classification') != 'accepted'
            or len(report.get('trials', [])) != 1):
        raise ValueError('Accepted report is not durably recorded')
    saved = report['trials'][0]
    if (saved.get('hdf5_path') != trial.get('hdf5_path')
            or saved.get('hdf5_sha256') != trial.get('hdf5_sha256')
            or saved.get('hdf5_integrity', {}).get('valid') is not True
            or saved.get('data_integrity', {}).get('valid') is not True):
        raise ValueError('Saved report does not match verified HDF5')
    saved_trial = load_report(directory / 'trial.json')
    if (not saved_trial or saved_trial.get('hdf5_sha256') != trial.get('hdf5_sha256')
            or saved_trial.get('hdf5_integrity', {}).get('valid') is not True):
        raise ValueError('Saved trial metadata does not match verified HDF5')
    image_root = directory / 'original_images'
    if (trial.get('hdf5_integrity', {}).get('storage_format') == 'direct_hdf5'
            and not image_root.exists()):
        # Direct recording embeds source-aligned training RGB in HDF5 and
        # deliberately does not create separate raw image files.
        return {'status': 'not_saved', 'frames_removed': 0,
                'hdf5_sha256': trial['hdf5_sha256']}

    task_id = report.get('task_id', '')
    attempt = directory.parent.parent.name
    if not re.fullmatch(r'[A-Za-z0-9_-]+', task_id) or not re.fullmatch(r'attempt_\d+', attempt):
        raise ValueError('Unsafe task or attempt identifier')
    if directory.parent.parent.parent.name != task_id:
        raise ValueError('Task identifier does not match trial directory')
    preview = root / 'review/media' / task_id / attempt / 'trial_001/trajectory.gif'
    if not preview.is_file():
        raise ValueError('10-second GIF preview is missing')
    try:
        with Image.open(preview) as animation:
            if animation.format != 'GIF' or animation.n_frames < 1:
                raise ValueError('Preview is not a GIF animation')
            total = 0
            for frame in range(animation.n_frames):
                animation.seek(frame)
                animation.load()
                total += int(animation.info.get('duration', 0))
            if total != 10_000:
                raise ValueError(f'GIF preview duration is {total} ms, not 10000 ms')
    except (OSError, EOFError) as exc:
        raise ValueError(f'Cannot read 10-second GIF preview: {exc}') from exc

    previous = trial.get('raw_frame_cleanup', {})
    if not image_root.exists() and previous.get('status') == 'cleaned':
        return dict(previous)
    if image_root.is_symlink() or not image_root.is_dir():
        raise ValueError('Original image root is missing or unsafe')
    cameras = ('external_cam_side', 'wrist_cam')
    if {p.name for p in image_root.iterdir()} != set(cameras):
        raise ValueError('Original image root has a missing or unknown camera entry')

    pattern = re.compile(r'^step_(\d+)\.(webp|png)$')
    verified = []
    hdf5_path = Path(trial['hdf5_path'])
    with h5py.File(hdf5_path, 'r') as data:
        for camera in cameras:
            frame_dir = image_root / camera
            if frame_dir.is_symlink() or not frame_dir.is_dir():
                raise ValueError(f'Camera frame directory is missing or unsafe: {camera}')
            group = data[f'recordings/cameras/{camera}']
            names = list(group['file_name'].asstr()[...])
            indices = np.asarray(group['frame_index'][...])
            encoded = group['encoded_bytes']
            if not names or len(names) != len(indices) or len(names) != len(encoded) or len(set(names)) != len(names):
                raise ValueError(f'HDF5 camera metadata is incomplete: {camera}')
            entries = {p.name: p for p in frame_dir.iterdir()}
            if set(entries) != set(names):
                raise ValueError(f'Camera {camera} has missing or unknown files')
            for row, (name, index) in enumerate(zip(names, indices)):
                match = pattern.fullmatch(name)
                entry = entries[name]
                if (not match or int(match.group(1)) != int(index)
                        or entry.is_symlink() or not entry.is_file()):
                    raise ValueError(f'Unsafe or mismatched camera frame: {camera}/{name}')
                expected = encoded[row].tobytes()
                if entry.read_bytes() != expected:
                    raise ValueError(f'Camera frame differs from HDF5: {camera}/{name}')
                verified.append((entry, expected))
    # Verify again after the complete preflight, before any deletion.
    if file_sha256(hdf5_path) != trial['hdf5_sha256']:
        raise ValueError('HDF5 changed during frame verification')
    for entry, expected in verified:
        if entry.is_symlink() or not entry.is_file() or entry.read_bytes() != expected:
            raise ValueError(f'Camera frame changed during verification: {entry}')

    deleted = 0
    try:
        for entry, _ in verified:
            entry.unlink()
            deleted += 1
        for camera in cameras:
            (image_root / camera).rmdir()  # Only succeeds if directory is empty.
        image_root.rmdir()
    except OSError as exc:
        raise OSError(f'Frame cleanup interrupted after {deleted}/{len(verified)} deletions: {exc}') from exc
    return {'status': 'cleaned', 'frames_removed': deleted, 'preview': str(preview),
            'hdf5_sha256': trial['hdf5_sha256']}


def classify(report, *, preview_only=False, allow_legacy_npy=False):
    if not report.get('completed') or report.get('error') or report.get('returncode', 0) != 0:
        return 'error', None
    trials = report.get('trials', [])
    if len(trials) != 1:
        return 'error', None
    trial = trials[0]
    collision = collision_flyaway_status(trial)
    if collision != 'complete':
        return 'collision_' + collision, None
    if not trial.get('success'):
        return 'failure', None
    visibility = trial.get('side_camera_visibility', {})
    if visibility.get('checked_control_steps', 0) <= 0:
        return 'visibility_unknown', None
    attempt = Path(report.get('config_path', '')).parent.name
    manually_reviewed = preview_only and user_confirmed_visibility(trial, report.get('task_id'), attempt)
    if visibility.get('all_objects_in_frame') is not True and not manually_reviewed:
        return 'out_of_view', None
    if preview_only:
        return 'accepted', None
    if (not trial.get('hdf5_path') and str(trial.get('trajectory_path', '')).lower().endswith('.npy')
            and not allow_legacy_npy):
        return 'data_error', {'valid': False, 'reason': 'Legacy NPY requires explicit --allow-legacy-npy'}
    data_check = validate_final_hdf5(trial) if trial.get('hdf5_path') else validate_trajectory(trial)
    return ('accepted' if data_check['valid'] else 'data_error'), data_check


def valid_preview(path):
    """Only reuse and clean up after a fully readable ten-second GIF."""
    try:
        with Image.open(path) as gif:
            if gif.format != 'GIF':
                return False
            return sum(int(frame.info.get('duration', 0)) for frame in ImageSequence.Iterator(gif)) == 10_000
    except (OSError, ValueError, TypeError):
        return False


def cleanup_preview_frames(trial, output):
    """Remove only temporary frame files after the attempt's GIF is verified."""
    directory = Path(trial['directory']).resolve()
    root = Path(output).resolve()
    if not directory.is_relative_to(root) or directory.name != 'trial_001':
        raise ValueError('Refusing frame cleanup outside the current collection trial')
    task = directory.parents[2].name
    attempt = directory.parents[1].name
    if not re.fullmatch(r'[A-Za-z0-9_-]+', task) or not re.fullmatch(r'attempt_\d+', attempt):
        raise ValueError('Invalid preview directory')
    gif = root / 'review/media' / task / attempt / 'trial_001/trajectory.gif'
    if not valid_preview(gif):
        raise ValueError('Cannot remove frames without a verified ten-second GIF')
    image_root = directory / 'original_images'
    removed = 0
    for camera in ('external_cam_side', 'wrist_cam'):
        camera_dir = image_root / camera
        if not camera_dir.is_dir():
            continue
        for frame in camera_dir.iterdir():
            if frame.is_file() and re.fullmatch(r'step_\d+\.(?:png|webp)', frame.name):
                frame.unlink()
                removed += 1
        if not any(camera_dir.iterdir()):
            camera_dir.rmdir()
    if image_root.is_dir() and not any(image_root.iterdir()):
        image_root.rmdir()
    (directory / 'initial.png').unlink(missing_ok=True)
    return {'status': 'cleaned', 'frames_removed': removed, 'gif': str(gif)}


def load_report(path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def update_report(report_path, report, status, data_check):
    report['collection_classification'] = status
    if data_check is not None:
        report['trials'][0]['data_integrity'] = data_check
        trial_json = Path(report['trials'][0]['directory']) / 'trial.json'
        write_json(trial_json, report['trials'][0])
    write_json(report_path, report)


def review(manifest, output):
    command = [sys.executable, str(ROOT / 'scripts/render_collection_review.py'),
               '--manifest', str(manifest), '--results-dir', str(output),
               '--output-dir', str(output / 'review')]
    completed = subprocess.run(command, cwd=ROOT, check=False)
    if completed.returncode:
        print(f'WARNING review refresh failed: exit={completed.returncode}', flush=True)
    return completed.returncode == 0


def should_refresh_review(*, preview_only, reused_report, initial_reviewed, gif_valid):
    """Reused previews were already incorporated by the startup refresh."""
    return not (preview_only and reused_report and initial_reviewed and gif_valid)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--max-attempts', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--timeout', type=int, default=1200)
    parser.add_argument('--minimum-free-gb', type=float, default=10.)
    parser.add_argument('--preview-only', action='store_true', help='Create GIF previews only; do not save training trajectories')
    parser.add_argument('--allow-legacy-npy', action='store_true',
                        help='Explicitly validate and convert existing legacy NPY attempts; new recordings still use HDF5')
    args = parser.parse_args()
    if args.max_attempts < 1:
        parser.error('max-attempts must be positive')
    manifest, output = args.manifest.resolve(), args.output_dir.resolve()
    manifest_data = json.loads(manifest.read_text())
    tasks = manifest_data['tasks']
    if args.preview_only and manifest_data.get('output_format') != 'preview':
        parser.error('--preview-only requires manifest output_format: preview')
    if len(tasks) != 50 or len({t['task_id'] for t in tasks}) != 50:
        raise ValueError('Expected exactly 50 distinct tasks')
    output.mkdir(parents=True, exist_ok=True)
    accepted = {}
    history = {t['task_id']: [] for t in tasks}
    consecutive_errors = 0
    # A restart can encounter dozens of immutable completed attempts. Build
    # their gallery once, including any updated manual-review decisions,
    # instead of rebuilding the full gallery for each report we re-register.
    initial_reviewed = review(manifest, output) if args.preview_only else False
    for attempt_number in range(1, args.max_attempts + 1):
        for task in tasks:
            name = task['task_id']
            if name in accepted:
                continue
            free_gb = shutil.disk_usage(output).free / 2**30
            if free_gb < args.minimum_free_gb:
                raise RuntimeError(f'Free disk space {free_gb:.1f} GiB is below the safety floor')
            attempt = output / name / f'attempt_{attempt_number:04d}'
            report_path = attempt / 'report.json'
            report = load_report(report_path)
            reused_report = bool(report and report.get('completed'))
            if report is None or not report.get('completed'):
                if attempt.exists() and any(attempt.iterdir()):
                    print(f'INCOMPLETE {name} {attempt.name}: retained, skipping this attempt', flush=True)
                    history[name].append({'attempt': attempt.name, 'status': 'incomplete'})
                    continue
                attempt.mkdir(parents=True, exist_ok=True)
                config = attempt / 'task.yaml'
                prepare_config(ROOT / task['config_path'], config, trials=1, scale=1., task=task,
                               num_envs=1, preserve_render_interval=True, save_trajectories=not args.preview_only)
                seed = (task_seed(args.seed, name) + (attempt_number - 1) * 1299709) % (2**31)
                command = [sys.executable, '-u', str(ROOT / 'scripts/run_benchmark_50.py'), '--worker',
                           '--config', str(config), '--task-id', name, '--robot', task['robot'],
                           '--env-task-name', task['env_task_name'], '--robot-cfg-name', task['robot_cfg_name'],
                           '--robot-type', task['robot_type'], '--trials', '1', '--num-envs', '1',
                           '--seed', str(seed), '--device', 'cuda:0',
                           '--preserve-render-interval', '--output-dir', str(attempt)]
                command.append('--preview-only' if args.preview_only else '--save-trajectories')
                print(f'RUN {name} robot={task["robot"]} attempt={attempt_number} seed={seed}', flush=True)
                with (attempt / 'run.log').open('w') as log:
                    process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                               env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'), start_new_session=True)
                    try:
                        returncode = process.wait(timeout=args.timeout)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
                        returncode = process.returncode
                    except BaseException:
                        os.killpg(process.pid, signal.SIGTERM)
                        raise
                report = load_report(report_path) or {'completed': False, 'error': 'worker did not produce a report'}
                report['returncode'] = returncode
                write_json(report_path, report)
            status, data_check = classify(report, preview_only=args.preview_only,
                                         allow_legacy_npy=args.allow_legacy_npy)
            if status == 'accepted' and not args.preview_only and not report['trials'][0].get('hdf5_path'):
                try:
                    verification = finalize_trajectory_hdf5(report['trials'][0], output,
                                                            allow_legacy_npy=args.allow_legacy_npy)
                    data_check = {**data_check, 'hdf5': verification,
                                  'hdf5_sha256': report['trials'][0]['hdf5_sha256']}
                except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
                    status = 'data_error'
                    data_check = {'valid': False, 'reason': f'HDF5 validation failed: {exc}'}
            update_report(report_path, report, status, data_check)
            if status == 'accepted' and not args.preview_only:
                # update_report persists the HDF5 checksum and verification
                # before the source is removed, allowing safe restart.
                sync_file_and_directory(report_path)
                sync_file_and_directory(Path(report['trials'][0]['directory']) / 'trial.json')
                cleanup_verified_temporary_npy(report['trials'][0], output)
            history[name].append({'attempt': attempt.name, 'status': status, 'report': str(report_path)})
            if status == 'accepted':
                trial = report['trials'][0]
                if args.preview_only:
                    accepted[name] = {'task_id': name, 'robot': task['robot'], 'attempt': attempt.name,
                                      'directory': trial['directory'], 'report': str(report_path)}
                else:
                    accepted[name] = {'task_id': name, 'robot': task['robot'], 'attempt': attempt.name,
                                  'trajectory': trial['hdf5_path'], 'hdf5_path': trial['hdf5_path'],
                                  'directory': trial['directory'],
                                  'report': str(report_path), 'data_integrity': data_check}
            consecutive_errors = consecutive_errors + 1 if status in ('error', 'data_error') else 0
            state = {'updated_at': datetime.now(timezone.utc).isoformat(), 'completed': len(accepted),
                     'total': len(tasks), 'accepted': accepted, 'history': history}
            status_file = output / ('preview_status.json' if args.preview_only else 'collection_status.json')
            accepted_file = output / ('preview_manifest.json' if args.preview_only else 'accepted_manifest.json')
            write_json(status_file, state)
            write_json(accepted_file, {'tasks': list(accepted.values())})
            gif = output / 'review/media' / name / attempt.name / 'trial_001/trajectory.gif'
            existing_gif_valid = valid_preview(gif) if args.preview_only and reused_report else False
            if should_refresh_review(preview_only=args.preview_only, reused_report=reused_report,
                                     initial_reviewed=initial_reviewed, gif_valid=existing_gif_valid):
                review(manifest, output)
            if args.preview_only and report.get('trials'):
                trial = report['trials'][0]
                gif_valid = existing_gif_valid or valid_preview(gif)
                if status == 'accepted' and not gif_valid:
                    accepted.pop(name, None)
                    status = 'preview_error'
                    history[name][-1]['status'] = status
                    state['completed'] = len(accepted)
                    print(f'PREVIEW_ERROR {name}: missing or invalid ten-second GIF', flush=True)
                # Old HDF5 attempts are left untouched. New preview attempts
                # retain only their GIF and small diagnostic metadata.
                temporary_frames = Path(trial['directory']) / 'original_images'
                initial_image = Path(trial['directory']) / 'initial.png'
                if (not report.get('save_trajectories', False) and gif_valid
                        and (temporary_frames.exists() or initial_image.exists())):
                    try:
                        cleanup = cleanup_preview_frames(trial, output)
                        print(f'PREVIEW_CLEANUP {name} {json.dumps(cleanup, ensure_ascii=False)}', flush=True)
                    except (OSError, ValueError) as exc:
                        print(f'WARNING preview frame cleanup failed: {exc}', flush=True)
                write_json(status_file, state)
                write_json(accepted_file, {'tasks': list(accepted.values())})
            print(f'RESULT {name} {status} accepted={len(accepted)}/{len(tasks)}', flush=True)
            if status == 'accepted' and not args.preview_only:
                try:
                    frame_cleanup = cleanup_verified_original_frames(report['trials'][0], report_path, output)
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    frame_cleanup = {'status': 'retained', 'reason': str(exc)}
                report['trials'][0]['raw_frame_cleanup'] = frame_cleanup
                accepted[name]['raw_frame_cleanup'] = frame_cleanup
                update_report(report_path, report, status, data_check)
                write_json(output / 'collection_status.json', state)
                write_json(output / 'accepted_manifest.json', {'tasks': list(accepted.values())})
                print(f'FRAME_CLEANUP {name} {json.dumps(frame_cleanup, ensure_ascii=False)}', flush=True)
            if consecutive_errors >= 3:
                raise RuntimeError('Three consecutive program/data errors; stop and inspect the shared cause')
    if args.preview_only:
        review(manifest, output)
    print(f'FINISHED accepted={len(accepted)}/{len(tasks)}; remaining={sorted(set(history)-set(accepted))}', flush=True)


if __name__ == '__main__':
    main()
