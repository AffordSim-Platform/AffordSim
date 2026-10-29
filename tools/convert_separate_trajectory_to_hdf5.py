"""Convert frame-separated AffordSim trajectories to the legacy episode HDF5 layout.

The saved source_frame_index selects observations; no motion, action, gripper,
state, or timestamp values are recomputed. Original RGB frames are center-cropped
and resized using the same helper as tools/np2hdf5.py.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import tempfile

import h5py
import numpy as np
from PIL import Image

from np2hdf5 import NPY_IMG_SHAPE, TARGET_IMG_SIZE, reshape_and_resize_images


FRAME_NAME = re.compile(r'^step_(\d+)\.(png|webp)$', re.IGNORECASE)
TASK_NAME = re.compile(r'^[A-Za-z0-9_-]+$')


def frame_map(directory: Path) -> dict[int, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f'Missing camera frame directory: {directory}')
    result = {}
    for path in directory.iterdir():
        match = FRAME_NAME.fullmatch(path.name)
        if not match or not path.is_file():
            continue
        index = int(match.group(1))
        if index in result:
            raise ValueError(f'Ambiguous duplicate frame {index}: {result[index]} and {path}')
        result[index] = path
    return result


def read_resized_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        rgb = np.asarray(image.convert('RGB'))
    expected_shape = (NPY_IMG_SHAPE[1], NPY_IMG_SHAPE[2], NPY_IMG_SHAPE[0])
    if rgb.shape != expected_shape:
        raise ValueError(f'Expected RGB image shape {expected_shape}, got {rgb.shape}: {path}')
    # np2hdf5 expects flattened CHW input. Reuse the original implementation so
    # crop coordinates, resize interpolation and resulting RGB layout match it.
    chw = rgb.transpose(2, 0, 1)
    return reshape_and_resize_images(chw[None], NPY_IMG_SHAPE, TARGET_IMG_SIZE)[0]


def load_inputs(trajectory: Path):
    original_npy = trajectory.read_bytes()
    data = np.load(io.BytesIO(original_npy), allow_pickle=True).item()
    if not isinstance(data, dict):
        raise ValueError(f'Trajectory must contain a dictionary: {trajectory}')
    convention = data.get('action_convention', data.get('transition_action_convention'))
    if not isinstance(convention, str) or not convention:
        raise ValueError(f'Missing action convention: {convention!r}')
    required = ['source_frame_index', 'next_frame_index', 'obs_policy_actions', 'obs_policy_q_pos', 'obs_policy_q_vel']
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f'Missing trajectory fields: {missing}')
    indices = np.asarray(data['source_frame_index'])
    if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer) or np.any(indices < 0):
        raise ValueError('source_frame_index must be a one-dimensional nonnegative integer array')
    if not len(indices):
        raise ValueError('Trajectory contains no transitions')
    next_indices = np.asarray(data['next_frame_index'])
    if (next_indices.shape != indices.shape or not np.issubdtype(next_indices.dtype, np.integer)
            or np.any(next_indices <= indices)):
        raise ValueError('next_frame_index must have one greater, integral index per source frame')
    arrays = {out: np.asarray(data[key]) for out, key in [
        ('action', 'obs_policy_actions'), ('qpos', 'obs_policy_q_pos'), ('qvel', 'obs_policy_q_vel')]}
    for name, array in arrays.items():
        if array.ndim != 2 or len(array) != len(indices) or not np.issubdtype(array.dtype, np.number):
            raise ValueError(f'{name} must be a numeric [T, D] array with T={len(indices)}; got {array.shape}, {array.dtype}')
        if not np.isfinite(array).all():
            raise ValueError(f'{name} contains non-finite values')
    image_root = trajectory.parent / 'original_images'
    cameras = {'front': frame_map(image_root / 'external_cam_side'),
               'wrist': frame_map(image_root / 'wrist_cam')}
    for name, mapping in cameras.items():
        missing = sorted((set(map(int, indices)) | set(map(int, next_indices))) - mapping.keys())
        if missing:
            raise FileNotFoundError(f'{name} camera is missing referenced frame indices: {missing[:12]}')
    return indices, next_indices, arrays, cameras, original_npy, convention


def write_trajectory_fields(group: h5py.Group, original_npy: bytes):
    """Expose ordinary source fields while retaining the original NPY byte-for-byte."""
    source = np.load(io.BytesIO(original_npy), allow_pickle=True).item()
    omitted = []
    for key, value in source.items():
        array = np.asarray(value)
        if not isinstance(key, str) or not key or '/' in key or array.dtype.kind not in 'biufcSU':
            omitted.append(str(key))
            continue  # Still preserved exactly in recordings/trajectory_npy.
        kwargs = {'compression': 'gzip'} if array.ndim and array.size else {}
        if array.dtype.kind == 'U':
            if array.ndim == 0:
                dataset = group.create_dataset(key, data=str(array.item()), dtype=h5py.string_dtype('utf-8'))
            else:
                dataset = group.create_dataset(key, data=array.astype(object), dtype=h5py.string_dtype('utf-8'), **kwargs)
        else:
            dataset = group.create_dataset(key, data=array, **kwargs)
        dataset.attrs['original_numpy_dtype'] = str(array.dtype)
    group.attrs['fields_available_only_in_original_npy'] = json.dumps(omitted)


def verify_trajectory_hdf5(trajectory: Path, hdf5_path: Path):
    """Verify numerical values, training images, and every lossless source byte.

    Raise on any discrepancy; return a compact validation record on success.
    The source NPY and its original_images directories must still be present.
    """
    trajectory, hdf5_path = Path(trajectory), Path(hdf5_path)
    indices, next_indices, arrays, cameras, original_npy, convention = load_inputs(trajectory)
    with h5py.File(hdf5_path, 'r') as data:
        for name, path in [('action', 'action'), ('qpos', 'observations/qpos'), ('qvel', 'observations/qvel')]:
            stored = data[path][...]
            if stored.dtype != arrays[name].dtype or not np.array_equal(stored, arrays[name]):
                raise ValueError(f'HDF5 {name} values or dtype differ from source')
        if data.attrs['action_convention'] != convention:
            raise ValueError('HDF5 action convention differs from source')
        if data['recordings/trajectory_npy'][...].tobytes() != original_npy:
            raise ValueError('Embedded original NPY bytes differ from source')
        original_fields = np.load(io.BytesIO(original_npy), allow_pickle=True).item()
        field_group = data['recordings/trajectory_fields']
        for key, dataset in field_group.items():
            original = np.asarray(original_fields[key])
            if dataset.attrs['original_numpy_dtype'] != str(original.dtype):
                raise ValueError(f'{key} original dtype metadata differs')
            value = dataset.asstr()[...] if original.dtype.kind == 'U' else dataset[...]
            if not np.array_equal(value, original):
                raise ValueError(f'{key} recorded trajectory field differs')
        if not np.array_equal(data['recordings/source_frame_index'][...], indices):
            raise ValueError('Embedded source frame indices differ')
        if not np.array_equal(data['recordings/next_frame_index'][...], next_indices):
            raise ValueError('Embedded next frame indices differ')
        for name, mapping in cameras.items():
            camera = 'external_cam_side' if name == 'front' else 'wrist_cam'
            group = data[f'recordings/cameras/{camera}']
            sorted_indices = sorted(mapping)
            if not np.array_equal(group['frame_index'][...], sorted_indices):
                raise ValueError(f'{camera} recorded frame index list differs')
            filenames = group['file_name'].asstr()[...]
            if list(filenames) != [mapping[i].name for i in sorted_indices]:
                raise ValueError(f'{camera} original frame filenames differ')
            for row, index in enumerate(sorted_indices):
                if group['encoded_bytes'][row].tobytes() != mapping[index].read_bytes():
                    raise ValueError(f'{camera} original frame bytes differ at index {index}')
            for row, index in enumerate(indices):
                if not np.array_equal(data[f'observations/images/{name}'][row], read_resized_rgb(mapping[int(index)])):
                    raise ValueError(f'{camera} selected/resized image differs at transition {row}')
    return {'valid': True, 'transitions': len(indices),
            'source_npy_sha256': hashlib.sha256(original_npy).hexdigest(),
            'camera_frames': {name: len(mapping) for name, mapping in cameras.items()}}


def convert_trajectory(trajectory: Path, destination: Path):
    trajectory, destination = trajectory.expanduser().resolve(), destination.expanduser().resolve()
    if not trajectory.is_file():
        raise FileNotFoundError(trajectory)
    if trajectory.suffix.lower() != '.npy':
        raise ValueError(f'Expected the temporary trajectory.npy source, got: {trajectory}')
    if destination == trajectory or destination.exists():
        raise FileExistsError(f'Refusing to overwrite input or existing output: {destination}')
    indices, next_indices, arrays, cameras, original_npy, convention = load_inputs(trajectory)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f'.{destination.name}.', suffix='.tmp', dir=destination.parent)
    os.close(fd)
    try:
        with h5py.File(temporary, 'w') as out:
            # Preserve numeric dtype and values exactly, including action precision.
            out.create_dataset('action', data=arrays['action'], compression='gzip')
            observations = out.create_group('observations')
            observations.create_dataset('qpos', data=arrays['qpos'], compression='gzip')
            observations.create_dataset('qvel', data=arrays['qvel'], compression='gzip')
            images = observations.create_group('images')
            height, width = TARGET_IMG_SIZE
            datasets = {name: images.create_dataset(name, shape=(len(indices), height, width, 3),
                                                     dtype=np.uint8, chunks=(1, height, width, 3),
                                                     compression='gzip') for name in cameras}
            for row, index in enumerate(indices):
                for name, mapping in cameras.items():
                    datasets[name][row] = read_resized_rgb(mapping[int(index)])
            recordings = out.create_group('recordings')
            recordings.create_dataset('trajectory_npy', data=np.frombuffer(original_npy, dtype=np.uint8), compression='gzip')
            write_trajectory_fields(recordings.create_group('trajectory_fields'), original_npy)
            recordings.create_dataset('source_frame_index', data=indices, compression='gzip')
            recordings.create_dataset('next_frame_index', data=next_indices, compression='gzip')
            original_cameras = recordings.create_group('cameras')
            for name, mapping in cameras.items():
                camera = 'external_cam_side' if name == 'front' else 'wrist_cam'
                group = original_cameras.create_group(camera)
                ordered = sorted(mapping)
                group.create_dataset('frame_index', data=np.asarray(ordered, dtype=np.int64))
                group.create_dataset('file_name', data=[mapping[i].name for i in ordered],
                                     dtype=h5py.string_dtype(encoding='utf-8'))
                encoded = group.create_dataset('encoded_bytes', shape=(len(ordered),), dtype=h5py.vlen_dtype(np.dtype('uint8')))
                for row, index in enumerate(ordered):
                    encoded[row] = np.frombuffer(mapping[index].read_bytes(), dtype=np.uint8)
            out.attrs['action_convention'] = convention
            out.attrs['action_source'] = 'obs_policy_actions'
            out.attrs['image_alignment'] = 'source_frame_index'
            out.attrs['image_color_order'] = 'RGB'
        verification = verify_trajectory_hdf5(trajectory, Path(temporary))
        # Atomic no-overwrite publication; an existing result is never replaced.
        os.link(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return {'trajectory': str(trajectory), 'hdf5': str(destination), 'transitions': len(indices),
            'action_dtype': str(arrays['action'].dtype), 'verification': verification}


def convert_manifest(manifest: Path, output_dir: Path, skip_existing: bool):
    data = json.loads(manifest.read_text())
    tasks = data.get('tasks')
    if not isinstance(tasks, list):
        raise ValueError('Accepted manifest must contain a tasks list')
    seen = set()
    failures = []
    for task in tasks:
        name = task.get('task_id', '')
        if not TASK_NAME.fullmatch(name) or name in seen:
            raise ValueError(f'Invalid or duplicate task_id in accepted manifest: {name!r}')
        seen.add(name)
        output = output_dir / name / 'episode_0.hdf5'
        if output.exists() and skip_existing:
            print(f'SKIP {name}: {output}', flush=True)
            continue
        try:
            source = Path(task['trajectory'])
            if not source.is_absolute():
                source = manifest.parent / source
            result = convert_trajectory(source, output)
            print(f'CONVERTED {name}: {result["transitions"]} transitions -> {output}', flush=True)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            failures.append((name, str(exc)))
            print(f'FAILED {name}: {exc}', flush=True)
    if failures:
        raise RuntimeError(f'{len(failures)} task(s) failed conversion; inspect FAILED lines above')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--trajectory', type=Path, help='Path to one trajectory.npy')
    source.add_argument('--accepted-manifest', type=Path, help='Collection accepted_manifest.json')
    parser.add_argument('--output', type=Path, help='HDF5 path for one trajectory')
    parser.add_argument('--output-dir', type=Path, help='Batch output root: TASK/episode_0.hdf5')
    parser.add_argument('--skip-existing', action='store_true', help='Skip existing batch outputs without validating them')
    args = parser.parse_args()
    if args.trajectory:
        if args.output is None or args.output_dir is not None or args.skip_existing:
            parser.error('--trajectory requires --output; --output-dir and --skip-existing apply only to batch mode')
        print(json.dumps(convert_trajectory(args.trajectory, args.output), indent=2))
    else:
        if args.output_dir is None or args.output is not None:
            parser.error('--accepted-manifest requires --output-dir, not --output')
        convert_manifest(args.accepted_manifest.expanduser().resolve(), args.output_dir.expanduser().resolve(), args.skip_existing)


if __name__ == '__main__':
    main()
