"""Merge accepted per-trajectory HDF5 files without changing their contents.

The output uses the ``traj_0/action``, ``traj_0/observations/...`` layout
read by this repository's raw ACT and BC datasets. One robot is selected at a
time: its original joint dimensions and action values are kept as recorded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import h5py
import numpy as np


def select_entries(manifest_path: Path, robot: str | None = None, task_ids: list[str] | None = None):
    manifest_path = Path(manifest_path).expanduser().resolve()
    raw = manifest_path.read_bytes()
    tasks = json.loads(raw).get('tasks')
    if not isinstance(tasks, list):
        raise ValueError('Accepted manifest must contain a tasks list')
    seen = set()
    for item in tasks:
        if not isinstance(item, dict) or not item.get('task_id') or not item.get('robot'):
            raise ValueError('Each accepted entry needs task_id and robot')
        if item['task_id'] in seen:
            raise ValueError(f"Duplicate task_id: {item['task_id']}")
        seen.add(item['task_id'])
    if task_ids:
        if len(set(task_ids)) != len(task_ids):
            raise ValueError('Task selection contains duplicates')
        unknown = set(task_ids) - seen
        if unknown:
            raise ValueError(f'Unknown accepted task(s): {sorted(unknown)}')
        chosen = [entry for entry in tasks if entry['task_id'] in set(task_ids)]
        if robot and any(entry['robot'] != robot for entry in chosen):
            raise ValueError('Selected task(s) include a different robot than --robot')
    else:
        chosen = list(tasks)
        if robot:
            chosen = [entry for entry in chosen if entry['robot'] == robot]
    if not chosen:
        raise ValueError('No accepted trajectories match the selection')
    robots = {entry['robot'] for entry in chosen}
    if len(robots) != 1:
        raise ValueError(f'Mixed robots in selection: {sorted(robots)}; choose one using --robot')
    robot = next(iter(robots))
    result = []
    seen_paths = set()
    for item in chosen:
        integrity = item.get('data_integrity')
        if isinstance(integrity, dict) and integrity.get('valid') is False:
            raise ValueError(f"Entry is not marked valid: {item['task_id']}")
        value = item.get('hdf5_path') or item.get('trajectory')
        if not value:
            raise ValueError(f"Missing finalized HDF5 path: {item['task_id']}")
        path = Path(value).expanduser()
        path = (path if path.is_absolute() else manifest_path.parent / path).resolve()
        if not path.is_file() or not h5py.is_hdf5(path):
            raise ValueError(f"Accepted trajectory must reference a finalized HDF5 file: {path}")
        if path in seen_paths:
            raise ValueError(f'Duplicate HDF5 source: {path}')
        seen_paths.add(path)
        result.append({'task_id': item['task_id'], 'robot': robot, 'path': path})
    return result, hashlib.sha256(raw).hexdigest()


def inspect_source(entry):
    path = entry['path']
    with h5py.File(path, 'r') as source:
        try:
            action = source['action']
            qpos = source['observations/qpos']
            qvel = source['observations/qvel']
            front = source['observations/images/front']
            wrist = source['observations/images/wrist']
        except KeyError as exc:
            raise ValueError(f"{entry['task_id']} lacks a required ACT/BC field: {exc}") from exc
        if action.ndim != 2 or qpos.ndim != 2 or action.shape[0] == 0 or qpos.shape[1] == 0:
            raise ValueError(f"{entry['task_id']} needs nonempty 2-D action and qpos arrays")
        n = action.shape[0]
        if front.ndim != 4 or wrist.ndim != 4 or front.shape[-1] != 3 or wrist.shape[-1] != 3:
            raise ValueError(f"{entry['task_id']} images must be [T, H, W, 3]")
        if qpos.shape[0] != n or qvel.shape != qpos.shape or front.shape[0] != n or wrist.shape[0] != n:
            raise ValueError(f"{entry['task_id']} has mismatched action, state, or image dimensions")
        if front.dtype != np.uint8 or wrist.dtype != np.uint8:
            raise ValueError(f"{entry['task_id']} training images must be uint8 RGB")
        convention = source.attrs.get('action_convention')
        if isinstance(convention, bytes):
            convention = convention.decode('utf-8')
        if not isinstance(convention, str) or not convention:
            raise ValueError(f"{entry['task_id']} has no declared action convention")
        joint_names = None
        if 'recordings/trajectory_fields/arm_joint_names' in source:
            joint_names = tuple(source['recordings/trajectory_fields/arm_joint_names'].asstr()[...].tolist())
            if len(joint_names) != qpos.shape[1] or len(set(joint_names)) != len(joint_names):
                raise ValueError(f"{entry['task_id']} has invalid arm joint names")
        return {'qpos_dim': qpos.shape[1], 'action_dim': action.shape[1], 'convention': convention,
                'front_shape': front.shape[1:], 'wrist_shape': wrist.shape[1:],
                'joint_names': joint_names, 'steps': n}


def merge_manifest(manifest_path: Path, output: Path, *, robot: str | None = None,
                   task_ids: list[str] | None = None):
    output = Path(output).expanduser()
    output = output.parent.resolve() / output.name
    if output.exists() or output.is_symlink():
        raise FileExistsError(f'Refusing to overwrite existing output: {output}')
    entries, manifest_sha = select_entries(manifest_path, robot, task_ids)
    if any(entry['path'] == output for entry in entries):
        raise ValueError('Output must differ from every input')
    schema = [inspect_source(entry) for entry in entries]
    expected = schema[0]
    for entry, item in zip(entries[1:], schema[1:]):
        for field in ('qpos_dim', 'action_dim', 'convention', 'front_shape', 'wrist_shape'):
            if item[field] != expected[field]:
                raise ValueError(f"{entry['task_id']} has incompatible {field}: {item[field]!r}, "
                                 f"expected {expected[field]!r}; trajectories are not padded or reinterpreted")
    named = [item['joint_names'] for item in schema if item['joint_names'] is not None]
    if named and any(names != named[0] for names in named):
        raise ValueError('Selected trajectories have different arm joint orderings')
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f'.{output.name}.', suffix='.tmp', dir=output.parent)
    os.close(fd)
    try:
        with h5py.File(temporary, 'w') as target:
            target.attrs['schema'] = 'affordsim_raw_grouped_v1'
            target.attrs['robot'] = entries[0]['robot']
            target.attrs['manifest_sha256'] = manifest_sha
            target.attrs['joint_order_verified_for_all'] = len(named) == len(entries)
            target.attrs['sources'] = json.dumps([
                {'task_id': entry['task_id'], 'hdf5_path': str(entry['path'])} for entry in entries])
            for index, entry in enumerate(entries):
                before = entry['path'].stat()
                with h5py.File(entry['path'], 'r') as source:
                    group = target.create_group(f'traj_{index}')
                    for key in source.keys():
                        source.copy(key, group, name=key)
                    for key, value in source.attrs.items():
                        group.attrs[key] = value
                after = entry['path'].stat()
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise RuntimeError(f"Source changed while merging: {entry['path']}")
        # Link is atomic and cannot replace an output created by another process.
        os.link(temporary, output)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return {'output': str(output), 'robot': entries[0]['robot'], 'trajectories': len(entries),
            'qpos_dim': expected['qpos_dim'], 'action_dim': expected['action_dim'],
            'joint_order_verified_for_all': len(named) == len(entries),
            'task_ids': [entry['task_id'] for entry in entries]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--manifest', type=Path, help='Path to accepted_manifest.json')
    source.add_argument('--episode-dir', type=Path, help='Directory of directly recorded episode_*.hdf5 files')
    parser.add_argument('--output', type=Path, required=True, help='New grouped HDF5 file (must not exist)')
    parser.add_argument('--robot', help='Select one robot, e.g. franka_long, ur5e, kinova, sawyer, xarm7')
    parser.add_argument('--task', dest='tasks', action='append', help='Include a specific task; repeat as needed')
    args = parser.parse_args()
    if args.episode_dir:
        if not args.robot or args.tasks:
            parser.error('--episode-dir requires --robot and does not accept --task')
        paths = sorted(args.episode_dir.expanduser().resolve().glob('episode_*.hdf5'))
        if not paths:
            parser.error('No episode_*.hdf5 files found')
        # Reuse the same robot/joint/action compatibility checks as manifests.
        with tempfile.TemporaryDirectory(prefix='affordsim_merge_') as tempdir:
            manifest = Path(tempdir) / 'manifest.json'
            manifest.write_text(json.dumps({'tasks': [
                {'task_id': path.stem, 'robot': args.robot, 'hdf5_path': str(path)} for path in paths]}))
            result = merge_manifest(manifest, args.output, robot=args.robot)
    else:
        result = merge_manifest(args.manifest, args.output, robot=args.robot, task_ids=args.tasks)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
