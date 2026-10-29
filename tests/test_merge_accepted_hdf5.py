"""Grouped HDF5 merge preserves accepted data and refuses unsafe mixing."""
import importlib.util
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('merge_accepted', ROOT / 'tools/merge_accepted_hdf5.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def source_file(path, joint_dim=6, *, joint_names=None, convention='legacy_ee_delta_source_gripper'):
    n = 5
    actions = np.arange(n * 7, dtype=np.float64).reshape(n, 7) / 1003
    actions[:, -1] = [1, 1, -1, -1, 1]
    with h5py.File(path, 'w') as h:
        h.attrs['action_convention'] = convention
        h.attrs['image_alignment'] = 'source_frame_index'
        h.create_dataset('action', data=actions, compression='gzip')
        obs = h.create_group('observations')
        obs.create_dataset('qpos', data=np.arange(n * joint_dim, dtype=np.float32).reshape(n, joint_dim))
        obs.create_dataset('qvel', data=np.arange(n * joint_dim, dtype=np.float32).reshape(n, joint_dim) / 10)
        imgs = obs.create_group('images')
        imgs.create_dataset('front', data=np.arange(n * 4 * 4 * 3, dtype=np.uint8).reshape(n, 4, 4, 3))
        imgs.create_dataset('wrist', data=np.full((n, 4, 4, 3), 73, dtype=np.uint8))
        extra = h.create_group('recordings')
        extra.create_dataset('trajectory_npy', data=np.array([0, 1, 253, 255], dtype=np.uint8))
        extra.attrs['note'] = 'preserve original bytes'
        if joint_names is not None:
            fields = extra.create_group('trajectory_fields')
            fields.create_dataset('arm_joint_names', data=joint_names, dtype=h5py.string_dtype())
    return actions


def manifest_file(path, entries):
    path.write_text(json.dumps({'tasks': entries}))
    return path


def entry(task_id, robot, source):
    return {'task_id': task_id, 'robot': robot, 'hdf5_path': str(source), 'data_integrity': {'valid': True}}


def compare_all_fields(src, dst):
    assert dict(src.attrs) == dict(dst.attrs)
    assert set(src.keys()) == set(dst.keys())
    for key in src.keys():
        left, right = src[key], dst[key]
        if isinstance(left, h5py.Group):
            compare_all_fields(left, right)
        else:
            assert left.dtype == right.dtype
            assert left.shape == right.shape
            assert left.compression == right.compression
            np.testing.assert_array_equal(left[...], right[...])
            assert dict(left.attrs) == dict(right.attrs)


def test_select_robot_merge_all_fields_and_preserve_action_values(tmp_path):
    a, b, other = (tmp_path / name for name in ('a.hdf5', 'b.hdf5', 'other.hdf5'))
    names = [f'joint_{i}' for i in range(6)]
    expected = source_file(a, 6, joint_names=names)
    source_file(b, 6, joint_names=names)
    source_file(other, 7)
    manifest = manifest_file(tmp_path / 'accepted_manifest.json', [
        entry('one', 'ur5e', a), entry('two', 'ur5e', b), entry('three', 'kinova', other)])
    output = tmp_path / 'grouped.hdf5'
    result = module.merge_manifest(manifest, output, robot='ur5e')
    assert result['task_ids'] == ['one', 'two']
    assert result['qpos_dim'] == 6 and result['joint_order_verified_for_all']
    with h5py.File(output, 'r') as h, h5py.File(a, 'r') as s1, h5py.File(b, 'r') as s2:
        assert set(h.keys()) == {'traj_0', 'traj_1'}
        compare_all_fields(s1, h['traj_0'])
        compare_all_fields(s2, h['traj_1'])
        np.testing.assert_array_equal(h['traj_0/action'][...], expected)
        assert h['traj_0/action'].dtype == expected.dtype
        assert h.attrs['robot'] == 'ur5e'


def test_mixed_robots_are_rejected_unless_an_explicit_robot_is_selected(tmp_path):
    a, b = tmp_path / 'a.hdf5', tmp_path / 'b.hdf5'
    source_file(a); source_file(b)
    manifest = manifest_file(tmp_path / 'manifest.json', [entry('one', 'ur5e', a), entry('two', 'kinova', b)])
    with pytest.raises(ValueError, match='Mixed robots'):
        module.merge_manifest(manifest, tmp_path / 'mix.hdf5')
    with pytest.raises(ValueError, match='different robot'):
        module.merge_manifest(manifest, tmp_path / 'mix.hdf5', robot='ur5e', task_ids=['one', 'two'])
    assert not (tmp_path / 'mix.hdf5').exists()


@pytest.mark.parametrize('incompatibility', ['dimension', 'joint_order', 'convention'])
def test_incompatible_trajectories_fail_without_output(tmp_path, incompatibility):
    a, b = tmp_path / 'a.hdf5', tmp_path / 'b.hdf5'
    source_file(a, 6, joint_names=[f'j{i}' for i in range(6)])
    dim = 7 if incompatibility == 'dimension' else 6
    names = [f'j{i}' for i in range(dim)]
    if incompatibility == 'joint_order':names.reverse()
    source_file(b, dim, joint_names=names, convention='other' if incompatibility == 'convention' else 'legacy_ee_delta_source_gripper')
    manifest = manifest_file(tmp_path / 'manifest.json', [entry('one', 'ur5e', a), entry('two', 'ur5e', b)])
    with pytest.raises(ValueError, match={'dimension': 'qpos_dim', 'joint_order': 'joint orderings', 'convention': 'convention'}[incompatibility]):
        module.merge_manifest(manifest, tmp_path / 'invalid.hdf5', robot='ur5e')
    assert not (tmp_path / 'invalid.hdf5').exists()


def test_existing_output_is_never_modified(tmp_path):
    a = tmp_path / 'a.hdf5'; source_file(a)
    manifest = manifest_file(tmp_path / 'manifest.json', [entry('one', 'ur5e', a)])
    output = tmp_path / 'existing.hdf5'; output.write_bytes(b'user-owned output')
    with pytest.raises(FileExistsError):
        module.merge_manifest(manifest, output)
    assert output.read_bytes() == b'user-owned output'


def test_dangling_output_symlink_is_never_followed(tmp_path):
    source = tmp_path / 'a.hdf5'; source_file(source)
    manifest = manifest_file(tmp_path / 'manifest.json', [entry('one', 'ur5e', source)])
    output = tmp_path / 'symlink.hdf5'; output.symlink_to(tmp_path / 'missing.hdf5')
    with pytest.raises(FileExistsError):
        module.merge_manifest(manifest, output)
    assert output.is_symlink() and not (tmp_path / 'missing.hdf5').exists()


def test_incompatible_image_dimensions_are_rejected(tmp_path):
    a, b = tmp_path / 'a.hdf5', tmp_path / 'b.hdf5'
    source_file(a); source_file(b)
    with h5py.File(b, 'a') as h:
        del h['observations/images/front']
        h['observations/images'].create_dataset('front', data=np.zeros((5, 5, 4, 3), dtype=np.uint8))
    manifest = manifest_file(tmp_path / 'manifest.json', [entry('one', 'ur5e', a), entry('two', 'ur5e', b)])
    with pytest.raises(ValueError, match='front_shape'):
        module.merge_manifest(manifest, tmp_path / 'bad.hdf5')
    assert not (tmp_path / 'bad.hdf5').exists()


def test_grouped_output_is_readable_by_existing_act_and_bc_loaders(tmp_path):
    source = tmp_path / 'source.hdf5'; source_file(source, 6)
    manifest = manifest_file(tmp_path / 'manifest.json', [entry('one', 'ur5e', source)])
    output = tmp_path / 'grouped.hdf5'; module.merge_manifest(manifest, output)
    for subdir, filename, klass in [('ACT', 'train_act.py', 'ACTDataset'), ('BC', 'train_bc.py', 'BCDataset')]:
        directory = ROOT / 'scripts/imitation_learning' / subdir
        sys.path.insert(0, str(directory))
        try:
            spec = importlib.util.spec_from_file_location(f'test_{subdir}_loader', directory / filename)
            training_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(training_module)
            dataset = getattr(training_module, klass)(str(output), remove_idle=False, action_chunk_size=1)
            assert len(dataset) == 5
            assert dataset[0]['state'].shape == (12,)  # Real UR 6D qpos + 6D qvel.
            assert dataset[0]['action'].shape == (1, 7)
        finally:
            sys.path.remove(str(directory))
