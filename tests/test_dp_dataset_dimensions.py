"""Validate real joint dimensions across DP training and validation data."""
import importlib.util
from pathlib import Path
import sys

import h5py
import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[1] / 'scripts/imitation_learning/dp/dataset.py'
spec = importlib.util.spec_from_file_location('dp_dataset_dimensions', PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def make_dataset(path, widths, *, has_qvel=True, action_widths=None, split=None):
    with h5py.File(path, 'w') as f:
        root = f.create_group('data')
        for i, width in enumerate(widths):
            demo = root.create_group(f'demo_{i}')
            demo.create_dataset('actions', data=np.zeros((5, (action_widths or [7]*len(widths))[i])))
            obs = demo.create_group('obs')
            obs.create_dataset('qpos', data=np.zeros((5, width)))
            if has_qvel:
                obs.create_dataset('qvel', data=np.ones((5, width)))
            obs.create_dataset('front', data=np.zeros((5, 8, 8, 3), dtype=np.uint8))
            obs.create_dataset('wrist', data=np.zeros((5, 8, 8, 3), dtype=np.uint8))
        if split is not None:
            mask = f.create_group('mask')
            for name, indices in split.items():
                mask.create_dataset(name, data=np.array([f'demo_{i}'.encode() for i in indices], dtype='S16'))


@pytest.mark.parametrize('joint_dim', [6, 7])
@pytest.mark.parametrize('use_qvel', [True, False, None])
def test_dataset_infers_actual_lowdim_without_padding(tmp_path, joint_dim, use_qvel):
    path = tmp_path / 'dataset.hdf5'; make_dataset(path, [joint_dim])
    ds = module.DiffusionPolicyDataset(str(path), pred_horizon=2, obs_horizon=2,
                                       filter_key=None, use_qvel=use_qvel)
    assert ds.qpos_dim == joint_dim
    assert ds.action_dim == 7
    assert ds.lowdim_dim == joint_dim * (1 if use_qvel is False else 2)
    assert ds[0]['obs_lowdim'].shape == (2, ds.lowdim_dim)
    assert ds[0]['actions'].shape == (2, 7)
    ds.close()


def test_missing_qvel_is_inferred_without_assuming_franka(tmp_path):
    path = tmp_path / 'no_vel.hdf5'; make_dataset(path, [6], has_qvel=False)
    ds = module.DiffusionPolicyDataset(str(path), pred_horizon=2, obs_horizon=2, filter_key=None)
    assert not ds.use_qvel and ds.lowdim_dim == 6
    ds.close()
    with pytest.raises(ValueError, match='lacks obs/qvel'):
        module.DiffusionPolicyDataset(str(path), pred_horizon=2, obs_horizon=2,
                                      filter_key=None, use_qvel=True)


@pytest.mark.parametrize(('widths', 'actions'), [([6, 7], [7, 7]), ([7, 7], [7, 8])])
def test_incompatible_demonstrations_fail_instead_of_being_padded(tmp_path, widths, actions):
    path = tmp_path / 'mixed.hdf5'; make_dataset(path, widths, action_widths=actions)
    with pytest.raises(ValueError, match='Inconsistent qpos/action dimensions'):
        module.DiffusionPolicyDataset(str(path), pred_horizon=2, obs_horizon=2, filter_key=None)


def test_train_and_validation_splits_are_checked_against_each_other(tmp_path):
    path = tmp_path / 'split.hdf5'; make_dataset(path, [6, 7], split={'train': [0], 'valid': [1]})
    train = module.DiffusionPolicyDataset(str(path), pred_horizon=2, obs_horizon=2, filter_key='train')
    valid = module.DiffusionPolicyDataset(str(path), pred_horizon=2, obs_horizon=2, filter_key='valid')
    with pytest.raises(ValueError, match='Train and validation dataset dimensions differ'):
        module.validate_dataset_compatibility(train, valid)
    train.close(); valid.close()


def test_automatic_qvel_rejects_inconsistent_availability(tmp_path):
    path = tmp_path / 'mixed_vel.hdf5'; make_dataset(path, [7, 7])
    with h5py.File(path, 'a') as f:
        del f['data/demo_1/obs/qvel']
    with pytest.raises(ValueError, match='inconsistent qvel availability'):
        module.DiffusionPolicyDataset(str(path), pred_horizon=2, obs_horizon=2, filter_key=None)
    ds = module.DiffusionPolicyDataset(str(path), pred_horizon=2, obs_horizon=2,
                                       filter_key=None, use_qvel=False)
    assert ds.lowdim_dim == 7
    ds.close()


def test_qpos_qvel_dimensions_must_match(tmp_path):
    path = tmp_path / 'invalid_vel.hdf5'; make_dataset(path, [6])
    with h5py.File(path, 'a') as f:
        del f['data/demo_0/obs/qvel']
        f['data/demo_0/obs'].create_dataset('qvel', data=np.zeros((5, 7)))
    with pytest.raises(ValueError, match='qvel shape'):
        module.DiffusionPolicyDataset(str(path), pred_horizon=2, obs_horizon=2, filter_key=None)


def test_empty_validation_split_keeps_training_schema(tmp_path):
    path = tmp_path / 'empty_valid.hdf5'; make_dataset(path, [6], split={'train': [0], 'valid': []})
    train = module.DiffusionPolicyDataset(str(path), pred_horizon=2, obs_horizon=2, filter_key='train')
    valid = module.DiffusionPolicyDataset(str(path), pred_horizon=2, obs_horizon=2, filter_key='valid')
    module.validate_dataset_compatibility(train, valid)
    assert valid.num_demos == 0 and len(valid) == 0
    train.close(); valid.close()


@pytest.mark.parametrize('joint_dim', [6, 7])
def test_inferred_dimensions_feed_actual_diffusion_policy_loss(tmp_path, joint_dim):
    import torch
    from torch.utils.data import DataLoader

    policy_path = str(PATH.parent)
    sys.path.insert(0, policy_path)
    try:
        from policy import DiffusionPolicy

        path = tmp_path / 'dataset.hdf5'; make_dataset(path, [joint_dim])
        ds = module.DiffusionPolicyDataset(str(path), pred_horizon=2, obs_horizon=2, filter_key=None)
        batch = next(iter(DataLoader(ds, batch_size=1)))
        model = DiffusionPolicy(action_dim=ds.action_dim, lowdim_dim=ds.lowdim_dim,
                                pred_horizon=2, obs_horizon=2, img_feature_dim=16,
                                lowdim_feature_dim=8, resize_shape=(32, 32), crop_shape=(32, 32),
                                pretrained_resnet=False, image_aug=False, down_dims=(16, 32),
                                diffusion_step_embed_dim=32, num_train_timesteps=8)
        old_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            loss = model.compute_loss(batch)
            assert loss.ndim == 0 and torch.isfinite(loss)
        finally:
            torch.set_num_threads(old_threads)
        ds.close()
    finally:
        sys.path.remove(policy_path)
