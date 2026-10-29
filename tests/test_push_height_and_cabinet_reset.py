"""Verify contact-height consistency and clean articulation resets on CPU."""
import ast
from pathlib import Path
import runpy
from types import SimpleNamespace
from typing import Optional
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from isaaclab.motion_collect.primitives.push import primitive_compute_push_pose

ROOT = Path(__file__).resolve().parents[1]


def test_lowered_push_keeps_approach_contact_and_sweep_at_consistent_heights():
    origins = np.array([[0., 0., 0.], [10., -5., 0.]])
    planned = []
    candidate = dict(candidate_index=0, grasp_center=np.array([0., 0., .04]),
                     rotation_matrix=np.eye(3), approach_dir=np.array([0., 1., 0.]))
    planner = SimpleNamespace(plan_to_pose=lambda **kw: (planned.append(kw['target_pos'].copy()), True))
    collector = SimpleNamespace(
        num_envs=2, task_config=None, motion_planner=planner,
        _get_env_origins=lambda: origins,
        get_current_ee_pose=lambda: (np.array([[.3, 0., .4]] * 2), np.array([[0., 1., 0., 0.]] * 2)),
        _get_object_pose_global=lambda name, i: (origins[i] + [.5, 0., .095], np.array([1., 0., 0., 0.])),
        _update_obstacles_for_env=lambda i: None,
        robot=SimpleNamespace(joint_names=['joint'], data=SimpleNamespace(
            joint_pos=torch.zeros((2, 1)), joint_vel=torch.zeros((2, 1)))),
    )
    params = dict(target='can', push_target=[.5, .3, .095], lift_z_delta=.15)
    baseline, lowered = {}, {}
    with patch('isaaclab.motion_collect.primitives.push.resolve_affordance_path', return_value='fixture.npz'), \
         patch('isaaclab.motion_collect.primitives.push.load_push_affordance_candidates', return_value=[candidate]):
        assert primitive_compute_push_pose(collector, params, baseline)
        assert primitive_compute_push_pose(collector, {**params, 'push_contact_z_offset': -.03}, lowered)
        with pytest.raises(ValueError, match='finite'):
            primitive_compute_push_pose(collector, {**params, 'push_contact_z_offset': float('nan')}, {})
    for field in ['pre_positions', 'down_positions', 'push_positions', 'lift_positions']:
        np.testing.assert_allclose(lowered['push_pose'][field] - baseline['push_pose'][field], [[0., 0., -.03]] * 2)
    np.testing.assert_allclose(lowered['push_pose']['push_target_positions'], baseline['push_pose']['push_target_positions'])
    np.testing.assert_allclose(planned[-2:], lowered['push_pose']['pre_positions'])
    np.testing.assert_allclose(lowered['push_pose']['down_positions'][:, 2], lowered['push_pose']['push_positions'][:, 2])


def test_articulation_reset_restores_velocities_and_drive_targets_for_selected_envs():
    path = ROOT / 'source/isaaclab_tasks/isaaclab_tasks/real2sim/tasks/mdp/events.py'
    node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'set_articulation_pose')
    namespace = dict(torch=torch, np=np, Optional=Optional, ManagerBasedRLEnv=object, SceneEntityCfg=object, Articulation=object)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    root = torch.zeros((4, 13)); root[:, 3] = 1.
    asset = SimpleNamespace(data=SimpleNamespace(default_root_state=root,
        default_joint_pos=torch.zeros((4, 2)), default_joint_vel=torch.zeros((4, 2))))
    for method in ['write_root_pose_to_sim', 'write_root_velocity_to_sim', 'write_joint_state_to_sim',
                   'set_joint_position_target', 'set_joint_velocity_target']:
        setattr(asset, method, Mock())
    class Scene(dict):
        env_origins = torch.tensor([[0., 0., 0.], [10., 0., 0.], [20., 0., 0.], [30., 0., 0.]])
    env = SimpleNamespace(scene=Scene(cabinet=asset), device='cpu')
    ids = torch.tensor([1, 3])
    namespace['set_articulation_pose'](env, ids, SimpleNamespace(name='cabinet'))
    for method in ['write_root_velocity_to_sim', 'set_joint_position_target', 'set_joint_velocity_target']:
        call = getattr(asset, method).call_args
        torch.testing.assert_close(call.kwargs['env_ids'], ids)
        assert torch.count_nonzero(call.args[0]) == 0
    torch.testing.assert_close(asset.write_root_pose_to_sim.call_args.args[0][:, 0], torch.tensor([10., 30.]))


def test_base_anchor_is_optional_and_does_not_increase_drawer_drive():
    loader = runpy.run_path(str(ROOT / 'source/isaaclab_tasks/isaaclab_tasks/real2sim/utils/task_config_loader.py'))
    cls = loader['ArticulationObjectConfig']
    assert cls.from_dict({'name': 'cabinet'}).fix_root_link is None
    cfg = cls.from_dict({'name': 'cabinet', 'fix_root_link': True,
                         'actuators': {'joint': {'stiffness': 0., 'damping': 1.}}})
    assert cfg.fix_root_link is True
    assert cfg.actuators['joint'].stiffness == 0.
    assert cfg.actuators['joint'].damping == 1.
