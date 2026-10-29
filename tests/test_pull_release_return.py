"""Check drawer release, clearance motions and post-return success on CPU."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from isaaclab.motion_collect.primitives.pull import primitive_execute_pull
from isaaclab.motion_collect.primitives.return_home import primitive_return_home


class Collector:
    def __init__(self):
        self.num_envs = 2
        self.target_prim_path = '/World/envs/env_0/cabinet'
        self._containers_registry = {'cabinet': {'facing_axis_local': '-x'}}
        half = np.sqrt(0.5)
        articulation = SimpleNamespace(
            data=SimpleNamespace(root_quat_w=torch.tensor([[1., 0., 0., 0.], [half, 0., 0., half]])),
            cfg=SimpleNamespace(spawn=SimpleNamespace(usd_path='fixture.usd')),
        )
        self.scene = SimpleNamespace(articulations={'cabinet': articulation})
        self.robot = SimpleNamespace(joint_names=['joint'], data=SimpleNamespace(
            joint_pos=torch.zeros((2, 1)), joint_vel=torch.zeros((2, 1))))
        self.motion_planner = SimpleNamespace(plan_to_pose_batch=Mock(return_value=([None, None], [True, True])))
        self.positions = np.array([[.6, .1, .3], [.6, .1, .3]])
        self.quaternions = np.array([[1., 0., 0., 0.], [1., 0., 0., 0.]])
        self.events = []
        self.ik_failures = {}
        self.cabinet_retreat_monitor = SimpleNamespace(start=Mock())

    def _find_object_in_scene(self, name):
        return name

    def _get_object_pose_global(self, name, env_idx):
        return np.zeros(3), np.array([1., 0., 0., 0.])

    def get_current_ee_pose(self):
        return self.positions.copy(), self.quaternions.copy()

    def execute_trajectory_batch(self, plans, gripper_states, *, desc, active_mask):
        self.events.append({'name': desc, 'gripper': list(gripper_states), 'active': list(active_mask)})

    def execute_ik_motion(self, *, target_positions, target_quaternions, gripper_states,
                          interpolation_steps, desc, active_mask):
        before = self.positions.copy()
        success = self.ik_failures.get(desc, [True, True])
        for i in range(self.num_envs):
            if active_mask[i] and success[i]:
                self.positions[i] = target_positions[i]
        self.events.append({'name': desc, 'before': before, 'target': np.array(target_positions),
                            'gripper': list(gripper_states), 'active': list(active_mask),
                            'steps': interpolation_steps})
        return any(success), success

    def control_gripper_batch(self, commands, *, steps, active_mask):
        self.events.append({'name': 'gripper', 'gripper': list(commands), 'active': list(active_mask)})


def pull_context():
    return {'grasp_pose': {'target_available_mask': [True, True]},
            'all_pick_pos': np.array([[.6, .1, .3], [.6, .1, .3]]),
            'all_pick_quat': np.array([[1., 0., 0., 0.], [1., 0., 0., 0.]]),
            'env_alive_mask': [True, True]}


def test_release_precedes_outward_retreat_then_lift_in_each_cabinet_frame():
    collector = Collector()
    context = pull_context()
    params = {'target': 'cabinet', 'release_after_pull': True,
              'post_release_retreat_distance': .08, 'post_release_lift_height': .10,
              'post_release_retreat_steps': 6}
    assert primitive_execute_pull(collector, params, context)
    names = [e['name'] for e in collector.events]
    assert names == ['Primitive Pull Pre-Grasp', 'Primitive Pull Grasp Descent', 'gripper',
                     'Primitive Pull Drawer', 'gripper', 'Primitive Pull Post-Release Retreat',
                     'Primitive Pull Post-Release Lift']
    assert collector.events[2]['gripper'] == [-1., -1.]
    assert collector.events[3]['gripper'] == [-1., -1.]
    for index in (4, 5, 6):
        assert collector.events[index]['gripper'] == [1., 1.]
    retreat, lift = collector.events[5:]
    np.testing.assert_allclose(retreat['target'] - retreat['before'], [[-.08, 0., 0.], [0., -.08, 0.]], atol=1e-7)
    np.testing.assert_allclose(lift['target'] - lift['before'], [[0., 0., .1], [0., 0., .1]], atol=1e-7)
    assert retreat['steps'] == lift['steps'] == 6
    assert context['current_gripper_state'] == 'open'
    collector.cabinet_retreat_monitor.start.assert_called_once()


def test_legacy_pull_without_clearance_retains_existing_sequence():
    collector = Collector()
    context = pull_context()
    assert primitive_execute_pull(collector, {'target': 'cabinet', 'release_after_pull': True}, context)
    assert [e['name'] for e in collector.events] == [
        'Primitive Pull Pre-Grasp', 'Primitive Pull Grasp Descent', 'gripper',
        'Primitive Pull Drawer', 'gripper']
    assert 'pull_after_return_check' not in context
    collector.cabinet_retreat_monitor.start.assert_not_called()


def test_failed_clearance_excludes_environment_from_lift_and_final_success():
    collector = Collector()
    collector.ik_failures['Primitive Pull Post-Release Retreat'] = [True, False]
    context = pull_context()
    assert primitive_execute_pull(collector, {'target': 'cabinet', 'post_release_retreat_distance': .08,
                                              'post_release_lift_height': .10}, context)
    lift = collector.events[-1]
    assert lift['active'] == [True, False]
    np.testing.assert_array_equal(lift['target'][1], lift['before'][1])
    assert context['plan_success_mask'] == [True, False]
    assert context['env_alive_mask'] == [True, False]


def test_clearance_requires_open_gripper():
    with pytest.raises(ValueError, match='release_after_pull'):
        primitive_execute_pull(Collector(), {'target': 'cabinet', 'release_after_pull': False,
                                             'post_release_retreat_distance': .08}, pull_context())


def test_drawer_reclosing_during_return_is_a_task_failure():
    collector = Collector()
    context = {'env_alive_mask': [True, True], 'current_gripper_state': 'open',
               'initial_ee_pos': [[.3, 0., .4], [.3, 0., .4]],
               'initial_ee_quat': [[1., 0., 0., 0.], [1., 0., 0., 0.]],
               'task_success_metrics': [{'joint_displacement_m': .10}, {'joint_displacement_m': .10}],
               'pull_after_return_check': {'target': 'cabinet', 'joint_name': 'joint_0',
                                           'joint_type': 'prismatic', 'threshold': .08}}
    evaluator = Mock(side_effect=[(True, {'joint_displacement_m': .10}),
                                  (False, {'joint_displacement_m': .01})])
    with patch('isaaclab.motion_collect.primitives.return_home._evaluate_pull_success_with_metrics', evaluator):
        assert primitive_return_home(collector, None, {'post_settle_steps': 0}, context)
    assert collector.events[0]['gripper'] == [1., 1.]
    assert context['env_alive_mask'] == [True, False]
    assert context['task_success_mask'] == [True, False]
    assert context['step_failure_reason_mask'][1] == 'task_failed'
    assert context['task_success_metrics'][1]['post_return_home']['joint_displacement_m'] == .01
