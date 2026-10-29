"""CPU regression tests for preserving per-environment raw gripper actions."""
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from isaaclab.motion_collect.hold_action import make_joint_hold_action
from isaaclab.motion_collect.motion_planner import MotionPlanner
from isaaclab.motion_collect.primitives.pour import primitive_execute_pour
from isaaclab.motion_collect.primitives.return_home import primitive_return_home


class FakeActionManager:
    def __init__(self, values):
        self.total_action_dim = 8
        self.active_terms = ["arm_action", "gripper_action"]
        self.action_term_dim = [7, 1]
        self.action = torch.zeros(len(values), 8)
        self.action[:, -1] = torch.tensor(values)


class FakeEnv:
    def __init__(self, values):
        self.env = SimpleNamespace(action_manager=FakeActionManager(values))
        self.actions = []

    def step(self, action):
        self.actions.append(action.detach().clone())
        self.env.action_manager.action.copy_(action)
        return None, None, None, None, None


class FakeRobot:
    def __init__(self, finger_positions):
        n = len(finger_positions)
        self.joint_names = [f"joint_{i}" for i in range(7)] + ["finger_1", "finger_2"]
        joints = torch.full((n, 9), 0.2)
        joints[:, 7] = torch.tensor(finger_positions)
        self.data = SimpleNamespace(joint_pos=joints, joint_vel=torch.zeros_like(joints))


class FakePlan:
    joint_names = [f"joint_{i}" for i in range(7)]

    def get_ordered_joint_state(self, joint_names):
        assert joint_names == self.joint_names
        class Ordered:
            position = torch.tensor([[0.4] * 7, [0.5] * 7])

            def __getitem__(self, index):
                return SimpleNamespace(position=self.position[index])

        return Ordered()


class GripperHoldTests(unittest.TestCase):
    def setUp(self):
        self.env = FakeEnv([-1.0, 0.0, 0.75])
        # Positive/zero physical positions cover Franka and other gripper
        # conventions, which must not determine the raw binary command.
        self.robot = FakeRobot([0.025, 0.8, 0.0])
        self.planner = MotionPlanner.__new__(MotionPlanner)

    def assert_commands(self, expected):
        self.assertTrue(self.env.actions)
        for action in self.env.actions:
            torch.testing.assert_close(action[:, -1], torch.tensor(expected))

    def test_gripper_control_preserves_each_inactive_raw_command(self):
        self.planner.control_gripper_batch([1, -1, -1], self.env, self.robot, 3,
                                           steps=3, active_mask=[False, True, False])
        self.assert_commands([-1.0, -1.0, 0.75])

    def test_all_inactive_after_failed_plan_keeps_grasp_closed(self):
        self.env.env.action_manager.action[:, -1] = -1
        self.planner.control_gripper_batch([0, 0, 0], self.env, self.robot, 3,
                                           steps=3, active_mask=[False] * 3)
        self.assert_commands([-1.0, -1.0, -1.0])

    def test_trajectory_preserves_raw_commands_of_inactive_rows(self):
        self.planner.execute_trajectory_batch([FakePlan()] * 3, [1, -1, -1],
                                              self.env, self.robot, 3,
                                              active_mask=[False, True, False])
        self.assert_commands([-1.0, -1.0, 0.75])

    def test_ik_failure_preserves_raw_commands_in_mixed_batch(self):
        self.planner.get_current_ee_pose = lambda env, n, kind: (np.zeros((n, 3)), np.tile([1, 0, 0, 0], (n, 1)))
        self.planner._build_interpolated_waypoints = lambda *args: (None, None)
        self.planner.solve_ik_trajectory = lambda *args: (np.zeros((3, 2, 7)), None, [False, True, False])
        _, mask = self.planner.execute_ik_motion(np.zeros((3, 3)), np.tile([1, 0, 0, 0], (3, 1)),
                                                  [1, -1, -1], self.env, self.robot, 3,
                                                  interpolation_steps=2, active_mask=[True, True, False])
        self.assertEqual(mask, [False, True, False])
        self.assert_commands([-1.0, -1.0, 0.75])

    def test_unrecognized_action_layout_is_rejected(self):
        self.env.env.action_manager.active_terms = ["gripper_action", "arm_action"]
        self.env.env.action_manager.action_term_dim = [1, 7]
        with self.assertRaisesRegex(RuntimeError, "gripper_action"):
            make_joint_hold_action(self.env, self.robot, 3)
        self.assertFalse(self.env.actions)

    def test_reset_raw_zero_is_preserved_not_guessed_from_joint_position(self):
        self.env.env.action_manager.action.zero_()
        action = make_joint_hold_action(self.env, self.robot, 3)
        torch.testing.assert_close(action[:, -1], torch.zeros(3))

    def make_collector(self):
        collector = SimpleNamespace(num_envs=3, device='cpu', env=self.env, robot=self.robot,
                                    target_prim_path=None, scene={})
        self.planner.plan_to_pose_batch = lambda *a, **k: ([FakePlan()] * 3, [True] * 3)
        collector.motion_planner = self.planner
        collector.execute_trajectory_batch = lambda plans, commands, **kw: self.planner.execute_trajectory_batch(
            plans, commands, self.env, self.robot, 3, **kw)
        return collector

    def test_return_home_settle_preserves_each_raw_gripper_command(self):
        collector = self.make_collector()
        self.planner.plan_to_pose_batch = lambda *a, **k: ([FakePlan()] * 3, [True, False, True])
        context = {'initial_ee_pos': np.zeros((3, 3)), 'initial_ee_quat': np.tile([1, 0, 0, 0], (3, 1)),
                   'current_gripper_state': 'closed', 'env_alive_mask': [True] * 3}
        self.assertTrue(primitive_return_home(collector, None, {'post_settle_steps': 2}, context))
        self.assert_commands([-1.0, 0.0, -1.0])

    def test_pour_tilt_failure_does_not_open_failed_environment(self):
        collector = self.make_collector()
        self.env.env.action_manager.action[:, -1] = -1.0
        collector.execute_ik_motion = lambda *a, **k: (True, [True, False, True])
        context = {'env_alive_mask': [True] * 3, 'held_object': 'scoop',
                   'pour_pose': {'positions': np.zeros((3, 3)), 'orientations': np.tile([1, 0, 0, 0], (3, 1)),
                                 'pour_directions': np.tile([1, 0, 0], (3, 1)), 'target_label': 'bowl'}}
        self.assertTrue(primitive_execute_pour(collector, {}, context))
        self.assertEqual(len(self.env.actions), 2 + 3)  # Move trajectory and three peak-hold steps.
        self.assert_commands([-1.0, -1.0, -1.0])


if __name__ == '__main__':
    unittest.main()
