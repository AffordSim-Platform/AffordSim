"""Exercise configurable pour IK segments without a simulator or GPU."""
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from isaaclab.motion_collect.primitives.pour import primitive_execute_pour


@pytest.mark.parametrize(
    ("params", "expected"),
    [
        ({}, [("Primitive Tilt Pour", 30), ("Primitive Return Upright", 10)]),
        ({"return_lift_height": .05}, [("Primitive Tilt Pour", 30),
                                        ("Primitive Lift Before Return", 5),
                                        ("Primitive Return Upright", 10)]),
        ({"tilt_interpolation_steps": 7, "return_interpolation_steps": 9},
         [("Primitive Tilt Pour", 7), ("Primitive Return Upright", 9)]),
        ({"return_lift_height": .05, "tilt_interpolation_steps": 7,
          "return_lift_interpolation_steps": 8, "return_interpolation_steps": 9},
         [("Primitive Tilt Pour", 7), ("Primitive Lift Before Return", 8),
          ("Primitive Return Upright", 9)]),
        ({"return_lift_height": .05, "tilt_interpolation_steps": 5,
          "return_lift_interpolation_steps": 5, "return_interpolation_steps": 5},
         [("Primitive Tilt Pour", 5), ("Primitive Lift Before Return", 5),
          ("Primitive Return Upright", 5)]),
        ({"tilt_interpolation_steps": 15, "return_interpolation_steps": 15, "hold_steps": 3},
         [("Primitive Tilt Pour", 15), ("Primitive Return Upright", 15)]),
    ],
)
def test_pour_ik_segment_interpolation(params, expected):
    calls = []
    holds = []
    collector = SimpleNamespace(
        num_envs=1,
        device="cpu",
        target_prim_path=None,
        scene={},
        robot=SimpleNamespace(joint_names=["joint"], data=SimpleNamespace(
            joint_pos=torch.zeros((1, 2)), joint_vel=torch.zeros((1, 2)))),
        motion_planner=SimpleNamespace(plan_to_pose_batch=lambda *args, **kw: ([None], [True])),
        execute_trajectory_batch=lambda *args, **kwargs: None,
        env=SimpleNamespace(step=lambda action: holds.append((calls[-1][0], action.clone()))),
    )

    def execute_ik_motion(**kwargs):
        calls.append((kwargs["desc"], kwargs["interpolation_steps"]))
        assert kwargs["gripper_states"] == [-1.0]
        return True, [True]

    collector.execute_ik_motion = execute_ik_motion
    context = {
        "env_alive_mask": [True],
        "held_object": "cup",
        "pour_pose": {"positions": np.zeros((1, 3)),
                      "orientations": np.array([[1., 0., 0., 0.]]),
                      "pour_directions": np.array([[1., 0., 0.]]),
                      "target_label": "bowl"},
    }
    with patch("isaaclab.motion_collect.primitives.pour.make_joint_hold_action",
               return_value=torch.zeros((1, 2))):
        assert primitive_execute_pour(collector, params, context)
    assert calls == expected
    assert len(holds) == params.get('hold_steps', 3)
    assert all(segment == 'Primitive Tilt Pour' for segment, _ in holds)
    assert all(action[0, -1] == -1 for _, action in holds)


def test_formal_recording_includes_all_three_peak_hold_steps():
    recorded = []
    collector = SimpleNamespace(
        num_envs=1, device="cpu", target_prim_path=None, scene={},
        robot=SimpleNamespace(joint_names=["joint"], data=SimpleNamespace(
            joint_pos=torch.zeros((1, 2)), joint_vel=torch.zeros((1, 2)))),
        motion_planner=SimpleNamespace(plan_to_pose_batch=lambda *a, **k: ([None], [True])),
        execute_trajectory_batch=lambda *a, **k: None,
        execute_ik_motion=lambda **kw: (True, [True]),
        env=SimpleNamespace(step=lambda action: ({"policy": {}}, None, False, False, {})),
        recorder=SimpleNamespace(preserve_idle_steps=True,
                                 add_step=lambda obs, action, term, trunc: recorded.append(action.clone())),
    )
    context = {"env_alive_mask": [True], "held_object": "cup", "pour_pose": {
        "positions": np.zeros((1, 3)), "orientations": np.array([[1., 0., 0., 0.]]),
        "pour_directions": np.array([[1., 0., 0.]]), "target_label": "bowl"}}
    with patch("isaaclab.motion_collect.primitives.pour.make_joint_hold_action",
               return_value=torch.zeros((1, 2))):
        assert primitive_execute_pour(collector, {"hold_steps": 3}, context)
    assert len(recorded) == 3
    assert all(action[0, -1] == -1 for action in recorded)
