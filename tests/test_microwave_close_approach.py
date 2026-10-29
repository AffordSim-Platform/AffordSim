"""Check approach clearance and failure handling without launching simulation."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from isaaclab.motion_collect.primitives.close_microwave import primitive_execute_close


def run_close(height=0.0, fail_descent=False):
    plans, motions = [], []
    origins = np.array([[0., 0., 0.], [3., 2., 0.]])
    root_positions = origins + [.77, -.07, .16]
    microwave = SimpleNamespace(data=SimpleNamespace(
        root_pos_w=torch.tensor(root_positions),
        root_quat_w=torch.tensor([[1., 0., 0., 0.]] * 2)))

    def plan(*args, **kwargs):
        plans.append(args[3].copy())
        return [None, None], [True, True]

    def move(**kwargs):
        motions.append(kwargs)
        successes = list(kwargs['active_mask'])
        if fail_descent and kwargs['desc'] == 'Primitive Microwave Pre-Close Descent':
            successes[0] = False
        return any(successes), successes

    collector = SimpleNamespace(
        num_envs=2, target_prim_path=None,
        _containers_registry={'microwave': {'facing_axis_local': '+x'}},
        _find_object_in_scene=lambda name: name,
        _get_object_pose_global=lambda name, i: (root_positions[i], [1., 0., 0., 0.]),
        _get_env_origins=lambda: origins,
        scene=SimpleNamespace(articulations={'microwave': microwave}),
        robot=SimpleNamespace(joint_names=['joint'], data=SimpleNamespace(
            joint_pos=torch.zeros((2, 1)), joint_vel=torch.zeros((2, 1)))),
        motion_planner=SimpleNamespace(plan_to_pose_batch=plan),
        execute_trajectory_batch=lambda *args, **kwargs: None,
        execute_ik_motion=move,
        control_gripper_batch=lambda *args, **kwargs: None,
    )
    context = {'env_alive_mask': [True, True]}
    primitive_execute_close(collector, {
        'target': 'microwave', 'pre_close_approach_height': height,
        'pre_close_descent_steps': 10, 'close_interpolation_steps': 5,
    }, context)
    return plans, motions, context


def test_elevated_approach_preserves_contact_pose_and_environment_origins():
    baseline, original_motion, _ = run_close()
    raised, motions, _ = run_close(.08)
    assert len(original_motion) == 1
    assert len(motions) == 2
    np.testing.assert_allclose(raised[0] - baseline[0], [[0., 0., .08]] * 2)
    np.testing.assert_allclose(motions[0]['target_positions'], baseline[0])
    np.testing.assert_allclose(motions[0]['target_positions'][0], [1.02, -.47, .16])
    np.testing.assert_allclose(motions[0]['target_positions'][0], motions[0]['target_positions'][1])
    np.testing.assert_allclose(motions[1]['target_positions'], original_motion[0]['target_positions'])
    np.testing.assert_allclose(motions[1]['target_quaternions'], original_motion[0]['target_quaternions'])
    assert [m['interpolation_steps'] for m in motions] == [10, 5]


def test_failed_descent_does_not_push_or_report_success_in_that_environment():
    _, motions, context = run_close(.08, fail_descent=True)
    assert motions[-1]['active_mask'] == [False, True]
    assert context['env_alive_mask'] == [False, True]
    assert context['plan_success_mask'] == [False, True]


@pytest.mark.parametrize('height', [-.08, float('nan'), float('inf')])
def test_invalid_clearance_rejected(height):
    with pytest.raises(ValueError, match='pre_close_approach_height'):
        run_close(height)
