"""Regression checks for reset semantics, infeasible plans and lossless frames."""
import ast
import copy
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
import unittest
from unittest.mock import Mock

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def definitions(relative, names, namespace):
    path = ROOT / relative
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    assert len(nodes) == len(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)


class EvaluationRepairs(unittest.TestCase):
    def test_object_mass_is_optional_and_requires_a_positive_finite_value(self):
        loader = load('mass_loader_test', 'source/isaaclab_tasks/isaaclab_tasks/real2sim/utils/task_config_loader.py')
        base = {'name': 'yellow_bowl', 'usd_path': 'bowl.usd'}
        self.assertIsNone(loader.ObjectConfig.from_dict(base).mass)
        self.assertEqual(loader.ObjectConfig.from_dict({**base, 'mass': 0.2}).mass, 0.2)
        for invalid in (0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                loader.ObjectConfig.from_dict({**base, 'mass': invalid})

    def test_per_object_depenetration_override_is_optional_and_positive(self):
        loader = load('depenetration_loader_test', 'source/isaaclab_tasks/isaaclab_tasks/real2sim/utils/task_config_loader.py')
        base = {'name': 'doraemoncup', 'usd_path': 'cup.usd'}
        self.assertIsNone(loader.ObjectConfig.from_dict(base).max_depenetration_velocity)
        self.assertEqual(loader.ObjectConfig.from_dict({**base, 'max_depenetration_velocity': 0.5}).max_depenetration_velocity, 0.5)
        for invalid in (0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                loader.ObjectConfig.from_dict({**base, 'max_depenetration_velocity': invalid})

    def test_fixed_and_partial_reset_ranges_do_not_become_success_bounds(self):
        loader = load('reset_loader_test', 'source/isaaclab_tasks/isaaclab_tasks/real2sim/utils/task_config_loader.py')
        source = {'enable_objects_randomization': True, 'objects': {
            'fixed': {'usd_path': 'fixed.usd', 'init_pos': [0.5, -0.2, 0.1]},
            'partial': {'usd_path': 'partial.usd', 'init_pos': [0.6, 0.2, 0.1],
                        'position_range': {'x': [0.4, 0.8]}, 'orientation_range': {'yaw': [-0.3, 0.3]}}},
            'articulations': {'door': {'usd_path': 'door.usd', 'init_pos': [0.8, 0.0, 0.2]}}}
        before = copy.deepcopy(source)
        cfg = loader.TaskConfig.from_dict(source)
        self.assertEqual(cfg.get_event_params()['fixed']['position_range'],
                         {'x': (0.5, 0.5), 'y': (-0.2, -0.2), 'z': (0.1, 0.1)})
        self.assertEqual(cfg.get_event_params()['partial']['position_range'],
                         {'x': (0.4, 0.8), 'y': (0.2, 0.2), 'z': (0.1, 0.1)})
        self.assertEqual(cfg.get_articulation_event_params()['door']['position_range']['x'], (0.8, 0.8))
        self.assertEqual(cfg.objects['fixed'].position_range, {})
        self.assertEqual(cfg.articulations['door'].position_range, {})
        self.assertEqual(cfg.objects['partial'].position_range, {'x': (0.4, 0.8)})
        self.assertEqual(source, before)

    def test_infeasible_pour_is_a_per_environment_plan_failure(self):
        ns = {'Any': Any, 'Dict': Dict, 'List': List, 'Optional': Optional,
              'np': np, 'Rotation': Rotation,
              'FAILURE_REASON_NONE': 'none', 'FAILURE_REASON_TARGET_MISSING': 'target_missing',
              'FAILURE_REASON_PLANNING_FAILED': 'planning_failed'}
        utils = 'source/isaaclab/isaaclab/motion_collect/primitives/utils.py'
        definitions(utils, {'NoFeasiblePourRotationError', 'search_best_pour_rotation',
                            '_normalize_xy', '_ensure_mask', '_ensure_reason_mask',
                            'get_prev_alive_mask', 'init_local_active_mask',
                            'mark_local_inactive_with_reason', 'commit_plan_failures'}, ns)
        ns['_quat_wxyz_to_xyzw'] = lambda q: [q[1], q[2], q[3], q[0]]
        ns['_quat_xyzw_to_wxyz'] = lambda q: [q[3], q[0], q[1], q[2]]
        error = ns['NoFeasiblePourRotationError']
        with self.assertRaises(error):
            ns['search_best_pour_rotation'](
                container_center_world=np.array([0.6, 0.0, 0.1]),
                effective_pour_world=np.array([0.2, 0.0, 0.2]), current_ee_world=np.zeros(3),
                object_center_world=np.array([0.1, 0.0, 0.1]), env_origin=np.zeros(3),
                lift_pos_robot=np.zeros(3), current_grasp_quat_wxyz=np.array([1., 0., 0., 0.]))
        ns.update({
            'compute_target_availability_mask': lambda collector, label: [True, True],
            'log_target_missing_envs': lambda *args: None,
            'resolve_affordance_path': lambda *args: 'fixture.npz',
            'prepare_pour_geometry': lambda *args, **kwargs: {'object_center_local': np.zeros(3)},
            'load_affordance_clusters': lambda *args: [np.array([0., 0., 0.1])],
            'load_affordance_centroid': lambda *args: np.zeros(3),
            '_quat_wxyz_to_matrix': lambda quat: Rotation.from_quat(ns['_quat_wxyz_to_xyzw'](quat)).as_matrix(),
            'select_effective_pour_point_local': lambda *args, **kwargs: (np.array([0., 0., 0.1]), {}),
            'search_best_pour_rotation': Mock(side_effect=[error('empty feasible set'),
                (np.zeros(3), np.zeros(3), np.array([1., 0., 0., 0.]), np.array([1., 0., 0.]), 0., {})]),
        })
        definitions('source/isaaclab/isaaclab/motion_collect/primitives/pour.py',
                    {'primitive_compute_pour_pose', 'primitive_execute_pour'}, ns)
        collector = SimpleNamespace(num_envs=2, task_config=None,
            _get_env_origins=lambda: np.zeros((2, 3)),
            get_current_ee_pose=lambda: (np.zeros((2, 3)), None),
            _get_object_pose_global=lambda *args: (np.zeros(3), np.array([1., 0., 0., 0.])))
        context = {'post_pick_success_mask': [True, True], 'env_alive_mask': [True, True],
                   'lift_positions': np.zeros((2, 3)), 'pick_target_label': 'cup',
                   'selected_grasp_centers_local': [np.zeros(3), np.zeros(3)],
                   'all_pick_quat': np.array([[1., 0., 0., 0.], [1., 0., 0., 0.]])}
        self.assertTrue(ns['primitive_compute_pour_pose'](collector, None, {'target': 'bowl'}, context))
        self.assertEqual(context['pour_pose']['pose_available_mask'], [False, True])
        self.assertEqual(context['pour_pose']['meta'][0]['status'], 'no_feasible_rotation')
        context['pour_pose']['pose_available_mask'] = [False, False]
        self.assertFalse(ns['primitive_execute_pour'](collector, {}, context))
        self.assertEqual(context['plan_success_mask'], [False, False])
        self.assertEqual(context['step_failure_reason_mask'], ['planning_failed', 'planning_failed'])

    def test_recorded_webp_has_identical_pixels_to_png(self):
        module = load('encoding_recorder_test', 'source/isaaclab_tasks/isaaclab_tasks/real2sim/utils/EpisodeBuffer.py')
        observation = {'policy': {'external_cam_side': np.random.default_rng(2).random((1, 3 * 80 * 60)).astype(np.float32)}}
        with tempfile.TemporaryDirectory() as directory:
            outputs = []
            for encoding, extension in [('png', 'png'), ('webp_lossless', 'webp')]:
                recorder = module.EpisodeDataRecorder(save_dir=Path(directory) / encoding,
                    save_npy=False, save_camera_images=False, save_original_images=True,
                    sim_camera_resolution=(80, 60), image_noise_std=0, original_image_format=encoding)
                recorder.start_episode(1, {})
                recorder.add_step(copy.deepcopy(observation), np.zeros((1, 8)), False, False)
                path = next((Path(directory) / encoding).rglob(f'step_*.{extension}'))
                with Image.open(path) as image:
                    outputs.append((image.mode, image.size, image.tobytes()))
            self.assertEqual(outputs[0], outputs[1])


if __name__ == '__main__':
    unittest.main()
