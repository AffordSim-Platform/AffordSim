"""CPU-only checks for opt-in live PhysX phase changes and reporting."""
import ast
import copy
from collections import Counter
from contextlib import nullcontext, redirect_stdout
import importlib.util
from io import StringIO
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


POLICY = load('phase_depenetration_test', 'source/isaaclab/isaaclab/motion_collect/phase_depenetration.py')
LOADER = load('phase_task_config_test', 'source/isaaclab_tasks/isaaclab_tasks/real2sim/utils/task_config_loader.py')
RUNNER = load('phase_runner_test', 'scripts/run_benchmark_50.py')


def vec(x, y, z):
    return SimpleNamespace(x=x, y=y, z=z)


def properties(value=2.0):
    return SimpleNamespace(max_depeneration_velocity=value, mass=0.25, moment=vec(.01, .02, .03),
                           cMassLocalPose=vec(0, 0, .02), max_contact_impulse=10.,
                           solver_position_iteration_count=4, solver_velocity_iteration_count=0)


class FakePrim:
    def __init__(self, path, valid=True):
        self.path, self.valid = path, valid
    def IsValid(self): return self.valid
    def HasAPI(self, api): return self.valid
    def GetPath(self): return self.path


class FakeStage:
    def __init__(self, paths): self.paths = set(paths)
    def GetPrimAtPath(self, path): return FakePrim(path, path in self.paths)


class FakeControl:
    def __init__(self, paths):
        self.handles = {p: i + 1 for i, p in enumerate(paths)}
        self.props = {h: properties() for h in self.handles.values()}
        self.fail_once = None
        self.corrupt_once = None
        self.calls = []
    def get_rigid_body(self, path): return self.handles.get(path, 0)
    def get_rigid_body_properties(self, handle): return copy.deepcopy(self.props[handle])
    def set_rigid_body_properties(self, handle, value):
        self.calls.append((handle, value.max_depeneration_velocity))
        if handle == self.fail_once:
            self.fail_once = None
            return False
        self.props[handle] = copy.deepcopy(value)
        if handle == self.corrupt_once:
            self.corrupt_once = None
            self.props[handle].mass = 99.0
        return True


class FakeUsd:
    @staticmethod
    def PrimRange(root): return [root]


class FakePhysx:
    class PhysxRigidBodyAPI: pass


CONFIG = {'objects': ['green_cube', 'doraemoncup'], 'reset': 2.0,
          'primitives': {'pick': 2.0, 'place': 0.5}}


def fixture():
    paths = [f'/World/envs/env_{env}/{name}' for env in range(2) for name in CONFIG['objects']]
    dc = FakeControl(paths)
    policy = POLICY.PhaseDepenetrationPolicy(FakeStage(paths), 2, copy.deepcopy(CONFIG),
        ['green_cube', 'doraemoncup', 'untouched'], usd=FakeUsd, physx_schema=FakePhysx,
        dc=dc, invalid_handle=0)
    return policy, dc, paths


class PhaseDepenetrationTests(unittest.TestCase):
    def test_parser_is_opt_in_and_rejects_invalid_policy(self):
        obj = {'green_cube': {'usd_path': 'cube.usd'}, 'doraemoncup': {'usd_path': 'cup.usd'}}
        base = {'objects': obj}
        self.assertIsNone(LOADER.TaskConfig.from_dict(base).phase_depenetration)
        parsed = LOADER.TaskConfig.from_dict({**base, 'phase_depenetration': CONFIG}).phase_depenetration
        self.assertEqual(parsed, CONFIG)
        release = {'objects': ['green_cube', 'doraemoncup'], 'reset': 2.0,
                   'primitives': {'pick': 2.0, 'release': .5}}
        self.assertEqual(LOADER.TaskConfig.from_dict({**base, 'phase_depenetration': release}).phase_depenetration, release)
        invalid = [
            {'objects': ['green_cube', 'green_cube'], 'reset': 2, 'primitives': {'pick': 2, 'place': .5}},
            {'objects': ['robot'], 'reset': 2, 'primitives': {'pick': 2, 'place': .5}},
            {'objects': ['green_cube'], 'reset': float('nan'), 'primitives': {'pick': 2, 'place': .5}},
            {'objects': ['green_cube'], 'reset': True, 'primitives': {'pick': 2, 'place': .5}},
            {'objects': ['green_cube'], 'reset': 2, 'primitives': {'pick': 2, 'place': 0}},
            {'objects': ['green_cube'], 'reset': 1e100, 'primitives': {'pick': 2, 'place': .5}},
            {'objects': ['green_cube'], 'reset': 2, 'primitives': {'pick': 2}},
            {'objects': ['green_cube'], 'reset': 2, 'primitives': {'pick': 2, 'place': .5, 'release': .5}},
        ]
        for case in invalid:
            with self.subTest(case=case), self.assertRaises(ValueError):
                LOADER.TaskConfig.from_dict({**base, 'phase_depenetration': case})

    def test_every_actor_transitions_and_restore_preserves_other_properties(self):
        policy, dc, paths = fixture()
        original = copy.deepcopy(dc.props)
        policy.begin_episode(1)
        policy.apply_reset()
        policy.apply_primitive('pick', [True, True])
        policy.apply_primitive('place', [True, False])
        self.assertEqual([e['phase'] for e in policy.events], ['reset', 'pick', 'place'])
        self.assertEqual(policy.events[-1]['active_envs'], [True, False])
        for env in range(2):
            for name in CONFIG['objects']:
                self.assertEqual(policy.events[-1]['physx_readback'][str(env)][name][f'/World/envs/env_{env}/{name}'], .5)
        self.assertTrue(all(p.max_depeneration_velocity == .5 for p in dc.props.values()))
        policy.begin_episode(2)
        policy.apply_reset()
        self.assertTrue(all(p.max_depeneration_velocity == 2.0 for p in dc.props.values()))
        policy.apply_primitive('place', [True, True])
        policy.restore_original()
        self.assertEqual(dc.props, original)

    def test_failed_update_and_unrelated_mutation_roll_back_all_actors(self):
        for failure in ('setter', 'mass'):
            policy, dc, _ = fixture()
            original = copy.deepcopy(dc.props)
            if failure == 'setter': dc.fail_once = 2
            else: dc.corrupt_once = 2
            policy.begin_episode(1)
            with self.subTest(failure=failure), self.assertRaisesRegex(RuntimeError, 'rolled back'):
                policy.apply_primitive('place', [True, True])
            self.assertEqual(dc.props, original)
            self.assertEqual(policy.events[-1]['status'], 'error')

    def test_missing_or_aliased_actor_is_rejected_before_writes(self):
        paths = [f'/World/envs/env_{i}/{name}' for i in range(2) for name in CONFIG['objects']]
        dc = FakeControl(paths)
        dc.handles[paths[-1]] = dc.handles[paths[0]]
        with self.assertRaisesRegex(RuntimeError, 'same PhysX actor'):
            POLICY.PhaseDepenetrationPolicy(FakeStage(paths), 2, CONFIG, CONFIG['objects'],
                usd=FakeUsd, physx_schema=FakePhysx, dc=dc, invalid_handle=0)
        self.assertEqual(dc.calls, [])

    def test_composite_selection_respects_environment_and_unknown_labels(self):
        policy, dc, _ = fixture()
        policy._config['primitives'] = {'pick': 2., 'release': .5}
        policy.begin_episode(1)
        policy.apply_reset()
        policy.apply_primitive('release', [True, False], objects=['green_cube', 'doraemoncup', 'door'])
        self.assertEqual(policy.events[-1]['selected_objects'], ['green_cube', 'doraemoncup'])
        self.assertEqual([p.max_depeneration_velocity for p in dc.props.values()], [.5, .5, 2., 2.])
        # A subsequent grasp does not disturb an already placed object.
        policy.apply_primitive('pick', [True, False], objects=['green_cube'])
        self.assertEqual([p.max_depeneration_velocity for p in dc.props.values()], [2., .5, 2., 2.])
        policy.apply_primitive('release', [True, True], objects=['door', None])
        self.assertEqual(policy.events[-1]['selected_objects'], [])
        self.assertEqual([p.max_depeneration_velocity for p in dc.props.values()], [2., .5, 2., 2.])
        policy.begin_episode(2)
        policy.apply_reset()
        self.assertTrue(all(p.max_depeneration_velocity == 2. for p in dc.props.values()))

    def test_hang_releases_after_insertion_with_only_surviving_envs(self):
        source = ROOT / 'source/isaaclab/isaaclab/motion_collect/primitives/hang.py'
        tree = ast.parse(source.read_text())
        func = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'primitive_execute_hang')
        stop = next(i for i, n in enumerate(func.body) if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == 'retreat_distance' for t in n.targets))
        func.body = func.body[:stop] + [ast.Return(value=ast.Name(id='local_active', ctx=ast.Load()))]
        ast.fix_missing_locations(func)
        def merge(active, segment):
            for i, value in enumerate(segment): active[i] = active[i] and bool(value)
        ns = {'Dict': dict, 'Any': object, 'np': np, 'FIXED_GRIPPER_STEPS': 3,
              'get_prev_alive_mask': lambda context, n: [True] * n,
              'init_local_active_mask': lambda context, n: [True] * n,
              'mark_local_inactive_with_reason': lambda *args: None,
              'FAILURE_REASON_TARGET_MISSING': 'target_missing',
              'merge_segment_into_local_active': merge}
        exec(compile(ast.Module(body=[func], type_ignores=[]), str(source), 'exec'), ns)
        policy, dc, _ = fixture()
        policy._config['primitives'] = {'pick': 2., 'release': .5}
        policy.begin_episode(1); policy.apply_reset()
        observed = []
        class Planner:
            def plan_to_pose_batch(self, *args, **kwargs): return [None, None], [True, False]
        class Collector:
            num_envs = 2
            motion_planner = Planner()
            robot = SimpleNamespace(data=SimpleNamespace(joint_pos=torch.zeros((2, 2)), joint_vel=torch.zeros((2, 2))), joint_names=[])
            target_prim_path = '/target'
            _phase_depenetration = policy
            def execute_trajectory_batch(self, *args, **kwargs): pass
            def execute_ik_motion(self, *args, **kwargs):
                observed.append(('insert', [p.max_depeneration_velocity for p in dc.props.values()]))
                return None, [True, False]
            def control_gripper_batch(self, actions, **kwargs):
                observed.append(('open', [p.max_depeneration_velocity for p in dc.props.values()], kwargs['active_mask']))
        pose = {'pre_positions': [[0, 0, 0]] * 2, 'positions': [[0, 0, 0]] * 2,
                'orientations': [[1, 0, 0, 0]] * 2, 'mount_directions': [[0, 0, 1]] * 2,
                'target_mount_local': [0, 0, 0], 'container_mount_local': [0, 0, 0],
                'held_object_label': 'green_cube', 'target_label': 'doraemoncup'}
        with redirect_stdout(StringIO()):
            result = ns['primitive_execute_hang'](Collector(), {'extra_insert_distance': 0}, {'hang_pose': pose})
        self.assertEqual(result, [True, False])
        self.assertEqual(observed, [('insert', [2.] * 4), ('open', [.5, .5, 2., 2.], [True, False])])
        self.assertEqual(policy.events[-1]['selected_objects'], ['green_cube', 'doraemoncup'])

    def test_episode_timing_and_trial_event_filtering(self):
        source = ROOT / 'source/isaaclab/isaaclab/motion_collect/datacollector.py'
        tree = ast.parse(source.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'DataCollector')
        methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in {'collect', '_run_primitives'}]
        ns = {'List': list, 'Dict': dict, 'Any': object, 'Counter': Counter, 'sys': sys,
              'torch': SimpleNamespace(no_grad=nullcontext, cuda=SimpleNamespace(empty_cache=lambda: None))}
        exec(compile(ast.Module(body=methods, type_ignores=[]), str(source), 'exec'), ns)
        policy, dc, paths = fixture()
        order, records = [], []
        class Collector:
            collect = ns['collect']
            _run_primitives = ns['_run_primitives']
            episodes = 2
            num_envs = 2
            env_task_name = 'fake'
            enable_motion_planning = False
            data_dir = None
            recorder = None
            _phase_depenetration = policy
            def rebuild_scene_desc_from_yaml(self): return [{'object': 'cube'}], 'place cube'
            def _init_environment(self): pass
            def _reset_and_stabilize(self):
                order.append(('reset', [p.max_depeneration_velocity for p in dc.props.values()]))
            def _build_primitive_context(self, **kwargs): return {**kwargs, 'held_object': 'green_cube'}
            def _get_primitive_steps(self): return [
                {'type': 'pick', 'params': {'target': 'green_cube'}},
                {'type': 'place', 'params': {'target': 'doraemoncup'}}]
            def _get_primitive_handlers(self):
                def handler(phase):
                    def run(mode, params, context):
                        order.append((phase, [p.max_depeneration_velocity for p in dc.props.values()]))
                        return True
                    return run
                return {'pick': handler('pick'), 'place': handler('place')}
            def _prepare_primitive_episode(self, context): return True
            def _normalize_primitive_mode(self, value): return value
            def episode_result_callback(self, result): records.extend(RUNNER.split_episode_result(result, 2))
        with redirect_stdout(StringIO()): Collector().collect()
        self.assertEqual([x[0] for x in order], ['reset', 'pick', 'place'] * 2)
        self.assertTrue(all(v == 2.0 for v in order[0][1] + order[1][1] + order[3][1] + order[4][1]))
        self.assertTrue(all(v == 0.5 for v in order[2][1] + order[5][1]))
        self.assertTrue(all(p.max_depeneration_velocity == 2.0 for p in dc.props.values()))
        self.assertEqual(len(records), 4)
        for r in records:
            events = r['phase_depenetration_events']
            self.assertEqual([e['phase'] for e in events], ['reset', 'pick', 'place'])
            self.assertTrue(all(e['status'] == 'verified' and e['env_active'] for e in events))
            self.assertEqual(set(events[-1]['physx_readback']), set(CONFIG['objects']))
            for name in CONFIG['objects']:
                self.assertEqual(list(events[-1]['physx_readback'][name].values()), [.5])

    def test_switch_failure_marks_program_error_and_restores_physics(self):
        source = ROOT / 'source/isaaclab/isaaclab/motion_collect/datacollector.py'
        tree = ast.parse(source.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'DataCollector')
        methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in {'collect', '_run_primitives'}]
        ns = {'List': list, 'Dict': dict, 'Any': object, 'Counter': Counter, 'sys': sys,
              'torch': SimpleNamespace(no_grad=nullcontext, cuda=SimpleNamespace(empty_cache=lambda: None))}
        exec(compile(ast.Module(body=methods, type_ignores=[]), str(source), 'exec'), ns)
        policy, dc, _ = fixture()
        original = copy.deepcopy(dc.props)
        dc.fail_once = 2
        records = []
        class Collector:
            collect = ns['collect']
            _run_primitives = ns['_run_primitives']
            episodes = 1
            num_envs = 2
            env_task_name = 'fake'
            enable_motion_planning = False
            data_dir = None
            recorder = None
            _phase_depenetration = policy
            def rebuild_scene_desc_from_yaml(self): return [{'object': 'cube'}], 'place cube'
            def _init_environment(self): pass
            def _reset_and_stabilize(self): pass
            def _build_primitive_context(self, **kwargs): return {**kwargs, 'held_object': 'green_cube'}
            def _get_primitive_steps(self): return [
                {'type': 'pick', 'params': {'target': 'green_cube'}},
                {'type': 'place', 'params': {'target': 'doraemoncup'}}]
            def _get_primitive_handlers(self): return {'pick': lambda *args: True, 'place': lambda *args: True}
            def _prepare_primitive_episode(self, context): return True
            def _normalize_primitive_mode(self, value): return value
            def episode_result_callback(self, result): records.extend(RUNNER.split_episode_result(result, 2))
        with redirect_stdout(StringIO()): Collector().collect()
        self.assertEqual(dc.props, original)
        self.assertEqual(len(records), 2)
        for record in records:
            self.assertTrue(record['program_error'])
            self.assertFalse(record['success'])
            self.assertEqual(record['phase_depenetration_events'][-1]['phase'], 'place')
            self.assertEqual(record['phase_depenetration_events'][-1]['status'], 'error')

    def test_release_switch_occurs_after_transport_and_only_for_active_envs(self):
        # Execute the actual place implementation through its release call;
        # truncate only the unrelated post-release evaluation/retreat.
        source = ROOT / 'source/isaaclab/isaaclab/motion_collect/primitives/place.py'
        tree = ast.parse(source.read_text())
        func = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'primitive_execute_place')
        stop = next(i for i, n in enumerate(func.body) if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == 'success_eval_enabled' for t in n.targets))
        func.body = func.body[:stop] + [ast.Return(value=ast.Name(id='local_active', ctx=ast.Load()))]
        ast.fix_missing_locations(func)
        def merge(active, segment):
            for i, value in enumerate(segment): active[i] = active[i] and bool(value)
        ns = {'Dict': dict, 'Any': object, 'np': np, 'FIXED_GRIPPER_STEPS': 3,
              'get_prev_alive_mask': lambda context, n: [True] * n,
              'init_local_active_mask': lambda context, n: [True] * n,
              'mark_local_inactive_with_reason': lambda *args: None,
              'FAILURE_REASON_TARGET_MISSING': 'target_missing',
              '_scatter_subset_plans': lambda n, ids, plans, ok: (plans, ok),
              'merge_segment_into_local_active': merge}
        exec(compile(ast.Module(body=[func], type_ignores=[]), str(source), 'exec'), ns)
        policy, dc, _ = fixture()
        policy._config = {'objects': CONFIG['objects'], 'reset': 2.0,
                          'primitives': {'pick': 2.0, 'release': .5}}
        policy.begin_episode(1); policy.apply_reset(); policy.apply_primitive('pick', [True, True])
        observed = []
        class Planner:
            def plan_to_pose_batch(self, *args, **kwargs): return [None, None], [True, False]
        class Collector:
            num_envs = 2
            motion_planner = Planner()
            robot = SimpleNamespace(data=SimpleNamespace(joint_pos=torch.zeros((2, 2)), joint_vel=torch.zeros((2, 2))), joint_names=[])
            target_prim_path = '/target'
            _phase_depenetration = policy
            def execute_trajectory_batch(self, *args, **kwargs):
                observed.append(('transport', [p.max_depeneration_velocity for p in dc.props.values()]))
            def control_gripper_batch(self, actions, **kwargs):
                observed.append(('open', [p.max_depeneration_velocity for p in dc.props.values()], actions, kwargs['active_mask']))
        context = {'held_object': 'green_cube', 'place_pose': {'target_label': 'doraemoncup', 'positions': [[0, 0, 0], [0, 0, 0]],
                                  'orientations': [[1, 0, 0, 0], [1, 0, 0, 0]]}}
        with redirect_stdout(StringIO()): result = ns['primitive_execute_place'](Collector(), {}, context)
        self.assertEqual(result, [True, False])
        self.assertEqual(observed[0], ('transport', [2.0] * 4))
        self.assertEqual(observed[1], ('open', [.5, .5, 2.0, 2.0], [1.0, 0.0], [True, False]))
        event = policy.events[-1]
        self.assertEqual((event['phase'], event['active_envs']), ('release', [True, False]))
        self.assertEqual(list(event['physx_readback']['0']['green_cube'].values()), [.5])
        self.assertEqual(list(event['physx_readback']['1']['green_cube'].values()), [2.0])

        # A failed release switch rolls back before any gripper-open command.
        policy2, dc2, _ = fixture()
        policy2._config = policy._config
        policy2.begin_episode(1); policy2.apply_reset(); policy2.apply_primitive('pick', [True, True])
        dc2.fail_once = 2
        collector2 = Collector(); collector2._phase_depenetration = policy2
        opened = []
        collector2.control_gripper_batch = lambda *args, **kwargs: opened.append(True)
        with redirect_stdout(StringIO()), self.assertRaisesRegex(RuntimeError, 'rolled back'):
            ns['primitive_execute_place'](collector2, {}, context)
        self.assertEqual(opened, [])
        self.assertTrue(all(p.max_depeneration_velocity == 2.0 for p in dc2.props.values()))
        self.assertEqual(policy2.events[-1]['status'], 'error')


if __name__ == '__main__':
    unittest.main()
