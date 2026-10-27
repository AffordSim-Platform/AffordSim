"""CPU checks for fixed-attempt evaluation and lightweight image recording."""

import copy
import ast
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import ModuleType, SimpleNamespace
from collections import Counter
from contextlib import nullcontext, redirect_stdout
from io import StringIO

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUNNER = load_file("benchmark_runner_test", ROOT / "scripts/run_benchmark_50.py")


def normal_collision_result():
    return {"status": "complete", "excluded": False, "events": [],
            "observed_substeps": 1, "thresholds": {"min_speed_m_s": 2.0}}


def excluded_collision_result():
    return {"status": "complete", "excluded": True, "observed_substeps": 2,
            "thresholds": {"min_speed_m_s": 2.0},
            "events": [{"object": "cup", "other_object": "rack", "collision_time_s": 0.1,
                        "detection_time_s": 0.2, "impulse_ns": 0.1,
                        "velocity_jump_m_s": 2.0, "displacement_m": 0.3}]}


def fake_classified_trials(count, successes):
    return [{"trial": i + 1, "program_error": False, "success": i < successes,
             "failure_reason": "none" if i < successes else "task_failed",
             "collision_flyaway": normal_collision_result()} for i in range(count)]


def detector_provenance():
    return {
        "collision_detector_source_sha256": RUNNER.digest(ROOT / "source/isaaclab/isaaclab/motion_collect/collision_flyaway.py"),
        "collision_detector_monitor_sha256": RUNNER.digest(ROOT / "source/isaaclab/isaaclab/motion_collect/collision_flyaway_monitor.py"),
        "collision_detector_collector_sha256": RUNNER.digest(ROOT / "source/isaaclab/isaaclab/motion_collect/datacollector.py"),
    }


class BenchmarkRunnerTests(unittest.TestCase):
    def test_resume_requires_successful_worker_exit_and_preserves_interrupted_recordings(self):
        for label, returncode, timed_out, should_resume in (
            ('successful', 0, False, True),
            ('missing_returncode', None, False, False),
            ('failed_returncode', 1, False, False),
            ('timed_out', 0, True, False),
        ):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = root / 'source.yaml'
                source.write_text(yaml.safe_dump({'objects': {'cup': {}},
                    'primitives': [{'type': 'place', 'params': {'success_eval': True}}]}))
                task = {'task_id': 'cup', 'config_path': str(source), 'robot': 'robot',
                        'env_task_name': 'env', 'robot_cfg_name': 'robot.yml', 'robot_type': 'robot', 'num_envs': 20}
                manifest = root / 'manifest.json'
                manifest.write_text(json.dumps({'tasks': [task]}))
                original = root / 'results/cup/round_00'
                original.mkdir(parents=True)
                (original / 'frame.webp').write_bytes(b'old-frame')
                trials = fake_classified_trials(20, 20)
                previous = {'completed': True, 'source_config_sha256': RUNNER.digest(source),
                    'seed': RUNNER.task_seed(42, 'cup'), 'robot': 'robot', 'num_envs': 20,
                    'range_scale': 1.0, 'summary': RUNNER.summarize(trials, 20), 'trials': trials,
                    **detector_provenance(),
                    'timed_out': timed_out}
                if returncode is not None:
                    previous['returncode'] = returncode
                RUNNER.write_json(original / 'report.json', previous)
                previous_bytes = (original / 'report.json').read_bytes()
                calls = []
                class WorkerDouble:
                    def __init__(self, command, **kwargs):
                        self.command = command
                        calls.append(command)
                    def wait(self, timeout):
                        value = lambda name: self.command[self.command.index(name) + 1]
                        RUNNER.write_json(Path(value('--output-dir')) / 'report.json', {
                            'completed': True, 'summary': previous['summary'], 'seed': int(value('--seed')),
                            'robot': 'robot', 'num_envs': int(value('--num-envs')), 'trials': trials,
                            **detector_provenance()})
                        return 0
                args = SimpleNamespace(manifest=str(manifest), task=None, output_dir=str(root / 'results'),
                    no_adapt=True, trials=20, num_envs=1, seed=42, resume=True, device='cpu', timeout=10,
                    stop_after_task=None)
                with patch.object(RUNNER.subprocess, 'Popen', WorkerDouble), redirect_stdout(StringIO()):
                    RUNNER.run_schedule(args)
                archives = list(original.parent.glob('round_00_previous_*'))
                if should_resume:
                    self.assertEqual(calls, [])
                    self.assertEqual(archives, [])
                    self.assertEqual((original / 'report.json').read_bytes(), previous_bytes)
                    self.assertEqual((original / 'frame.webp').read_bytes(), b'old-frame')
                else:
                    self.assertEqual(len(calls), 1)
                    self.assertEqual(len(archives), 1)
                    self.assertEqual((archives[0] / 'report.json').read_bytes(), previous_bytes)
                    self.assertEqual((archives[0] / 'frame.webp').read_bytes(), b'old-frame')

    def test_scheduler_task_batch_sizes_resume_and_stop_at_task_boundary(self):
        self.assertEqual(RUNNER.task_num_envs({"task_id": "default"}, 5, 20), 5)
        for invalid in (0, -1, 3, 2.5, "5", True, None):
            with self.assertRaises(ValueError):
                RUNNER.task_num_envs({"task_id": "invalid", "num_envs": invalid}, 5, 20)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "source.yaml"
            config.write_text(yaml.safe_dump({"objects": {"cup": {"position_range": {"x": [0.4, 0.6]}}},
                "primitives": [{"type": "place", "params": {"success_eval": True}}]}))
            manifest = root / "manifest.json"
            tasks = [{"task_id": name, "num_envs": n, "config_path": str(config), "robot": "robot",
                      "env_task_name": "env", "robot_cfg_name": "robot.yml", "robot_type": "robot"}
                     for name, n in (("first", 5), ("second", 10))]
            manifest.write_text(json.dumps({"tasks": tasks}))
            stop = root / "STOP"
            args = SimpleNamespace(manifest=str(manifest), task=None, output_dir=str(root / "results"),
                no_adapt=False, trials=20, num_envs=1, seed=42, resume=True, device="cpu", timeout=10,
                stop_after_task=str(stop))
            commands = []
            stop_on_first = True
            class WorkerDouble:
                def __init__(self, command, **kwargs):
                    self.command = command
                    commands.append(command)
                def wait(self, timeout):
                    value = lambda name: self.command[self.command.index(name) + 1]
                    destination = Path(value("--output-dir"))
                    trials = fake_classified_trials(20, 5)
                    report = {"completed": True, "summary": RUNNER.summarize(trials, 20), "trials": trials,
                        "seed": int(value("--seed")), "robot": "robot", "num_envs": int(value("--num-envs")),
                        **detector_provenance()}
                    RUNNER.write_json(destination / "report.json", report)
                    # A marker raised in round 0 must still allow this task's
                    # round 1 and 2; it must not start the second task.
                    if stop_on_first and value("--task-id") == "first":
                        stop.touch()
                    return 0
            with patch.object(RUNNER.subprocess, "Popen", WorkerDouble), redirect_stdout(StringIO()):
                first = RUNNER.run_schedule(args)
                self.assertEqual(len(commands), 3)
                self.assertTrue(first["stopped"])
                self.assertEqual(first["remaining_task_ids"], ["second"])
                self.assertEqual(first["tasks"][0]["num_envs"], 5)
                self.assertTrue(all(command[command.index("--num-envs") + 1] == "5" for command in commands))
                stop.unlink()
                stop_on_first = False
                second = RUNNER.run_schedule(args)
                self.assertFalse(second["stopped"])
                self.assertEqual(len(commands), 6)  # First task resumed; only second task ran.
                self.assertTrue(all(command[command.index("--num-envs") + 1] == "10" for command in commands[3:]))
                args.num_envs = 20  # Per-task overrides also govern resume eligibility.
                RUNNER.run_schedule(args)
                self.assertEqual(len(commands), 6)

    def test_rgb_only_cameras_preserve_recorded_rgb_and_task_conditions(self):
        source = ROOT / "source/isaaclab/isaaclab/motion_collect/datacollector.py"
        tree = ast.parse(source.read_text())
        helper = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_configure_rgb_only_collection_cameras")
        namespace = {"Dict": dict, "Any": object}
        exec(compile(ast.Module(body=[helper], type_ignores=[]), str(source), "exec"), namespace)
        configure = namespace["_configure_rgb_only_collection_cameras"]
        camera = lambda name: SimpleNamespace(width=640, height=480, data_types=["rgb", "distance_to_image_plane"],
            prim_path=name, update_period=0.0, calibration={"unchanged": [1, 2, 3]})
        term = lambda camera_name, data_type: SimpleNamespace(params={"sensor_cfg": SimpleNamespace(name=camera_name), "data_type": data_type})
        side, wrist = camera("external_cam_side"), camera("wrist_cam")
        rgb_side, rgb_wrist = term("external_cam_side", "rgb"), term("wrist_cam", "rgb")
        reset_events = {"reset_object_pose": "unchanged"}
        config = SimpleNamespace(
            scene=SimpleNamespace(external_cam_side=side, wrist_cam=wrist, top_rgb_camera=object(), top_depth_camera=object(), robot=object()),
            observations=SimpleNamespace(policy=SimpleNamespace(external_cam_side=rgb_side, wrist_cam=rgb_wrist,
                semantic_segmentation_side=term("external_cam_side", "semantic_segmentation"))),
            sim={"dt": 0.00833333, "decimation": 30}, events=reset_events,
        )
        original_robot = config.scene.robot
        primitives = [{"type": "pick", "mode": "affordance"}, {"type": "place", "params": {"success_eval": True}}]
        task = SimpleNamespace(runtime=SimpleNamespace(use_background_replace=False), get_primitives=lambda **kwargs: copy.deepcopy(primitives))
        result = configure(config, task)
        self.assertIs(config.scene.external_cam_side, side)
        self.assertIs(config.scene.wrist_cam, wrist)
        self.assertIs(config.observations.policy.external_cam_side, rgb_side)
        self.assertIs(config.observations.policy.wrist_cam, rgb_wrist)
        self.assertIsNone(config.scene.top_rgb_camera)
        self.assertIsNone(config.scene.top_depth_camera)
        self.assertIsNone(config.observations.policy.semantic_segmentation_side)
        self.assertIs(config.scene.robot, original_robot)
        self.assertIs(config.events, reset_events)
        self.assertEqual(config.sim, {"dt": 0.00833333, "decimation": 30})
        self.assertEqual(task.get_primitives(), primitives)
        for name in ("external_cam_side", "wrist_cam"):
            self.assertEqual(result["cameras"][name]["data_types"], ["rgb"])
            self.assertEqual(result["cameras"][name]["width"], 640)
            self.assertEqual(result["cameras"][name]["height"], 480)
            self.assertEqual(getattr(config.scene, name).calibration, {"unchanged": [1, 2, 3]})

        # Incompatible dependencies are rejected before any scene mutation.
        for case in ("anygrasp", "background", "top_observation"):
            blocked = SimpleNamespace(scene=SimpleNamespace(external_cam_side=camera("side"), wrist_cam=camera("wrist"),
                top_rgb_camera=object(), top_depth_camera=object()), observations=SimpleNamespace(policy=SimpleNamespace()))
            top_before = blocked.scene.top_rgb_camera
            blocked_task = SimpleNamespace(runtime=SimpleNamespace(use_background_replace=case == "background"),
                get_primitives=lambda **kwargs: [{"type": "pick", "mode": "anygrasp" if case == "anygrasp" else "gt"}])
            if case == "top_observation":
                blocked.observations.policy.top = term("top_rgb_camera", "rgb")
            with self.assertRaises(ValueError):
                configure(blocked, blocked_task)
            self.assertIs(blocked.scene.top_rgb_camera, top_before)
            self.assertEqual(blocked.scene.external_cam_side.data_types, ["rgb", "distance_to_image_plane"])

    def test_actual_collector_keeps_partial_success_in_five_environment_batches(self):
        # Run the production primitive loop and collect/callback code with CPU
        # environment doubles; a partial batch must retain its successful trial.
        source = ROOT / "source/isaaclab/isaaclab/motion_collect/datacollector.py"
        tree = ast.parse(source.read_text())
        collector = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DataCollector")
        methods = [n for n in collector.body if isinstance(n, ast.FunctionDef) and n.name in {"collect", "_run_primitives"}]
        namespace = {
            "List": list, "Dict": dict, "Any": object, "Counter": Counter,
            "torch": SimpleNamespace(no_grad=nullcontext, cuda=SimpleNamespace(empty_cache=lambda: None)),
        }
        exec(compile(ast.Module(body=methods, type_ignores=[]), str(source), "exec"), namespace)
        outcomes = [[False, False, False, False, True], [False] * 5,
                    [False, True, False, False, False], [False] * 5]
        evaluated = []
        def physical_eval(mode, params, context):
            self.assertTrue(params["success_eval"])
            evaluated.append(context["episode_num"])
            mask = outcomes[context["episode_num"] - 1]
            context["task_success_mask"] = mask
            return all(mask)  # Even a handler's aggregate False must preserve survivors.
        records = []
        class CollectorDouble:
            collect = namespace["collect"]
            _run_primitives = namespace["_run_primitives"]
            episodes = 4
            num_envs = 5
            env_task_name = "cpu-evaluation-check"
            enable_motion_planning = False
            data_dir = None
            recorder = None
            manual_episode_reset = True
            def rebuild_scene_desc_from_yaml(self): return [{"object": "cup"}], "place cup"
            def _init_environment(self): pass
            def _reset_and_stabilize(self): pass
            def _build_primitive_context(self, **kwargs): return kwargs
            def _get_primitive_steps(self): return [{"type": "place", "params": {"success_eval": True}}]
            def _get_primitive_handlers(self): return {"place": physical_eval}
            def _prepare_primitive_episode(self, context): return True
            def _normalize_primitive_mode(self, value): return value
            def episode_result_callback(self, result): records.extend(RUNNER.split_episode_result(result, 5))
        with redirect_stdout(StringIO()):
            stats = CollectorDouble().collect()
        # This CPU collector double has no physical detector. Missing evidence
        # must not be silently counted as normal collisions.
        self.assertIsNone(RUNNER.summarize(records, 20)["success_rate"])
        for record in records:
            record["collision_flyaway"] = normal_collision_result()
        summary = RUNNER.summarize(records, 20)
        self.assertEqual(evaluated, [1, 2, 3, 4])
        self.assertEqual([r["trial"] for r in records if r["success"]], [5, 12])
        self.assertEqual(summary["valid_trials"], 20)
        self.assertEqual(summary["success_rate"], 0.1)
        self.assertEqual(summary["failure_reasons"], {"task_failed": 18})
        self.assertEqual(stats["successful_envs"], 2)
        self.assertEqual(stats["env_success_rate"], 0.1)
        self.assertEqual(stats["success_rate"], 0.5)  # Batch rate is separate.

    def test_manual_episode_reset_is_opt_in_and_preserves_reset_events(self):
        # Exercise the actual helper without starting Isaac Sim on the CPU host.
        source = ROOT / "source/isaaclab/isaaclab/motion_collect/datacollector.py"
        tree = ast.parse(source.read_text())
        collector = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DataCollector")
        init = next(n for n in collector.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
        defaults = dict(zip([a.arg for a in init.args.args][-len(init.args.defaults):], init.args.defaults))
        self.assertIs(ast.literal_eval(defaults["manual_episode_reset"]), False)
        self.assertIs(ast.literal_eval(defaults["rgb_only_cameras"]), False)
        helper = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_disable_automatic_episode_resets")
        namespace = {"List": list}
        exec(compile(ast.Module(body=[helper], type_ignores=[]), str(source), "exec"), namespace)
        term_type = type("TerminationTermCfg", (), {})
        mock_managers = ModuleType("isaaclab.managers")
        mock_managers.TerminationTermCfg = term_type
        events = SimpleNamespace(reset_scene=object(), randomize_object_pose=object())
        original_events = vars(events).copy()
        for terms in (SimpleNamespace(time_out=term_type(), object_dropping=term_type(), already_disabled=None),
                      {"time_out": term_type(), "object_dropping": term_type(), "already_disabled": None}):
            config = SimpleNamespace(terminations=terms, events=events, episode_length_s=100.0)
            with patch.dict("sys.modules", {"isaaclab.managers": mock_managers}):
                names = namespace["_disable_automatic_episode_resets"](config)
            self.assertEqual(names, ["time_out", "object_dropping"])
            values = terms if isinstance(terms, dict) else vars(terms)
            self.assertTrue(all(value is None for value in values.values()))
            self.assertIs(config.events, events)
            self.assertEqual(vars(config.events), original_events)
            self.assertEqual(config.episode_length_s, 100.0)

    def test_vectorized_episodes_split_exactly_twenty_trials_and_metadata(self):
        metadata = {
            "initial_env_states": [{"env_idx": i, "objects": {"cup": {"x": i}}} for i in range(4)],
            "randomization": {"texture_randomization_applied": {"assignments": [
                {"env_idx": i, "texture_file": f"texture_{i}.png"} for i in range(4)
            ]}},
        }
        records = []
        for episode in range(1, 6):
            result = {
                "episode": episode, "success": True,
                "successful_envs": [True, False, True, False],
                "failure_reasons": ["none", "planning_failed", "none", "task_failed"],
                "execution_errors": [], "execution_error_envs": [False] * 4,
                "initial_metadata": [metadata] * 4, "recorded_steps": 99,
                "collision_flyaway": [normal_collision_result() for _ in range(4)],
            }
            split = RUNNER.split_episode_result(result, 4)
            for i, trial in enumerate(split):
                self.assertEqual(trial["env_idx"], i)
                self.assertEqual(trial["initial_metadata"][0]["initial_env_states"], [metadata["initial_env_states"][i]])
                assignments = trial["initial_metadata"][0]["randomization"]["texture_randomization_applied"]["assignments"]
                self.assertEqual(assignments, [{"env_idx": i, "texture_file": f"texture_{i}.png"}])
            records.extend(split)
        self.assertEqual([r["trial"] for r in records], list(range(1, 21)))
        self.assertEqual(len(metadata["initial_env_states"]), 4)
        summary = RUNNER.summarize(records, 20)
        self.assertEqual(summary["valid_trials"], 20)
        self.assertEqual(summary["success_rate"], 0.5)
        self.assertEqual(summary["failure_reasons"], {"planning_failed": 5, "task_failed": 5})

    def test_vectorized_program_error_does_not_relabel_previously_failed_env(self):
        result = {
            "episode": 2, "success": False, "successful_envs": [False, False],
            "failure_reasons": ["task_failed", "planning_failed"],
            "execution_errors": ["RuntimeError: primitive broken"],
            "execution_error_envs": [False, True], "recorded_steps": 9,
            "collision_flyaway": [normal_collision_result(), normal_collision_result()],
        }
        split = RUNNER.split_episode_result(result, 2)
        self.assertEqual([r["trial"] for r in split], [3, 4])
        self.assertFalse(split[0]["program_error"])
        self.assertEqual(split[0]["failure_reason"], "task_failed")
        self.assertTrue(split[1]["program_error"])
        self.assertEqual(RUNNER.summarize(split, 20)["valid_trials"], 1)

    def test_task_seeds_are_stable_and_fixed_ranges_are_not_retuned(self):
        first = RUNNER.task_seed(42, "pick_banana_place_bowl")
        self.assertEqual(first, RUNNER.task_seed(42, "pick_banana_place_bowl"))
        self.assertNotEqual(first, RUNNER.task_seed(42, "pick_banana_place_plate"))
        self.assertNotEqual(first, RUNNER.task_seed(43, "pick_banana_place_bowl"))
        fixed = {"objects": {"cup": {"position_range": {"x": [0.5, 0.5], "z": [0.1, 0.4]}, "orientation_range": {"yaw": [0, 0]}}}}
        self.assertFalse(RUNNER.has_tunable_pose_ranges(fixed))
        fixed["objects"]["cup"]["orientation_range"]["yaw"] = [-0.2, 0.2]
        self.assertTrue(RUNNER.has_tunable_pose_ranges(fixed))

    def test_program_errors_are_excluded_from_success_rate(self):
        trials = [
            {"program_error": False, "success": True, "failure_reason": "none", "collision_flyaway": normal_collision_result()},
            {"program_error": False, "success": False, "failure_reason": "planning_failed", "collision_flyaway": normal_collision_result()},
            {"program_error": True, "success": False, "failure_reason": "program_error"},
        ]
        result = RUNNER.summarize(trials, 20)
        self.assertEqual(result["completed_trials"], 3)
        self.assertEqual(result["valid_trials"], 2)
        self.assertEqual(result["requested_trials"], 20)
        self.assertEqual(result["success_rate"], 0.5)
        self.assertEqual(result["failure_reasons"], {"planning_failed": 1})
        self.assertIsNone(RUNNER.summarize([trials[-1]], 20)["success_rate"])

    def test_collision_exclusion_counts_and_unknown_are_not_normalized_as_failures(self):
        def trial(success, detection, program_error=False):
            return {"success": success, "program_error": program_error,
                    "failure_reason": "none" if success else "task_failed",
                    "collision_flyaway": detection}
        excluded = excluded_collision_result()
        assessed = [trial(True, normal_collision_result()), trial(False, normal_collision_result()),
                    trial(False, excluded), trial(True, excluded)]
        out = RUNNER.summarize(assessed, 4)
        self.assertEqual((out["completed_trials"], out["valid_trials"], out["excluded_trials"]), (4, 2, 2))
        self.assertEqual(out["successes"], 1)
        self.assertEqual(out["success_rate"], 0.5)
        self.assertEqual(out["raw_successes"], 2)
        self.assertEqual(out["raw_success_rate"], 0.5)
        self.assertTrue(out["collision_assessment_complete"])
        self.assertEqual(out["failure_reasons"], {"task_failed": 1})
        all_excluded = RUNNER.summarize([trial(False, excluded)] * 10, 10)
        self.assertEqual(all_excluded["excluded_trials"], 10)
        self.assertEqual(all_excluded["valid_trials"], 0)
        self.assertTrue(all_excluded["collision_assessment_complete"])
        self.assertIsNone(all_excluded["success_rate"])
        for invalid in (None, {}, {"status": "unavailable", "excluded": None, "events": []},
                        {"status": "error", "excluded": None, "events": []},
                        {"status": "complete", "excluded": True, "events": []},
                        {"status": "complete", "excluded": "false", "events": []}):
            partial = RUNNER.summarize([trial(True, normal_collision_result()), trial(False, invalid)], 2)
            self.assertEqual(partial["collision_unknown_trials"], 1)
            self.assertEqual(partial["valid_trials"], 1)
            self.assertIsNone(partial["success_rate"])
            self.assertFalse(partial["collision_assessment_complete"])
        error = RUNNER.summarize([trial(False, None, True)], 1)
        self.assertEqual(error["program_error_trials"], 1)
        self.assertEqual(error["collision_unknown_trials"], 0)
        self.assertFalse(error["collision_assessment_complete"])

    def test_ten_attempts_cap_historical_20_environment_configuration(self):
        self.assertEqual(RUNNER.task_num_envs({"task_id": "a", "num_envs": 20}, 1, 10), 10)
        self.assertEqual(RUNNER.task_num_envs({"task_id": "b", "num_envs": 5}, 1, 10), 5)
        with self.assertRaises(ValueError):
            RUNNER.task_num_envs({"task_id": "c", "num_envs": 3}, 1, 10)

    def test_legacy_twenty_attempt_report_is_preserved_for_new_ten_attempt_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "source.yaml"
            config.write_text(yaml.safe_dump({"objects": {"cup": {}}, "primitives": [{"type": "pick"}]}))
            task = {"task_id": "cup", "num_envs": 20, "config_path": str(config), "robot": "robot",
                    "env_task_name": "env", "robot_cfg_name": "robot.yml", "robot_type": "robot"}
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"tasks": [task]}))
            previous = {"completed": True, "returncode": 0, "timed_out": False,
                        "summary": {"requested_trials": 20, "completed_trials": 20,
                                    "valid_trials": 20, "successes": 10, "success_rate": 0.5}}
            round_dir = root / "results/cup/round_00"
            round_dir.mkdir(parents=True)
            RUNNER.write_json(round_dir / "report.json", previous)
            (round_dir / "frame.webp").write_bytes(b"preserve")
            old_report = (round_dir / "report.json").read_bytes()
            args = SimpleNamespace(manifest=str(manifest), task=None, output_dir=str(root / "results"),
                no_adapt=False, trials=10, num_envs=1, seed=42, resume=True, device="cpu", timeout=10,
                stop_after_task=None)
            with patch.object(RUNNER.subprocess, "Popen", side_effect=AssertionError("must not start")), redirect_stdout(StringIO()):
                result = RUNNER.run_schedule(args)
            self.assertEqual(result["tasks"][0]["status"], "existing_report_preserved")
            self.assertEqual((round_dir / "report.json").read_bytes(), old_report)
            self.assertEqual((round_dir / "frame.webp").read_bytes(), b"preserve")
            self.assertEqual(list(round_dir.parent.glob("round_00_previous_*")), [])

    def test_unknown_and_zero_eligible_do_not_trigger_randomization_shrinking(self):
        for status in ("unknown", "all_excluded"):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                config = root / "source.yaml"
                config.write_text(yaml.safe_dump({"objects": {"cup": {"position_range": {"x": [0, 1]}}},
                                                  "primitives": [{"type": "pick"}]}))
                task = {"task_id": "cup", "num_envs": 20, "config_path": str(config), "robot": "robot",
                        "env_task_name": "env", "robot_cfg_name": "robot.yml", "robot_type": "robot"}
                manifest = root / "manifest.json"
                manifest.write_text(json.dumps({"tasks": [task]}))
                args = SimpleNamespace(manifest=str(manifest), task=None, output_dir=str(root / "results"),
                    no_adapt=False, trials=10, num_envs=1, seed=42, resume=True, device="cpu", timeout=10,
                    stop_after_task=None)
                calls = []
                class WorkerDouble:
                    def __init__(self, command, **kwargs):
                        self.command = command
                        calls.append(command)
                    def wait(self, timeout):
                        arg = lambda flag: self.command[self.command.index(flag) + 1]
                        trials = fake_classified_trials(10, 0)
                        for trial in trials:
                            trial["collision_flyaway"] = ({"status": "unavailable", "excluded": None, "events": []}
                                if status == "unknown" else excluded_collision_result())
                        RUNNER.write_json(Path(arg("--output-dir")) / "report.json", {
                            "completed": True, "trials": trials, "summary": RUNNER.summarize(trials, 10),
                            "seed": int(arg("--seed")), "robot": "robot", "num_envs": int(arg("--num-envs")),
                            **detector_provenance()})
                        return 0
                with patch.object(RUNNER.subprocess, "Popen", WorkerDouble), redirect_stdout(StringIO()):
                    result = RUNNER.run_schedule(args)
                self.assertEqual(len(calls), 1)
                self.assertEqual(calls[0][calls[0].index("--num-envs") + 1], "10")
                self.assertEqual(result["tasks"][0]["status"],
                                 "collision_detection_incomplete" if status == "unknown" else "no_eligible_trials")

    def test_split_records_preserve_per_environment_detector_evidence(self):
        excluded = excluded_collision_result()
        excluded["events"][0]["pair"] = ["a", "b"]
        result = {"episode": 1, "success": True, "successful_envs": [True, False],
                  "failure_reasons": ["none", "task_failed"], "recorded_steps": 5,
                  "collision_flyaway": [normal_collision_result(), excluded]}
        records = RUNNER.split_episode_result(result, 2)
        self.assertEqual([RUNNER.collision_flyaway_status(r) for r in records], ["complete", "excluded"])
        records[1]["collision_flyaway"]["events"][0]["pair"].append("changed")
        self.assertEqual(result["collision_flyaway"][1]["events"][0]["pair"], ["a", "b"])
        result["collision_flyaway"] = [normal_collision_result()]
        with self.assertRaises(ValueError):
            RUNNER.split_episode_result(result, 2)

    def test_tuning_changes_only_xy_yaw_ranges_and_runtime(self):
        source_data = {
            "simulation": {"decimation": 30, "dt": 0.00833333, "render_interval": 5},
            "runtime": {"use_background_replace": True,
                        "da3_background": {"image_path": "backgrounds/coffee_shop/view.png", "keep_table": False}},
            "objects": {"cup": {
                "usd_path": "cup.usd", "init_pos": [0.5, 0.2, 0.1],
                "position_range": {"x": [0.4, 0.6], "y": [0.2, 0.2], "z": [0.1, 0.3]},
                "orientation_range": {"yaw": [-0.6, 0.4], "roll": [-0.1, 0.1]},
            }},
            "primitives": [{"type": "pick", "mode": "affordance", "params": {"target": "cup"}}],
            "texture_randomization": {"enabled": True, "avoid_repeats": True},
        }
        original = copy.deepcopy(source_data)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.yaml"
            output = Path(directory) / "output.yaml"
            source.write_text(yaml.safe_dump(source_data))
            task = {"env_task_name": "env", "robot_cfg_name": "robot.yml"}
            result = RUNNER.prepare_config(source, output, trials=20, scale=0.5, task=task, num_envs=4)
            self.assertEqual(yaml.safe_load(source.read_text()), original)
        cup = result["objects"]["cup"]
        np.testing.assert_allclose(cup["position_range"]["x"], [0.45, 0.55])
        self.assertEqual(cup["position_range"]["z"], [0.1, 0.3])
        self.assertEqual(cup["position_range"]["y"], [0.2, 0.2])
        np.testing.assert_allclose(cup["orientation_range"]["yaw"], [-0.35, 0.15])
        self.assertEqual(cup["orientation_range"]["roll"], [-0.1, 0.1])
        self.assertEqual(cup["init_pos"], original["objects"]["cup"]["init_pos"])
        self.assertEqual(result["primitives"], original["primitives"])
        self.assertFalse(result["runtime"]["save_npy"])
        self.assertFalse(result["runtime"]["use_background_replace"])
        self.assertNotIn("da3_background", result["runtime"])
        self.assertTrue(result["texture_randomization"]["avoid_repeats"])
        self.assertEqual(result["scene"]["num_envs"], 4)
        self.assertEqual(result["runtime"]["episodes"], 5)
        self.assertEqual(result["simulation"], {"decimation": 30, "dt": 0.00833333, "render_interval": 30})

    def test_image_only_recording_keeps_frames_without_npy_or_observation_buffer(self):
        recorder_module = load_file(
            "episode_recorder_test",
            ROOT / "source/isaaclab_tasks/isaaclab_tasks/real2sim/utils/EpisodeBuffer.py",
        )
        with tempfile.TemporaryDirectory() as directory:
            recorder = recorder_module.EpisodeDataRecorder(
                save_dir=Path(directory), save_npy=False, save_camera_images=False,
                save_original_images=True, sim_camera_resolution=(80, 60),
            )
            recorder.start_episode(1, {"purpose": "successful-trial"})
            image = np.ones((1, 3 * 80 * 60), dtype=np.float32)
            obs = {"policy": {"external_cam_side": image}}
            recorder.add_step(obs, np.zeros((1, 8)), False, False)
            self.assertEqual(recorder.episode_buffers[0]["observations"], [])
            self.assertEqual(recorder.current_step_count, 1)
            recorder.save_episode(success=True)
            self.assertTrue((Path(directory) / "original_images/external_cam_side/env0_ep1/step_0000.png").is_file())
            self.assertFalse(list(Path(directory).rglob("*.npy")))
            recorder.start_episode(2, {"purpose": "failed-trial"})
            recorder.add_step(obs, np.zeros((1, 8)), False, False)
            recorder.episode_buffers = [None]
            self.assertTrue((Path(directory) / "original_images/external_cam_side/env0_ep2/step_0000.png").is_file())


if __name__ == "__main__":
    unittest.main()
