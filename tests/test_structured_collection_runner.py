"""Formal collection is opt-in and retains an isolated trajectory per attempt."""
from contextlib import redirect_stdout
import importlib.util
from io import StringIO
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("structured_collection_runner", ROOT / "scripts/run_benchmark_50.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_formal_recorder_flags_are_opt_in():
    recorder = SimpleNamespace(preserve_idle_steps=False, separate_image_observations=False)
    runner.configure_formal_recorder(recorder, False)
    assert not recorder.preserve_idle_steps and not recorder.separate_image_observations
    runner.configure_formal_recorder(recorder, True)
    assert recorder.preserve_idle_steps and recorder.separate_image_observations


@pytest.mark.parametrize('width', [6, 7])
def test_formal_recorder_measures_arm_state_in_actual_action_order(width):
    # Deliberately use a robot articulation order different from action order.
    order = [f'j{i}' for i in range(width)]
    articulation_order = ['finger'] + list(reversed(order))
    positions = torch.arange(width + 1, dtype=torch.float32)[None, :]
    velocities = positions + 100
    robot = SimpleNamespace(joint_names=articulation_order,
                            data=SimpleNamespace(joint_pos=positions, joint_vel=velocities))
    body = SimpleNamespace(_joint_names=order, action_dim=width)
    manager = SimpleNamespace(get_term=lambda name: body if name == 'body' else None)
    recorded = []
    recorder = SimpleNamespace(preserve_idle_steps=False, separate_image_observations=False,
                               add_step=lambda obs, *args, **kwargs: recorded.append(obs))
    assert runner.configure_formal_recorder(recorder, True, robot, manager) == order
    obs = {'policy': {'q_pos': torch.full((1, width), -99.), 'camera': 'untouched'}}
    recorder.add_step(obs, torch.full((1, width + 1), -555.), False, False)
    states = recorded[0]['policy']
    torch.testing.assert_close(states['q_pos'], torch.arange(width, 0, -1, dtype=torch.float32)[None, :])
    torch.testing.assert_close(states['q_vel'], torch.arange(width, 0, -1, dtype=torch.float32)[None, :] + 100)
    assert states['camera'] == 'untouched'
    assert torch.all(obs['policy']['q_pos'] == -99)  # Do not mutate environment observations.
    robot.data.joint_pos.zero_()
    assert states['q_pos'].sum().item() > 0  # Samples are snapshots, not mutable aliases.
    assert runner.configure_formal_recorder(recorder, True, robot, manager) == order


def test_formal_recorder_rejects_unknown_arm_joint():
    robot = SimpleNamespace(joint_names=['j0'], data=SimpleNamespace())
    manager = SimpleNamespace(get_term=lambda name: SimpleNamespace(_joint_names=['missing'], action_dim=1))
    recorder = SimpleNamespace(add_step=lambda *args: None)
    with pytest.raises(ValueError, match='missing or ambiguous'):
        runner.configure_formal_recorder(recorder, True, robot, manager)


def test_disabled_recorder_does_not_inject_or_override_states():
    called = []
    recorder = SimpleNamespace(add_step=lambda *args: called.append(args))
    original = recorder.add_step
    runner.configure_formal_recorder(recorder, False, object(), object())
    assert recorder.add_step is original
    assert not hasattr(recorder, '_formal_arm_joint_names')


def test_runtime_copy_has_correct_robot_and_opt_in_data(tmp_path):
    source = tmp_path / "source.yaml"
    source.write_text(yaml.safe_dump({"robot": {"type": "franka"},
                                     "simulation": {"decimation": 30, "render_interval": 5},
                                     "primitives": [{"type": "pick"}]}))
    task = {"env_task_name": "env", "robot_cfg_name": "ur.yml", "robot_type": "ur5e"}
    kwargs = dict(trials=1, scale=1, task=task, preserve_render_interval=True)
    evaluated = runner.prepare_config(source, tmp_path / "eval.yaml", **kwargs)
    collected = runner.prepare_config(source, tmp_path / "formal.yaml", save_trajectories=True, **kwargs)
    assert evaluated["runtime"]["save_npy"] is False
    assert collected["runtime"]["save_npy"] is True
    assert collected["runtime"]["save_original_images"] is True
    assert collected["robot"] == {"type": "ur5e", "cfg_name": "ur.yml"}
    assert collected["simulation"]["render_interval"] == 5
    assert yaml.safe_load(source.read_text())["robot"]["type"] == "franka"


def test_successful_trajectories_are_moved_to_distinct_attempts_without_overwrite(tmp_path):
    recording = tmp_path / "recording"
    recording.mkdir()
    expected = {"obs_policy_actions": np.array([[1., 2., 3.]])}
    np.save(recording / "episode_0000.npy", expected)
    target = tmp_path / "success/trial_001"
    result = Path(runner.stage_structured_trajectory(recording, target, 1, 0, 1, allow_legacy_npy=True))
    np.testing.assert_array_equal(np.load(result, allow_pickle=True).item()["obs_policy_actions"],
                                  expected["obs_policy_actions"])
    assert not (recording / "episode_0000.npy").exists()
    np.save(recording / "episode_0000.npy", expected)
    with pytest.raises(FileExistsError):
        runner.stage_structured_trajectory(recording, target, 1, 0, 1, allow_legacy_npy=True)
    assert (recording / "episode_0000.npy").exists()
    with pytest.raises(FileNotFoundError):
        runner.stage_structured_trajectory(recording, tmp_path / "success/trial_002", 2, 0, 1, allow_legacy_npy=True)


@pytest.mark.parametrize('width', [6, 7])
def test_formal_trajectory_records_arm_joint_names_and_checks_state_shape(tmp_path, width):
    recording = tmp_path / 'recording'; recording.mkdir()
    names = [f'j{i}' for i in range(width)]
    payload = {'obs_policy_actions': np.zeros((3, 7)),
               'obs_policy_q_pos': np.arange(3 * width).reshape(3, width),
               'obs_policy_q_vel': np.ones((3, width))}
    np.save(recording / 'episode_0000.npy', payload)
    path = runner.stage_structured_trajectory(recording, tmp_path / 'success', 1, 0, 1, names, allow_legacy_npy=True)
    result = np.load(path, allow_pickle=True).item()
    assert result['arm_joint_names'].tolist() == names
    np.testing.assert_array_equal(result['obs_policy_q_pos'], payload['obs_policy_q_pos'])
    np.save(recording / 'episode_0000.npy', payload)
    with pytest.raises(ValueError, match='Invalid measured'):
        runner.stage_structured_trajectory(recording, tmp_path / 'other', 1, 0, 1, names[:-1], allow_legacy_npy=True)


def test_schedule_passes_formal_flag_and_refuses_to_resume_if_npy_disappears(tmp_path):
    source = tmp_path / "source.yaml"
    source.write_text(yaml.safe_dump({"primitives": [{"type": "pick"}]}))
    task = {"task_id": "test", "config_path": str(source), "robot": "ur5e", "num_envs": 1,
            "env_task_name": "env", "robot_cfg_name": "ur.yml", "robot_type": "ur5e"}
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"tasks": [task]}))
    args = SimpleNamespace(manifest=str(manifest), task=None, output_dir=str(tmp_path / "out"),
        no_adapt=True, trials=1, num_envs=1, seed=42, resume=True, device="cpu", timeout=10,
        preserve_render_interval=True, save_trajectories=True)
    commands = []

    class WorkerDouble:
        def __init__(self, command, **kwargs):
            self.command = command
            commands.append(command)

        def wait(self, timeout):
            value = lambda name: self.command[self.command.index(name) + 1]
            output = Path(value("--output-dir"))
            trajectory = output / "success/trial_001/trajectory.npy"
            trajectory.parent.mkdir(parents=True)
            np.save(trajectory, {"obs_policy_actions": np.array([[0., 0.]])})
            collision = {"status": "complete", "excluded": False, "events": [],
                         "observed_substeps": 1, "thresholds": {"min_speed_m_s": 2}}
            trials = [{"trial": 1, "success": True, "program_error": False,
                       "failure_reason": "none", "collision_flyaway": collision,
                       "trajectory_path": str(trajectory)}]
            report = {"completed": True, "summary": runner.summarize(trials, 1), "trials": trials,
                      "seed": int(value("--seed")), "robot": "ur5e", "num_envs": 1,
                      "save_trajectories": True, "preserve_render_interval": True,
                      "collision_detector_source_sha256": runner.digest(ROOT / "source/isaaclab/isaaclab/motion_collect/collision_flyaway.py"),
                      "collision_detector_monitor_sha256": runner.digest(ROOT / "source/isaaclab/isaaclab/motion_collect/collision_flyaway_monitor.py"),
                      "collision_detector_collector_sha256": runner.digest(ROOT / "source/isaaclab/isaaclab/motion_collect/datacollector.py")}
            runner.write_json(output / "report.json", report)
            return 0

    with patch.object(runner.subprocess, "Popen", WorkerDouble), redirect_stdout(StringIO()):
        result = runner.run_schedule(args)
        assert result["tasks"][0]["status"] == "passed"
        assert "--save-trajectories" in commands[0]
        runtime = yaml.safe_load((tmp_path / "out/test/round_00/task.yaml").read_text())
        assert runtime["runtime"]["save_npy"] is True
        assert runtime["robot"]["type"] == "ur5e"
        runner.run_schedule(args)
        assert len(commands) == 1
        (tmp_path / "out/test/round_00/success/trial_001/trajectory.npy").unlink()
        result = runner.run_schedule(args)
        assert result["tasks"][0]["status"] == "existing_report_preserved"
        assert len(commands) == 1
