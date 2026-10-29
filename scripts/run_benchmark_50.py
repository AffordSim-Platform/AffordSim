"""Run fixed-attempt collection evaluations with one local simulator process per task.

The manifest contains ``tasks`` entries with task_id, config_path, robot,
env_task_name, robot_cfg_name and robot_type. Optional task-level num_envs
overrides --num-envs for all rounds of that task. Normal mode schedules workers;
--worker runs one task directly. Reports are flushed after every attempt.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]


def cabinet_initial_state_valid(joints, tolerance=0.01):
    """All drawer joints must be close to their configured closed position."""
    return bool(joints) and all(math.isfinite(value) and abs(value) <= tolerance
                                for value in joints.values())


def cabinet_initialization_target(task_id, primitives, articulations):
    """Resolve the drawer to check from the active task, not a fixed model name."""
    if task_id not in {'pull_cabinet', 'pull_cabinet7'}:
        return None
    targets = [step.get('params', {}).get('target') for step in primitives if step.get('type') == 'pull']
    if len(targets) != 1 or targets[0] not in articulations:
        raise ValueError(f'Expected one configured pull articulation for {task_id}: {targets}')
    return targets[0]


def local_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def task_seed(base_seed, task_id):
    """Stable across processes, task order and filtered/resumed schedules."""
    offset = int(hashlib.sha256(task_id.encode("utf-8")).hexdigest()[:8], 16)
    return (int(base_seed) + offset) % (2**31)


def task_num_envs(task, default, trials):
    value = task.get("num_envs", default)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"Task {task.get('task_id', '<unknown>')} num_envs must be a positive integer dividing {trials}, got {value!r}")
    # A historical 20-env configuration can run the new 10-attempt protocol
    # without launching unnecessary environments. Smaller configured batches
    # must still divide the attempt count exactly.
    value = min(value, trials)
    if trials % value:
        raise ValueError(f"Task {task.get('task_id', '<unknown>')} num_envs must be a positive integer dividing {trials}, got {value!r}")
    return value


def has_tunable_pose_ranges(data):
    for section in ("objects", "articulations"):
        for obj in (data.get(section) or {}).values():
            if not isinstance(obj, dict):
                continue
            for field, axes in (("position_range", ("x", "y")), ("orientation_range", ("yaw",))):
                for axis in axes:
                    interval = (obj.get(field) or {}).get(axis)
                    if isinstance(interval, (list, tuple)) and len(interval) == 2:
                        if float(interval[1]) - float(interval[0]) > 1e-12:
                            return True
    return False


def prepare_config(source, destination, *, trials, scale, task, num_envs=1,
                   preserve_render_interval=False, save_trajectories=False):
    """Keep object/primitive definitions; shrink existing XY/yaw intervals only."""
    import yaml

    data = yaml.safe_load(Path(source).read_text())
    if not isinstance(data, dict) or not data.get("primitives"):
        raise ValueError(f"Task YAML has no primitive sequence: {source}")
    data = copy.deepcopy(data)
    if num_envs <= 0 or trials % num_envs:
        raise ValueError("num_envs must be positive and divide the total trials")
    data.setdefault("scene", {})["num_envs"] = num_envs
    simulation = data.setdefault("simulation", {})
    # The default renders once at each recorded control-step boundary. An
    # explicit opt-in retains the task's own render interval instead.
    if not preserve_render_interval:
        simulation["render_interval"] = int(simulation.get("decimation", 30))
    for section in ("objects", "articulations"):
        for obj in (data.get(section) or {}).values():
            if not isinstance(obj, dict):
                continue
            for field, axes in (("position_range", ("x", "y")), ("orientation_range", ("yaw",))):
                ranges = obj.get(field) or {}
                for axis in axes:
                    interval = ranges.get(axis)
                    if not isinstance(interval, (list, tuple)) or len(interval) != 2:
                        continue
                    lo, hi = map(float, interval)
                    if hi < lo:
                        raise ValueError(f"Reversed interval in {section}.{field}.{axis}: {interval}")
                    center = (lo + hi) / 2
                    radius = (hi - lo) * scale / 2
                    ranges[axis] = [center - radius, center + radius]
    runtime = data.setdefault("runtime", {})
    runtime.update({
        "headless": True, "episodes": trials // num_envs, "enable_motion_planning": True,
        "task_name": task["env_task_name"], "save_npy": bool(save_trajectories),
        "save_camera_images": False, "save_original_images": True,
        "save_composite_images": False, "save_mask_images": False,
        "save_depth_images": False, "use_background_replace": False,
    })
    # Benchmark footage must expose the actual randomized tabletop. A DA3
    # selection otherwise re-enables replacement in RuntimeConfig.from_dict,
    # even when use_background_replace is False. Only alter this runtime copy.
    runtime.pop("da3_background", None)
    data.setdefault("robot", {})["cfg_name"] = task["robot_cfg_name"]
    if task.get("robot_type"):
        data["robot"]["type"] = task["robot_type"]
    texture = data.setdefault("texture_randomization", {})
    texture["enabled"] = True
    texture.setdefault("texture_dir", "models/random_pic/table")
    texture.setdefault("targets", {})["table"] = {
        "prim_path": "/World/envs/env_{i}/table/Xform_01/Xform/leg/node_/Cube",
        "texture_dir": "models/random_pic/table",
    }
    data["enable_objects_randomization"] = True
    # Evaluation uses the declared ranges, not a training curriculum.
    data["enable_objects_randomization_curriculum"] = False
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False))
    return data


def stage_structured_trajectory(recording_dir, target_dir, episode, env_idx, num_envs, arm_joint_names=None,
                                *, allow_legacy_npy=False):
    """Move the recorder's successful episode into its own attempt directory.

    The recorder indexes episodes from zero across environments. The target is
    unique to one attempted episode, including collision-excluded successes.
    Missing or duplicate data is an error rather than a silently lost recording.
    """
    if not recording_dir:
        raise FileNotFoundError("Recorder directory is missing for a successful trajectory")
    # The current recorder writes HDF5 directly. Retain the older NPY branch
    # below solely for staging legacy recordings, never convert a new episode
    # to NPY or change its actions while moving it into the trial directory.
    episode_number = ((episode - 1) * num_envs + env_idx)
    hdf5_source = Path(recording_dir) / f"episode_{episode_number:04d}.hdf5"
    if hdf5_source.is_file():
        import h5py
        import numpy as np

        destination = Path(target_dir) / "trajectory.hdf5"
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite trajectory: {destination}")
        with h5py.File(hdf5_source, 'r') as data:
            action = data['action']
            qpos, qvel = data['observations/qpos'], data['observations/qvel']
            if (action.ndim != 2 or not len(action) or qpos.ndim != 2
                    or qvel.shape != qpos.shape or len(qpos) != len(action)
                    or not all(np.isfinite(values[...]).all() for values in (action, qpos, qvel))):
                raise ValueError('Invalid direct HDF5 action or joint state')
            if arm_joint_names is not None:
                names = list(arm_joint_names)
                stored = list(data['recordings/trajectory_fields/arm_joint_names'].asstr()[...])
                if (not names or len(set(names)) != len(names) or len(names) != qpos.shape[1]
                        or names != stored):
                    raise ValueError('HDF5 arm joint names do not match measured joint state')
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(hdf5_source), str(destination))
        return str(destination)
    if not allow_legacy_npy:
        raise FileNotFoundError(f"Direct HDF5 episode is missing: {hdf5_source}")
    source = Path(recording_dir) / f"episode_{((episode - 1) * num_envs + env_idx):04d}.npy"
    if not source.is_file():
        raise FileNotFoundError(f"Structured trajectory was not saved: {source}")
    destination = Path(target_dir) / "trajectory.npy"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite trajectory: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if arm_joint_names is None:
        shutil.move(str(source), str(destination))
    else:
        import numpy as np

        data = np.load(source, allow_pickle=True).item()
        names = list(arm_joint_names)
        if not names or len(set(names)) != len(names):
            raise ValueError(f"Invalid ordered arm joint names: {names}")
        rows = len(data['obs_policy_actions'])
        for key in ('obs_policy_q_pos', 'obs_policy_q_vel'):
            states = np.asarray(data[key])
            if states.shape != (rows, len(names)) or not np.isfinite(states).all():
                raise ValueError(f"Invalid measured {key} shape/values: {states.shape}; expected {(rows, len(names))}")
        data['arm_joint_names'] = np.asarray(names, dtype=str)
        temporary = destination.with_suffix('.npy.tmp')
        with temporary.open('wb') as stream:
            np.save(stream, data, allow_pickle=True)
        temporary.replace(destination)
        source.unlink()
    return str(destination)


def configure_formal_recorder(recorder, enabled, robot=None, action_manager=None):
    """Preserve timing and record actual arm state in the action term's order."""
    if enabled:
        recorder.preserve_idle_steps = True
        # Direct HDF5 embeds the camera observations; removing them here
        # makes the episode writer fail and would lose the training images.
        recorder.separate_image_observations = not bool(getattr(recorder, 'save_hdf5', False))
        if robot is None or action_manager is None:
            return None
        if hasattr(recorder, '_formal_arm_joint_names'):
            return list(recorder._formal_arm_joint_names)
        term = action_manager.get_term('body')
        names = list(term._joint_names)
        if not names or len(names) != len(set(names)) or len(names) != term.action_dim:
            raise ValueError(f"Arm action term has invalid joint ordering: {names}")
        all_names = list(robot.joint_names)
        if len(all_names) != len(set(all_names)) or any(name not in all_names for name in names):
            raise ValueError(f"Arm action joints are missing or ambiguous in robot joint names: {names}")
        indices = [all_names.index(name) for name in names]
        original_add_step = recorder.add_step

        def add_step_with_measured_arm(obs, *args, **kwargs):
            if not isinstance(obs, dict) or not isinstance(obs.get('policy'), dict):
                raise TypeError('Formal collection needs a policy observation dictionary')
            updated = dict(obs)
            policy = dict(obs['policy'])
            # Snapshot the articulation state after the action, at exactly the
            # same step as the observation and image passed to add_step.
            policy['q_pos'] = robot.data.joint_pos[:, indices].clone()
            policy['q_vel'] = robot.data.joint_vel[:, indices].clone()
            updated['policy'] = policy
            return original_add_step(updated, *args, **kwargs)

        recorder.add_step = add_step_with_measured_arm
        recorder._formal_arm_joint_names = tuple(names)
        return names
    return None


def collision_flyaway_status(trial):
    """Return complete/excluded/unknown without inferring missing evidence.

    An excluded event must carry evidence. Missing/malformed detector data and
    unavailable or error statuses remain unknown, including legacy trials.
    """
    detection = trial.get("collision_flyaway")
    if not isinstance(detection, dict) or detection.get("status") != "complete":
        return "unknown"
    excluded, events = detection.get("excluded"), detection.get("events")
    if type(excluded) is not bool or not isinstance(events, list):
        return "unknown"
    observed = detection.get("observed_substeps")
    if (type(observed) is not int or observed <= 0 or
            not isinstance(detection.get("thresholds"), dict) or not detection["thresholds"]):
        return "unknown"
    if excluded != bool(events):
        return "unknown"
    required = {"object", "other_object", "collision_time_s", "detection_time_s",
                "impulse_ns", "velocity_jump_m_s", "displacement_m"}
    if any(not isinstance(event, dict) or not required.issubset(event) for event in events):
        return "unknown"
    return "excluded" if excluded else "complete"


def summarize(trials, expected):
    """Summarize fixed attempts and preserve raw and detector-qualified counts."""
    healthy = [trial for trial in trials if not trial.get("program_error", False)]
    excluded = [trial for trial in healthy if collision_flyaway_status(trial) == "excluded"]
    unknown = [trial for trial in healthy if collision_flyaway_status(trial) == "unknown"]
    valid = [trial for trial in healthy if collision_flyaway_status(trial) == "complete"]
    successes = sum(bool(trial.get("success")) for trial in valid)
    raw_successes = sum(bool(trial.get("success")) for trial in healthy)
    failures = Counter(trial.get("failure_reason", "unknown") for trial in valid if not trial.get("success"))
    return {
        "requested_trials": expected, "completed_trials": len(trials),
        "valid_trials": len(valid), "program_error_trials": len(trials) - len(healthy),
        "excluded_trials": len(excluded), "collision_unknown_trials": len(unknown),
        "collision_assessment_complete": (len(trials) == expected and not unknown
                                          and len(healthy) == len(trials)),
        "successes": successes,
        "success_rate": successes / len(valid) if valid and not unknown else None,
        "raw_successes": raw_successes,
        "raw_success_rate": raw_successes / len(healthy) if healthy else None,
        "failure_reasons": dict(failures),
    }


def assessed_report_matches(report, expected):
    """Reject stale or forged summaries when resuming a completed round."""
    trials = report.get("trials")
    if not isinstance(trials, list) or len(trials) != expected:
        return False
    if any(not isinstance(t, dict) or type(t.get("trial")) is not int for t in trials):
        return False
    if sorted(t["trial"] for t in trials) != list(range(1, expected + 1)):
        return False
    actual = summarize(trials, expected)
    summary = report.get("summary")
    return bool(isinstance(summary, dict) and actual["collision_assessment_complete"]
                and all(summary.get(key) == value for key, value in actual.items()))


def split_episode_result(result, num_envs):
    """Convert one vectorized episode into independent, correctly attributed trials."""
    if len(result["successful_envs"]) != num_envs or len(result["failure_reasons"]) != num_envs:
        raise ValueError("Episode result masks do not match num_envs")
    shared_errors = list(result.get("execution_errors", []))
    error_mask = result.get("execution_error_envs", [bool(shared_errors)] * num_envs)
    if len(error_mask) != num_envs:
        raise ValueError("Episode error mask does not match num_envs")
    metadata_list = result.get("initial_metadata", [])
    success_metrics = result.get("task_success_metrics", [])
    flyaway = result.get("collision_flyaway")
    if flyaway is not None and (not isinstance(flyaway, list) or len(flyaway) != num_envs):
        raise ValueError("Collision detector results must match num_envs")
    records = []
    episode = int(result["episode"])
    for env_idx in range(num_envs):
        errors = shared_errors if error_mask[env_idx] else []
        success = bool(result["success"]) and bool(result["successful_envs"][env_idx]) and not errors
        metadata = copy.deepcopy(metadata_list[env_idx]) if env_idx < len(metadata_list) else {}
        states = metadata.get("initial_env_states")
        if isinstance(states, list):
            metadata["initial_env_states"] = [state for state in states if state.get("env_idx") == env_idx]
        applied = metadata.get("randomization", {}).get("texture_randomization_applied", {})
        if isinstance(applied.get("assignments"), list):
            applied["assignments"] = [assignment for assignment in applied["assignments"] if assignment.get("env_idx", env_idx) == env_idx]
        metadata["env_idx"] = env_idx
        records.append({
            "trial": (episode - 1) * num_envs + env_idx + 1,
            "episode": episode, "env_idx": env_idx,
            "success": success, "program_error": bool(errors),
            "failure_reason": "program_error" if errors else result["failure_reasons"][env_idx],
            "execution_errors": errors,
            "recorded_steps": result["recorded_steps"],
            "initial_metadata": [metadata],
            "task_success_metrics": copy.deepcopy(success_metrics[env_idx]) if env_idx < len(success_metrics) else {},
            "collision_flyaway": copy.deepcopy(flyaway[env_idx]) if flyaway is not None else {
                "status": "unavailable", "excluded": None, "events": [],
                "reason": "no collision detector result for this episode",
            },
        })
        if "phase_depenetration_events" in result:
            events = []
            for event in result["phase_depenetration_events"]:
                item = {key: copy.deepcopy(value) for key, value in event.items()
                        if key not in {"active_envs", "physx_readback"}}
                active = event.get("active_envs", [])
                item["env_active"] = bool(active[env_idx]) if env_idx < len(active) else False
                if "physx_readback" in event:
                    item["physx_readback"] = copy.deepcopy(event["physx_readback"].get(str(env_idx), {}))
                events.append(item)
            records[-1]["phase_depenetration_events"] = events
    return records


def call_ik_with_override(delegate, override, record_call, *motion_args, **motion_kwargs):
    """Override a collector IK call without changing ordinary collection.

    The bound collector interface permits the interpolation count and
    description as either positional or keyword arguments. Record the value
    originally requested by the primitive and the count actually passed on.
    """
    if override is None:
        return delegate(*motion_args, **motion_kwargs)
    if type(override) is not int or override <= 0:
        raise ValueError("interpolation_steps override must be a positive integer")
    args = list(motion_args)
    kwargs = dict(motion_kwargs)
    if len(args) > 3:
        requested = args[3]
        args[3] = override
    else:
        requested = kwargs.get("interpolation_steps", 30)
        kwargs["interpolation_steps"] = override
    desc = args[4] if len(args) > 4 else kwargs.get("desc", "IK Motion")
    record_call({"requested": requested, "effective": override, "desc": str(desc)})
    return delegate(*args, **kwargs)


def run_worker(args):
    # Pin every project module to this checkout before starting Isaac Sim.
    from test_backgrounds import local_modules
    from isaaclab.app import AppLauncher

    output = local_path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = local_path(args.config)
    report = {
        "task_id": args.task_id, "robot": args.robot, "repository": str(ROOT),
        "config_path": str(config), "config_sha256": digest(config),
        "seed": args.seed, "num_envs": args.num_envs,
        "manual_episode_reset": True,
        "original_image_encoding": "webp_lossless",
        "rgb_only_cameras": True,
        "episodes": args.trials // args.num_envs, "trials": [], "completed": False,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "modules": local_modules(), "summary": summarize([], args.trials),
        "preserve_render_interval": bool(getattr(args, "preserve_render_interval", False)),
        "save_trajectories": bool(getattr(args, "save_trajectories", False)),
        "interpolation_steps_override": getattr(args, "interpolation_steps", None),
        "interpolation_steps_calls": [],
    }
    detector_source = ROOT / "source/isaaclab/isaaclab/motion_collect/collision_flyaway.py"
    report["collision_detector_source_sha256"] = digest(detector_source) if detector_source.is_file() else None
    detector_monitor = ROOT / "source/isaaclab/isaaclab/motion_collect/collision_flyaway_monitor.py"
    report["collision_detector_monitor_sha256"] = digest(detector_monitor) if detector_monitor.is_file() else None
    report["collision_detector_collector_sha256"] = digest(ROOT / "source/isaaclab/isaaclab/motion_collect/datacollector.py")
    if args.verify_depenetration is not None:
        physics_source = ROOT / "source/isaaclab_tasks/isaaclab_tasks/real2sim/tasks/pick_r2s_env_cfg.py"
        report["physics_source_sha256"] = digest(physics_source)
        report["expected_max_depenetration_velocity"] = args.verify_depenetration
        report["expected_depenetration_overrides"] = dict(
            (key, float(value)) for key, value in (item.rsplit("=", 1) for item in args.verify_depenetration_override))
    report_path = output / "report.json"
    write_json(report_path, report)
    app = None
    try:
        app = AppLauncher(headless=True, enable_cameras=True, device=args.device).app
        from isaaclab.motion_collect.datacollector import DataCollector
        from isaaclab_tasks.real2sim.utils.task_config_loader import load_task_config
        import numpy as np
        from PIL import Image

        random.seed(args.seed)
        np.random.seed(args.seed)

        task_config = load_task_config(str(config))
        if getattr(args, "save_trajectories", False) and not task_config.runtime.save_npy:
            raise ValueError("--save-trajectories requires runtime.save_npy: true in the worker config")
        phase_config = getattr(task_config, "phase_depenetration", None)
        if phase_config is not None:
            source_files = {
                "policy": ROOT / "source/isaaclab/isaaclab/motion_collect/phase_depenetration.py",
                "collector": ROOT / "source/isaaclab/isaaclab/motion_collect/datacollector.py",
                "loader": ROOT / "source/isaaclab_tasks/isaaclab_tasks/real2sim/utils/task_config_loader.py",
                "place": ROOT / "source/isaaclab/isaaclab/motion_collect/primitives/place.py",
                "hang": ROOT / "source/isaaclab/isaaclab/motion_collect/primitives/hang.py",
            }
            report["phase_depenetration"] = {"config": copy.deepcopy(phase_config),
                                              "source_sha256": {key: digest(path) for key, path in source_files.items()}}
        report["simulation"] = {key: getattr(task_config.simulation, key) for key in ("dt", "decimation", "render_interval")}

        class BenchmarkCollector(DataCollector):
            def execute_ik_motion(self, *motion_args, **motion_kwargs):
                def record_call(entry):
                    report["interpolation_steps_calls"].append(entry)
                    write_json(report_path, report)

                return call_ik_with_override(
                    super().execute_ik_motion, getattr(args, "interpolation_steps", None),
                    record_call, *motion_args, **motion_kwargs,
                )

            def _init_environment(self):
                super()._init_environment()
                from isaaclab.motion_collect.tools.camera_framing import FrameVisibilityMonitor
                self.visibility_monitor = FrameVisibilityMonitor(self)
                self.cabinet_retreat_monitor = None
                primitives = self._get_primitive_steps()
                monitor_enabled = any(step.get('type') == 'return_home' and
                    step.get('params', {}).get('monitor_robot_cabinet_contact', False)
                    for step in primitives)
                if monitor_enabled:
                    from cabinet_retreat_monitor import RobotCabinetContactMonitor, UnavailableRetreatContactMonitor
                    try:
                        cabinet_name = cabinet_initialization_target(
                            args.task_id, primitives, self.task_config.articulations)
                        if cabinet_name is None:
                            raise ValueError('Contact monitoring requires a configured cabinet pull task')
                        self.cabinet_retreat_monitor = RobotCabinetContactMonitor(self, cabinet_name)
                        manager = self.env.env.recorder_manager
                        original_post_physics = manager.record_post_physics_decimation_step
                        def post_physics_with_retreat_contacts():
                            result = original_post_physics()
                            self.cabinet_retreat_monitor.after_substep()
                            return result
                        manager.record_post_physics_decimation_step = post_physics_with_retreat_contacts
                    except Exception as exc:
                        reason = f'{type(exc).__name__}: {exc}'
                        self.cabinet_retreat_monitor = UnavailableRetreatContactMonitor(self.num_envs, reason)
                        print(f'[WARN] Cabinet retreat contact monitor unavailable: {reason}', flush=True)
                original_step = self.env.step

                def step_with_visibility(*args, **kwargs):
                    result = original_step(*args, **kwargs)
                    self.visibility_monitor.sample()
                    return result

                if self.task_config.visibility_check is not None:
                    self.env.step = step_with_visibility
                mass_expectations = {name: obj.mass for name, obj in self.task_config.objects.items()
                                     if obj.mass is not None}
                if args.verify_mass:
                    name, value = args.verify_mass.rsplit("=", 1)
                    expected = float(value)
                    if name in mass_expectations and abs(mass_expectations[name] - expected) > 1e-8:
                        raise RuntimeError(f"Requested and configured masses disagree for {name}")
                    mass_expectations[name] = expected
                mass_records = {}
                for name, expected in mass_expectations.items():
                    asset = self.scene[name]
                    masses = asset.root_physx_view.get_masses().cpu().numpy()
                    inertias = asset.root_physx_view.get_inertias().cpu().numpy()
                    if not np.isfinite(masses).all() or not np.allclose(masses, expected, atol=1e-5, rtol=0):
                        raise RuntimeError(f"Unexpected simulated mass for {name}: {masses}")
                    if not np.isfinite(inertias).all() or not np.any(inertias > 0):
                        raise RuntimeError(f"Invalid simulated inertia for {name}: {inertias}")
                    mass_records[name] = {"asset": name, "expected": expected,
                                          "actual": masses.tolist(), "inertias": inertias.tolist()}
                    if args.verify_mass and name == args.verify_mass.rsplit("=", 1)[0]:
                        report["verified_mass_kg"] = mass_records[name]
                    write_json(report_path, report)
                    print("MASS_VERIFIED", name, masses.tolist(), flush=True)
                if mass_records:
                    report["verified_masses_kg"] = mass_records
                    write_json(report_path, report)
                if args.verify_depenetration is None:
                    return
                from pxr import PhysxSchema, Usd
                measured = {}
                names = list(self.task_config.objects) + list(self.task_config.articulations)
                overrides = report["expected_depenetration_overrides"]
                if set(overrides) - set(names):
                    raise RuntimeError(f"Unknown depenetration override assets: {set(overrides) - set(names)}")
                for name in names:
                    prim = self.world.stage.GetPrimAtPath(f"/World/envs/env_0/{name}")
                    if not prim.IsValid():
                        raise RuntimeError(f"Missing task asset prim: {name}")
                    values = [PhysxSchema.PhysxRigidBodyAPI(child).GetMaxDepenetrationVelocityAttr().Get()
                              for child in Usd.PrimRange(prim) if child.HasAPI(PhysxSchema.PhysxRigidBodyAPI)]
                    expected = overrides.get(name, args.verify_depenetration)
                    if not values or any(v is None or abs(v - expected) > 1e-6 for v in values):
                        raise RuntimeError(f"Unexpected max_depenetration_velocity for {name}: {values}")
                    measured[name] = values
                report["max_depenetration_velocity_verified"] = measured
                write_json(report_path, report)
                print("DEPenetration_VERIFIED", json.dumps(measured), flush=True)

            def _prepare_primitive_episode(self, context):
                if self.recorder is not None:
                    self.recorder.original_image_format = "webp_lossless"
                    # Preview-only collection records held poses in the video
                    # without creating structured imitation-learning data.
                    if getattr(args, "preview_only", False):
                        self.recorder.preserve_idle_steps = True
                    self.formal_arm_joint_names = configure_formal_recorder(
                        self.recorder, getattr(args, "save_trajectories", False),
                        robot=self.robot, action_manager=self.env.env.action_manager)
                    if self.formal_arm_joint_names:
                        report['arm_joint_names'] = list(self.formal_arm_joint_names)
                already = context.get("_prepared", False)
                prepared = super()._prepare_primitive_episode(context)
                if prepared and not already:
                    self.visibility_monitor.reset()
                    self.visibility_monitor.sample()
                    # These drawer assets can open during settling, before the
                    # robot acts. Record the real post-settle joints so an
                    # already-open cabinet cannot count as a successful pull.
                    cabinet_name = cabinet_initialization_target(
                        args.task_id, self._get_primitive_steps(), self.task_config.articulations)
                    if cabinet_name:
                        cabinet = self.scene[cabinet_name]
                        joint_values = cabinet.data.joint_pos.detach().cpu().numpy()
                        self._initial_cabinet_joints = [
                            dict(zip(cabinet.joint_names, map(float, row))) for row in joint_values
                        ]
                        report['cabinet_initialization'] = {
                            'fixed_base': bool(cabinet.is_fixed_base),
                            'joint_names': list(cabinet.joint_names),
                            'joint_stiffness': cabinet.data.joint_stiffness.detach().cpu().tolist(),
                            'joint_damping': cabinet.data.joint_damping.detach().cpu().tolist(),
                            'post_settle_joint_positions': self._initial_cabinet_joints,
                        }
                        print('INITIAL_CABINET_JOINTS', json.dumps(self._initial_cabinet_joints), flush=True)
                    # Even a plan failure with zero motion retains an initial RGB frame.
                    camera = self.scene["external_cam_side"]
                    for env_idx in range(self.num_envs):
                        number = (int(context["episode_num"]) - 1) * self.num_envs + env_idx + 1
                        start = output / "pending" / f"trial_{number:03d}"
                        start.mkdir(parents=True, exist_ok=True)
                        rgb = camera.data.output["rgb"][env_idx, ..., :3].detach().cpu().numpy()
                        if rgb.dtype != np.uint8:
                            rgb = np.clip(rgb * (255 if rgb.max() <= 1 else 1), 0, 255).astype(np.uint8)
                        Image.fromarray(rgb).save(start / "initial.png")
                return prepared

        def on_episode(result):
            report["disabled_termination_terms"] = collector.disabled_termination_terms
            report["collection_camera_configuration"] = collector.collection_camera_configuration
            for trial in split_episode_result(result, args.num_envs):
                number = trial["trial"]
                env_idx = trial["env_idx"]
                retreat_monitor = getattr(collector, 'cabinet_retreat_monitor', None)
                if retreat_monitor is not None:
                    trial['cabinet_retreat_contact'] = retreat_monitor.results()[env_idx]
                initial_joints = getattr(collector, '_initial_cabinet_joints', None)
                if initial_joints is not None:
                    joint_positions = initial_joints[env_idx]
                    trial['initial_articulation_joints'] = joint_positions
                    if not cabinet_initial_state_valid(joint_positions):
                        trial['success_before_initialization_check'] = trial['success']
                        trial['success'] = False
                        trial['failure_reason'] = 'cabinet_open_before_robot_action'
                if collector.task_config.visibility_check is not None:
                    trial["side_camera_visibility"] = collector.visibility_monitor.summary(env_idx)
                episode = trial["episode"]
                classification = collision_flyaway_status(trial)
                status = ("program_error" if trial["program_error"] else
                          "collision_excluded" if classification == "excluded" else
                          "collision_unknown" if classification == "unknown" else
                          "success" if trial["success"] else "failure")
                target = output / status / f"trial_{number:03d}"
                if getattr(args, "save_trajectories", False) and target.exists():
                    raise FileExistsError(f"Refusing to overwrite attempt directory: {target}")
                target.mkdir(parents=True, exist_ok=True)
                initial = output / "pending" / f"trial_{number:03d}" / "initial.png"
                if initial.is_file():
                    initial.replace(target / "initial.png")
                recording = result.get("recording_dir")
                frame_counts = {}
                if recording:
                    recording = Path(recording)
                    for category in ("original_images", "composite_images", "mask_images", "camera_images", "depth_images"):
                        category_path = recording / category
                        if not category_path.is_dir():
                            continue
                        for camera in category_path.iterdir():
                            source = camera / f"env{env_idx}_ep{episode}"
                            if not source.is_dir():
                                continue
                            destination = target / category / camera.name
                            destination.parent.mkdir(parents=True, exist_ok=True)
                            shutil.move(str(source), str(destination))
                            frame_counts[f"{category}/{camera.name}"] = sum(1 for p in destination.iterdir() if p.is_file())
                trajectory = None
                if getattr(args, "save_trajectories", False) and trial["success"]:
                    trajectory = stage_structured_trajectory(
                        recording, target, episode, env_idx, args.num_envs,
                        arm_joint_names=collector.formal_arm_joint_names)
                trial.update({"frame_counts": frame_counts, "directory": str(target),
                              "trajectory_path": trajectory})
                write_json(target / "trial.json", trial)
                report["trials"].append(trial)
                report["summary"] = summarize(report["trials"], args.trials)
                write_json(report_path, report)
                print("BENCHMARK_TRIAL", json.dumps({k:v for k,v in trial.items() if k != "initial_metadata"}), flush=True)
            if len(report["trials"]) >= 3 and all(t["program_error"] for t in report["trials"][-3:]):
                raise RuntimeError("Three consecutive program-error trials; repair the configuration before continuing")

        collector = BenchmarkCollector(
            env_task_name=args.env_task_name or task_config.runtime.task_name,
            task_config_path=str(config), robot_cfg_name=args.robot_cfg_name or task_config.robot.cfg_name,
            robot_type=args.robot_type, headless=True, enable_cameras=True,
            enable_motion_planning=True, device=args.device, num_envs=args.num_envs,
            episodes=args.trials // args.num_envs, seed=args.seed, data_dir=output / "recordings",
            task_name=task_config.task.name, save_npy=bool(getattr(args, "save_trajectories", False)), save_camera_images=False,
            save_original_images=True, episode_result_callback=on_episode,
            manual_episode_reset=True,
            rgb_only_cameras=True,
            detect_collision_flyaway=True,
        )
        report["collector_stats"] = collector.collect()
        report["summary"] = summarize(report["trials"], args.trials)
        report["completed"] = len(report["trials"]) == args.trials and not report["summary"]["program_error_trials"]
        report["modules_after_collection"] = local_modules()
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(report_path, report)
        print("BENCHMARK_RESULT", json.dumps(report["summary"]), flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        if app is not None and report["completed"]:
            app.close(skip_cleanup=True)
        # Failure cleanup can hang in native camera destructors. All data is flushed.
        os._exit(0 if report["completed"] else 1)


def run_schedule(args):
    import yaml

    manifest_path = local_path(args.manifest)
    manifest = json.loads(manifest_path.read_text())
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("Manifest must contain a nonempty tasks list")
    selected = set(args.task or [])
    if selected:
        missing = selected - {task["task_id"] for task in tasks}
        if missing:
            raise ValueError(f"Unknown task ids: {sorted(missing)}")
        tasks = [task for task in tasks if task["task_id"] in selected]
    output = local_path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    scales = [1.0] if args.no_adapt else [1.0, 0.75, 0.5]
    # Validate all selected batch sizes before starting any simulator workers.
    environment_counts = [task_num_envs(task, args.num_envs, args.trials) for task in tasks]
    if getattr(args, "save_trajectories", False) and (args.trials != 1 or any(n != 1 for n in environment_counts)):
        raise ValueError("--save-trajectories requires one attempt and one environment per worker")
    stop_file = local_path(args.stop_after_task) if getattr(args, "stop_after_task", None) else None
    summary = {"manifest": str(manifest_path), "manifest_sha256": digest(manifest_path), "tasks": [],
               "stop_after_task_file": str(stop_file) if stop_file is not None else None, "stopped": False}
    for task_index, (task, num_envs) in enumerate(zip(tasks, environment_counts)):
        minimum_free_gb = getattr(args, "minimum_free_gb", 0.0)
        if minimum_free_gb and shutil.disk_usage(output).free < minimum_free_gb * (1024 ** 3):
            summary.update({"stopped": True, "stop_reason": "insufficient_free_disk_space",
                            "remaining_task_ids": [item["task_id"] for item in tasks[task_index:]]})
            write_json(output / "summary.json", summary)
            print(f"STOP_LOW_DISK minimum_free_gb={minimum_free_gb}", flush=True)
            break
        # Never interrupt a worker or the current task's adaptive rounds. Keep
        # the marker in place so another invocation cannot accidentally resume.
        if stop_file is not None and stop_file.exists():
            summary.update({"stopped": True, "stop_reason": "stop_after_task_file_exists",
                            "stopped_after_task": tasks[task_index - 1]["task_id"] if task_index else None,
                            "remaining_task_ids": [item["task_id"] for item in tasks[task_index:]]})
            write_json(output / "summary.json", summary)
            print(f"STOP_AFTER_TASK {stop_file}; no further task started", flush=True)
            break
        for key in ("task_id", "config_path", "robot", "env_task_name", "robot_cfg_name", "robot_type"):
            if not task.get(key):
                raise ValueError(f"Missing task manifest field {key}: {task}")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", task["task_id"]):
            raise ValueError(f"Unsafe task id: {task['task_id']}")
        source = local_path(task["config_path"])
        seed = task_seed(args.seed, task["task_id"])
        source_data = yaml.safe_load(source.read_text())
        tunable = has_tunable_pose_ranges(source_data)
        detector_source = ROOT / "source/isaaclab/isaaclab/motion_collect/collision_flyaway.py"
        detector_hash = digest(detector_source) if detector_source.is_file() else None
        detector_monitor = ROOT / "source/isaaclab/isaaclab/motion_collect/collision_flyaway_monitor.py"
        monitor_hash = digest(detector_monitor) if detector_monitor.is_file() else None
        phase_source_hashes = None
        if source_data.get("phase_depenetration") is not None:
            phase_source_hashes = {
                "policy": digest(ROOT / "source/isaaclab/isaaclab/motion_collect/phase_depenetration.py"),
                "collector": digest(ROOT / "source/isaaclab/isaaclab/motion_collect/datacollector.py"),
                "loader": digest(ROOT / "source/isaaclab_tasks/isaaclab_tasks/real2sim/utils/task_config_loader.py"),
                "place": digest(ROOT / "source/isaaclab/isaaclab/motion_collect/primitives/place.py"),
                "hang": digest(ROOT / "source/isaaclab/isaaclab/motion_collect/primitives/hang.py"),
            }
        task_summary = {"task_id": task["task_id"], "robot": task["robot"], "seed": seed, "num_envs": num_envs, "rounds": []}
        summary["tasks"].append(task_summary)
        for round_index, scale in enumerate(scales):
            round_dir = output / task["task_id"] / f"round_{round_index:02d}"
            report_path = round_dir / "report.json"
            expected_config = round_dir / "task.yaml"
            previous = json.loads(report_path.read_text()) if report_path.is_file() else None
            resume = bool(
                args.resume and previous and previous.get("completed")
                and previous.get("returncode") == 0 and not previous.get("timed_out", False)
                and previous.get("summary", {}).get("requested_trials") == args.trials
                and previous.get("summary", {}).get("collision_assessment_complete") is True
                and assessed_report_matches(previous, args.trials)
                and detector_hash is not None
                and previous.get("collision_detector_source_sha256") == detector_hash
                and monitor_hash is not None
                and previous.get("collision_detector_monitor_sha256") == monitor_hash
                and previous.get("collision_detector_collector_sha256") == digest(ROOT / "source/isaaclab/isaaclab/motion_collect/datacollector.py")
                and previous.get("source_config_sha256") == digest(source)
                and previous.get("seed") == seed and previous.get("robot") == task["robot"]
                and previous.get("range_scale") == scale
                and previous.get("num_envs", 1) == num_envs
                and previous.get("preserve_render_interval", False) == bool(getattr(args, "preserve_render_interval", False))
                and previous.get("save_trajectories", False) == bool(getattr(args, "save_trajectories", False))
                and (not getattr(args, "save_trajectories", False) or all(
                    not t.get("success") or (t.get("trajectory_path") and Path(t["trajectory_path"]).is_file())
                    for t in previous.get("trials", [])))
                and previous.get("interpolation_steps_override") == getattr(args, "interpolation_steps", None)
                and (getattr(args, "interpolation_steps", None) is None or (
                    isinstance(previous.get("interpolation_steps_calls"), list)
                    and all(isinstance(call, dict) and call.get("effective") == args.interpolation_steps
                            for call in previous["interpolation_steps_calls"])))
                and (phase_source_hashes is None or
                     previous.get("phase_depenetration", {}).get("source_sha256") == phase_source_hashes)
                and (getattr(args, "verify_depenetration", None) is None or (
                    previous.get("expected_max_depenetration_velocity") == args.verify_depenetration
                    and previous.get("max_depenetration_velocity_verified")
                    and previous.get("expected_depenetration_overrides", {}) == dict(
                        (key, float(value)) for key, value in (item.rsplit("=", 1) for item in getattr(args, "verify_depenetration_override", [])))
                    and previous.get("physics_source_sha256") == digest(ROOT / "source/isaaclab_tasks/isaaclab_tasks/real2sim/tasks/pick_r2s_env_cfg.py")))
                and (getattr(args, "verify_mass", None) is None or (
                    previous.get("verified_mass_kg", {}).get("asset") == args.verify_mass.rsplit("=", 1)[0]
                    and previous.get("verified_mass_kg", {}).get("expected") == float(args.verify_mass.rsplit("=", 1)[1])))
            )
            if not resume:
                # Complete historical attempts must not be renamed or silently
                # reinterpreted under a different number of trials, detector, or
                # source configuration. Use a fresh output tree for new results.
                if previous and previous.get("completed") and previous.get("returncode") == 0 and not previous.get("timed_out", False):
                    old = copy.deepcopy(previous.get("summary", {}))
                    task_summary["rounds"].append({"round": round_index, "range_scale": previous.get("range_scale"),
                                                   "report_path": str(report_path), "completed": True,
                                                   "historical_unassessed": True, **old})
                    task_summary["status"] = "existing_report_preserved"
                    task_summary["reason"] = "Existing completed report does not match the requested collision-assessed protocol; use a new output directory."
                    write_json(output / "summary.json", summary)
                    break
                if round_dir.exists() and any(round_dir.iterdir()):
                    archive = round_dir.with_name(round_dir.name + "_previous_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
                    round_dir.rename(archive)
                round_dir.mkdir(parents=True, exist_ok=True)
                prepare_config(source, expected_config, trials=args.trials, scale=scale, task=task, num_envs=num_envs,
                               preserve_render_interval=getattr(args, "preserve_render_interval", False),
                               save_trajectories=getattr(args, "save_trajectories", False))
                command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker",
                           "--config", str(expected_config), "--task-id", task["task_id"],
                           "--robot", task["robot"], "--env-task-name", task["env_task_name"],
                           "--robot-cfg-name", task["robot_cfg_name"], "--robot-type", task["robot_type"],
                           "--trials", str(args.trials), "--seed", str(seed), "--device", args.device,
                           "--num-envs", str(num_envs),
                           "--output-dir", str(round_dir)]
                if getattr(args, "preserve_render_interval", False):
                    command.append("--preserve-render-interval")
                if getattr(args, "save_trajectories", False):
                    command.append("--save-trajectories")
                if getattr(args, "interpolation_steps", None) is not None:
                    command.extend(["--interpolation-steps", str(args.interpolation_steps)])
                if getattr(args, "verify_depenetration", None) is not None:
                    command.extend(["--verify-depenetration", str(args.verify_depenetration)])
                    for item in getattr(args, "verify_depenetration_override", []):
                        command.extend(["--verify-depenetration-override", item])
                if getattr(args, "verify_mass", None):
                    command.extend(["--verify-mass", args.verify_mass])
                print(f"RUN {task['task_id']} {task['robot']} round={round_index} range_scale={scale} num_envs={num_envs}", flush=True)
                env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
                timed_out = False
                with (round_dir / "run.log").open("w") as log:
                    process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
                    try:
                        returncode = process.wait(timeout=args.timeout)
                    except subprocess.TimeoutExpired:
                        timed_out = True
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
                        returncode = process.returncode
                    except KeyboardInterrupt:
                        os.killpg(process.pid, signal.SIGTERM)
                        raise
                previous = json.loads(report_path.read_text()) if report_path.is_file() else {"completed": False, "summary": summarize([], args.trials)}
                previous.update({"source_config_sha256": digest(source), "range_scale": scale, "returncode": returncode, "timed_out": timed_out})
                if timed_out:
                    previous["completed"] = False
                    previous["error"] = f"Worker exceeded {args.timeout} seconds"
                write_json(report_path, previous)
            task_summary["rounds"].append({"round": round_index, "range_scale": scale, "report_path": str(report_path), "completed": previous.get("completed", False), **previous.get("summary", {})})
            write_json(output / "summary.json", summary)
            print("ROUND_RESULT", task["task_id"], json.dumps(task_summary["rounds"][-1]), flush=True)
            if not previous.get("completed"):
                task_summary["status"] = "needs_repair"
                break
            rate = previous["summary"]["success_rate"]
            if rate is None:
                task_summary["status"] = ("collision_detection_incomplete" if
                    previous["summary"].get("collision_unknown_trials", 0) else "no_eligible_trials")
                break
            if rate is not None and rate >= 0.5:
                task_summary["status"] = "passed"
                break
            if not tunable:
                task_summary["status"] = "below_threshold_fixed_ranges"
                task_summary["tuning_skipped_reason"] = "All declared XY/yaw intervals are fixed; reducing their widths would repeat the same experiment."
                break
        else:
            task_summary["status"] = "below_threshold"
        write_json(output / "summary.json", summary)
        if task_summary["status"] == "needs_repair" and getattr(args, "stop_on_error", False):
            summary.update({"stopped": True, "stop_reason": "task_needs_repair",
                            "stopped_after_task": task["task_id"],
                            "remaining_task_ids": [item["task_id"] for item in tasks[task_index + 1:]]})
            write_json(output / "summary.json", summary)
            break
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="benchmarks/affordsim_50.json")
    parser.add_argument("--output-dir", default="outputs/local_benchmark/results")
    parser.add_argument("--task", action="append", help="Select a task_id; repeat for several")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--num-envs", type=int, default=1, help="Default parallel environments; a task's manifest num_envs overrides this; must divide --trials")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--preserve-render-interval", action="store_true",
                        help="Keep each source YAML's render_interval instead of setting it to decimation")
    parser.add_argument("--save-trajectories", action="store_true",
                        help="Save structured successful .npy trajectories alongside the original RGB frames")
    parser.add_argument("--preview-only", action="store_true",
                        help="Record held frames for a visual preview without structured trajectory data")
    parser.add_argument("--interpolation-steps", type=int,
                        help="Override all collector IK interpolation motions for this run")
    parser.add_argument("--timeout", type=float, default=7200, help="Per-task, per-round wall-time limit in seconds")
    parser.add_argument("--verify-depenetration", type=float, help="Verify and record the spawned task assets' PhysX maximum depenetration velocity")
    parser.add_argument("--verify-depenetration-override", action="append", default=[], metavar="ASSET=VALUE", help="Expected per-asset value; repeat for additional overrides")
    parser.add_argument("--verify-mass", metavar="ASSET=KG", help="Verify and record a rigid object's simulated mass in every environment")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-adapt", action="store_true")
    parser.add_argument("--stop-after-task", metavar="PATH", help="Stop before starting the next task when this marker file exists; finish the current task first")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop scheduling after an incomplete task")
    parser.add_argument("--minimum-free-gb", type=float, default=0.0, help="Stop between tasks when free disk space falls below this many GiB")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--config")
    parser.add_argument("--task-id", default="task")
    parser.add_argument("--robot", default="franka_long")
    parser.add_argument("--env-task-name")
    parser.add_argument("--robot-cfg-name")
    parser.add_argument("--robot-type", default="franka_fr3")
    args = parser.parse_args()
    if args.preview_only and args.save_trajectories:
        parser.error("--preview-only and --save-trajectories are mutually exclusive")
    if args.trials <= 0 or args.timeout <= 0:
        parser.error("--trials and --timeout must be positive")
    if args.interpolation_steps is not None and args.interpolation_steps <= 0:
        parser.error("--interpolation-steps must be a positive integer")
    if args.num_envs <= 0 or args.trials % args.num_envs:
        parser.error("--num-envs must be positive and divide --trials exactly")
    if args.save_trajectories and args.worker and (args.trials != 1 or args.num_envs != 1):
        parser.error("--save-trajectories requires one attempt and one environment per worker")
    if args.worker:
        if not args.config:
            parser.error("--worker requires --config")
        run_worker(args)
    else:
        run_schedule(args)


if __name__ == "__main__":
    main()
