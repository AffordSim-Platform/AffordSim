"""
Runtime lifecycle and cross-process restore helpers for the scene editor.
"""

import copy
import os
import sys
import tempfile

import yaml as pyyaml
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
from isaaclab.sim.utils import stage as stage_utils

import config as scene_create_config
from scene_core import SceneManager
from sim_config import build_scene_cfg


def _resolve_project_path(path: str) -> str:
    """Resolve a project-relative path to an absolute path."""
    raw_path = str(path or "").strip()
    if not raw_path:
        return scene_create_config.DEFAULT_SCENE_USD_PATH
    if os.path.isabs(raw_path):
        return raw_path
    return os.path.abspath(os.path.join(scene_create_config.PROJECT_ROOT, raw_path))


def get_scene_usd_abs_path() -> str:
    """Return the fixed table scene USD path used by the editor."""
    return _resolve_project_path(scene_create_config.DEFAULT_SCENE_USD_PATH)


def extract_scene_labels_from_yaml(yaml_path: str) -> list[str]:
    """Read strict labels from a task YAML before building the simulation scene."""
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(yaml_path)

    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = pyyaml.safe_load(f) or {}

    labels = []
    seen = set()
    for section_name in ("objects", "articulations"):
        section = cfg.get(section_name, {}) or {}
        if not isinstance(section, dict):
            continue
        for obj_val in section.values():
            usd_path = obj_val.get("usd_path", "")
            label = os.path.basename(str(usd_path)).split(".")[0].strip()
            if label and label not in seen:
                labels.append(label)
                seen.add(label)
    return labels


def destroy_runtime(runtime: dict | None):
    """Tear down the current scene runtime before rebuilding it."""
    if not runtime:
        return

    scene = runtime.get("scene")
    sim = runtime.get("sim")

    if scene is not None:
        del scene

    if sim is not None:
        try:
            if not sim.has_gui():
                sim.stop()
        except Exception:
            pass
        try:
            sim.clear_all_callbacks()
        except Exception:
            pass
        try:
            sim.clear_instance()
        except Exception:
            pass

    try:
        stage_utils.clear_stage()
    except Exception:
        pass


def create_runtime(
    args_cli,
    active_labels: list[str],
    generator,
    runtime: dict | None = None,
) -> dict:
    """Build a fresh simulation runtime for the requested label subset."""
    destroy_runtime(runtime)
    scene_name = scene_create_config.DEFAULT_SCENE_NAME
    scene_create_config.apply_table_bounds(scene_name)

    scene_cfg, sim_object_labels, sim_object_types = build_scene_cfg(
        active_labels,
        num_envs=args_cli.num_envs,
        env_spacing=2.0,
        scene_usd_path=get_scene_usd_abs_path(),
        scene_name=scene_name,
    )
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([5, 0.0, 2], [0.8, 0.0, 0.1])
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    manager = SceneManager(
        scene,
        generator,
        sim_object_labels=sim_object_labels,
        sim_object_types=sim_object_types,
    )
    print(
        f"[INFO] Runtime initialized with {len(sim_object_labels)} object slots "
        f"and scene '{scene_name}'."
    )
    return {
        "sim": sim,
        "scene": scene,
        "manager": manager,
        "sim_dt": sim.get_physics_dt(),
        "scene_name": scene_name,
    }


def sync_and_step_runtime(runtime: dict, num_steps: int = 50):
    """Sync scene state into the simulator and advance a few frames."""
    runtime["manager"].sync_to_simulation()
    for _ in range(num_steps):
        runtime["sim"].step()


def _make_serializable(value):
    """Convert nested tuples to lists so scene state can round-trip through YAML safely."""
    if isinstance(value, dict):
        return {key: _make_serializable(val) for key, val in value.items()}
    if isinstance(value, tuple):
        return [_make_serializable(item) for item in value]
    if isinstance(value, list):
        return [_make_serializable(item) for item in value]
    return value


def write_restore_state(
    current_scene: list,
    pending_add_label: str | None = None,
) -> str:
    """Persist scene state into a temporary YAML file for cross-process restart."""
    payload = {
        "scene": _make_serializable(copy.deepcopy(current_scene)),
        "pending_add_label": pending_add_label,
    }
    fd, restore_path = tempfile.mkstemp(prefix="scene_create_restore_", suffix=".yaml", dir="/tmp")
    os.close(fd)
    with open(restore_path, "w", encoding="utf-8") as f:
        pyyaml.safe_dump(payload, f, sort_keys=False)
    return restore_path


def load_restore_state(restore_path: str) -> tuple[list, str | None]:
    """Load previously saved scene state for editor restart."""
    with open(restore_path, "r", encoding="utf-8") as f:
        payload = pyyaml.safe_load(f) or {}
    scene_data = payload.get("scene", []) or []
    pending_add_label = payload.get("pending_add_label")
    try:
        os.remove(restore_path)
    except OSError:
        pass
    return scene_data, pending_add_label


def build_restart_argv(script_path: str, extra_args: list[str] | None = None) -> list[str]:
    """Build argv for restarting the current editor script."""
    filtered_args = []
    skip_next = False
    for arg in sys.argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if arg == "--restore-state":
            skip_next = True
            continue
        if arg.startswith("--restore-state="):
            continue
        if arg == "--initial-yaml":
            skip_next = True
            continue
        if arg.startswith("--initial-yaml="):
            continue
        filtered_args.append(arg)
    return [sys.executable, script_path, *filtered_args, *(extra_args or [])]


def restart_editor_with_state(
    script_path: str,
    current_scene: list,
    pending_add_label: str | None = None,
):
    """Restart the whole editor process and restore the scene in the new process."""
    restore_path = write_restore_state(current_scene, pending_add_label)
    restart_argv = build_restart_argv(script_path, ["--restore-state", restore_path])
    if pending_add_label:
        print(f"[INFO] Restarting editor to load additional asset '{pending_add_label}'.")
    else:
        print("[INFO] Restarting editor to load generated scene.")
    sys.stdout.flush()
    sys.stderr.flush()
    os.execv(sys.executable, restart_argv)


def restart_editor_with_yaml(script_path: str, yaml_path: str):
    """Restart the editor and load object layout from YAML in the fresh process."""
    restart_argv = build_restart_argv(
        script_path,
        ["--initial-yaml", yaml_path],
    )
    print(f"[INFO] Restarting editor to load YAML object layout: {yaml_path}")
    sys.stdout.flush()
    sys.stderr.flush()
    os.execv(sys.executable, restart_argv)


def restart_editor_fresh(script_path: str):
    """Restart the editor to a clean initial state without restoring the current scene."""
    restart_argv = build_restart_argv(script_path)
    print("[INFO] Restarting editor for fresh scene generation.")
    sys.stdout.flush()
    sys.stderr.flush()
    os.execv(sys.executable, restart_argv)
