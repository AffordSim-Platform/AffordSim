"""
Task primitive compilation and YAML export.
"""

import copy
import json
import os
import re

from ruamel.yaml import YAML

import config
from scene_utils import (
    build_semantic_tags,
    get_articulation_labels,
    get_scene_object_type,
    get_sim_object_type_map,
    load_containers_registry,
    make_flow_list,
    yamlify_value,
)


def _rel_to_project(p: str) -> str:
    """Return *p* as a forward-slash path relative to the project root for YAML export."""
    return os.path.relpath(p, config.PROJECT_ROOT).replace(os.sep, "/")


ALLOWED_TASK_PRIMITIVES = {
    "pick",
    "place",
    "pour",
    "push",
    "pull",
    "open",
    "close",
    "hang",
    "hold",
    "return_home",
}
TARGET_REQUIRED_PRIMITIVES = {
    "pick",
    "place",
    "pour",
    "push",
    "pull",
    "open",
    "close",
    "hang",
}
SCENE_ARTICULATION_PRIMITIVES = {"open", "close", "pull"}
PICK_MODE_ALLOWED = {"affordance"}
PLACE_MODE_ALLOWED = {"in", "on", "pose"}
DEFAULT_ARTICULATION_ACTUATOR = {
    "all_joints": {
        "joint_names_expr": [".*"],
        "effort_limit": 87.0,
        "velocity_limit": 2.175,
        "stiffness": 400.0,
        "damping": 80.0,
    }
}


def _get_scene_entry(scene: list, label: str) -> dict | None:
    """Return a scene entry by exact label."""
    for obj in scene:
        if obj.get("label") == label:
            return obj
    return None


def _next_primitive_type(primitives: list, start_index: int) -> str | None:
    """Return the next primitive type after start_index, if any."""
    for item in primitives[start_index + 1 :]:
        prim_type = str(item.get("type", "")).strip()
        if prim_type:
            return prim_type
    return None


def _affordance_exists(label: str, action: str, filename: str) -> bool:
    """Check convention-based affordance files used by runtime primitives."""
    affordance_path = os.path.join(
        config.PROJECT_ROOT,
        "pc_dataset_generate",
        "pc",
        f"{label}_{action}",
        filename,
    )
    return os.path.exists(affordance_path)


def _build_pick_primitive(target: str, mode: str) -> dict:
    if mode == "hang":
        return {
            "type": "pick",
            "mode": "hang",
            "params": {
                "target": target,
                "pre_pick_offset": [0.0, 0.0, 0.08],
                "pick_interpolation_steps": 10,
                "lift_interpolation_steps": 10,
            },
        }
    return {
        "type": "pick",
        "mode": "affordance",
        "params": {
            "target": target,
            "pre_pick_offset": [0.0, 0.0, 0.10],
            "pick_interpolation_steps": 15,
            "lift_interpolation_steps": 15,
            "lift_offset": 0.1,
        },
    }


def _build_place_primitive(target: str, mode: str) -> dict:
    if mode == "pose":
        raise ValueError("Use _build_place_pose_primitive() for place pose mode.")
    if mode == "in":
        return {
            "type": "place",
            "mode": "in",
            "params": {
                "target": target,
                "ik_interpolation_steps": 30,
                "success_eval": True,
            },
        }
    return {
        "type": "place",
        "mode": "on",
        "params": {
            "target": target,
            "on_height_margin": 0.01,
            "ik_interpolation_steps": 30,
            "post_release_lift_offset": [0.0, 0.0, 0.08],
            "post_release_lift_steps": 10,
            "success_eval": True,
        },
    }


def _build_place_pose_primitive(position: list[float]) -> dict:
    return {
        "type": "place",
        "mode": "pose",
        "params": {
            "position": [round(float(position[0]), 4), round(float(position[1]), 4), round(float(position[2]), 4)],
            "success_eval": False,
        },
    }


def _build_pour_primitive(target: str) -> dict:
    return {
        "type": "pour",
        "params": {
            "target": target,
            "tilt_angle_deg": 90.0,
            "move_hover_height": 0.15,
            "tilt_lift_height": 0.08,
            "tilt_horizontal_offset": 0.05,
            "success_eval": False,
        },
    }


def _build_push_primitive(target: str, position_xy: list[float], target_z: float) -> dict:
    return {
        "type": "push",
        "mode": "object_aligned",
        "params": {
            "target": target,
            "push_target": [round(float(position_xy[0]), 4), round(float(position_xy[1]), 4), round(float(target_z), 4)],
            "pre_height": 0.15,
            "pre_push_backoff": 0.08,
            "push_contact_retreat": 0.02,
            "lift_z_delta": 0.15,
            "down_interpolation_steps": 5,
            "push_interpolation_steps": 30,
            "lift_interpolation_steps": 10,
            "success_eval": False,
        },
    }


def _build_pull_primitive(target: str) -> dict:
    return {
        "type": "pull",
        "params": {
            "target": target,
            "pre_grasp_offset": 0.08,
            "pick_interpolation_steps": 10,
            "pull_distance": 0.20,
            "pull_interpolation_steps": 30,
            "release_after_pull": True,
            "success_eval": False,
        },
    }


def _build_open_primitive(target: str) -> dict:
    return {
        "type": "open",
        "params": {
            "target": target,
            "pre_pick_offset": 0.10,
            "pick_interpolation_steps": 10,
            "fast_pull_distance": 0.30,
            "fast_pull_interpolation_steps": 5,
            "lift_before_pre_insert": 0.30,
            "lift_interpolation_steps": 10,
            "pre_insert_outward_offset": 0.03,
            "align_above_interpolation_steps": 10,
            "door_tangent_move_distance": 0.20,
            "tangent_move_interpolation_steps": 10,
            "success_eval": False,
        },
    }


def _build_close_primitive(target: str) -> dict:
    return {
        "type": "close",
        "params": {
            "target": target,
            "pre_close_facing_offset": 0.25,
            "pre_close_open_normal_offset": 0.40,
            "close_facing_offset": 0.15,
            "close_open_normal_backoff": 0.15,
            "close_height_offset": 0.0,
            "close_interpolation_steps": 30,
            "success_eval": False,
        },
    }


def _build_hang_primitive(target: str) -> dict:
    return {
        "type": "hang",
        "mode": "mount",
        "params": {
            "target": target,
            "pre_hang_distance": 0.10,
            "insert_interpolation_steps": 20,
            "extra_insert_distance": 0.05,
            "extra_insert_interpolation_steps": 10,
            "post_release_retreat_distance": 0.10,
            "retreat_interpolation_steps": 10,
            "success_eval": False,
        },
    }


def _build_hold_primitive() -> dict:
    return {
        "type": "hold",
        "params": {
            "lift_height": 0.10,
            "hold_steps": 10,
            "success_eval": False,
        },
    }


def _build_return_home_primitive() -> dict:
    return {"type": "return_home", "params": {"post_settle_steps": 20}}


def _compute_table_resting_z(scene_entry: dict) -> float | None:
    """Compute the stable table placement center height for a scene entry."""
    if not isinstance(scene_entry, dict):
        return None
    bottom_offset = scene_entry.get("_bottom_offset")
    if bottom_offset is None:
        position_z = scene_entry.get("position", {}).get("z")
        height = scene_entry.get("_height")
        if position_z is None or height is None:
            return None
        bottom_offset = float(height) / 2.0
    return round(config.SAFE_Z + float(bottom_offset) + 0.005, 4)


def _attach_affordance_queries(compiled: dict, source: dict) -> dict:
    """Carry the LLM-provided affordance_queries field through to the compiled primitive."""
    queries = source.get("affordance_queries")
    if isinstance(queries, list) and queries:
        compiled["affordance_queries"] = [
            dict(q) for q in queries if isinstance(q, dict)
        ]
    return compiled


def compile_primitives_for_yaml(scene: list, parsed_plan: dict) -> tuple[list | None, list[str]]:
    """Validate parsed primitive plan and compile it into YAML primitives."""
    primitives = parsed_plan.get("primitives", []) if isinstance(parsed_plan, dict) else []
    if not isinstance(primitives, list) or not primitives:
        return None, ["Task parser did not return any primitives."]

    scene_labels = {obj.get("label") for obj in scene}
    articulation_labels = get_articulation_labels()
    containers_registry = load_containers_registry()
    errors = []
    compiled = []
    held_object_label = None

    for index, primitive in enumerate(primitives, start=1):
        if not isinstance(primitive, dict):
            errors.append(f"Primitive #{index} is not a JSON object.")
            continue
        prim_type = str(primitive.get("type", "")).strip()
        if prim_type not in ALLOWED_TASK_PRIMITIVES:
            errors.append(f"Primitive #{index} has unsupported type '{prim_type}'.")
            continue

        raw_position_xy = primitive.get("position_xy")
        requested_mode = str(primitive.get("mode") or "").strip().lower()
        if prim_type == "place" and not requested_mode and isinstance(raw_position_xy, list) and len(raw_position_xy) == 2:
            requested_mode = "pose"
        place_pose_mode = prim_type == "place" and requested_mode == "pose"

        target = primitive.get("target")
        if prim_type in TARGET_REQUIRED_PRIMITIVES and not place_pose_mode:
            if not isinstance(target, str) or not target.strip():
                errors.append(f"Primitive #{index} ({prim_type}) requires a non-empty target label.")
                continue
            target = target.strip()
            if target not in scene_labels:
                errors.append(f"Primitive #{index} ({prim_type}) references target '{target}' not present in scene.")
                continue
        else:
            if target not in (None, ""):
                errors.append(f"Primitive #{index} ({prim_type}) must not include a target.")
                continue
            target = None

        if prim_type == "pick":
            if held_object_label is not None:
                errors.append(f"Primitive #{index} (pick) cannot run while another object is still held.")
                continue
            requested_mode = str(primitive.get("mode") or "affordance").strip()
            if requested_mode not in PICK_MODE_ALLOWED:
                errors.append(f"Primitive #{index} (pick) has unsupported mode '{requested_mode}'.")
                continue
            next_type = _next_primitive_type(primitives, index - 1)
            compiled_pick_mode = "hang" if next_type == "hang" else requested_mode
            if compiled_pick_mode == "hang":
                if not _affordance_exists(target, "pick", "grasp_result_hang.npz"):
                    errors.append(
                        f"Primitive #{index} (pick to hang) requires pc_dataset_generate/pc/{target}_pick/grasp_result_hang.npz."
                    )
                    continue
            elif not _affordance_exists(target, "pick", "grasp_result.npz"):
                errors.append(f"Primitive #{index} (pick) requires pc_dataset_generate/pc/{target}_pick/grasp_result.npz.")
                continue
            compiled.append(_attach_affordance_queries(_build_pick_primitive(target, compiled_pick_mode), primitive))
            held_object_label = target
            continue

        if prim_type == "place":
            if held_object_label is None:
                errors.append(f"Primitive #{index} (place) requires a prior pick.")
                continue
            requested_mode = str(primitive.get("mode") or "on").strip().lower()
            if requested_mode not in PLACE_MODE_ALLOWED:
                errors.append(f"Primitive #{index} (place) has unsupported mode '{requested_mode}'.")
                continue
            if requested_mode == "pose":
                position_xy = primitive.get("position_xy")
                if not isinstance(position_xy, list) or len(position_xy) != 2:
                    errors.append(f"Primitive #{index} (place pose) requires position_xy=[x, y].")
                    continue
                try:
                    x = float(position_xy[0])
                    y = float(position_xy[1])
                except (TypeError, ValueError):
                    errors.append(f"Primitive #{index} (place pose) has invalid numeric position_xy={position_xy}.")
                    continue
                held_scene_entry = _get_scene_entry(scene, held_object_label)
                if held_scene_entry is None:
                    errors.append(
                        f"Primitive #{index} (place pose) held object '{held_object_label}' is missing scene data."
                    )
                    continue
                target_z = _compute_table_resting_z(held_scene_entry)
                if target_z is None:
                    errors.append(
                        f"Primitive #{index} (place pose) could not compute resting z for held object '{held_object_label}'."
                    )
                    continue
                compiled.append(_attach_affordance_queries(_build_place_pose_primitive([x, y, target_z]), primitive))
                held_object_label = None
                continue
            compiled.append(_attach_affordance_queries(_build_place_primitive(target, requested_mode), primitive))
            held_object_label = None
            continue

        if prim_type == "pour":
            if held_object_label is None:
                errors.append(f"Primitive #{index} (pour) requires a prior pick.")
                continue
            if not _affordance_exists(held_object_label, "pourout", "affordance_clusters.npz"):
                errors.append(
                    f"Primitive #{index} (pour) requires pc_dataset_generate/pc/{held_object_label}_pourout/affordance_clusters.npz."
                )
                continue
            if not _affordance_exists(target, "pourin", "affordance_clusters.npz"):
                errors.append(
                    f"Primitive #{index} (pour) requires pc_dataset_generate/pc/{target}_pourin/affordance_clusters.npz."
                )
                continue
            compiled.append(_attach_affordance_queries(_build_pour_primitive(target), primitive))
            continue

        if prim_type == "push":
            if held_object_label is not None:
                errors.append(f"Primitive #{index} (push) cannot run while holding an object.")
                continue
            position_xy = primitive.get("position_xy")
            if not isinstance(position_xy, list) or len(position_xy) != 2:
                errors.append(f"Primitive #{index} (push) requires position_xy=[x, y].")
                continue
            try:
                x = float(position_xy[0])
                y = float(position_xy[1])
            except (TypeError, ValueError):
                errors.append(f"Primitive #{index} (push) has invalid numeric position_xy={position_xy}.")
                continue
            scene_entry = _get_scene_entry(scene, target)
            if scene_entry is None:
                errors.append(f"Primitive #{index} (push) target '{target}' is missing scene pose data.")
                continue
            target_z = scene_entry.get("position", {}).get("z")
            if target_z is None:
                errors.append(f"Primitive #{index} (push) target '{target}' is missing z position.")
                continue
            if not _affordance_exists(target, "push", "grasp_result.npz"):
                errors.append(f"Primitive #{index} (push) requires pc_dataset_generate/pc/{target}_push/grasp_result.npz.")
                continue
            compiled.append(_attach_affordance_queries(_build_push_primitive(target, [x, y], float(target_z)), primitive))
            continue

        if prim_type in SCENE_ARTICULATION_PRIMITIVES:
            if held_object_label is not None:
                errors.append(f"Primitive #{index} ({prim_type}) cannot run while holding an object.")
                continue
            if target not in articulation_labels:
                errors.append(f"Primitive #{index} ({prim_type}) target '{target}' is not an articulation in scene.")
                continue
            container_meta = containers_registry.get(target)
            if not isinstance(container_meta, dict) or container_meta.get("facing_axis_local") is None:
                errors.append(
                    f"Primitive #{index} ({prim_type}) target '{target}' must be registered in containers.yaml with facing_axis_local."
                )
                continue
            if prim_type == "open" and not _affordance_exists(target, "open", "grasp_result.npz"):
                errors.append(f"Primitive #{index} (open) requires pc_dataset_generate/pc/{target}_open/grasp_result.npz.")
                continue
            if prim_type == "pull" and not _affordance_exists(target, "pull", "grasp_result.npz"):
                errors.append(f"Primitive #{index} (pull) requires pc_dataset_generate/pc/{target}_pull/grasp_result.npz.")
                continue
            if prim_type == "open":
                compiled.append(_attach_affordance_queries(_build_open_primitive(target), primitive))
            elif prim_type == "close":
                compiled.append(_attach_affordance_queries(_build_close_primitive(target), primitive))
            else:
                compiled.append(_attach_affordance_queries(_build_pull_primitive(target), primitive))
            continue

        if prim_type == "hang":
            if held_object_label is None:
                errors.append(f"Primitive #{index} (hang) requires a prior pick.")
                continue
            if not _affordance_exists(held_object_label, "hang", "affordance_clusters.npz"):
                errors.append(
                    f"Primitive #{index} (hang) requires pc_dataset_generate/pc/{held_object_label}_hang/affordance_clusters.npz."
                )
                continue
            if not _affordance_exists(target, "hang", "affordance_clusters.npz"):
                errors.append(
                    f"Primitive #{index} (hang) requires pc_dataset_generate/pc/{target}_hang/affordance_clusters.npz."
                )
                continue
            if not _affordance_exists(target, "hang", "labeled_result.pt") and not _affordance_exists(target, "hang", "labeled_result.npz"):
                errors.append(
                    f"Primitive #{index} (hang) requires pc_dataset_generate/pc/{target}_hang/labeled_result.pt or labeled_result.npz."
                )
                continue
            compiled.append(_attach_affordance_queries(_build_hang_primitive(target), primitive))
            held_object_label = None
            continue

        if prim_type == "hold":
            if held_object_label is None:
                errors.append(f"Primitive #{index} (hold) requires a prior pick.")
                continue
            compiled.append(_attach_affordance_queries(_build_hold_primitive(), primitive))
            continue

        if prim_type == "return_home":
            compiled.append(_attach_affordance_queries(_build_return_home_primitive(), primitive))

    return (compiled if not errors else None), errors


def export_to_yaml(
    scene: list,
    task: str,
    primitives: list,
    random_config: dict = None,
    visual_config: dict = None,
):
    """Export scene configuration to YAML file."""
    print(f"[INFO] Saving to: {config.SAVE_YAML_PATH}")

    yaml_data = {
        "simulation": {"decimation": 10, "dt": 0.00833333, "render_interval": 2, "episode_length_s": 20},
        "scene": {
            "num_envs": 1,
            "env_spacing": 10,
        },
        "task": {"name": "pick task", "description": task},
        "runtime": {
            "headless": False,
            "episodes": 50,
            "save_npy": True,
            "save_camera_images": True,
            "save_composite_images": False,
            "save_mask_images": False,
            "save_original_images": False,
            "save_depth_images": False,
            "enable_motion_planning": True,
            "data_dir": _rel_to_project(os.path.join(config.PROJECT_ROOT, "data", "r2s2r_data", "pick")),
            "task_name": "FrankaFr3-OurGripper-Joint-VisPick-v0",
        },
        "robot": {"cfg_name": "franka_fr3_our.yml", "type": "franka_fr3_default"},
        "offsets": {"grasp": make_flow_list([0, 0, 0.0]), "place": make_flow_list([0, 0, 0.0])},
    }

    try:
        with open(config.ALL_OBJECTS_JSON, "r", encoding="utf-8") as f:
            object_metadata = {
                item.get("label"): item
                for item in json.load(f)
                if item.get("label")
            }
    except Exception as e:
        print(f"[WARN] Failed to load object metadata for scale export: {e}")
        object_metadata = {}

    needs_type_lookup = any(obj.get("_scene_object_type") not in {"rigid", "articulation"} for obj in scene)
    sim_object_type_map = get_sim_object_type_map() if needs_type_lookup else {}
    objects_dict = {}
    articulations_dict = {}
    for obj in scene:
        label = obj["label"]
        obj_key = re.sub(r"[^a-zA-Z0-9_]", "_", label)
        p, o = obj["position"], obj["orientation"]
        h = float(obj.get("_height", 0.05))
        scale = obj.get("_yaml_scale", object_metadata.get(label, {}).get("scale", [1.0, 1.0, 1.0]))
        object_type = get_scene_object_type(obj, sim_object_type_map)

        entry = {
            "usd_path": _rel_to_project(f"{config.ASSET_ROOT}/{label}.usd"),
            "semantic_tags": build_semantic_tags(label, obj.get("_semantic_tags")),
            "scale": make_flow_list(scale),
            "init_pos": make_flow_list([p["x"], p["y"], p["z"]]),
            "init_rot": make_flow_list([o["w"], o["x"], o["y"], o["z"]]),
        }

        if random_config:
            d = random_config["pos_delta"]
            entry["position_range"] = {
                "x": make_flow_list([round(p["x"] - d["back"], 3), round(p["x"] + d["front"], 3)]),
                "y": make_flow_list([round(p["y"] - d["right"], 3), round(p["y"] + d["left"], 3)]),
                "z": make_flow_list([round(p["z"], 3), round(p["z"], 3)]),
            }
            entry["orientation_range"] = {"yaw": make_flow_list(random_config.get("rot_range", [0, 0]))}

        if object_type == "articulation":
            entry["disable_gravity"] = bool(obj.get("_disable_gravity", False))
            entry["enabled_self_collisions"] = bool(obj.get("_enabled_self_collisions", True))
            entry["joint_pos"] = copy.deepcopy(obj.get("_joint_pos", {}) or {})
            entry["actuators"] = yamlify_value(copy.deepcopy(obj.get("_actuators", DEFAULT_ARTICULATION_ACTUATOR)))
            articulations_dict[obj_key] = entry
        else:
            entry["height"] = round(h, 3)
            objects_dict[obj_key] = entry

    yaml_data["objects"] = objects_dict
    if articulations_dict:
        yaml_data["articulations"] = articulations_dict
    yaml_data["primitives"] = [yamlify_value(primitive) for primitive in primitives]
    if visual_config is None:
        visual_config = {}
    yaml_data["enable_random_lights"] = visual_config.get("enable_random_lights", False)
    yaml_data["image_noise_std"] = visual_config.get("image_noise_std", 0)
    yaml_data["image_speckle_noise_std"] = visual_config.get("image_speckle_noise_std", 0.5)
    yaml_data["enable_objects_randomization"] = bool(random_config)
    yaml_data["enable_objects_randomization_curriculum"] = False
    yaml_data["texture_randomization"] = {
        "enabled": visual_config.get("enable_texture_randomization", True),
        "texture_dir": _rel_to_project(config.TEXTURE_DIR),
        "targets": {
            "table": {
                "prim_path": config.get_scene_texture_target(),
                "texture_dir": _rel_to_project(config.TEXTURE_DIR),
            }
        },
    }

    os.makedirs(os.path.dirname(config.SAVE_YAML_PATH), exist_ok=True)
    yaml = YAML()
    yaml.preserve_quotes = True
    with open(config.SAVE_YAML_PATH, "w", encoding="utf-8") as f:
        f.write("# Auto-generated by Scene System\n")
        yaml.dump(yaml_data, f)

    print(" YAML saved successfully.")
