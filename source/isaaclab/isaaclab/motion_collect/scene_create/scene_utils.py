"""
Shared geometry, YAML, and parsing helpers for scene creation.
"""

import json
import math
import os
import re

from ruamel.yaml.comments import CommentedSeq
from ruamel.yaml.scalarstring import DoubleQuotedScalarString

import config


TASK_CONTAINERS_YAML = os.path.join(
    config.PROJECT_ROOT,
    "source/isaaclab_tasks/isaaclab_tasks/real2sim/tasks/config/containers.yaml",
)

SUPPORTED_SCENE_RELATIONS = {
    "inside",
    "on_top",
    "left_of",
    "right_of",
    "front",
    "behind",
    "near",
}

SCENE_RELATION_ALIASES = {
    "in": "inside",
    "inside": "inside",
    "into": "inside",
    "within": "inside",
    "in_side": "inside",
    "on": "on_top",
    "onto": "on_top",
    "on_top": "on_top",
    "on_top_of": "on_top",
    "on_the_top_of": "on_top",
    "on_top_of_the": "on_top",
    "atop": "on_top",
    "above": "on_top",
    "left": "left_of",
    "left_of": "left_of",
    "on_the_left_of": "left_of",
    "to_the_left_of": "left_of",
    "right": "right_of",
    "right_of": "right_of",
    "on_the_right_of": "right_of",
    "to_the_right_of": "right_of",
    "front": "front",
    "front_of": "front",
    "in_front_of": "front",
    "to_the_front_of": "front",
    "ahead_of": "front",
    "behind": "behind",
    "back_of": "behind",
    "in_back_of": "behind",
    "to_the_back_of": "behind",
    "near": "near",
    "next_to": "near",
    "close_to": "near",
    "close_by": "near",
    "beside": "near",
}


def make_flow_list(data_list):
    """Create inline YAML list [a, b, c]."""
    seq = CommentedSeq(data_list)
    seq.fa.set_flow_style()
    return seq


def yamlify_value(value):
    """Convert Python values to YAML-friendly values with inline lists."""
    if isinstance(value, list):
        return make_flow_list([yamlify_value(item) for item in value])
    if isinstance(value, dict):
        return {key: yamlify_value(val) for key, val in value.items()}
    if isinstance(value, str) and value == "on":
        return DoubleQuotedScalarString("on")
    return value


def get_obb_corners(x, y, dim_x, dim_y, angle):
    """Calculate rotated rectangle corners."""
    c, s = math.cos(angle), math.sin(angle)
    hx, hy = dim_x / 2.0, dim_y / 2.0
    corners = [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)]
    return [(lx * c - ly * s + x, lx * s + ly * c + y) for lx, ly in corners]


def check_obb_collision(pos1, dim1, angle1, pos2, dim2, angle2):
    """SAT collision detection for oriented bounding boxes."""
    corners1 = get_obb_corners(pos1[0], pos1[1], dim1[0], dim1[1], angle1)
    corners2 = get_obb_corners(pos2[0], pos2[1], dim2[0], dim2[1], angle2)

    for corners in [corners1, corners2]:
        for i in range(4):
            p1, p2 = corners[i], corners[(i + 1) % 4]
            edge = (p2[0] - p1[0], p2[1] - p1[1])
            length = math.hypot(*edge)
            if length < 1e-6:
                continue
            axis = (-edge[1] / length, edge[0] / length)
            proj1 = [x * axis[0] + y * axis[1] for x, y in corners1]
            proj2 = [x * axis[0] + y * axis[1] for x, y in corners2]
            if max(proj1) < min(proj2) or max(proj2) < min(proj1):
                return False
    return True


def get_world_axis_half_extents(dim_x: float, dim_y: float, angle: float) -> tuple[float, float]:
    """Return the OBB half extents projected onto world X/Y axes."""
    half_x = dim_x / 2.0
    half_y = dim_y / 2.0
    cos_a = abs(math.cos(angle))
    sin_a = abs(math.sin(angle))
    return (
        cos_a * half_x + sin_a * half_y,
        sin_a * half_x + cos_a * half_y,
    )


def load_containers_registry() -> dict:
    """Load registered container metadata for articulation/place validation."""
    if not os.path.exists(TASK_CONTAINERS_YAML):
        return {}
    try:
        import yaml as pyyaml

        with open(TASK_CONTAINERS_YAML, "r", encoding="utf-8") as f:
            data = pyyaml.safe_load(f) or {}
        return data if isinstance(data, dict) else {}
    except Exception as e:
        print(f"[WARN] Failed to load containers registry: {e}")
        return {}


def get_sim_object_type_map() -> dict[str, str]:
    """Return label  to  object type from the available asset registry."""
    try:
        from sim_config import get_asset_registry

        return {
            label: asset_info.get("object_type", "rigid")
            for label, asset_info in get_asset_registry().items()
        }
    except Exception as e:
        print(f"[WARN] Failed to query sim object types: {e}")
        return {}


def get_articulation_labels() -> set[str]:
    """Return the set of labels whose USD assets are articulations."""
    return {
        label
        for label, object_type in get_sim_object_type_map().items()
        if object_type == "articulation"
    }


def get_scene_object_type(scene_entry: dict, sim_object_type_map: dict[str, str]) -> str:
    """Resolve the exported object type for a scene entry."""
    explicit_type = scene_entry.get("_scene_object_type")
    if explicit_type in {"rigid", "articulation"}:
        return explicit_type
    return sim_object_type_map.get(scene_entry.get("label", ""), "rigid")


def build_semantic_tags(label: str, semantic_tags=None) -> list:
    """Normalize semantic tags to YAML-friendly nested lists."""
    tags = semantic_tags or [["class", label]]
    normalized = []
    for tag in tags:
        if isinstance(tag, (list, tuple)):
            normalized.append(make_flow_list(list(tag)))
        else:
            normalized.append(tag)
    return normalized


def canonicalize_scene_relation(raw_relation: str) -> str | None:
    """Normalize free-form relation words to the editor's canonical relation set."""
    relation = str(raw_relation or "").strip().lower()
    if not relation:
        return None
    if relation in SUPPORTED_SCENE_RELATIONS:
        return relation
    normalized = re.sub(r"[^a-z0-9]+", "_", relation).strip("_")
    if normalized in SUPPORTED_SCENE_RELATIONS:
        return normalized
    return SCENE_RELATION_ALIASES.get(normalized)


def normalize_scale_values(scale_value, fallback: list[float] | tuple[float, ...]) -> list[float]:
    """Normalize scale values to a 3-float list, or fall back on parse failure."""
    try:
        values = [float(v) for v in scale_value]
        if len(values) != 3:
            raise ValueError("scale must have exactly 3 values")
        return values
    except Exception:
        return [float(v) for v in fallback]


def scales_match(scale_a: list[float] | tuple[float, ...], scale_b: list[float] | tuple[float, ...], tol: float = 1e-6) -> bool:
    """Return True when two xyz scales match within tolerance."""
    if len(scale_a) != 3 or len(scale_b) != 3:
        return False
    return all(abs(float(a) - float(b)) <= tol for a, b in zip(scale_a, scale_b))

