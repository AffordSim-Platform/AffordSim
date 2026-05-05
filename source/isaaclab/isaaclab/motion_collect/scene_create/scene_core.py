"""
Core scene parsing, generation, and editing logic.
"""

import copy
import json
import math
import os
import random
import re

import torch
from openai import OpenAI

import config
from scene_utils import (
    canonicalize_scene_relation,
    check_obb_collision,
    get_world_axis_half_extents,
    normalize_scale_values,
    scales_match,
)


class LLMClient:
    """Handles LLM-based instruction parsing."""

    def __init__(self):
        self.model = config.MODEL
        self.client = OpenAI(
            api_key=config.API_KEY,
            base_url=config.BASE_URL,
            timeout=config.TIMEOUT,
            max_retries=0,
        )

    def _call_llm(self, system_prompt: str, user_message: str) -> dict:
        """Call LLM and parse JSON response."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.0,
            )
            content = response.choices[0].message.content
            content = content.replace("```json", "").replace("```", "").strip()
            return json.loads(content)
        except Exception as e:
            print(f"[LLM ERROR] {e}")
            return {}

    def parse_scene_instruction(self, instruction: str) -> dict:
        """Parse scene generation instruction."""
        system_prompt = """You are a Scene Graph Parser.

Convert the user's scene description into a JSON object with exactly this schema:
{
  "specified_objects": ["..."],
  "random_objects_count": 0,
  "relationships": [
    {"child": "...", "relation": "...", "parent": "..."}
  ]
}

Rules:
1. Output JSON only. No markdown. No explanation.
2. `specified_objects` must contain only short object reference phrases. Do not include verbs, conjunctions,
   politeness words, or full clauses. Good: "banana", "yellow bowl", "microwave". Bad: "the kettle and then pour".
3. `child` and `parent` must also be short object reference phrases only.
4. Do not invent extra disambiguating attributes that the user did not say. Never add colors, materials, sizes,
   indices, or serial numbers on your own. If the user says "kettle", output "kettle", not "blue_kettle".
5. If the user mentions a count of random objects, put the integer in `random_objects_count`. Otherwise use 0.
6. Do not duplicate the same object mention multiple times in `specified_objects`.
7. Output relation values using only these canonical relations:
   - "inside", "on_top"
   - "left_of", "right_of", "front", "behind"
   - "near"
8. Normalize common phrasing to the canonical relations above:
   - in / into / inside -> "inside"
   - on / onto / on top of / atop -> "on_top"
   - left / on the left of / to the left of -> "left_of"
   - right / on the right of / to the right of -> "right_of"
   - in front of / ahead of -> "front"
   - back of / behind / in back of -> "behind"
   - near / next to / close to / beside -> "near"
9. Ignore filler words like "please", "carefully", "then", "after that", "could you".
10. If a relation is not clearly expressed, omit it instead of inventing one.

Example input:
"Please create a plate with a banana inside, and a fork on the left of the plate."

Example output:
{
  "specified_objects": ["plate", "banana", "fork"],
  "random_objects_count": 0,
  "relationships": [
    {"child": "banana", "relation": "inside", "parent": "plate"},
    {"child": "fork", "relation": "left_of", "parent": "plate"}
  ]
}"""
        result = self._call_llm(system_prompt, instruction)
        print(f"[LLM] Parsed scene instruction: {result}")
        return {
            "specified_objects": result.get("specified_objects", []),
            "random_objects_count": result.get("random_objects_count", 0),
            "relationships": result.get("relationships", []),
        }

    def parse_task_primitives(self, instruction: str, scene_labels: list[str]) -> dict:
        """Parse a task description into a primitive sequence JSON plan."""
        system_prompt = """You are a task primitive parser.

Convert the user's natural-language task into a JSON object with this format:
{
  "primitives": [
    {"type": "...", "target": "...", "mode": "...", "position_xy": [x, y],
     "affordance_queries": [{"object": "...", "action": "..."}]}
  ]
}

Rules:
1. Allowed primitive types:
   pick, place, pour, push, pull, open, close, hang, hold, return_home
2. Output JSON only. No markdown. No explanation.
3. Each `target` must be only a short object reference phrase. Never include conjunctions, transition words,
   or the next action in `target`. Good: "kettle", "bowl", "plate". Bad: "kettle and then pour in the bowl".
4. Never invent extra disambiguating attributes that the user did not say. Do not add colors, materials, sizes,
   indices, or serial numbers on your own. If the user says "kettle", output "kettle", not "blue_kettle" or
   "white_kettle10". Only use an exact label when the user's wording already uniquely specifies it, such as
   "blue kettle" or "white kettle 10".
5. `target` is required for: pick, place(in/on), pour, push, pull, open, close, hang.
6. `target` must be omitted for: place(pose), hold, return_home.
7. Only `pick` and `place` may include `mode`.
8. If pick mode is not explicitly specified, use "affordance".
9. If place mode says in/into/inside, use "in".
10. If place mode says on/onto/on top of, use "on".
11. If the user specifies an absolute placement location such as "place at x=..., y=..." or "place to (x, y)",
    use `{"type": "place", "mode": "pose", "position_xy": [x, y]}` and omit `target`.
12. If place relation is not explicitly specified, default to "on".
13. For push, output `position_xy: [x, y]` only when the instruction explicitly provides x and y.
14. For place pose, output `position_xy: [x, y]` only when the instruction explicitly provides x and y.
15. Do not output `position_xy` for non-push, non-place-pose primitives.
16. Preserve execution order exactly as implied by the instruction.
17. Ignore filler words like "please", "then", "after that", "carefully", "finally".

Affordance query rules:
A1. `affordance_queries` is a list of {"object": "<short label>", "action": "<verb>"} entries describing
    which objects need an affordance region annotated and for which action.
A2. The only allowed `action` values are:
    "pick", "place", "pour", "push", "pull", "open", "close", "hang".
    Never use any other verb (no "grasp", "receive", "lift", etc.).
A3. The `object` field must be a short object reference phrase, following the same rules as `target`.
    Never invent disambiguating attributes the user did not say.
A4. Emit `affordance_queries` for: pick, place(in/on), pour, push, pull, open, close, hang.
    Do NOT emit `affordance_queries` for: place(pose), hold, return_home (omit the field entirely).
A5. For pick, push, pull, open, close, hang, place(in/on): output exactly one entry whose `object`
    equals the primitive's `target`, and whose `action` equals the primitive's `type`.
A6. For pour: output exactly two entries, both with `action: "pour"`:
    - one with `object` = the in-hand object (the most recent prior `pick`'s target),
    - one with `object` = the pour primitive's `target` (the receiving container).
    If no prior pick is visible in the instruction, emit only the receiving-container entry.
A7. Preserve the order of entries as: in-hand object first, then target container (for pour); target only
    (for the other action types).

Example input:
"Pick the kettle and then pour in the bowl, then place the kettle on the plate."

Example output:
{
  "primitives": [
    {"type": "pick", "target": "kettle", "mode": "affordance",
     "affordance_queries": [{"object": "kettle", "action": "pick"}]},
    {"type": "pour", "target": "bowl",
     "affordance_queries": [
       {"object": "kettle", "action": "pour"},
       {"object": "bowl", "action": "pour"}
     ]},
    {"type": "place", "target": "plate", "mode": "on",
     "affordance_queries": [{"object": "plate", "action": "place"}]}
  ]
}

Example input:
"Pick the kettle, then pick the cup."

Example output:
{
  "primitives": [
    {"type": "pick", "target": "kettle", "mode": "affordance",
     "affordance_queries": [{"object": "kettle", "action": "pick"}]},
    {"type": "pick", "target": "cup", "mode": "affordance",
     "affordance_queries": [{"object": "cup", "action": "pick"}]}
  ]
}

Example input:
"Pick the mug and then place it at x = 0.5, y = -0.4."

Example output:
{
  "primitives": [
    {"type": "pick", "target": "mug", "mode": "affordance",
     "affordance_queries": [{"object": "mug", "action": "pick"}]},
    {"type": "place", "mode": "pose", "position_xy": [0.5, -0.4]}
  ]
}

Example input:
"Open the drawer."

Example output:
{
  "primitives": [
    {"type": "open", "target": "drawer",
     "affordance_queries": [{"object": "drawer", "action": "open"}]}
  ]
}"""
        user_message = (
            "Available scene labels:\n"
            f"{json.dumps(scene_labels, ensure_ascii=True)}\n\n"
            "Task instruction:\n"
            f"{instruction}"
        )
        result = self._call_llm(system_prompt, user_message)
        print(f"[LLM] Parsed task primitives: {result}")
        return {
            "primitives": result.get("primitives", []) if isinstance(result, dict) else []
        }


class SceneGenerator:
    """Generates scene layouts from natural language instructions."""

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
        self.objects_db = self._load_objects_db()

    def _load_objects_db(self) -> dict:
        """Load object database, indexed by exact label."""
        try:
            with open(config.ALL_OBJECTS_JSON, "r", encoding="utf-8") as f:
                all_data = json.load(f)
            self._raw_objects = all_data
            mapping = {}
            for item in all_data:
                label = item.get("label", "").strip()
                if label:
                    mapping[label] = item
            return mapping
        except Exception as e:
            print(f"[ERROR] Loading objects data: {e}")
            return {}

    def find_object(self, label: str):
        """Find object by exact label match."""
        return self.objects_db.get(label.strip())

    def generate_scene_from_parsed(self, parsed: dict) -> list:
        """Generate complete scene from a pre-parsed scene specification."""
        specified = parsed.get("specified_objects", [])
        random_count = parsed.get("random_objects_count", 0)
        relationships = parsed.get("relationships", [])
        instances = {}

        for label in specified:
            label = label.strip()
            obj_data = self.find_object(label)
            if obj_data is None:
                print(f"[WARN] No exact match for '{label}' in label.json")
                similar = [k for k in self.objects_db if label.lower() in k.lower()]
                if similar:
                    print(f"       Similar labels: {similar[:10]}")
                continue
            if label in instances:
                print(f"[WARN] Duplicate label '{label}' ignored (one sim slot per USD)")
                continue
            instances[label] = {"label": label, "data": obj_data, "parent": None, "relation": None}
            print(f"[MATCH] '{label}' -> exact match")

        used_labels = set(instances.keys())
        available = [item for item in self._raw_objects if item.get("label", "") not in used_labels]
        if random_count > 0 and available:
            selected = random.sample(available, min(random_count, len(available)))
            for item in selected:
                label = item.get("label", "").strip()
                if label and label not in instances:
                    instances[label] = {"label": label, "data": item, "parent": None, "relation": None}

        for rel in relationships:
            child = rel.get("child", "").strip()
            parent = rel.get("parent", "").strip()
            raw_relation = rel.get("relation")
            relation = canonicalize_scene_relation(raw_relation)
            if child in instances and parent in instances:
                if relation is None:
                    print(
                        f"[WARN] Unsupported relationship '{raw_relation}' for child '{child}' "
                        f"and parent '{parent}', skipping relation"
                    )
                    continue
                instances[child]["parent"] = parent
                instances[child]["relation"] = relation
            else:
                if child not in instances:
                    print(f"[WARN] Relationship child '{child}' not found in scene")
                if parent not in instances:
                    print(f"[WARN] Relationship parent '{parent}' not found in scene")

        placement_queue = []
        unplaced = list(instances.values())
        for _ in range(10):
            placed_labels = {item["label"] for item in placement_queue}
            remaining = []
            for item in unplaced:
                if item["parent"] is None or item["parent"] in placed_labels:
                    placement_queue.append(item)
                else:
                    remaining.append(item)
            unplaced = remaining
            if not unplaced:
                break
        placement_queue.extend(unplaced)

        scene = []
        placed_map = {}
        for idx, item in enumerate(placement_queue):
            parent = placed_map.get(item["parent"])
            placement = None
            if parent and item["relation"]:
                placement = self._place_relative(item["data"], parent, item["relation"], scene)
                if placement is None:
                    print(
                        f"[WARN] Failed to apply relationship '{item['relation']}' for "
                        f"'{item['label']}', falling back to table placement"
                    )
            if placement is None:
                placement = self._place_on_table(item["data"], scene)
            if placement:
                placement["placement_order"] = idx + 1
                placement["_logic_id"] = item["label"]
                scene.append(placement)
                placed_map[item["label"]] = placement
            else:
                print(f"[WARN] Failed to place '{item['label']}'")

        print(f"[INFO] Generated scene with {len(scene)} objects.")
        return scene

    def _place_on_table(self, obj_data: dict, current_scene: list, max_attempts: int = 200) -> dict:
        """Find valid position on table surface."""
        scale = obj_data.get("scale", [1.0, 1.0, 1.0])
        dim_x = obj_data["dimensions_cm"]["length"] / 100.0 * scale[0]
        dim_y = obj_data["dimensions_cm"]["width"] / 100.0 * scale[1]
        raw_bottom_offset = obj_data.get(
            "bottom_offset_m",
            obj_data["dimensions_cm"]["height"] / 100.0 / 2.0,
        )
        bottom_offset = raw_bottom_offset * scale[2]
        placement_z = config.SAFE_Z + bottom_offset + 0.005

        for _ in range(max_attempts):
            angle = random.uniform(0, 2 * math.pi)
            x = random.uniform(config.TABLE_MIN_X, config.TABLE_MAX_X)
            y = random.uniform(config.TABLE_MIN_Y, config.TABLE_MAX_Y)
            if not self._check_collision((x, y), (dim_x, dim_y), angle, current_scene):
                return self._create_placement(obj_data, x, y, placement_z, angle, dim_x, dim_y)
        return None

    def _is_center_within_table_bounds(self, x: float, y: float) -> bool:
        """Check whether an XY center lies inside the editable tabletop region."""
        return (
            config.TABLE_MIN_X <= x <= config.TABLE_MAX_X
            and config.TABLE_MIN_Y <= y <= config.TABLE_MAX_Y
        )

    def _place_relative(self, child_data: dict, parent: dict, relation: str, current_scene: list) -> dict:
        """Place object relative to parent based on relation type."""
        p_pos = parent["position"]
        p_dim_x, p_dim_y = parent.get("_dims", (0.1, 0.1))
        p_angle = float(parent.get("_angle", 0.0))
        p_half_world_x, p_half_world_y = get_world_axis_half_extents(p_dim_x, p_dim_y, p_angle)
        p_height = parent.get("_height", 0.05)
        p_bottom_offset = parent.get("_bottom_offset", p_height / 2.0)
        c_scale = child_data.get("scale", [1.0, 1.0, 1.0])
        c_dim_x = child_data["dimensions_cm"]["length"] / 100.0 * c_scale[0]
        c_dim_y = child_data["dimensions_cm"]["width"] / 100.0 * c_scale[1]
        raw_c_bottom_offset = child_data.get(
            "bottom_offset_m",
            child_data["dimensions_cm"]["height"] / 100.0 / 2.0,
        )
        c_bottom_offset = raw_c_bottom_offset * c_scale[2]

        max_attempts = 100
        ignore_parent = False
        base_z = config.SAFE_Z + c_bottom_offset + 0.005
        near_candidates: list[tuple[float, float, float, float]] = []

        if relation in ["inside", "on_top"]:
            ignore_parent = True
            p_top_z = p_pos["z"] + (p_height - p_bottom_offset)
            if relation == "inside":
                p_mid_z = p_pos["z"] - p_bottom_offset + p_height * 0.5
                base_z = p_mid_z + c_bottom_offset
            else:
                base_z = p_top_z + c_bottom_offset + 0.002

            def sample():
                return 0, 0

        elif relation in ["left_of", "right_of", "front", "behind"]:
            gap_min, gap_max = 0.02, 0.15
        elif relation == "near":
            near_min = max((p_dim_x + c_dim_x) / 2.0, (p_dim_y + c_dim_y) / 2.0) + config.MARGIN
            near_max = near_min + 0.18
            max_attempts = 160

            def sample():
                theta = random.uniform(0.0, 2.0 * math.pi)
                radius = random.uniform(near_min, near_max)
                return radius * math.cos(theta), radius * math.sin(theta)

        else:
            return None

        for _ in range(max_attempts):
            angle = random.uniform(0, 2 * math.pi)
            c_half_world_x, c_half_world_y = get_world_axis_half_extents(c_dim_x, c_dim_y, angle)
            if relation in ["left_of", "right_of", "front", "behind"]:
                gap = random.uniform(gap_min, gap_max)
                if relation == "left_of":
                    dx = random.uniform(-p_half_world_x, p_half_world_x)
                    dy = p_half_world_y + c_half_world_y + gap
                elif relation == "right_of":
                    dx = random.uniform(-p_half_world_x, p_half_world_x)
                    dy = -(p_half_world_y + c_half_world_y + gap)
                elif relation == "front":
                    dx = p_half_world_x + c_half_world_x + gap
                    dy = random.uniform(-p_half_world_y, p_half_world_y)
                else:
                    dx = -(p_half_world_x + c_half_world_x + gap)
                    dy = random.uniform(-p_half_world_y, p_half_world_y)
            else:
                dx, dy = sample()

            x = p_pos["x"] + dx
            y = p_pos["y"] + dy
            if not self._is_center_within_table_bounds(x, y):
                continue

            ignore_id = parent["label"] if ignore_parent else None
            if self._check_collision((x, y), (c_dim_x, c_dim_y), angle, current_scene, ignore_id):
                continue

            if relation == "near":
                near_candidates.append((math.hypot(dx, dy), x, y, angle))
                continue

            return self._create_placement(child_data, x, y, base_z, angle, c_dim_x, c_dim_y)

        if relation == "near" and near_candidates:
            _, x, y, angle = min(near_candidates, key=lambda item: item[0])
            return self._create_placement(child_data, x, y, base_z, angle, c_dim_x, c_dim_y)
        return None

    def _check_collision(self, pos: tuple, dims: tuple, angle: float, scene: list, ignore_id: str = None) -> bool:
        """Check collision against all objects in scene."""
        margin_dims = (dims[0] + config.MARGIN, dims[1] + config.MARGIN)
        for obj in scene:
            if ignore_id and obj["label"] == ignore_id:
                continue
            ex_dims = obj.get("_dims", (0.1, 0.1))
            ex_pos = (obj["position"]["x"], obj["position"]["y"])
            ex_angle = obj.get("_angle", 0.0)
            ex_margin = (ex_dims[0] + config.MARGIN, ex_dims[1] + config.MARGIN)
            if check_obb_collision(pos, margin_dims, angle, ex_pos, ex_margin, ex_angle):
                return True
        return False

    def _create_placement(self, obj_data: dict, x: float, y: float, z: float, angle: float, dim_x: float, dim_y: float) -> dict:
        """Create standardized placement dictionary."""
        scale = obj_data.get("scale", [1.0, 1.0, 1.0])
        qw = math.cos(angle / 2.0)
        qz = math.sin(angle / 2.0)
        h = obj_data.get("dimensions_cm", {}).get("height", 5.0) / 100.0 * scale[2]
        raw_bottom_offset = obj_data.get(
            "bottom_offset_m",
            obj_data.get("dimensions_cm", {}).get("height", 5.0) / 100.0 / 2.0,
        )
        bottom_offset = raw_bottom_offset * scale[2]
        return {
            "label": obj_data.get("label", "unknown"),
            "position": {"x": round(x, 3), "y": round(y, 3), "z": round(z, 3)},
            "orientation": {"w": round(qw, 4), "x": 0.0, "y": 0.0, "z": round(qz, 4)},
            "_dims": (dim_x, dim_y),
            "_height": h,
            "_bottom_offset": bottom_offset,
            "_angle": angle,
        }


class SceneManager:
    """Manages scene state and simulation synchronization."""

    def __init__(
        self,
        sim_scene,
        generator: SceneGenerator,
        sim_object_labels: list[str] | None = None,
        sim_object_types: dict[str, str] | None = None,
    ):
        self.sim_scene = sim_scene
        self.generator = generator
        self.current_scene = []
        self.sim_object_labels = list(sim_object_labels or [])
        self.sim_object_types = dict(sim_object_types or {})
        with open(config.ALL_OBJECTS_JSON, "r", encoding="utf-8") as f:
            self.all_objects_data = json.load(f)

    def update_scene(self, scene_data: list):
        """Update current scene description."""
        self.current_scene = scene_data

    def _is_center_within_table_bounds(self, candidate: dict) -> bool:
        """Check whether the candidate center stays within table bounds."""
        pos = candidate.get("position", {})
        x = float(pos.get("x", 0.0))
        y = float(pos.get("y", 0.0))
        return (
            config.TABLE_MIN_X <= x <= config.TABLE_MAX_X
            and config.TABLE_MIN_Y <= y <= config.TABLE_MAX_Y
        )

    def _validate_pose_edit(self, obj_index: int, candidate: dict) -> tuple[bool, str]:
        """Validate a manual pose edit before committing it to scene state."""
        if not self._is_center_within_table_bounds(candidate):
            return False, "object center goes out of table bounds"
        dims = candidate.get("_dims", (0.1, 0.1))
        angle = float(candidate.get("_angle", 0.0))
        pos = candidate.get("position", {})
        xy = (float(pos.get("x", 0.0)), float(pos.get("y", 0.0)))
        other_scene = [obj for i, obj in enumerate(self.current_scene) if i != obj_index]
        if self.generator._check_collision(xy, dims, angle, other_scene):
            return False, "object collides with another object"
        return True, ""

    def try_move_object(self, obj_index: int, new_x: float, new_y: float) -> tuple[bool, str]:
        """Try to move an object and commit only if validation passes."""
        if not (0 <= obj_index < len(self.current_scene)):
            return False, "invalid object index"
        candidate = copy.deepcopy(self.current_scene[obj_index])
        candidate["position"]["x"] = round(float(new_x), 3)
        candidate["position"]["y"] = round(float(new_y), 3)
        ok, reason = self._validate_pose_edit(obj_index, candidate)
        if not ok:
            return False, reason
        self.current_scene[obj_index] = candidate
        return True, ""

    def try_rotate_object(self, obj_index: int, delta_deg: float) -> tuple[bool, str]:
        """Try to rotate an object around Z and commit only if validation passes."""
        if not (0 <= obj_index < len(self.current_scene)):
            return False, "invalid object index"
        candidate = copy.deepcopy(self.current_scene[obj_index])
        delta_rad = math.radians(float(delta_deg))
        new_angle = float(candidate.get("_angle", 0.0)) + delta_rad
        candidate["_angle"] = new_angle
        candidate["orientation"] = {
            "w": round(math.cos(new_angle / 2.0), 4),
            "x": 0.0,
            "y": 0.0,
            "z": round(math.sin(new_angle / 2.0), 4),
        }
        ok, reason = self._validate_pose_edit(obj_index, candidate)
        if not ok:
            return False, reason
        self.current_scene[obj_index] = candidate
        return True, ""

    def add_object(self, label: str):
        """Add object to scene by exact label match."""
        label = label.strip()
        obj_data = self.generator.find_object(label)
        if obj_data is None:
            print(f"[WARN] No exact match for '{label}' in label.json")
            similar = [k for k in self.generator.objects_db if label.lower() in k.lower()]
            if similar:
                print(f"       Similar labels: {similar[:10]}")
            return False
        if any(obj["label"] == label for obj in self.current_scene):
            print(f"[WARN] '{label}' already in scene")
            return False

        context = []
        for item in self.current_scene:
            matched = next((x for x in self.all_objects_data if x["label"] == item["label"]), None)
            temp = item.copy()
            if temp.get("_dims") and temp.get("_height") is not None:
                context.append(temp)
                continue
            if matched and "dimensions_cm" in matched:
                d = matched["dimensions_cm"]
                s = matched.get("scale", [1.0, 1.0, 1.0])
                temp["_dims"] = (d.get("length", 10) / 100 * s[0], d.get("width", 10) / 100 * s[1])
                temp["_height"] = d.get("height", 5) / 100 * s[2]
                raw_bottom_offset = matched.get("bottom_offset_m", d.get("height", 5) / 100 / 2.0)
                temp["_bottom_offset"] = raw_bottom_offset * s[2]
            context.append(temp)

        placement_probe = self.generator._place_on_table(obj_data, context, max_attempts=2000)
        if placement_probe is None:
            print(f"[WARN] No space for '{label}'")
            return False
        if label not in self.sim_object_labels:
            return "requires_scene_rebuild"
        placement_probe["_logic_id"] = label
        self.current_scene.append(placement_probe)
        print(f"[SUCCESS] Added '{label}'")
        return True

    def remove_object(self, label: str):
        """Remove object from scene by exact label match."""
        label = label.strip()
        to_remove = None
        for obj in self.current_scene:
            if obj.get("label", "") == label:
                to_remove = obj
                break
        if to_remove:
            self.current_scene.remove(to_remove)
            print(f"[SUCCESS] Removed '{label}'")
            return True
        print(f"[WARN] '{label}' not found in scene. Current objects:")
        for obj in self.current_scene:
            print(f"       - {obj.get('label', 'unknown')}")
        return False

    def sync_to_simulation(self):
        """Write current scene state to Isaac Lab simulation."""
        scene_by_label = {obj["label"]: obj for obj in self.current_scene}
        for index, obj_id in enumerate(self.sim_object_labels, start=1):
            obj_name = f"object_{index}"
            obj_type = self.sim_object_types.get(obj_name, "rigid")
            if obj_type == "articulation":
                if obj_name not in self.sim_scene.articulations:
                    continue
                sim_obj = self.sim_scene.articulations[obj_name]
            else:
                if obj_name not in self.sim_scene.rigid_objects:
                    continue
                sim_obj = self.sim_scene.rigid_objects[obj_name]

            if obj_id in scene_by_label:
                data = scene_by_label[obj_id]
                pos = [data["position"]["x"], data["position"]["y"], data["position"]["z"]]
                rot = [
                    data["orientation"]["w"],
                    data["orientation"]["x"],
                    data["orientation"]["y"],
                    data["orientation"]["z"],
                ]
            else:
                pos = [-10.0, -10.0 - index, 1.5]
                rot = [1.0, 0.0, 0.0, 0.0]

            pose = torch.tensor([pos + rot], dtype=torch.float32, device=sim_obj.device)
            sim_obj.write_root_pose_to_sim(pose)
            sim_obj.write_root_velocity_to_sim(torch.zeros((1, 6), dtype=torch.float32, device=sim_obj.device))

    def find_key_by_label(self, label: str) -> str:
        """Find object in scene by exact label match."""
        if not label:
            return None
        label = label.strip()
        for obj in self.current_scene:
            if obj.get("label", "") == label:
                return label
        return None

    def get_scene_labels(self) -> list[str]:
        """Return exact labels currently present in scene."""
        return [obj.get("label") for obj in self.current_scene if obj.get("label")]

    def load_from_yaml(self, yaml_path: str) -> bool:
        """Load scene from YAML configuration."""
        if not os.path.exists(yaml_path):
            print(f"[ERROR] File not found: {yaml_path}")
            return False

        try:
            import yaml as pyyaml

            with open(yaml_path, "r", encoding="utf-8") as f:
                cfg = pyyaml.safe_load(f)

            objects_dict = cfg.get("objects", {}) or {}
            articulations_dict = cfg.get("articulations", {}) or {}
            if not objects_dict and not articulations_dict:
                print("[WARN] No objects or articulations in YAML")
                return False

            object_metadata = {
                item.get("label"): item
                for item in self.all_objects_data
                if item.get("label")
            }
            scale_mismatches = []

            def _collect_scale_mismatches(entries: dict, object_type: str):
                for entry_name, obj_val in entries.items():
                    usd_path = obj_val.get("usd_path", "")
                    label = os.path.basename(usd_path).split(".")[0]
                    if not label:
                        continue
                    matched = object_metadata.get(label)
                    if matched is None:
                        continue
                    default_scale = normalize_scale_values(
                        matched.get("scale", [1.0, 1.0, 1.0]),
                        [1.0, 1.0, 1.0],
                    )
                    yaml_scale = normalize_scale_values(obj_val.get("scale", default_scale), default_scale)
                    if not scales_match(yaml_scale, default_scale):
                        scale_mismatches.append(
                            {
                                "entry_name": entry_name,
                                "object_type": object_type,
                                "label": label,
                                "yaml_scale": yaml_scale,
                                "default_scale": default_scale,
                            }
                        )

            _collect_scale_mismatches(objects_dict, "rigid")
            _collect_scale_mismatches(articulations_dict, "articulation")
            if scale_mismatches:
                print("[ERROR] Refusing to load YAML with non-default scales.")
                for mismatch in scale_mismatches:
                    print(
                        "       "
                        f"{mismatch['object_type']} '{mismatch['entry_name']}' "
                        f"(label='{mismatch['label']}') uses scale={mismatch['yaml_scale']} "
                        f"but label.json default is {mismatch['default_scale']}"
                    )
                return False

            reconstructed = []

            def _reconstruct_entries(entries: dict, object_type: str, start_order: int) -> int:
                next_idx = start_order
                for _, obj_val in entries.items():
                    usd_path = obj_val.get("usd_path", "")
                    label = os.path.basename(usd_path).split(".")[0]
                    if not label:
                        print(f"[WARN] Missing usd_path in {object_type} entry, skipping")
                        continue
                    matched = object_metadata.get(label)
                    if matched is None:
                        print(f"[WARN] Label '{label}' (from {usd_path}) not in label.json, skipping")
                        continue

                    default_scale = normalize_scale_values(
                        matched.get("scale", [1.0, 1.0, 1.0]),
                        [1.0, 1.0, 1.0],
                    )
                    scale = normalize_scale_values(obj_val.get("scale", default_scale), default_scale)

                    if "dimensions_cm" in matched:
                        d = matched["dimensions_cm"]
                        dim_x = d.get("length", 10) / 100 * scale[0]
                        dim_y = d.get("width", 10) / 100 * scale[1]
                        height = d.get("height", 5) / 100 * scale[2]
                    else:
                        dim_x, dim_y, height = 0.1, 0.1, 0.05

                    rot = obj_val.get("init_rot", [1, 0, 0, 0])
                    pos = obj_val.get("init_pos", [0, 0, 0])
                    angle = 2 * math.atan2(rot[3], rot[0])
                    raw_bo = matched.get("bottom_offset_m", height / 2.0 / scale[2])
                    bottom_offset = raw_bo * scale[2]

                    scene_entry = {
                        "label": label,
                        "position": {"x": pos[0], "y": pos[1], "z": pos[2]},
                        "orientation": {"w": rot[0], "x": rot[1], "y": rot[2], "z": rot[3]},
                        "_dims": (dim_x, dim_y),
                        "_height": height,
                        "_bottom_offset": bottom_offset,
                        "_angle": angle,
                        "placement_order": next_idx,
                        "_logic_id": label,
                        "_scene_object_type": object_type,
                        "_yaml_scale": scale,
                        "_semantic_tags": copy.deepcopy(obj_val.get("semantic_tags", [["class", label]])),
                    }
                    if object_type == "articulation":
                        scene_entry["_disable_gravity"] = bool(obj_val.get("disable_gravity", False))
                        scene_entry["_enabled_self_collisions"] = bool(obj_val.get("enabled_self_collisions", True))
                        scene_entry["_joint_pos"] = copy.deepcopy(obj_val.get("joint_pos", {}) or {})
                        scene_entry["_actuators"] = copy.deepcopy(obj_val.get("actuators", {}) or {})
                    reconstructed.append(scene_entry)
                    next_idx += 1
                return next_idx

            next_order = _reconstruct_entries(objects_dict, "rigid", 1)
            _reconstruct_entries(articulations_dict, "articulation", next_order)
            self.current_scene = reconstructed
            print(
                f"[SUCCESS] Loaded {len(reconstructed)} scene entries from YAML "
                f"({len(objects_dict)} rigid, {len(articulations_dict)} articulation definitions)"
            )
            return True
        except Exception as e:
            print(f"[ERROR] Loading YAML: {e}")
            import traceback

            traceback.print_exc()
            return False
