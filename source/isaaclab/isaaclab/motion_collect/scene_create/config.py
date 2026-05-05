# config.py
import os

# Qwen API Settings (DashScope)
API_KEY = os.getenv("DASHSCOPE_API_KEY", "sk-18f3f0aa86044683bd58ec537de79a6f")
BASE_URL = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
MODEL = os.getenv("DASHSCOPE_MODEL", "qwen-vl-plus")
TIMEOUT = float(os.getenv("SCENE_LLM_TIMEOUT", "30"))

# Paths
CUR_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = os.path.abspath(os.path.join(CUR_DIR, "../../../../.."))

ASSET_ROOT = os.path.join(PROJECT_ROOT, "models/all_object")
SCENE_ASSET_ROOT = os.path.join(PROJECT_ROOT, "models/scene")
DEFAULT_SCENE_NAME = "table"
DEFAULT_SCENE_USD_PATH = os.path.join(SCENE_ASSET_ROOT, "table.usd")
SCENE_SELECTION_ORDER = [DEFAULT_SCENE_NAME]
TEXTURE_DIR = os.path.join(PROJECT_ROOT, "models/random_pic")
ALL_OBJECTS_JSON = os.path.join(ASSET_ROOT, "label.json")
TASK_CONFIG_ROOT = os.path.join(
    PROJECT_ROOT,
    "source/isaaclab_tasks/isaaclab_tasks/real2sim/tasks/config/task",
)
DEFAULT_LOAD_YAML_PATH = os.path.join(
    TASK_CONFIG_ROOT,
    "pick/pick_banana_place_bowl.yaml",
)

SAVE_YAML_PATH = os.path.join(
    TASK_CONFIG_ROOT,
    "pick/scene_editor_output.yaml",
)

# Table Constraints
TABLE_BOUNDS_BY_SCENE = {
    "table": {
        "center": [0.63, 0.0, 0.025],
        "length_x": 0.5,
        "width_y": 1.5,
        "min_x": 0.38,
        "max_x": 0.88,
        "min_y": -0.75,
        "max_y": 0.75,
        "safe_z": 0.025,
    },
}


# Texture-randomization Cube prim suffix per scene, relative to "/World/envs/env_{i}/table/".
SCENE_TEXTURE_TARGETS = {
    "table": "Xform_01/Xform/leg/node_/Cube",
}


def get_scene_texture_target(scene_name: str | None = None) -> str:
    """Return the per-env Cube prim path template for the given static scene."""
    key = str(scene_name or "table").strip().lower()
    suffix = SCENE_TEXTURE_TARGETS.get(key, SCENE_TEXTURE_TARGETS["table"])
    return f"/World/envs/env_{{i}}/table/{suffix}"


def get_table_bounds(scene_name: str | None = None) -> dict:
    """Return tabletop bounds for a scene name, falling back to the default table scene."""
    key = str(scene_name or "table").strip().lower()
    return dict(TABLE_BOUNDS_BY_SCENE.get(key, TABLE_BOUNDS_BY_SCENE["table"]))


def apply_table_bounds(scene_name: str | None = None) -> dict:
    """Apply scene-specific tabletop bounds to the legacy module-level constants."""
    bounds = get_table_bounds(scene_name)
    global TABLE_MIN_X, TABLE_MAX_X, TABLE_MIN_Y, TABLE_MAX_Y, SAFE_Z
    TABLE_MIN_X = bounds["min_x"]
    TABLE_MAX_X = bounds["max_x"]
    TABLE_MIN_Y = bounds["min_y"]
    TABLE_MAX_Y = bounds["max_y"]
    SAFE_Z = bounds["safe_z"]
    return bounds


_DEFAULT_TABLE_BOUNDS = apply_table_bounds("table")
MARGIN = 0.02
