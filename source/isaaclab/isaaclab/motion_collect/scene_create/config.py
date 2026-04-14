# config.py
import os

# API Settings
API_KEY = "your key"
BASE_URL = ""

# Paths
CUR_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = os.path.abspath(os.path.join(CUR_DIR, "../../.."))  

ASSET_ROOT = os.path.join(PROJECT_ROOT, "models/all_object")
TEXTURE_DIR = os.path.join(PROJECT_ROOT, "models/random_pic")
ALL_OBJECTS_JSON = os.path.join(ASSET_ROOT, "label.json")

SAVE_YAML_PATH = os.path.join(
    PROJECT_ROOT,
    "source/isaaclab_tasks/isaaclab_tasks/real2sim/tasks/pick/config/sceneconfig/task/pick/pick_redcup_lower.yaml"
)

# Table Constraints
# TABLE_CENTER_X = 0.68
# TABLE_WIDTH_X = 0.40
TABLE_MIN_X = 0.38
TABLE_MAX_X = 0.88
TABLE_MIN_Y = -0.75
TABLE_MAX_Y = 0.75
SAFE_Z = 0.025
MARGIN = 0.02
