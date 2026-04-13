import yaml
from pathlib import Path
from isaaclab_tasks.real2sim import REAL2SIM_ROOT_DIR

def load_config(config_path: str = "config.yaml"):
    """Load configuration from YAML file"""
    # Note: Ensure this path is correct in your environment
    config_path = f"{REAL2SIM_ROOT_DIR}/source/isaaclab_tasks/isaaclab_tasks/real2sim/tasks/pick/config/sceneconfig/{config_path}"
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg