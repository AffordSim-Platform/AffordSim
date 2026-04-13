import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Any

@dataclass
class SceneConfig:
    """Base scene configuration"""
    name: str
    scene_tag: int
    description: str = ""
    grasp_offset: np.ndarray = 0
    place_offset: np.ndarray = 0
    objects: Optional[Dict[str, Any]] = None
    randomization_config: Optional[Dict[str, Any]] = None
    fixed_poses: Optional[Dict[str, Any]] = None