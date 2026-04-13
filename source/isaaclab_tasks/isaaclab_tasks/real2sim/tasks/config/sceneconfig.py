import numpy as np
from typing import Dict
from isaaclab.scene.real2sim_scene_cfg import SceneConfig
from .pick_up_cube import PICK_UP_CUBE_RANDOM, PICK_UP_CUBE_FIXED
from .pick_up_cube_place_bowl import PICK_UP_CUBE_PLACE_BOWL_RANDOM, PICK_UP_CUBE_PLACE_BOWL_FIXED
from .pick_up_mug_place_cup import PICK_UP_MUG_PLACE_BOWL_RANDOM, PICK_UP_MUG_PLACE_BOWL_FIXED

# ============================================================================
# Scene Registry
# ============================================================================


SCENE_CONFIGS: Dict[str, Dict[str, SceneConfig]] = {
    "pick up cube": {
        "Random": PICK_UP_CUBE_RANDOM,
        "Fixed": PICK_UP_CUBE_FIXED,
        "None": SceneConfig(
            name="pick up cube", 
            scene_tag=2, 
            description="No initialization",
            grasp_offset=np.array([0, 0, 0.12]),
            place_offset=np.array([0, 0, 0.12]),
            objects={
                "target": "_005_mustard", #_005_mustard basic_block_red
                "container": None,}, # None _024_bowl
        ),
    },
    "pick up cube and place in bowl": {
        "Random": PICK_UP_CUBE_PLACE_BOWL_RANDOM,
        "Fixed": PICK_UP_CUBE_PLACE_BOWL_FIXED,
        "None": SceneConfig(
            name="pick up cube and place in bowl", 
            scene_tag=3, 
            description="No initialization",
            grasp_offset=np.array([0, 0, 0.18]),
            place_offset=np.array([0, 0, 0.28]),
            objects={
                "target": "basic_block_red",
                "container": "_24_bowl_red",},
        ),
    },
    "pick up mug and place in bowl": {
        "Random": PICK_UP_MUG_PLACE_BOWL_RANDOM,
        "Fixed": PICK_UP_MUG_PLACE_BOWL_FIXED,
        "None": SceneConfig(
            name="pick up mug and place in bowl", 
            scene_tag=1, 
            description="No initialization",
            grasp_offset=np.array([0, 0, 0.1]),
            place_offset=np.array([0, 0, 0.2]),
            objects={
                "target": "basic_block_red",# basic_block_red _10_potted_meat_can
                "container": "_25_mug",},
        ),
    }
}

