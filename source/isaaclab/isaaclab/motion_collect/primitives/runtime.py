from functools import partial
from typing import Any, Dict, List

from .close_microwave import primitive_close
from .hang import primitive_hang
from .hold import primitive_hold
from .pick import primitive_pick
from .place import primitive_place
from .pour import primitive_pour
from .pull import primitive_pull
from .push import primitive_push
from .return_home import primitive_return_home
from .open_microwave import primitive_open


def build_primitive_context(
    collector,
    episode_num: int,
    scene_desc: List[dict],
    task_desc: str,
) -> Dict[str, Any]:
    """Construct shared primitive context."""
    return {
        "episode_num": episode_num,
        "task_name": collector.task_name,
        "plan_success_mask": [True] * collector.num_envs,
        "task_success_mask": [True] * collector.num_envs,
        "env_alive_mask": [True] * collector.num_envs,
        "env_failure_reasons": ["none"] * collector.num_envs,
        "step_failure_reason_mask": None,
        "scene_desc": scene_desc,
        "task_desc": task_desc,
        "grasp_pose": None,
        "pick_container_context": None,
        "place_pose": None,
        "pour_pose": None,
        "hang_pose": None,
        "held_object": None,
        "target_container": None,
        "primitive_executed_motion": False,
        "current_gripper_state": "open",
    }


def build_primitive_handlers(collector) -> Dict[str, Any]:
    """Build primitive handler registry for a collector instance."""
    return {
        "pick": partial(primitive_pick, collector),
        "place": partial(primitive_place, collector),
        "pour": partial(primitive_pour, collector),
        "hang": partial(primitive_hang, collector),
        "hold": partial(primitive_hold, collector),
        "open": partial(primitive_open, collector),
        "close": partial(primitive_close, collector),
        "pull": partial(primitive_pull, collector),
        "push": partial(primitive_push, collector),
        "return_home": partial(primitive_return_home, collector),
    }
