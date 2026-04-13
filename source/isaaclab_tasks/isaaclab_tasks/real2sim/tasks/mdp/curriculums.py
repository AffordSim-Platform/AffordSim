from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING
import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def gradual_reward_weight_modification( # Curriculum function to linearly change a reward weight
    env: ManagerBasedRLEnv, 
    env_ids: Sequence[int],
    term_name: str,
    initial_weight: float,
    final_weight: float,
    start_it: int,
    end_it: int,
):
    """Curriculum that gradually modifies a reward weight between an initial and final value over a range of steps."""
    # current_it = getattr(env, 'current_learning_iteration_for_curriculums', 0)
    # current_it = env.current_learning_iteration_for_curriculums
    current_it = env.common_step_counter // 24  
    if current_it < start_it:
        return

    if current_it >= end_it:
        new_weight = final_weight
    else:
        new_weight = (current_it - start_it) / (end_it - start_it) * (final_weight - initial_weight) + initial_weight

    # print(f"[Curriculum] Setting reward weight for '{term_name}' to {new_weight:.4f} at iteration {current_it}")
    term_cfg = env.reward_manager.get_term_cfg(term_name)
    term_cfg.weight = new_weight
    env.reward_manager.set_term_cfg(term_name, term_cfg)

    # return new_weight

def expand_position_range_all_axes(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    term_name: str,
    asset_name: str,
    initial_ranges: dict[str, tuple[float, float]],
    final_ranges: dict[str, tuple[float, float]],
    start_it: int,
    end_it: int,
) -> None:
    """
    Curriculum learning: gradually expand the random range of object positions.

    Behavior:
    - current_it < start_it: use initial_ranges
    - start_it <= current_it < end_it: linearly interpolate from initial_ranges to final_ranges
    - current_it >= end_it: use final_ranges

    Args:
        env: Environment instance.
        env_ids: Environment IDs.
        term_name: Curriculum term name (for logging).
        asset_name: Object name (e.g. "basic_block_red").
        initial_ranges: Initial range dict {'x': (min, max), 'y': (min, max), 'z': (min, max)}.
        final_ranges: Final range dict.
        start_it: Iteration at which expansion starts.
        end_it: Iteration at which final range is reached.
    """
    if not hasattr(env, '_curriculum_last_it'):
        env._curriculum_last_it = {}
    last_it = env._curriculum_last_it.get(term_name, 0)
    current_it = env.common_step_counter // 24
    # current_it = getattr(env, 'current_learning_iteration_for_curriculums', 0)
    if current_it > last_it :
        print(f"[Curriculum] expand_position_range_all_axes called at iteration {current_it} for asset '{asset_name}' with term '{term_name}'")
    env._curriculum_last_it[term_name] = current_it
    # Compute progress
    if current_it < start_it:
        progress = 0.0
    elif current_it >= end_it:
        progress = 1.0
    else:
        progress = (current_it - start_it) / (end_it - start_it)
    
    # Initialize storage
    if not hasattr(env, '_curriculum_position_range'):
        env._curriculum_position_range = {}
    if asset_name not in env._curriculum_position_range:
        env._curriculum_position_range[asset_name] = {}
    
    # Update all axes
    for axis in ['x', 'y', 'z']:
        if axis in initial_ranges and axis in final_ranges:
            init_range = initial_ranges[axis]
            final_range = final_ranges[axis]
            
            current_min = init_range[0] + progress * (final_range[0] - init_range[0])
            current_max = init_range[1] + progress * (final_range[1] - init_range[1])
            
            env._curriculum_position_range[asset_name][axis] = (current_min, current_max)
    # print(f"[Curriculum] Asset '{asset_name}' range updated to ({env._curriculum_position_range[asset_name]}) at iteration {current_it}")