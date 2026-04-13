from isaaclab.scene.real2sim_scene_cfg import SceneConfig
from isaaclab_tasks.real2sim.tasks.pick.config.sceneconfig.sceneconfig import SCENE_CONFIGS


def get_scene_config(scene_name: str, randomize_mode: str) -> SceneConfig:
   
    if scene_name not in SCENE_CONFIGS:
        available = list(SCENE_CONFIGS.keys())
        raise ValueError(f"Scene '{scene_name}' not found. Available: {available}")
    
    if randomize_mode not in SCENE_CONFIGS[scene_name]:
        available = list(SCENE_CONFIGS[scene_name].keys())
        raise ValueError(f"Mode '{randomize_mode}' not found for scene '{scene_name}'. Available: {available}")
    
    return SCENE_CONFIGS[scene_name][randomize_mode]

def apply_scene_config(env_cfg, scene_config: SceneConfig):

    if scene_config.randomization_config is not None:
        env_cfg.set_randomization_config(scene_config.randomization_config)
    elif scene_config.fixed_poses is not None:
        env_cfg.set_fixed_poses(scene_config.fixed_poses)
    else:
        # No initialization
        env_cfg.disable_initialization()
    env_cfg.set_scene(scene_config.scene_tag)