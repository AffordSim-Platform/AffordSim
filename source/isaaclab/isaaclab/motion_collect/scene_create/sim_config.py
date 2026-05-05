import glob
import json
import os
from pathlib import Path

import numpy as np
from pxr import Usd, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass


_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
ASSET_PATH = os.path.abspath(os.path.join(_CUR_DIR, "../../../../..", "models"))
ALL_OBJECT_DIR = os.path.join(ASSET_PATH, "all_object")


def _load_scale_map() -> dict[str, tuple[float, float, float]]:
    """Load default asset scales from label.json."""
    scale_map = {}
    label_json_path = os.path.join(ALL_OBJECT_DIR, "label.json")
    if os.path.exists(label_json_path):
        with open(label_json_path, "r", encoding="utf-8") as f:
            label_data = json.load(f)
        for item in label_data:
            label = item.get("label")
            if label:
                scale_map[label] = tuple(item.get("scale", [1.0, 1.0, 1.0]))
    return scale_map


def _is_articulation_usd(usd_path: str) -> bool:
    """Check if a USD file contains an articulation (has joints)."""
    try:
        stage = Usd.Stage.Open(usd_path)
        for prim in stage.Traverse():
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                return True
            if prim.IsA(UsdPhysics.Joint):
                return True
        return False
    except Exception:
        return False


def _build_asset_registry() -> dict[str, dict]:
    """Scan all available USD assets without pre-registering them into the scene."""
    scale_map = _load_scale_map()
    registry = {}
    for usd_path in sorted(glob.glob(os.path.join(ALL_OBJECT_DIR, "*.usd"))):
        label = Path(usd_path).stem
        registry[label] = {
            "usd_path": usd_path,
            "object_type": "articulation" if _is_articulation_usd(usd_path) else "rigid",
            "scale": scale_map.get(label, (1.0, 1.0, 1.0)),
        }
    return registry


ASSET_REGISTRY = _build_asset_registry()


def get_asset_registry() -> dict[str, dict]:
    """Return the asset registry keyed by strict label."""
    return ASSET_REGISTRY


def get_available_asset_labels() -> list[str]:
    """Return all known asset labels."""
    return sorted(ASSET_REGISTRY.keys())


@configclass
class BaseSceneCfg(InteractiveSceneCfg):
    light_sphere = AssetBaseCfg(
        prim_path="/World/light_sphere",
        spawn=sim_utils.SphereLightCfg(intensity=0.0, color=(1.0, 1.0, 1.0)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(1.0, 0.0, 2.0)),
    )
    light_disk = AssetBaseCfg(
        prim_path="/World/light_disk",
        spawn=sim_utils.DiskLightCfg(intensity=0.0, color=(1.0, 1.0, 1.0)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-1.0, 0.0, 2.5)),
    )
    light_distant = AssetBaseCfg(
        prim_path="/World/light_distant",
        spawn=sim_utils.DistantLightCfg(intensity=2500.0, color=(1.0, 1.0, 1.0)),
        init_state=AssetBaseCfg.InitialStateCfg(rot=(0.9238795, 0.3826834, 0.0, 0.0)),
    )
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.8)),
    )
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ASSET_PATH}/table.usd",
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0, 0, 0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )


def _build_table_cfg(scene_usd_path: str | None = None):
    """Create the static scene/table asset config."""
    return AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=scene_usd_path or f"{ASSET_PATH}/table.usd",
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0, 0, 0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )


def _build_object_cfg(slot_name: str, label: str, asset_info: dict):
    """Create a scene asset config for one requested label."""
    prim_path = "{ENV_REGEX_NS}/" + slot_name
    init_pos = [10.0, 0.0, 0.25]
    init_rot = [1, 0, 0, 0]
    usd_cfg = UsdFileCfg(usd_path=asset_info["usd_path"], scale=asset_info["scale"])

    if asset_info["object_type"] == "articulation":
        return ArticulationCfg(
            prim_path=prim_path,
            init_state=ArticulationCfg.InitialStateCfg(pos=init_pos, rot=init_rot),
            spawn=usd_cfg,
            actuators={
                "all_joints": ImplicitActuatorCfg(
                    joint_names_expr=[".*"],
                    effort_limit_sim=87.0,
                    velocity_limit_sim=2.175,
                    stiffness=400.0,
                    damping=80.0,
                ),
            },
        )

    return RigidObjectCfg(
        prim_path=prim_path,
        init_state=RigidObjectCfg.InitialStateCfg(pos=init_pos, rot=init_rot),
        spawn=usd_cfg,
    )


def build_scene_cfg(
    active_labels: list[str],
    num_envs: int = 1,
    env_spacing: float = 2.0,
    scene_usd_path: str | None = None,
    scene_name: str | None = None,
):
    """Build a scene config instance for the requested subset of labels.

    `scene_name` is kept for compatibility. The editor does not spawn a robot.
    """
    ordered_labels = []
    seen = set()
    missing = []
    for raw_label in active_labels:
        label = str(raw_label).strip()
        if not label or label in seen:
            continue
        seen.add(label)
        asset_info = ASSET_REGISTRY.get(label)
        if asset_info is None:
            missing.append(label)
            continue
        ordered_labels.append(label)

    if missing:
        raise ValueError(f"Unknown scene labels requested: {sorted(missing)}")

    attrs = {
        "__module__": __name__,
        "table": _build_table_cfg(scene_usd_path),
    }
    slot_labels = []
    slot_types = {}
    for index, label in enumerate(ordered_labels, start=1):
        slot_name = f"object_{index}"
        asset_info = ASSET_REGISTRY[label]
        attrs[slot_name] = _build_object_cfg(slot_name, label, asset_info)
        slot_labels.append(label)
        slot_types[slot_name] = asset_info["object_type"]

    dynamic_cfg_cls = configclass(type("DynamicSceneCfg", (BaseSceneCfg,), attrs))
    scene_cfg = dynamic_cfg_cls(num_envs=num_envs, env_spacing=env_spacing)
    return scene_cfg, slot_labels, slot_types
