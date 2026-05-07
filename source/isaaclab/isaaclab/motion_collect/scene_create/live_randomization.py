"""Live preview helpers for light and texture randomization in the scene editor.

These run inside the editor's Isaac Sim app right after the user confirms the
related YAML option, so the change is visible in the viewport before export.
"""

import os
import random

import omni.usd
from pxr import Sdf, UsdGeom, UsdShade

import config


_LOOKS_SCOPE = "/Looks"


def _get_stage():
    return omni.usd.get_context().get_stage()


_LIGHT_RANDOMIZATION_CONFIGS = [
    {
        "prim_path": "/World/light_sphere",
        "intensity_range": (500.0, 14000.0),
        "color_range": ((0.35, 0.30, 0.25), (1.0, 0.95, 0.85)),
        "position_range": ((-2.8, -2.8, 0.8), (2.8, 2.8, 4.2)),
    },
    {
        "prim_path": "/World/light_disk",
        "intensity_range": (300.0, 12000.0),
        "color_range": ((0.30, 0.35, 0.45), (1.0, 0.95, 1.0)),
        "position_range": ((-3.5, -3.0, 0.7), (3.5, 3.0, 4.5)),
    },
    {
        "prim_path": "/World/light_distant",
        "intensity_range": (100.0, 5000.0),
        "color_range": ((0.45, 0.40, 0.35), (1.0, 1.0, 0.95)),
        "direction_range": ((-85.0, -85.0, -180.0), (85.0, 85.0, 180.0)),
    },
]


def _set_attr(prim, attr_name: str, value) -> bool:
    attr = prim.GetAttribute(attr_name)
    if not attr.IsValid():
        return False
    attr.Set(value)
    return True


def _set_xform_value(prim, attr_name: str, value) -> bool:
    attr = prim.GetAttribute(attr_name)
    if attr.IsValid():
        attr.Set(value)
        return True

    xformable = UsdGeom.Xformable(prim)
    if attr_name == "xformOp:translate":
        xformable.AddTranslateOp().Set(value)
        return True
    if attr_name == "xformOp:rotateXYZ":
        xformable.AddRotateXYZOp().Set(value)
        return True
    return False


def _sample_range(range_pair):
    lo, hi = range_pair
    return (
        random.uniform(lo[0], hi[0]),
        random.uniform(lo[1], hi[1]),
        random.uniform(lo[2], hi[2]),
    )


def _disable_light(prim):
    intensity_attr = prim.GetAttribute("inputs:intensity")
    if intensity_attr.IsValid():
        intensity_attr.Set(0.0)


def apply_random_lights_preview(runtime: dict | None = None) -> bool:
    """Preview the same three-light randomization pattern used during collection."""
    stage = _get_stage()
    valid_configs = [
        item for item in _LIGHT_RANDOMIZATION_CONFIGS
        if stage.GetPrimAtPath(item["prim_path"]).IsValid()
    ]
    if not valid_configs:
        paths = ", ".join(item["prim_path"] for item in _LIGHT_RANDOMIZATION_CONFIGS)
        print(f"[live-rand] WARN: no preview lights found at {paths}")
        return False

    chosen = random.choice(valid_configs)
    chosen_path = chosen["prim_path"]
    for item in _LIGHT_RANDOMIZATION_CONFIGS:
        prim = stage.GetPrimAtPath(item["prim_path"])
        if not prim.IsValid():
            continue
        if item["prim_path"] != chosen_path:
            _disable_light(prim)
            continue

        intensity = random.uniform(*item["intensity_range"])
        color = _sample_range(item["color_range"])
        _set_attr(prim, "inputs:intensity", intensity)
        _set_attr(prim, "inputs:color", color)

        position = None
        direction = None
        if "position_range" in item:
            position = _sample_range(item["position_range"])
            _set_xform_value(prim, "xformOp:translate", position)
        if "direction_range" in item:
            direction = _sample_range(item["direction_range"])
            _set_xform_value(prim, "xformOp:rotateXYZ", direction)

        print(f"[live-rand] activated={chosen_path} intensity={intensity:.1f} color={color}")
        if position is not None:
            print(f"[live-rand] position={position}")
        if direction is not None:
            print(f"[live-rand] direction={direction}")
    return True


def _resolve_texture_dir(scene_name: str | None) -> str:
    """Pick a texture sub-folder for a given scene; fall back to root TEXTURE_DIR."""
    candidates = [os.path.join(config.TEXTURE_DIR, "table"), config.TEXTURE_DIR]
    for path in candidates:
        if os.path.isdir(path) and any(f.endswith(".png") for f in os.listdir(path)):
            return path
    return config.TEXTURE_DIR


def apply_random_texture_preview(
    num_envs: int = 1,
) -> bool:
    """Bind a randomly chosen PNG texture onto the table scene Cube prim.

    `config.get_scene_texture_target` returns the table scene's per-env Cube
    prim template.
    """
    scene_name = "table"
    target_template = config.get_scene_texture_target(scene_name)
    texture_dir = _resolve_texture_dir(scene_name)

    textures = [f for f in os.listdir(texture_dir) if f.endswith(".png")]
    if not textures:
        print(f"[live-rand] WARN: no .png textures in {texture_dir}")
        return False

    stage = _get_stage()
    if not stage.GetPrimAtPath(_LOOKS_SCOPE).IsValid():
        UsdGeom.Scope.Define(stage, _LOOKS_SCOPE)

    bound_any = False
    for env_idx in range(max(1, int(num_envs))):
        mesh_prim_path = target_template.format(i=env_idx)
        mesh_prim = stage.GetPrimAtPath(mesh_prim_path)
        if not mesh_prim.IsValid():
            print(f"[live-rand] WARN: prim not found for {scene_name}: {mesh_prim_path}")
            continue

        tex_file = random.choice(textures)
        texture_path = os.path.join(texture_dir, tex_file)

        mtl_path = f"/Looks/PreviewTexMaterial_{scene_name}_env_{env_idx}"
        shader_path = f"{mtl_path}/Shader"

        material = UsdShade.Material.Define(stage, mtl_path)
        shader = UsdShade.Shader.Define(stage, shader_path)
        shader.SetSourceAsset("OmniPBR.mdl", "mdl")
        shader.SetSourceAssetSubIdentifier("OmniPBR", "mdl")

        diffuse_input = shader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset)
        diffuse_input.Set(Sdf.AssetPath(texture_path))

        shader_output = shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
        material.CreateSurfaceOutput("mdl").ConnectToSource(shader_output)
        material.CreateVolumeOutput("mdl").ConnectToSource(shader_output)
        material.CreateDisplacementOutput("mdl").ConnectToSource(shader_output)

        UsdShade.MaterialBindingAPI(mesh_prim).Bind(
            material, UsdShade.Tokens.strongerThanDescendants
        )
        print(f"[live-rand] bound texture {tex_file}  to  {mesh_prim_path}")
        bound_any = True

    return bound_any
