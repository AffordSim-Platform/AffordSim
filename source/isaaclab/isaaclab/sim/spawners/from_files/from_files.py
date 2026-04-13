# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import isaacsim.core.utils.prims as prim_utils
import omni.kit.commands
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

# from Isaac Sim 4.2 onwards, pxr.Semantics is deprecated
try:
    import Semantics
except ModuleNotFoundError:
    from pxr import Semantics

from isaaclab.sim import converters, schemas
from isaaclab.sim.utils import bind_physics_material, bind_visual_material, clone, select_usd_variants
from isaaclab.sim.utils.stage import get_current_stage, is_current_stage_in_memory
from isaaclab.utils.assets import check_usd_path_with_timeout

if TYPE_CHECKING:
    from . import from_files_cfg

# import logger
logger = logging.getLogger(__name__)


@clone
def spawn_from_usd(
    prim_path: str,
    cfg: from_files_cfg.UsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn an asset from a USD file and override the settings with the given config.

    In the case of a USD file, the asset is spawned at the default prim specified in the USD file.
    If a default prim is not specified, then the asset is spawned at the root prim.

    In case a prim already exists at the given prim path, then the function does not create a new prim
    or throw an error that the prim already exists. Instead, it just takes the existing prim and overrides
    the settings with the given config.

    .. note::
        This function is decorated with :func:`clone` that resolves prim path into list of paths
        if the input prim path is a regex pattern. This is done to support spawning multiple assets
        from a single and cloning the USD prim at the given path expression.

    Args:
        prim_path: The prim path or pattern to spawn the asset at. If the prim path is a regex pattern,
            then the asset is spawned at all the matching prim paths.
        cfg: The configuration instance.
        translation: The translation to apply to the prim w.r.t. its parent prim. Defaults to None, in which
            case the translation specified in the USD file is used.
        orientation: The orientation in (w, x, y, z) to apply to the prim w.r.t. its parent prim. Defaults to None,
            in which case the orientation specified in the USD file is used.
        **kwargs: Additional keyword arguments, like ``clone_in_fabric``.

    Returns:
        The prim of the spawned asset.

    Raises:
        FileNotFoundError: If the USD file does not exist at the given path.
    """
    # spawn asset from the given usd file
    return _spawn_from_usd_file(prim_path, cfg.usd_path, cfg, translation, orientation)


@clone
def spawn_from_urdf(
    prim_path: str,
    cfg: from_files_cfg.UrdfFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn an asset from a URDF file and override the settings with the given config.

    It uses the :class:`UrdfConverter` class to create a USD file from URDF. This file is then imported
    at the specified prim path.

    In case a prim already exists at the given prim path, then the function does not create a new prim
    or throw an error that the prim already exists. Instead, it just takes the existing prim and overrides
    the settings with the given config.

    .. note::
        This function is decorated with :func:`clone` that resolves prim path into list of paths
        if the input prim path is a regex pattern. This is done to support spawning multiple assets
        from a single and cloning the USD prim at the given path expression.

    Args:
        prim_path: The prim path or pattern to spawn the asset at. If the prim path is a regex pattern,
            then the asset is spawned at all the matching prim paths.
        cfg: The configuration instance.
        translation: The translation to apply to the prim w.r.t. its parent prim. Defaults to None, in which
            case the translation specified in the generated USD file is used.
        orientation: The orientation in (w, x, y, z) to apply to the prim w.r.t. its parent prim. Defaults to None,
            in which case the orientation specified in the generated USD file is used.
        **kwargs: Additional keyword arguments, like ``clone_in_fabric``.

    Returns:
        The prim of the spawned asset.

    Raises:
        FileNotFoundError: If the URDF file does not exist at the given path.
    """
    # urdf loader to convert urdf to usd
    urdf_loader = converters.UrdfConverter(cfg)
    # spawn asset from the generated usd file
    return _spawn_from_usd_file(prim_path, urdf_loader.usd_path, cfg, translation, orientation)


def spawn_ground_plane(
    prim_path: str,
    cfg: from_files_cfg.GroundPlaneCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawns a ground plane into the scene.

    This function loads the USD file containing the grid plane asset from Isaac Sim. It may
    not work with other assets for ground planes. In those cases, please use the `spawn_from_usd`
    function.

    Note:
        This function takes keyword arguments to be compatible with other spawners. However, it does not
        use any of the kwargs.

    Args:
        prim_path: The path to spawn the asset at.
        cfg: The configuration instance.
        translation: The translation to apply to the prim w.r.t. its parent prim. Defaults to None, in which
            case the translation specified in the USD file is used.
        orientation: The orientation in (w, x, y, z) to apply to the prim w.r.t. its parent prim. Defaults to None,
            in which case the orientation specified in the USD file is used.
        **kwargs: Additional keyword arguments, like ``clone_in_fabric``.

    Returns:
        The prim of the spawned asset.

    Raises:
        ValueError: If the prim path already exists.
    """
    # Spawn Ground-plane
    if not prim_utils.is_prim_path_valid(prim_path):
        prim_utils.create_prim(prim_path, usd_path=cfg.usd_path, translation=translation, orientation=orientation)
    else:
        raise ValueError(f"A prim already exists at path: '{prim_path}'.")

    # Create physics material
    if cfg.physics_material is not None:
        cfg.physics_material.func(f"{prim_path}/physicsMaterial", cfg.physics_material)
        # Apply physics material to ground plane
        # Note: newer Isaac Sim (5.x) ground plane USD assets may not expose a child prim
        # of type "Plane". First try "Plane", then fall back to "Mesh", so we never pass
        # ``None`` to ``bind_physics_material``.
        collision_prim = prim_utils.get_first_matching_child_prim(
            prim_path, predicate=lambda x: prim_utils.get_prim_type_name(x) == "Plane"
        )
        if collision_prim is None:
            collision_prim = prim_utils.get_first_matching_child_prim(
                prim_path, predicate=lambda x: prim_utils.get_prim_type_name(x) == "Mesh"
            )
        if collision_prim is None:
            # Newer Isaac Sim (5.x) default ground-plane USD assets are purely
            # visual and ship without any collision geometry. In that case we
            # create an (invisible) infinite collision plane ourselves so that
            # objects actually collide with the ground instead of falling through.
            stage_for_plane = get_current_stage()
            collision_prim_path = f"{prim_path}/CollisionPlane"
            collision_plane = UsdGeom.Plane.Define(stage_for_plane, Sdf.Path(collision_prim_path))
            # orient the plane so its normal points up (+Z) and make it large
            collision_plane.CreateAxisAttr().Set("Z")
            collision_plane.CreateWidthAttr().Set(cfg.size[0])
            collision_plane.CreateLengthAttr().Set(cfg.size[1])
            # keep the helper plane visible so it also acts as the visual ground
            UsdGeom.Imageable(collision_plane).MakeVisible()
            # give it a displayColor so it is not pitch black in the viewport
            if cfg.color is not None:
                collision_plane.CreateDisplayColorAttr().Set([Gf.Vec3f(*cfg.color)])
            else:
                collision_plane.CreateDisplayColorAttr().Set([Gf.Vec3f(0.5, 0.5, 0.5)])
            # enable collisions on it
            UsdPhysics.CollisionAPI.Apply(collision_plane.GetPrim())
            collision_prim = collision_plane.GetPrim()
            logger.warning(
                f"No 'Plane'/'Mesh' child prim found under '{prim_path}'. Created a"
                f" fallback visible collision plane at '{collision_prim_path}'."
            )

        collision_prim_path = prim_utils.get_prim_path(collision_prim)
        bind_physics_material(collision_prim_path, f"{prim_path}/physicsMaterial")

    # Scale only the mesh
    # Warning: This is specific to the default grid plane asset.
    if prim_utils.is_prim_path_valid(f"{prim_path}/Environment"):
        # compute scale from size
        scale = (cfg.size[0] / 100.0, cfg.size[1] / 100.0, 1.0)
        # apply scale to the mesh
        prim_utils.set_prim_property(f"{prim_path}/Environment", "xformOp:scale", scale)

    # Change the color of the plane
    # Warning: This is specific to the default grid plane asset.
    if cfg.color is not None:
        # avoiding this step if stage is in memory since the "ChangePropertyCommand" kit command
        # is not supported in stage in memory
        if is_current_stage_in_memory():
            logger.warning(
                "Ground plane color modification is not supported while the stage is in memory. Skipping operation."
            )

        else:
            shader_prim_path = f"{prim_path}/Looks/theGrid/Shader"
            prop_path = f"{shader_prim_path}.inputs:diffuse_tint"

            # Newer Isaac Sim ground plane USD assets may not contain the
            # "Looks/theGrid/Shader" prim. In that case ``ChangePropertyCommand``
            # receives an invalid property reference and raises a RuntimeError,
            # so we skip the color override when the shader prim is absent.
            if prim_utils.is_prim_path_valid(shader_prim_path):
                # change the color
                omni.kit.commands.execute(
                    "ChangePropertyCommand",
                    prop_path=Sdf.Path(prop_path),
                    value=Gf.Vec3f(*cfg.color),
                    prev=None,
                    type_to_create_if_not_exist=Sdf.ValueTypeNames.Color3f,
                )
            else:
                logger.warning(
                    f"Could not find shader prim at '{shader_prim_path}' to set the ground"
                    " plane color. Skipping color override."
                )
    # Remove the light from the ground plane
    # It isn't bright enough and messes up with the user's lighting settings
    stage = get_current_stage()
    sphere_light_path = f"{prim_path}/SphereLight"
    if prim_utils.is_prim_path_valid(sphere_light_path):
        omni.kit.commands.execute(
            "ToggleVisibilitySelectedPrims", selected_paths=[sphere_light_path], stage=stage
        )

    prim = prim_utils.get_prim_at_path(prim_path)
    # Apply semantic tags
    if hasattr(cfg, "semantic_tags") and cfg.semantic_tags is not None:
        # note: taken from replicator scripts.utils.utils.py
        for semantic_type, semantic_value in cfg.semantic_tags:
            # deal with spaces by replacing them with underscores
            semantic_type_sanitized = semantic_type.replace(" ", "_")
            semantic_value_sanitized = semantic_value.replace(" ", "_")
            # set the semantic API for the instance
            instance_name = f"{semantic_type_sanitized}_{semantic_value_sanitized}"
            sem = Semantics.SemanticsAPI.Apply(prim, instance_name)
            # create semantic type and data attributes
            sem.CreateSemanticTypeAttr().Set(semantic_type)
            sem.CreateSemanticDataAttr().Set(semantic_value)

    # Apply visibility
    prim_utils.set_prim_visibility(prim, cfg.visible)

    # return the prim
    return prim


"""
Helper functions.
"""


def _spawn_from_usd_file(
    prim_path: str,
    usd_path: str,
    cfg: from_files_cfg.FileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn an asset from a USD file and override the settings with the given config.

    In case a prim already exists at the given prim path, then the function does not create a new prim
    or throw an error that the prim already exists. Instead, it just takes the existing prim and overrides
    the settings with the given config.

    Args:
        prim_path: The prim path or pattern to spawn the asset at. If the prim path is a regex pattern,
            then the asset is spawned at all the matching prim paths.
        usd_path: The path to the USD file to spawn the asset from.
        cfg: The configuration instance.
        translation: The translation to apply to the prim w.r.t. its parent prim. Defaults to None, in which
            case the translation specified in the generated USD file is used.
        orientation: The orientation in (w, x, y, z) to apply to the prim w.r.t. its parent prim. Defaults to None,
            in which case the orientation specified in the generated USD file is used.
        **kwargs: Additional keyword arguments, like ``clone_in_fabric``.

    Returns:
        The prim of the spawned asset.

    Raises:
        FileNotFoundError: If the USD file does not exist at the given path.
    """
    # check if usd path exists with periodic logging until timeout
    if not check_usd_path_with_timeout(usd_path):
        if "4.5" in usd_path:
            usd_5_0_path = usd_path.replace("http", "https").replace("/4.5", "/5.0")
            if not check_usd_path_with_timeout(usd_5_0_path):
                raise FileNotFoundError(f"USD file not found at path at either: '{usd_path}' or '{usd_5_0_path}'.")
            usd_path = usd_5_0_path
        else:
            raise FileNotFoundError(f"USD file not found at path at: '{usd_path}'.")

    # spawn asset if it doesn't exist.
    if not prim_utils.is_prim_path_valid(prim_path):
        # add prim as reference to stage
        prim_utils.create_prim(
            prim_path,
            usd_path=usd_path,
            translation=translation,
            orientation=orientation,
            scale=cfg.scale,
        )
    else:
        logger.warning(f"A prim already exists at prim path: '{prim_path}'.")

    # modify variants
    if hasattr(cfg, "variants") and cfg.variants is not None:
        select_usd_variants(prim_path, cfg.variants)

    # modify rigid body properties
    if cfg.rigid_props is not None:
        schemas.modify_rigid_body_properties(prim_path, cfg.rigid_props)
    # modify collision properties
    if cfg.collision_props is not None:
        schemas.modify_collision_properties(prim_path, cfg.collision_props)
    # modify mass properties
    if cfg.mass_props is not None:
        schemas.modify_mass_properties(prim_path, cfg.mass_props)

    # modify articulation root properties
    if cfg.articulation_props is not None:
        schemas.modify_articulation_root_properties(prim_path, cfg.articulation_props)
    # modify tendon properties
    if cfg.fixed_tendons_props is not None:
        schemas.modify_fixed_tendon_properties(prim_path, cfg.fixed_tendons_props)
    if cfg.spatial_tendons_props is not None:
        schemas.modify_spatial_tendon_properties(prim_path, cfg.spatial_tendons_props)
    # define drive API on the joints
    # note: these are only for setting low-level simulation properties. all others should be set or are
    #  and overridden by the articulation/actuator properties.
    if cfg.joint_drive_props is not None:
        schemas.modify_joint_drive_properties(prim_path, cfg.joint_drive_props)

    # modify deformable body properties
    if cfg.deformable_props is not None:
        schemas.modify_deformable_body_properties(prim_path, cfg.deformable_props)

    # apply visual material
    if cfg.visual_material is not None:
        if not cfg.visual_material_path.startswith("/"):
            material_path = f"{prim_path}/{cfg.visual_material_path}"
        else:
            material_path = cfg.visual_material_path
        # create material
        cfg.visual_material.func(material_path, cfg.visual_material)
        # apply material
        bind_visual_material(prim_path, material_path)

    # return the prim
    return prim_utils.get_prim_at_path(prim_path)