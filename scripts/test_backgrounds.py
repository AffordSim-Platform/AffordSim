"""Validate the bundled DA3 views and render them with a local robot task.

Run with the Isaac Sim Python environment. --validate-only needs NumPy,
Pillow, SciPy and PyYAML. All project modules are pinned to this checkout.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for package in ("isaaclab", "isaaclab_assets", "isaaclab_tasks", "isaaclab_rl", "isaaclab_mimic"):
    sys.path.insert(0, str(ROOT / "source" / package))
sys.path.insert(0, str(ROOT / "curobo" / "src"))


def local_modules() -> dict:
    origins = {}
    for name in ("isaaclab", "isaaclab_tasks", "isaaclab_assets", "curobo"):
        spec = importlib.util.find_spec(name)
        if spec is None or spec.origin is None:
            raise RuntimeError(f"Missing local module: {name}")
        path = Path(spec.origin).resolve()
        if not path.is_relative_to(ROOT):
            raise RuntimeError(f"Refusing non-local {name}: {path}")
        origins[name] = str(path)
    return origins


def prepare_views(args, output: Path):
    import numpy as np
    import yaml

    # Load only this pure-Python utility; importing isaaclab_tasks before
    # SimulationApp would trigger simulator-dependent environment registration.
    path = ROOT / "source/isaaclab_tasks/isaaclab_tasks/real2sim/utils/da3_background.py"
    spec = importlib.util.spec_from_file_location("local_da3_background", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    task_path = Path(args.task_config).resolve()
    task = yaml.safe_load(task_path.read_text())
    configs = output / "configs"
    configs.mkdir(parents=True, exist_ok=True)
    records = []
    manifest_path = ROOT / "backgrounds/background_cameras.json"
    manifest = json.loads(manifest_path.read_text())
    cameras = manifest.get("cameras")
    if not isinstance(cameras, dict) or not cameras:
        raise ValueError("background_cameras.json must map image paths to camera calibration")
    image_base = (manifest_path.parent / manifest["image_path_base"]).resolve()
    scene_views = {}
    for image_key, camera in cameras.items():
        scene = camera["scene_id"]
        index = camera["view_index"]
        images = scene_views.setdefault(scene, {})
        if index in images:
            raise ValueError(f"Ambiguous view {index} for scene {scene} in {manifest_path}")
        images[index] = (image_base / image_key).resolve()
    if len(args.views) != len(set(args.views)):
        raise ValueError("--views must not contain duplicate indices")
    for scene in sorted(scene_views):
        for index in args.views:
            if index not in scene_views[scene]:
                raise ValueError(f"View {index} is not listed for scene {scene} in {manifest_path}")
            image_path = scene_views[scene][index]
            calibration = module.load_da3_background_image(image_path, manifest_path, (640, 480))
            config = copy.deepcopy(task)
            config["scene"]["num_envs"] = 1
            config["enable_objects_randomization"] = False
            config["enable_random_lights"] = False
            config.setdefault("texture_randomization", {})["enabled"] = False
            config["image_noise_std"] = 0
            config["image_speckle_noise_std"] = 0
            config["runtime"].update({
                "headless": True,
                "episodes": 1,
                "save_original_images": True,
                "save_composite_images": True,
                "save_mask_images": True,
                "sim_camera_resolution": [640, 480],
                "background_image_size": [640, 480],
                "use_background_replace": True,
                "cam_wrist_background": "",
                "da3_background": {
                    "image_path": str(image_path.relative_to(ROOT)),
                    "keep_table": True,
                },
            })
            stem = f"{scene}_view_{index:02d}"
            config_path = configs / f"{stem}.yaml"
            config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
            # Independently verify projection through inverse camera pose and
            # the exported world-to-camera transform at task object centers.
            from scipy.spatial.transform import Rotation
            quat = calibration["rot"]
            rotation = Rotation.from_quat([*quat[1:], quat[0]]).as_matrix()
            points = np.asarray([obj["init_pos"] for obj in config["objects"].values()])
            from_pose = (points - np.asarray(calibration["pos"])) @ rotation
            extrinsic = np.asarray(calibration["extrinsics_w2c_user"])
            from_matrix = points @ extrinsic[:3, :3].T + extrinsic[:3, 3]
            np.testing.assert_allclose(from_pose, from_matrix, atol=1e-6)
            k = np.asarray(calibration["intrinsic_matrix"]).reshape(3, 3)
            pixels_h = from_matrix @ k.T
            pixels = pixels_h[:, :2] / pixels_h[:, 2:]
            if not (from_matrix[:, 2] > 0).all():
                raise ValueError(f"Task objects are behind the camera: {stem}")
            if not ((pixels >= 0).all() and (pixels < [640, 480]).all()):
                raise ValueError(f"Task objects are outside the selected view: {stem}")
            records.append({
                "scene": scene, "stem": stem,
                "config_path": str(config_path), "calibration": calibration,
                "object_pixels": pixels.tolist(),
            })
    return records


def render_views(args, output, records, report):
    from isaaclab.app import AppLauncher

    # Use the installed simulator's experience, not another AffordSim checkout.
    # A source-only checkout may not yet have the optional apps/ link.
    if not args.experience and not (ROOT / "apps").is_dir():
        sim_spec = importlib.util.find_spec("isaacsim")
        args.experience = str(Path(sim_spec.origin).parent / "apps/isaacsim.exp.base.python.kit")
    app = AppLauncher(args).app
    env = None
    try:
        import gymnasium as gym
        import numpy as np
        from PIL import Image, ImageDraw
        from pxr import PhysxSchema, Usd
        import torch
        import isaaclab_tasks  # registers this checkout's tasks
        from isaaclab_tasks.real2sim.tasks.pick_r2s_env_cfg import (
            create_pick_env_cfg_from_yaml, get_pick_env_cfg_class_by_name,
        )
        from isaaclab_tasks.real2sim.utils.EpisodeBuffer import replace_background_with_segmentation

        report["modules_after_startup"] = local_modules()
        task_name = "FrankaFr3-OurGripper-Joint-VisPick-v0"
        cls = get_pick_env_cfg_class_by_name(task_name)
        cfg = create_pick_env_cfg_from_yaml(
            records[0]["config_path"], cls, num_envs=1, device=args.device, seed=42,
        )
        cfg.scene.external_cam_side.update_latest_camera_pose = True
        env = gym.make(task_name, cfg=cfg).unwrapped
        env.reset()
        robot = env.scene["robot"]
        camera = env.scene["external_cam_side"]
        action = torch.zeros((1, env.action_manager.total_action_dim), device=env.device)
        arm_ids, _ = robot.find_joints(["fr3_joint.*"], preserve_order=True)
        action[:, :7] = robot.data.default_joint_pos[:, arm_ids]
        action[:, -1] = 1.0
        for _ in range(args.steps):
            env.step(action)

        initial = records[0]["calibration"]
        np.testing.assert_allclose(
            camera.data.intrinsic_matrices[0].cpu().numpy(),
            np.asarray(initial["intrinsic_matrix"]).reshape(3, 3), atol=1e-3,
        )
        np.testing.assert_allclose(
            camera.data.pos_w[0].cpu().numpy() - env.scene.env_origins[0].cpu().numpy(),
            initial["pos"], atol=1e-5,
        )
        assert abs(np.dot(camera.data.quat_w_ros[0].cpu().numpy(), initial["rot"])) > 1 - 1e-5

        # Check the authored PhysX attributes, not merely the Python config.
        physics = {}
        for name in cfg.task_config.objects:
            root_prim = env.sim.stage.GetPrimAtPath(f"/World/envs/env_0/{name}")
            values = [
                PhysxSchema.PhysxRigidBodyAPI(prim).GetMaxDepenetrationVelocityAttr().Get()
                for prim in Usd.PrimRange(root_prim) if prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI)
            ]
            if not values or any(value is None or abs(value - 2.0) > 1e-6 for value in values):
                raise AssertionError(f"Incorrect max_depenetration_velocity for {name}: {values}")
            physics[name] = values
        report["max_depenetration_velocity"] = physics
        report["task"] = task_name
        report["settling_control_steps"] = args.steps
        report["physics_steps_during_settling"] = args.steps * cfg.decimation
        contact_sheet = Image.new("RGB", (640 * len(args.views), 510 * (len(records) // len(args.views))), "white")
        draw = ImageDraw.Draw(contact_sheet)
        for number, record in enumerate(records):
            # Exercise every generated YAML through the real config factory.
            view_cfg = create_pick_env_cfg_from_yaml(record["config_path"], cls, num_envs=1, device=args.device)
            calibration = view_cfg.task_config.runtime.da3_background
            expected = record["calibration"]
            assert view_cfg.task_config.runtime.cam_side_background == expected["image_path"]
            assert tuple(view_cfg.scene.external_cam_side.offset.pos) == tuple(expected["pos"])
            assert tuple(view_cfg.scene.external_cam_side.offset.rot) == tuple(expected["rot"])
            matrix = torch.tensor(calibration["intrinsic_matrix"], device=env.device).reshape(1, 3, 3)
            camera.set_intrinsic_matrices(matrix)
            camera.set_world_poses(
                positions=torch.tensor([calibration["pos"]], device=env.device) + env.scene.env_origins,
                orientations=torch.tensor([calibration["rot"]], device=env.device),
                convention="ros",
            )
            for _ in range(8):
                env.sim.render()
            camera.update(env.physics_dt, force_recompute=True)
            rgb = camera.data.output["rgb"][0, ..., :3].cpu().numpy()
            seg = camera.data.output["semantic_segmentation"][0].cpu().numpy()
            background = np.asarray(Image.open(calibration["image_path"]).convert("RGB"))
            composite, mask = replace_background_with_segmentation(rgb, seg, background)
            foreground_fraction = float((mask > 0).mean())
            if not 0.005 < foreground_fraction < 0.85:
                raise AssertionError(f"Unexpected foreground coverage for {record['stem']}: {foreground_fraction}")
            np.testing.assert_allclose(camera.data.intrinsic_matrices[0].cpu().numpy(), matrix[0].cpu().numpy(), atol=1e-3)
            np.testing.assert_allclose(
                camera.data.pos_w[0].cpu().numpy() - env.scene.env_origins[0].cpu().numpy(),
                calibration["pos"], atol=1e-5,
            )
            for suffix, array in [("rgb", rgb), ("mask", mask), ("composite", np.round(composite * 255).astype(np.uint8))]:
                Image.fromarray(array).save(output / f"{record['stem']}_{suffix}.png")
            image = Image.fromarray(np.round(composite * 255).astype(np.uint8))
            x = (number % len(args.views)) * 640
            y = (number // len(args.views)) * 510
            contact_sheet.paste(image, (x, y + 30))
            draw.text((x + 8, y + 8), f"Scene {number // len(args.views) + 1}, view {calibration['view_index']}, {calibration['angle_deg']:g} deg", fill="black")
            record["foreground_fraction"] = foreground_fraction
            record["rendered"] = True
            print(f"PASS {record['stem']}: foreground={foreground_fraction:.3f}", flush=True)
        contact_sheet.save(output / "contact_sheet.png")
        report["rendering_passed"] = True
    except Exception as exc:
        import traceback
        report["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
        raise
    finally:
        # Kit can terminate Python during shutdown, so persist evidence first.
        (output / "render_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
        sys.stdout.flush()
        sys.stderr.flush()
        if "error" in report:
            # Kit's immediate shutdown otherwise replaces a failed assertion's
            # exit status with zero. The OS releases this process's GPU context.
            os._exit(1)
        # Outputs are already written synchronously. Kit's immediate shutdown
        # avoids a hang when closing a stage with tiled render products.
        app.close(skip_cleanup=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--task-config", default=str(ROOT / "source/isaaclab_tasks/isaaclab_tasks/real2sim/tasks/config/task/pick/pick_banana_place_plate.yaml"))
    parser.add_argument("--views", type=int, nargs="+", default=[2, 4, 6, 8])
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--output-dir", default=str(ROOT / "outputs/background_validation"))
    # Keep metadata checks usable without importing the simulator.
    if "--validate-only" not in sys.argv:
        from isaaclab.app import AppLauncher
        AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {"repository": str(ROOT), "modules": local_modules(), "rendering_passed": False}
    try:
        records = prepare_views(args, output)
        report["views"] = records
        report["metadata_passed"] = True
        print(f"Validated {len(records)} calibrated views in {ROOT}", flush=True)
        if not args.validate_only:
            if not args.enable_cameras:
                raise ValueError("Rendering requires --enable_cameras")
            render_views(args, output, records, report)
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        name = "metadata_report.json" if args.validate_only else "render_report.json"
        (output / name).write_text(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
