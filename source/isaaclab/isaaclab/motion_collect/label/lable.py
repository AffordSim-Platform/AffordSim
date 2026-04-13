import argparse
import os
import numpy as np
import random
import json
import base64
import cv2
import torch
from pathlib import Path
import glob
from scipy.spatial.transform import Rotation as R
import open3d as o3d  # for visualization debugging

# Isaac Lab imports
from isaaclab.app import AppLauncher
from openai import OpenAI

# 1. Parse arguments
parser = argparse.ArgumentParser(description="Multi-View VLM + Single-View Depth")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--show_pcd", action="store_true", default=False, help="Show point cloud visualization window.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 2. Launch simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.sensors import CameraCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg

# ==========================================
# Global configuration & API
# ==========================================
client = OpenAI(
    api_key="sk-18f3f0aa86044683bd58ec537de79a6f", 
    timeout=120,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

ASSET_PATH = "/home/hk/xuhaofan/Real2Sim2Real-Pipeline-auto_task_aff/models"
ALL_OBJECT_DIR = os.path.join(ASSET_PATH, "all_object")

# Top camera parameters (consistent with SceneCfg)
TOP_CAM_WIDTH = 1280
TOP_CAM_HEIGHT = 720
TOP_INTRINSICS = np.array([
    [927.17, 0.0, 651.32],
    [0.0, 927.37, 349.62],
    [0.0, 0.0, 1.0]
])

# Camera position and orientation (consistent with SceneCfg)
TOP_CAM_POS = np.array([0.68, 0.0, 1.0])  # Camera 1m above the table center
TOP_CAM_ROT_QUAT = [0.707, 0, 0, 0.707]   # Quaternion (w,x,y,z) in convention="opengl"

# Table height
TABLE_HEIGHT = 0.025  # meters

# Object center position (for ROI)
OBJECT_CENTER_XY = [0.68, 0.0]

# ==========================================
# Helper functions: VLM & point cloud measurement
# ==========================================

def encode_image(img_rgb):
    """Encode image to Base64"""
    if img_rgb is None: return ""
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode('.jpg', img_bgr)
    return base64.b64encode(buf).decode('utf-8')

def get_label_multiview(top_img, front_img, left_img):
    """Send three views to the VLM"""
    b64_top = encode_image(top_img)
    b64_front = encode_image(front_img)
    b64_left = encode_image(left_img)

    prompt = (
        "You are given three views of the same object: Top, Front, and Left. "
        "Based on these images, identify the object. "
        "Output format: 'Color Type Shape' (e.g., 'Blue Coffee Can', 'Red Ceramic Mug'). "
        "Focus on the object in the center. Do not output any other text."
    )

    try:
        resp = client.chat.completions.create(
            model="qwen-vl-max",
            messages=[{
                "role": "user", 
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_top}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_front}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_left}"}},
                ]
            }]
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"VLM Error: {e}")
        return "Unknown Object"


def reconstruct_and_measure_top(depth_image, rgb_image, cam_sensor):
    """
    Reconstruct point cloud and measure dimensions using the top camera depth image.

    Args:
        depth_image: Depth image (H, W) in meters
        rgb_image: RGB image (H, W, 3)
        cam_sensor: Isaac Lab camera sensor object, used to get accurate pose

    Returns:
        length, width, height (cm), point_cloud, colors
    """
    h, w = depth_image.shape
    
    # Get intrinsics from camera sensor
    # Isaac Lab camera intrinsic matrix format
    fx = TOP_INTRINSICS[0, 0]
    fy = TOP_INTRINSICS[1, 1]
    cx = TOP_INTRINSICS[0, 2]
    cy = TOP_INTRINSICS[1, 2]
    
    print(f"  [Debug] Image size: {w}x{h}")
    print(f"  [Debug] Intrinsics: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
    print(f"  [Debug] Depth range: {depth_image.min():.3f} - {depth_image.max():.3f} m")
    
    # 1. Filter valid depth values
    # Top camera is ~1m - 0.025m = 0.975m from table
    # Object height typically 0.02m ~ 0.3m, so depth range should be 0.67m ~ 0.96m
    min_depth = 0.5   # Minimum depth (filter noise)
    max_depth = 1.2   # Maximum depth (filter background)
    
    mask = (depth_image > min_depth) & (depth_image < max_depth) & np.isfinite(depth_image)
    
    valid_count = np.sum(mask)
    print(f"  [Debug] Valid depth pixels: {valid_count} / {h*w}")
    
    if valid_count < 100:
        print("  [Warning] Too few valid depth pixels!")
        return 0, 0, 0, None, None
    
    # 2. Pixel coordinates -> Camera frame
    v_map, u_map = np.indices((h, w))
    z_cam = depth_image[mask]
    u = u_map[mask]
    v = v_map[mask]
    
    # Standard pinhole camera model back-projection
    x_cam = (u - cx) * z_cam / fx
    y_cam = (v - cy) * z_cam / fy
    
    # Points in camera frame (OpenGL convention: X-right, Y-up, Z-toward camera back)
    # For a downward-looking camera, adjustment based on convention is needed
    points_cam = np.stack([x_cam, y_cam, z_cam], axis=1)
    
    # 3. Camera coordinates -> World coordinates
    # Using camera pose from configuration
    cam_pos = TOP_CAM_POS
    cam_quat = TOP_CAM_ROT_QUAT  # [w, x, y, z]
    
    # OpenGL convention: camera looks toward -Z direction
    # Quaternion [0.707, 0, 0, 0.707] represents 90 degree rotation around X axis (camera facing down)
    # scipy requires [x, y, z, w] format
    r = R.from_quat([cam_quat[1], cam_quat[2], cam_quat[3], cam_quat[0]])
    R_cam_to_world = r.as_matrix()
    
    print(f"  [Debug] Camera position: {cam_pos}")
    print(f"  [Debug] Camera rotation matrix:\n{R_cam_to_world}")
    
    # Due to OpenGL convention, camera Z axis points toward camera back
    # Depth values are distances along the optical axis (-Z)
    # Therefore z must be negated before transformation
    points_cam_opengl = points_cam.copy()
    points_cam_opengl[:, 2] = -points_cam_opengl[:, 2]  # OpenGL: depth along -Z
    
    # Transform to world coordinates
    points_world = (R_cam_to_world @ points_cam_opengl.T).T + cam_pos
    
    print(f"  [Debug] World points range:")
    print(f"    X: {points_world[:,0].min():.3f} - {points_world[:,0].max():.3f}")
    print(f"    Y: {points_world[:,1].min():.3f} - {points_world[:,1].max():.3f}")
    print(f"    Z: {points_world[:,2].min():.3f} - {points_world[:,2].max():.3f}")
    
    # 4. ROI cropping - only keep object region above the table
    roi_half_size = 0.25  # ROI radius 25cm
    z_threshold = TABLE_HEIGHT + 0.005  # Points 5mm above table height count as object
    
    roi_mask = (
        (points_world[:, 0] > OBJECT_CENTER_XY[0] - roi_half_size) & 
        (points_world[:, 0] < OBJECT_CENTER_XY[0] + roi_half_size) &
        (points_world[:, 1] > OBJECT_CENTER_XY[1] - roi_half_size) & 
        (points_world[:, 1] < OBJECT_CENTER_XY[1] + roi_half_size) &
        (points_world[:, 2] > z_threshold)  # Z > table height
    )
    
    obj_points = points_world[roi_mask]
    print(f"  [Debug] Points in ROI: {len(obj_points)}")
    
    # Extract corresponding colors
    obj_colors = None
    if rgb_image is not None:
        rgb_flat = rgb_image[mask]
        obj_colors = rgb_flat[roi_mask] / 255.0
    
    if len(obj_points) < 50:
        print("  [Warning] Too few points in ROI for measurement!")
        return 0, 0, 0, obj_points if len(obj_points) > 0 else None, obj_colors
    
    # 5. Dimension measurement
    # Height = object highest point - table height
    z_max = np.percentile(obj_points[:, 2], 98)  # 98th percentile to avoid noise
    z_min = np.percentile(obj_points[:, 2], 2)   # 2nd percentile as bottom
    height_m = z_max - TABLE_HEIGHT
    
    # Alternatively use z_max - z_min as height (more robust)
    height_m_alt = z_max - z_min
    print(f"  [Debug] Height calculation: z_max={z_max:.4f}, z_min={z_min:.4f}")
    print(f"  [Debug] Height (from table): {height_m*100:.2f} cm")
    print(f"  [Debug] Height (z_max-z_min): {height_m_alt*100:.2f} cm")
    
    # Use more robust height calculation
    height_m = max(height_m, height_m_alt)
    
    # Length/Width = axis-aligned bounding box along image coordinate axes
    # When top camera looks down:
    #   - Image X axis corresponds to world Y axis (or -Y, depends on camera orientation)
    #   - Image Y axis corresponds to world X axis (or -X)
    #
    # Camera at (0.68, 0, 1) looking down, rot=(0.707, 0, 0, 0.707) rotated 90 degrees around X:
    #   - Image u (horizontal/X) direction -> world Y direction
    #   - Image v (vertical/Y) direction -> world X direction
    #
    # Definitions:
    #   - Width: along image X axis -> world Y axis range
    #   - Length: along image Y axis -> world X axis range
    
    # Directly use world coordinate axis-aligned bounding box
    x_min, x_max = obj_points[:, 0].min(), obj_points[:, 0].max()
    y_min, y_max = obj_points[:, 1].min(), obj_points[:, 1].max()
    
    # World X range -> image Y direction -> length
    # World Y range -> image X direction -> width
    length_m = x_max - x_min  # along image Y axis
    width_m = y_max - y_min   # along image X axis
    
    print(f"  [Debug] AABB: X=[{x_min:.4f}, {x_max:.4f}], Y=[{y_min:.4f}, {y_max:.4f}]")
    print(f"  [Debug] Length (img Y / world X): {length_m:.4f} m")
    print(f"  [Debug] Width (img X / world Y): {width_m:.4f} m")
    
    # Convert to centimeters and return
    return length_m * 100, width_m * 100, height_m * 100, obj_points, obj_colors


def reconstruct_and_measure_top_v2(depth_image, rgb_image, cam_sensor):
    """
    Alternative approach: directly use Isaac Lab camera sensor pose data
    """
    h, w = depth_image.shape
    
    # Try to get data from sensor
    try:
        # Isaac Lab camera may provide world-frame point cloud
        if hasattr(cam_sensor.data, 'output') and 'pointcloud' in cam_sensor.data.output:
            points_world = cam_sensor.data.output['pointcloud'][0].cpu().numpy()
            print(f"  [Debug] Using sensor pointcloud directly, shape: {points_world.shape}")
        else:
            # If no point cloud output available, use manual reconstruction
            return reconstruct_and_measure_top(depth_image, rgb_image, cam_sensor)
    except:
        return reconstruct_and_measure_top(depth_image, rgb_image, cam_sensor)
    
    # Subsequent processing same as above...
    # ROI cropping
    roi_half_size = 0.25
    z_threshold = TABLE_HEIGHT + 0.005
    
    roi_mask = (
        (points_world[:, 0] > OBJECT_CENTER_XY[0] - roi_half_size) & 
        (points_world[:, 0] < OBJECT_CENTER_XY[0] + roi_half_size) &
        (points_world[:, 1] > OBJECT_CENTER_XY[1] - roi_half_size) & 
        (points_world[:, 1] < OBJECT_CENTER_XY[1] + roi_half_size) &
        (points_world[:, 2] > z_threshold)
    )
    
    obj_points = points_world[roi_mask]
    
    if len(obj_points) < 50:
        return 0, 0, 0, None, None
    
    # Dimension measurement
    z_max = np.percentile(obj_points[:, 2], 98)
    height_m = z_max - TABLE_HEIGHT
    
    points_2d = obj_points[:, :2].astype(np.float32)
    rect = cv2.minAreaRect(points_2d)
    (w_box, h_box) = rect[1]
    length_m = max(w_box, h_box)
    width_m = min(w_box, h_box)
    
    return length_m * 100, width_m * 100, height_m * 100, obj_points, None


# ==========================================
# Scene configuration
# ==========================================

CAM_COMMON_PROPS = sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
    intrinsic_matrix=[927.17, 0.0, 651.32, 0.0, 927.37, 349.62, 0.0, 0.0, 1.0],
    height=720, width=1600, focal_length=24.0
)

@configclass
class SceneCfg(InteractiveSceneCfg):
    dome_light = AssetBaseCfg(
        prim_path="/World/dome_light",
        spawn=sim_utils.DomeLightCfg(intensity=1500.0),
    )
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane", 
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.8)),
    )
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/table",
        spawn=sim_utils.UsdFileCfg(usd_path=f"{ASSET_PATH}/table.usd"),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0, 0, 0), rot=(1, 0, 0, 0)),
    )

    # Top camera: RGB + Depth
    # Located 1m above the table center, looking down
    # convention="opengl": X-right, Y-up, Z-toward camera back (camera looks toward -Z)
    top_camera = CameraCfg(
        prim_path="/World/Camera/top_rgb_camera", 
        height=TOP_CAM_HEIGHT,
        width=TOP_CAM_WIDTH,
        data_types=["rgb", "depth"],  
        spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
            focal_length=24.0,
            clipping_range=(0.1, 10.0),
            intrinsic_matrix=[
                927.17, 0.0, 651.32,
                0.0, 927.37, 349.62,
                0.0, 0.0, 1.0
            ],
            height=TOP_CAM_HEIGHT,
            width=TOP_CAM_WIDTH,
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.68, 0.0, 1.0),  # 1m above table center
            rot=(0.707, 0, 0, 0.707),  # 90 degree rotation around X axis, camera facing down
            convention="opengl"
        ),
    )

    # Front camera
    front_camera = CameraCfg(
        prim_path="/World/Camera/front_camera", 
        height=720, width=1600, 
        data_types=["rgb"], 
        spawn=CAM_COMMON_PROPS,
        offset=CameraCfg.OffsetCfg(pos=(1.1, 0, 0.06), rot=(0.707, 0, -0.707, 0), convention="ros"),
    )
    
    # Left camera
    left_camera = CameraCfg(
        prim_path="/World/Camera/left_camera", 
        height=720, width=1600, 
        data_types=["rgb"], 
        spawn=CAM_COMMON_PROPS,
        offset=CameraCfg.OffsetCfg(pos=(0.68, 0.6, 0.06), rot=(0.707, 0.707, 0, 0), convention="ros"),
    )

    pass  # objects are added dynamically below

# Scan all USD files and dynamically add object configs to SceneCfg
_usd_files = sorted(glob.glob(os.path.join(ALL_OBJECT_DIR, "*.usd")))
OBJECT_IDS = []
for _usd in _usd_files:
    _stem = Path(_usd).stem  # e.g. "001"
    _idx = len(OBJECT_IDS) + 1
    _attr_name = f"object_{_idx}"
    _cfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/" + _attr_name,
        init_state=RigidObjectCfg.InitialStateCfg(pos=[10.0 + _idx, 0.0, 0.25], rot=[1, 0, 0, 0]),
        spawn=UsdFileCfg(usd_path=_usd, scale=(1, 1, 1)),
    )
    setattr(SceneCfg, _attr_name, _cfg)
    OBJECT_IDS.append(_stem)

NUM_OBJECTS = len(OBJECT_IDS)
print(f"[INFO] Found {NUM_OBJECTS} USD files: {OBJECT_IDS}")

# Will be populated after scene init with {1: bottom_offset, 2: bottom_offset, ...}
# bottom_offset = distance from object origin to its bottom (i.e. -local_z_min)
OBJECT_BOTTOM_OFFSETS = {}


def compute_object_bottom_offsets():
    """After scene init, compute each object's origin-to-bottom offset via world bbox."""
    from pxr import UsdGeom, Usd
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])

    for i in range(1, NUM_OBJECTS + 1):
        prim_path = f"/World/envs/env_0/object_{i}"
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            print(f"  [WARN] Prim not found: {prim_path}, using default offset=0.1")
            OBJECT_BOTTOM_OFFSETS[i] = 0.1
            continue

        # Get current world position of the object origin (from init_state)
        origin_z = 0.25  # init spawn height
        # Use the XformCache to get the actual world transform
        xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        world_xform = xform_cache.GetLocalToWorldTransform(prim)
        origin_z = world_xform.ExtractTranslation()[2]

        # Compute world bounding box
        world_bbox = bbox_cache.ComputeWorldBound(prim)
        world_range = world_bbox.ComputeAlignedRange()
        world_z_min = world_range.GetMin()[2]
        world_z_max = world_range.GetMax()[2]

        # Distance from object origin to its bottom in world space
        bottom_offset = origin_z - world_z_min
        full_height = world_z_max - world_z_min
        OBJECT_BOTTOM_OFFSETS[i] = bottom_offset
        print(f"  [INFO] object_{i} ({OBJECT_IDS[i-1]}): "
              f"origin_z={origin_z:.4f}, world_z=[{world_z_min:.4f}, {world_z_max:.4f}], "
              f"bottom_offset={bottom_offset:.4f} m, full_height={full_height:.4f} m")


def set_position(scene, target_id):
    target_id = int(target_id)
    bottom_offset = OBJECT_BOTTOM_OFFSETS.get(target_id, 0.1)
    # Place so the bottom of the object is just above the table
    place_z = TABLE_HEIGHT + bottom_offset + 0.005
    center_pos = torch.tensor([0.68, 0.0, place_z], device=scene.device)

    for i in range(1, NUM_OBJECTS + 1):
        name = f"object_{i}"
        if name not in scene.rigid_objects:
            continue
        obj = scene.rigid_objects[name]

        if i == target_id:
            rot = torch.tensor([1, 0, 0, 0], device=scene.device)
            state = torch.cat([center_pos, rot])
            obj.write_root_pose_to_sim(state.unsqueeze(0))
            obj.write_root_velocity_to_sim(torch.zeros((1, 6), device=scene.device))
        else:
            hidden_pos = torch.tensor([-10.0, -10.0 + i, 0.0], device=scene.device)
            rot = torch.tensor([1, 0, 0, 0], dtype=torch.float32, device=scene.device)
            obj.write_root_pose_to_sim(torch.cat([hidden_pos, rot]).unsqueeze(0))
            obj.write_root_velocity_to_sim(torch.zeros((1, 6), device=scene.device))


# ==========================================
# Main logic
# ==========================================

def run_simulator(sim, scene):
    top_cam = scene["top_camera"]
    front_cam = scene["front_camera"]
    left_cam = scene["left_camera"]
    
    sim_dt = sim.get_physics_dt()
    count = 0
    all_data = []

    while simulation_app.is_running():
        cycle = 30
        idx = int(count / cycle) + 1
        if idx > NUM_OBJECTS:
            break

        if count % cycle == 0:
            obj_id = OBJECT_IDS[idx - 1]  # e.g. "001", "019"
            print(f"\n{'='*50}")
            print(f"--- Processing Object {idx}/{NUM_OBJECTS} (ID: {obj_id}) ---")
            print(f"{'='*50}")
            set_position(scene, idx)
            
            # Physics stabilization
            for _ in range(30):  # Increased stabilization steps
                scene.write_data_to_sim()
                sim.step()
                scene.update(sim_dt)
            
            # --- 1. Capture images ---
            rgb_top = top_cam.data.output["rgb"][0].cpu().numpy()[..., :3]
            depth_top = top_cam.data.output["depth"][0].cpu().numpy().squeeze()
            
            rgb_front = front_cam.data.output["rgb"][0].cpu().numpy()[..., :3]
            rgb_left = left_cam.data.output["rgb"][0].cpu().numpy()[..., :3]
            
            # Save debug images
            debug_dir = Path("/home/hk/xuhaofan/Real2Sim2Real-Pipeline-auto_task_aff/models/all_object/debug")
            debug_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(debug_dir / f"obj_{obj_id}_top_rgb.jpg"), cv2.cvtColor(rgb_top, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(debug_dir / f"obj_{obj_id}_front_rgb.jpg"), cv2.cvtColor(rgb_front, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(debug_dir / f"obj_{obj_id}_left_rgb.jpg"), cv2.cvtColor(rgb_left, cv2.COLOR_RGB2BGR))

            # Save depth image visualization
            depth_vis = (depth_top - depth_top.min()) / (depth_top.max() - depth_top.min() + 1e-6)
            depth_vis = (depth_vis * 255).astype(np.uint8)
            cv2.imwrite(str(debug_dir / f"obj_{obj_id}_depth.jpg"), depth_vis)
            
            # --- 2. Point cloud reconstruction ---
            print("  > Measuring dimensions (Top View)...")
            l, w, h, pcd_pts, pcd_col = reconstruct_and_measure_top(depth_top, rgb_top, top_cam)
            print(f"  > Final Dimensions: L={l:.2f}, W={w:.2f}, H={h:.2f} cm")
            
            # --- 3. VLM recognition ---
            print("  > Querying VLM (3 Views)...")
            label = get_label_multiview(rgb_top, rgb_front, rgb_left)
            print(f"  > Label: {label}")
            
            # --- 4. Visualization ---
            if args_cli.show_pcd and pcd_pts is not None and len(pcd_pts) > 0:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pcd_pts.astype(np.float64))
                if pcd_col is not None:
                    pcd.colors = o3d.utility.Vector3dVector(pcd_col.astype(np.float64))
                else:
                    # No color available, use default color
                    pcd.paint_uniform_color([0.5, 0.5, 0.5])
                    
                # Add coordinate frame and table plane as reference
                axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
                
                # Create table reference plane
                table_plane = o3d.geometry.TriangleMesh.create_box(width=0.5, height=0.5, depth=0.001)
                table_plane.translate([OBJECT_CENTER_XY[0]-0.25, OBJECT_CENTER_XY[1]-0.25, TABLE_HEIGHT])
                table_plane.paint_uniform_color([0.8, 0.8, 0.8])
                
                o3d.visualization.draw_geometries(
                    [pcd, axis, table_plane], 
                    window_name=f"Obj {obj_id}: {label} (L={l:.1f}, W={w:.1f}, H={h:.1f} cm)"
                )

            bottom_off = OBJECT_BOTTOM_OFFSETS.get(idx, h / 200.0)
            all_data.append({
                "id": obj_id,
                "label": label,
                "dimensions_cm": {"length": round(l, 2), "width": round(w, 2), "height": round(h, 2)},
                "bottom_offset_m": round(bottom_off, 4)
            })

        scene.write_data_to_sim()
        sim.step()
        count += 1
        scene.update(sim_dt)

    if all_data:
        p = Path("/home/hk/xuhaofan/Real2Sim2Real-Pipeline-auto_task_aff/models/all_object/label.json")
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, 'w') as f:
            json.dump(all_data, f, indent=4, ensure_ascii=False)
        print(f"\nData saved to {p}")


def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01, device=args_cli.device))
    sim.set_camera_view([2.5, 2.5, 2.5], [0, 0, 0])
    scene = InteractiveScene(SceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0))
    sim.reset()

    print("\n[INFO] Computing object bounding boxes...")
    compute_object_bottom_offsets()
    print(f"[INFO] Done. {len(OBJECT_BOTTOM_OFFSETS)} objects measured.\n")

    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()