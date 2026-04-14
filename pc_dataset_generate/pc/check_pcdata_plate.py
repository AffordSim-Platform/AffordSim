import os

import numpy as np
import open3d as o3d
from unit_threshold_helper import DEFAULT_UNIT_THRESHOLD, attach_visualization, run_threshold_debug_loop

try:
    import torch
except ImportError:
    torch = None

from check_pcdata import (
    EE_OFFSET_Z,
    GRIPPER_FINGER_DEPTH,
    GRIPPER_FINGER_THICKNESS,
    GRIPPER_FINGER_WIDTH,
    GRIPPER_MAX_OPENING,
    TILT_ANGLE_DEG,
    cluster_affordance_points,
    create_grasp_visualization,
    generate_approach_directions,
    sample_grasp_centers,
)


PT_CANDIDATE_NAMES = ("labeled_result.pt", "label_result.pt", "lable_result.pt")
NPZ_CANDIDATE_NAMES = ("labeled_result.npz", "label_result.npz", "lable_result.npz")

# ============================================================================
# 输入输出配置
# ============================================================================
INPUT_DIR = "yellow_bowl_pick"
OUTPUT_FILE = "grasp_result.npz"
DEFAULT_UNIT_THRESHOLD_VALUE = DEFAULT_UNIT_THRESHOLD


def ensure_parent_dir(path: str):
    parent_dir = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent_dir, exist_ok=True)


def _to_numpy(array_like):
    if hasattr(array_like, "cpu"):
        array_like = array_like.cpu()
    if hasattr(array_like, "numpy"):
        return array_like.numpy()
    return np.asarray(array_like)


def _load_pt_data(pt_path):
    if torch is None:
        raise ImportError("torch 未安装，无法读取 .pt 点云文件")

    data = torch.load(pt_path, map_location="cpu")
    return _to_numpy(data["points"]), _to_numpy(data["labels"])


def _load_npz_data(npz_path):
    with np.load(npz_path) as data:
        return data["points"], data["labels"]


def _iter_candidate_files(base_dir):
    for name in PT_CANDIDATE_NAMES:
        path = os.path.join(base_dir, name)
        if os.path.isfile(path):
            yield path

    other_pt_files = sorted(
        os.path.join(base_dir, file_name)
        for file_name in os.listdir(base_dir)
        if file_name.endswith(".pt") and file_name not in PT_CANDIDATE_NAMES
    )
    for path in other_pt_files:
        yield path

    for name in NPZ_CANDIDATE_NAMES:
        path = os.path.join(base_dir, name)
        if os.path.isfile(path):
            yield path


def resolve_input_path(script_dir, input_dir=None, input_file=None):
    if input_file is not None:
        input_path = input_file if os.path.isabs(input_file) else os.path.join(script_dir, input_file)
        if not os.path.isfile(input_path):
            raise FileNotFoundError(f"指定输入文件不存在: {input_path}")
        return input_path

    candidate_dirs = []
    if input_dir is not None:
        resolved_dir = input_dir if os.path.isabs(input_dir) else os.path.join(script_dir, input_dir)
        if not os.path.isdir(resolved_dir):
            raise FileNotFoundError(f"指定输入目录不存在: {resolved_dir}")
        candidate_dirs.append(resolved_dir)
    else:
        default_dir = os.path.join(script_dir, "yellow_bowl_pick")
        if os.path.isdir(default_dir):
            candidate_dirs.append(default_dir)

        extra_dirs = sorted(
            os.path.join(script_dir, entry)
            for entry in os.listdir(script_dir)
            if entry.endswith("plate_pick") and os.path.isdir(os.path.join(script_dir, entry))
        )
        for path in extra_dirs:
            if path not in candidate_dirs:
                candidate_dirs.append(path)

    searched_dirs = []
    for candidate_dir in candidate_dirs:
        searched_dirs.append(candidate_dir)
        for path in _iter_candidate_files(candidate_dir):
            return path

    if input_dir is not None:
        raise FileNotFoundError(f"目录下未找到可用点云标注文件: {candidate_dirs[0]}")

    raise FileNotFoundError(
        "未找到可用的 plate 输入数据。"
        f" 已搜索目录: {searched_dirs if searched_dirs else [script_dir]}"
    )


def load_points_and_labels(script_dir, input_dir=None, input_file=None):
    input_path = resolve_input_path(script_dir, input_dir=input_dir, input_file=input_file)
    type_path = os.path.dirname(input_path)

    if input_path.endswith(".pt"):
        points, labels = _load_pt_data(input_path)
    elif input_path.endswith(".npz"):
        points, labels = _load_npz_data(input_path)
    else:
        raise ValueError(f"不支持的输入文件类型: {input_path}")

    return points, labels, input_path, type_path


def compute_radial_direction(aff_points, all_points):
    if len(aff_points) < 3:
        return None

    aff_centroid = aff_points.mean(axis=0)
    obj_centroid = all_points.mean(axis=0)
    direction = aff_centroid - obj_centroid
    direction[2] = 0.0
    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        return None
    return direction / norm


def compute_vertical_grip_axis(approach_dir):
    approach_dir = np.asarray(approach_dir, dtype=np.float64)
    approach_dir /= np.linalg.norm(approach_dir)

    z_up = np.array([0.0, 0.0, 1.0])
    grip_axis = z_up - np.dot(z_up, approach_dir) * approach_dir
    grip_norm = np.linalg.norm(grip_axis)

    if grip_norm < 1e-6:
        fallback = np.array([0.0, 1.0, 0.0])
        grip_axis = fallback - np.dot(fallback, approach_dir) * approach_dir
        grip_axis /= np.linalg.norm(grip_axis)
        return grip_axis

    return grip_axis / grip_norm


def score_vertical_grasp(approach_dir, grip_axis, grasp_center,
                         all_points, labels,
                         gripper_opening=GRIPPER_MAX_OPENING,
                         finger_depth=GRIPPER_FINGER_DEPTH,
                         finger_width=GRIPPER_FINGER_WIDTH,
                         finger_thickness=GRIPPER_FINGER_THICKNESS,
                         radial_dir=None):
    binormal = np.cross(approach_dir, grip_axis)
    bn_norm = np.linalg.norm(binormal)
    if bn_norm < 1e-8:
        return None
    binormal /= bn_norm

    grip_axis = np.cross(binormal, approach_dir)
    grip_axis /= np.linalg.norm(grip_axis)

    R = np.column_stack([binormal, grip_axis, approach_dir])
    all_local = (all_points - grasp_center) @ R
    aff_mask = labels == 1
    non_aff_local = all_local[~aff_mask]
    aff_local = all_local[aff_mask]

    half_open = gripper_opening / 2.0
    half_width = finger_width / 2.0

    def in_box(pts, xr, yr, zr):
        return ((pts[:, 0] >= xr[0]) & (pts[:, 0] <= xr[1]) &
                (pts[:, 1] >= yr[0]) & (pts[:, 1] <= yr[1]) &
                (pts[:, 2] >= zr[0]) & (pts[:, 2] <= zr[1]))

    n_closing = int(in_box(non_aff_local,
                           [-half_width, half_width],
                           [-half_open, half_open],
                           [-finger_depth, 0]).sum())
    if n_closing > 3:
        return None

    n_left = int(in_box(non_aff_local,
                        [-half_width, half_width],
                        [half_open, half_open + finger_thickness],
                        [-finger_depth, 0]).sum())
    n_right = int(in_box(non_aff_local,
                         [-half_width, half_width],
                         [-half_open - finger_thickness, -half_open],
                         [-finger_depth, 0]).sum())
    n_palm = int(in_box(non_aff_local,
                        [-half_width, half_width],
                        [-half_open - finger_thickness, half_open + finger_thickness],
                        [-finger_depth - 0.02, -finger_depth]).sum())
    n_approach = int(in_box(all_local,
                            [-half_width, half_width],
                            [-half_open - finger_thickness, half_open + finger_thickness],
                            [-finger_depth - 0.06, -finger_depth - 0.02]).sum())

    n_finger_col = n_left + n_right
    collision_score = max(0.0, 1.0 - n_finger_col * 0.2 - n_closing * 0.3 - n_palm * 0.15)
    clearance_score = max(0.0, 1.0 - n_approach * 0.05)

    in_grip = in_box(aff_local,
                     [-half_width, half_width],
                     [-half_open, half_open],
                     [-finger_depth, 0])
    n_aff_in_grip = int(in_grip.sum())
    coverage_score = n_aff_in_grip / max(len(aff_local), 1)

    if n_aff_in_grip > 0:
        aff_in = aff_local[in_grip]
        aff_center_y = aff_in[:, 1].mean()
        aff_spread_y = aff_in[:, 1].ptp()
    else:
        aff_center_y = aff_local[:, 1].mean()
        aff_spread_y = aff_local[:, 1].ptp()

    centering = 1.0 - min(abs(aff_center_y) / (half_open + 1e-6), 1.0)
    narrowness = 1.0 - min(aff_spread_y / gripper_opening, 1.0)
    antipodal_score = centering * 0.5 + narrowness * 0.5

    if aff_spread_y > gripper_opening and n_aff_in_grip < 3:
        return None

    target_tilt_rad = np.radians(TILT_ANGLE_DEG)
    target_z = -np.sin(target_tilt_rad)
    tilt_error = abs(approach_dir[2] - target_z)
    tilt_score = max(0.0, 1.0 - tilt_error * 2.0)

    vertical_grip_score = abs(grip_axis[2])
    side_horizontal_score = 1.0 - abs(binormal[2])

    if n_aff_in_grip > 0:
        aff_in = aff_local[in_grip]
        dist_to_wall = half_open - np.abs(aff_in[:, 1])
        contact_score = np.mean(np.clip(1.0 - dist_to_wall / half_open, 0.0, 1.0))
    else:
        contact_score = 0.0

    approach_align_score = 0.5
    if radial_dir is not None:
        approach_horiz = np.array([approach_dir[0], approach_dir[1], 0.0])
        ah_norm = np.linalg.norm(approach_horiz)
        if ah_norm > 1e-6:
            approach_horiz /= ah_norm
            alignment = np.dot(approach_horiz, -radial_dir)
            approach_align_score = (alignment + 1.0) / 2.0
        else:
            approach_align_score = 0.3

    horiz_penalty = 0.0
    approach_elevation = np.degrees(np.arcsin(np.clip(-approach_dir[2], -1, 1)))
    if approach_elevation < 15.0:
        horiz_penalty = 0.5 * (1.0 - approach_elevation / 15.0)

    total = (collision_score * 0.14 +
             clearance_score * 0.06 +
             coverage_score * 0.14 +
             antipodal_score * 0.08 +
             tilt_score * 0.14 +
             contact_score * 0.08 +
             approach_align_score * 0.14 +
             vertical_grip_score * 0.14 +
             side_horizontal_score * 0.08 -
             horiz_penalty * 0.05)
    total = max(0.0, total)

    return {
        "total_score": total,
        "collision_score": collision_score,
        "clearance_score": clearance_score,
        "coverage_score": coverage_score,
        "antipodal_score": antipodal_score,
        "tilt_score": tilt_score,
        "contact_score": contact_score,
        "approach_align_score": approach_align_score,
        "vertical_grip_score": vertical_grip_score,
        "side_horizontal_score": side_horizontal_score,
        "grip_axis": grip_axis,
        "n_collisions": n_finger_col + n_closing + n_palm,
        "n_closing": n_closing,
        "aff_width": aff_spread_y,
        "n_aff_in_grip": n_aff_in_grip,
        "approach_elevation": approach_elevation,
    }


def compute_best_vertical_grasp(points, labels):
    aff_points, global_centroid = cluster_affordance_points(points, labels)
    centers = sample_grasp_centers(aff_points)
    radial_dir = compute_radial_direction(aff_points, points)

    if radial_dir is not None:
        print(f"[INFO] affordance 朝外方向: [{radial_dir[0]:.3f}, {radial_dir[1]:.3f}, {radial_dir[2]:.3f}]")

    results = []

    for center, local_pts, center_name in centers:
        approach_dirs = generate_approach_directions(local_pts, points, center)

        for approach_dir, approach_name in approach_dirs:
            grip_axis = compute_vertical_grip_axis(approach_dir)
            info = score_vertical_grasp(
                approach_dir,
                grip_axis,
                center,
                points,
                labels,
                radial_dir=radial_dir,
            )
            if info is None:
                continue

            info["approach_dir"] = approach_dir
            info["grasp_center"] = center
            info["source"] = f"{center_name}/{approach_name}"
            results.append(info)

    if not results:
        print("[WARN] 所有候选都被淘汰! 使用 fallback (竖着抓)")
        tilt_rad = np.radians(TILT_ANGLE_DEG)
        approach = np.array([np.cos(tilt_rad), 0.0, -np.sin(tilt_rad)])
        approach /= np.linalg.norm(approach)
        grip_axis = compute_vertical_grip_axis(approach)
        binormal = np.cross(approach, grip_axis)
        binormal /= np.linalg.norm(binormal)
        grip_axis = np.cross(binormal, approach)
        grip_axis /= np.linalg.norm(grip_axis)
        return {
            "grasp_center": global_centroid,
            "approach_dir": approach,
            "grip_axis": grip_axis,
            "binormal": binormal,
            "rotation_matrix": np.column_stack([binormal, grip_axis, approach]),
            "score": 0.0,
            "source": "fallback(vertical)",
            "all_candidates": [],
            "aff_points": aff_points,
        }

    results.sort(key=lambda x: x["total_score"], reverse=True)

    print(f"\n{'='*132}")
    print(f"  {'候选':<36} {'总分':>6} {'碰撞':>6} {'通道':>6} {'覆盖':>6} "
          f"{'对心':>6} {'倾角':>6} {'贴壁':>6} {'朝向':>6} {'竖抓':>6} {'侧向':>6} {'碰数':>5} {'闭碰':>5} {'夹aff':>5} {'宽mm':>6}")
    print(f"  {'-'*126}")
    for r in results[:20]:
        print(f"  {r['source']:<36} {r['total_score']:>6.3f} "
              f"{r['collision_score']:>6.3f} "
              f"{r['clearance_score']:>6.3f} "
              f"{r['coverage_score']:>6.3f} "
              f"{r['antipodal_score']:>6.3f} "
              f"{r['tilt_score']:>6.3f} "
              f"{r['contact_score']:>6.3f} "
              f"{r['approach_align_score']:>6.3f} "
              f"{r['vertical_grip_score']:>6.3f} "
              f"{r['side_horizontal_score']:>6.3f} "
              f"{r['n_collisions']:>5} "
              f"{r['n_closing']:>5} "
              f"{r['n_aff_in_grip']:>5} "
              f"{r['aff_width'] * 1000:>6.1f}")
    if len(results) > 20:
        print(f"  ... 还有 {len(results) - 20} 个有效候选")
    print(f"{'='*132}")
    print(f"[INFO] 总计有效候选: {len(results)}")

    best = results[0]
    approach_dir = best["approach_dir"]
    grip_axis = compute_vertical_grip_axis(approach_dir)
    binormal = np.cross(approach_dir, grip_axis)
    binormal /= np.linalg.norm(binormal)
    grip_axis = np.cross(binormal, approach_dir)
    grip_axis /= np.linalg.norm(grip_axis)

    elevation = np.degrees(np.arcsin(np.clip(-approach_dir[2], -1, 1)))
    print(f"[INFO] 最优 approach 与水平面夹角: {elevation:.1f}° (目标: {TILT_ANGLE_DEG}°)")
    print(f"[INFO] grip_axis 竖直分量: {abs(grip_axis[2]):.3f}")

    return {
        "grasp_center": best["grasp_center"],
        "approach_dir": approach_dir,
        "grip_axis": grip_axis,
        "binormal": binormal,
        "rotation_matrix": np.column_stack([binormal, grip_axis, approach_dir]),
        "score": best["total_score"],
        "source": best["source"],
        "all_candidates": results,
        "aff_points": aff_points,
    }


def find_plate_pick_dirs(script_dir):
    return sorted(
        os.path.join(script_dir, entry)
        for entry in os.listdir(script_dir)
        if entry.endswith("plate_pick") and os.path.isdir(os.path.join(script_dir, entry))
    )


def process_type_path(type_path, default_threshold=DEFAULT_UNIT_THRESHOLD_VALUE):
    script_dir_of_type = os.path.dirname(type_path)
    input_dir_name = os.path.basename(type_path)
    raw_points, labels, input_path, type_path = load_points_and_labels(
        script_dir_of_type, input_dir=input_dir_name
    )
    print(f"[INFO] 加载: {input_path}")
    save_path = os.path.join(type_path, OUTPUT_FILE)
    ensure_parent_dir(save_path)

    def preview_fn(points, threshold, converted, coord_range):
        print(f"[INFO] 总点数: {len(points)}, label=0: {(labels == 0).sum()}, label=1: {(labels == 1).sum()}")
        print(f"[INFO] 目标倾斜角: {TILT_ANGLE_DEG}°")
        colors = np.zeros((len(labels), 3))
        colors[labels == 0] = [0.0, 1.0, 0.0]
        colors[labels == 1] = [1.0, 0.0, 0.0]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])
        grasp_result = compute_best_vertical_grasp(points, labels)

        print(f"\n{'='*60}")
        print("  竖着抓结果")
        print(f"  中心点:     [{grasp_result['grasp_center'][0]:.5f}, "
              f"{grasp_result['grasp_center'][1]:.5f}, {grasp_result['grasp_center'][2]:.5f}]")
        print(f"  Approach:   [{grasp_result['approach_dir'][0]:.4f}, "
              f"{grasp_result['approach_dir'][1]:.4f}, {grasp_result['approach_dir'][2]:.4f}]")
        print(f"  Grip axis:  [{grasp_result['grip_axis'][0]:.4f}, "
              f"{grasp_result['grip_axis'][1]:.4f}, {grasp_result['grip_axis'][2]:.4f}]")
        print(f"  得分:       {grasp_result['score']:.4f}")
        print(f"  来源:       {grasp_result['source']}")
        print(f"  grip竖直:   {abs(grasp_result['grip_axis'][2]):.3f}")
        elevation = np.degrees(np.arcsin(np.clip(-grasp_result['approach_dir'][2], -1, 1)))
        print(f"  倾斜角:     {elevation:.1f}° (目标: {TILT_ANGLE_DEG}°)")
        ee_pos = grasp_result["grasp_center"] - grasp_result["approach_dir"] * EE_OFFSET_Z
        print(f"  EE位置:     [{ee_pos[0]:.5f}, {ee_pos[1]:.5f}, {ee_pos[2]:.5f}]")
        print(f"{'='*60}")

        grasp_geometries = create_grasp_visualization(grasp_result)
        all_geometries = [pcd, coord_frame] + grasp_geometries
        return attach_visualization(
            {"grasp_result": grasp_result, "ee_pos": ee_pos},
            all_geometries,
            window_name=f"Plate 竖着抓 ({os.path.basename(type_path)}, 倾斜{TILT_ANGLE_DEG}°)",
            width=1280,
            height=720,
        )

    def accept_fn(payload, threshold, converted, coord_range):
        grasp_result = payload["grasp_result"]
        ee_pos = payload["ee_pos"]
        np.savez(
            save_path,
            grasp_center=grasp_result["grasp_center"],
            approach_dir=grasp_result["approach_dir"],
            grip_axis=grasp_result["grip_axis"],
            binormal=grasp_result["binormal"],
            rotation_matrix=grasp_result["rotation_matrix"],
            score=np.array(grasp_result["score"]),
            ee_position=ee_pos,
            tilt_angle=np.array(TILT_ANGLE_DEG),
            grip_vertical_component=np.array(abs(grasp_result["grip_axis"][2])),
            unit_threshold=np.array(threshold),
            unit_converted=np.array(int(converted)),
        )
        print(f"[INFO] 抓取结果已保存: {save_path}")

    return run_threshold_debug_loop(
        raw_points, os.path.basename(type_path), preview_fn, accept_fn, default_threshold
    )


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    type_path = os.path.join(script_dir, INPUT_DIR)

    if not os.path.isdir(type_path):
        raise FileNotFoundError(f"目标文件夹不存在: {type_path}")

    print(f"[INFO] 处理文件夹: {os.path.basename(type_path)}")
    print("[INFO] 键位: Y/Enter=保存, U/↑=阈值×10, D/↓=阈值÷10, Q/Esc/X=退出")
    process_type_path(type_path)
