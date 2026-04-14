import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:
    print("[警告] 未安装 torch，将使用 numpy 替代")
    torch = None

try:
    import open3d as o3d
except ImportError:
    print("[错误] 请先安装 open3d:  pip install open3d")
    sys.exit(1)

try:
    import cv2
except ImportError:
    cv2 = None
    print("[警告] 未安装 opencv-python，图像读取功能受限")


# K = np.array([
#     [796.44,    0,      640],
#     [   0,    796.44,   640],
#     [   0,       0,        1]
# ], dtype=np.float64)

# DEPTH_SCALE = 1000

# CAMERAS = {
#     'front': np.array([
#         [-0.032170, 0.698593, -0.714796, 1.168809],
#         [0.998000, 0.061387, 0.015079, -0.027872],
#         [0.054413, -0.712881, -0.699170, 0.686328],
#         [0.000000, 0.000000, 0.000000, 1.000000],
#     ]),

#     'side' : np.array([
#         [-1.000000, -0.000000, 0.000000, 0.614220],
#         [0.000000, 0.500000, -0.866025, 0.983940],
#         [0.000000, -0.866025, -0.500000, 0.672690],
#         [0.000000, 0.000000, 0.000000, 1.000000],
#     ]),

#     'right': np.array([
#         [1.000000, -0.000000, 0.000000, 0.614220],
#         [0.000000, -0.500000, 0.866025, -0.983940],
#         [-0.000000, -0.866025, -0.500000, 0.672690],
#         [0.000000, 0.000000, 0.000000, 1.000000],
#     ]),

#     'top': np.array([
#         [0.017695, 0.999271, -0.033840, 0.717000],
#         [0.997591, -0.015374, 0.067638, -0.168000],
#         [0.067069, -0.034955, -0.997136, 0.602000],
#         [0.000000, 0.000000, 0.000000, 1.000000],
#     ]),
# }

# =============================================================================
# 交互式标注 (保持原版)
# =============================================================================

LABEL_COLORS = np.array([
    [0.7, 0.7, 0.7],   # 0: 未标注 (灰)
    [1.0, 0.0, 0.0],   # 1: 红
    [0.0, 1.0, 0.0],   # 2: 绿
    [0.0, 0.0, 1.0],   # 3: 蓝
    [1.0, 1.0, 0.0],   # 4: 黄
    [1.0, 0.0, 1.0],   # 5: 品红
    [0.0, 1.0, 1.0],   # 6: 青
    [1.0, 0.5, 0.0],   # 7: 橙
    [0.5, 0.0, 1.0],   # 8: 紫
    [0.0, 0.5, 0.0],   # 9: 深绿
], dtype=np.float64)

LABEL_NAMES = ['未标注', '红', '绿', '蓝', '黄', '品红', '青', '橙', '紫', '深绿']


class PointCloudLabeler:
    def __init__(self, pcd: o3d.geometry.PointCloud):
        self.pcd = pcd
        self.points = np.asarray(pcd.points).copy()
        self.n_points = len(self.points)

        if pcd.has_colors():
            self.original_colors = np.asarray(pcd.colors).copy()
        else:
            self.original_colors = np.ones((self.n_points, 3)) * 0.7

        self.labels = np.zeros(self.n_points, dtype=np.int64)
        self.current_label = 1
        self.history = []
        self.kdtree = o3d.geometry.KDTreeFlann(pcd)

    def _print_help(self):
        print("""
╔══════════════════════════════════════════════════════╗
║               点云交互式标注工具                       ║
╠══════════════════════════════════════════════════════╣
║  E → 框选标注    N → 点选+邻域    V → 预览           ║
║  1-9 → 切换标签  U → 撤销  R → 重置  S → 保存       ║
║  Q → 退出                                            ║
╚══════════════════════════════════════════════════════╝
""")

    def _update_colors(self):
        colors = self.original_colors.copy()
        for lbl in range(1, int(self.labels.max()) + 1):
            mask = self.labels == lbl
            if mask.any():
                colors[mask] = LABEL_COLORS[lbl % len(LABEL_COLORS)]
        self.pcd.colors = o3d.utility.Vector3dVector(colors)

    def _label_indices(self, indices, label):
        if len(indices) == 0:
            return
        self.history.append((indices.copy(), self.labels[indices].copy()))
        self.labels[indices] = label
        self._update_colors()
        labeled = np.sum(self.labels > 0)
        name = LABEL_NAMES[label] if label < len(LABEL_NAMES) else str(label)
        print(f"  ✓ 标注 {len(indices)} 个点 → label {label} ({name})  "
              f"[已标注 {labeled}/{self.n_points}]")

    def _undo(self):
        if not self.history:
            print("  无可撤销操作")
            return
        indices, old = self.history.pop()
        self.labels[indices] = old
        self._update_colors()
        print(f"  ✓ 撤销 {len(indices)} 个点")

    def _reset(self):
        self.history.clear()
        self.labels[:] = 0
        self._update_colors()
        print("  ✓ 所有标注已清除")

    def _print_status(self):
        labeled = np.sum(self.labels > 0)
        name = LABEL_NAMES[self.current_label] if self.current_label < len(LABEL_NAMES) else str(self.current_label)
        print(f"\n── label={self.current_label}({name}) | 已标注 {labeled}/{self.n_points} ──")

    def _edit_crop_mode(self):
        print(f"\n  [框选模式] label = {self.current_label}")
        print("  → K → 画框 → C 确认 → 关闭窗口")

        self._update_colors()
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window(window_name=f"框选 | label={self.current_label}", width=1280, height=720)
        vis.add_geometry(self.pcd)
        vis.run()
        vis.destroy_window()

        cropped = vis.get_cropped_geometry()
        cropped_pts = np.asarray(cropped.points)

        if len(cropped_pts) == 0 or len(cropped_pts) == self.n_points:
            print("  未框选，跳过")
            return

        indices = []
        for pt in cropped_pts:
            [_, idx, _] = self.kdtree.search_knn_vector_3d(pt, 1)
            indices.append(idx[0])
        indices = np.unique(np.array(indices))

        mode = input(f"  框内 {len(indices)} 个点. 标注 [I]框内/[O]框外? (默认 I): ").strip().upper()
        if mode == 'O':
            indices = np.setdiff1d(np.arange(self.n_points), indices)
        self._label_indices(indices, self.current_label)

    def _vertex_pick_mode(self):
        radius_s = input(f"  邻域半径 (默认 0.03): ").strip()
        radius = float(radius_s) if radius_s else 0.03

        self._update_colors()
        vis = o3d.visualization.VisualizerWithVertexSelection()
        vis.create_window(window_name=f"点选 | Shift+点击", width=1280, height=720)
        vis.add_geometry(self.pcd)
        vis.run()
        vis.destroy_window()

        picked = vis.get_picked_points()
        if not picked:
            print("  未选点，跳过")
            return

        all_indices = set()
        for p in picked:
            [k, nbr_idx, _] = self.kdtree.search_radius_vector_3d(self.points[p.index], radius)
            all_indices.update(nbr_idx)
        self._label_indices(np.array(list(all_indices), dtype=np.int64), self.current_label)

    def _preview(self):
        self._update_colors()
        unique, counts = np.unique(self.labels, return_counts=True)
        print("  当前标注分布:")
        for u, c in zip(unique, counts):
            name = LABEL_NAMES[u] if u < len(LABEL_NAMES) else str(u)
            print(f"    label {u} ({name}): {c} ({c/self.n_points*100:.1f}%)")
        o3d.visualization.draw_geometries([self.pcd], window_name="预览", width=1280, height=720)

    def get_result(self):
        result = {
            'points': self.points.copy(),
            'colors': self.original_colors.copy(),
            'labels': self.labels.copy(),
        }
        if torch is not None:
            result['points_tensor'] = torch.from_numpy(result['points']).float()
            result['colors_tensor'] = torch.from_numpy(result['colors']).float()
            result['labels_tensor'] = torch.from_numpy(result['labels']).long()
        return result

    def save_result(self, output_path):
        result = self.get_result()
        if output_path.endswith('.pt') and torch is not None:
            torch.save({
                'points': result['points_tensor'],
                'colors': result['colors_tensor'],
                'labels': result['labels_tensor'],
            }, output_path)
        else:
            if not output_path.endswith('.npz'):
                output_path += '.npz'
            np.savez(output_path, points=result['points'], colors=result['colors'], labels=result['labels'])

        unique, counts = np.unique(result['labels'], return_counts=True)
        print(f"\n  ✓ 已保存: {output_path}")
        for u, c in zip(unique, counts):
            name = LABEL_NAMES[u] if u < len(LABEL_NAMES) else str(u)
            print(f"    label {u} ({name}): {c} ({c/self.n_points*100:.1f}%)")
        return output_path

    def run_interactive(self, output_path="labeled_result.pt"):
        self._update_colors()
        print(f"\n[INFO] 点云: {self.n_points} 个点")
        self._print_help()

        while True:
            self._print_status()
            cmd = input("  命令 [E/N/V/1-9/U/R/S/H/Q]: ").strip().upper()

            if cmd == 'Q':
                if np.sum(self.labels > 0) > 0:
                    if input("  保存后退出? (y/n): ").strip().lower() == 'y':
                        self.save_result(output_path)
                break
            elif cmd == 'S':  self.save_result(output_path)
            elif cmd == 'U':  self._undo()
            elif cmd == 'V':  self._preview()
            elif cmd == 'E':  self._edit_crop_mode()
            elif cmd == 'N':  self._vertex_pick_mode()
            elif cmd == 'H':  self._print_help()
            elif cmd == 'R':
                if input("  确认重置? (y/n): ").strip().lower() == 'y':
                    self._reset()
            elif cmd in [str(i) for i in range(10)]:
                self.current_label = int(cmd)
                name = LABEL_NAMES[self.current_label] if self.current_label < len(LABEL_NAMES) else ''
                print(f"  ✓ 切换到 label {self.current_label} ({name})")
            else:
                print("  未知命令，按 H 查看帮助")


class SimpleLabeler:
    def __init__(self, pcd):
        self.pcd = pcd
        self.points = np.asarray(pcd.points).copy()
        self.n_points = len(self.points)
        self.original_colors = (np.asarray(pcd.colors).copy() if pcd.has_colors()
                                else np.ones((self.n_points, 3)) * 0.7)
        self.labels = np.zeros(self.n_points, dtype=np.int64)
        self.history = []
        self.current_label = 1

    def _update_colors(self):
        colors = self.original_colors.copy()
        for lbl in range(1, max(1, int(self.labels.max()) + 1)):
            mask = self.labels == lbl
            if mask.any():
                colors[mask] = LABEL_COLORS[lbl % len(LABEL_COLORS)]
        self.pcd.colors = o3d.utility.Vector3dVector(colors)

    def get_result(self):
        r = {'points': self.points.copy(), 'colors': self.original_colors.copy(),
             'labels': self.labels.copy()}
        if torch:
            r['points_tensor'] = torch.from_numpy(r['points']).float()
            r['colors_tensor'] = torch.from_numpy(r['colors']).float()
            r['labels_tensor'] = torch.from_numpy(r['labels']).long()
        return r

    def run(self, output_path="labeled_result.pt"):
        mins, maxs = self.points.min(0), self.points.max(0)
        print(f"\n  点云范围: X[{mins[0]:.4f},{maxs[0]:.4f}] "
              f"Y[{mins[1]:.4f},{maxs[1]:.4f}] Z[{mins[2]:.4f},{maxs[2]:.4f}]")

        while True:
            print(f"\n  label={self.current_label} | [V]预览 [B]框选 [1-9]标签 [S]保存 [Q]退出")
            cmd = input("  命令: ").strip().upper()
            if cmd == 'Q': break
            elif cmd == 'V':
                self._update_colors()
                o3d.visualization.draw_geometries([self.pcd], width=1280, height=720)
            elif cmd == 'S':
                r = self.get_result()
                if output_path.endswith('.pt') and torch:
                    torch.save({k: v for k, v in r.items() if 'tensor' in k}, output_path)
                else:
                    np.savez(output_path.replace('.pt', '.npz'),
                             **{k: v for k, v in r.items() if 'tensor' not in k})
                print(f"  ✓ 已保存 {output_path}")
            elif cmd in [str(i) for i in range(10)]:
                self.current_label = int(cmd)
            elif cmd == 'B':
                try:
                    print("  输入范围 (回车=不限):")
                    x0 = float(input(f"    x_min[{mins[0]:.4f}]: ") or mins[0])
                    x1 = float(input(f"    x_max[{maxs[0]:.4f}]: ") or maxs[0])
                    y0 = float(input(f"    y_min[{mins[1]:.4f}]: ") or mins[1])
                    y1 = float(input(f"    y_max[{maxs[1]:.4f}]: ") or maxs[1])
                    z0 = float(input(f"    z_min[{mins[2]:.4f}]: ") or mins[2])
                    z1 = float(input(f"    z_max[{maxs[2]:.4f}]: ") or maxs[2])
                    m = ((self.points[:, 0] >= x0) & (self.points[:, 0] <= x1) &
                         (self.points[:, 1] >= y0) & (self.points[:, 1] <= y1) &
                         (self.points[:, 2] >= z0) & (self.points[:, 2] <= z1))
                    idx = np.where(m)[0]
                    if len(idx) > 0:
                        self.history.append((idx.copy(), self.labels[idx].copy()))
                        self.labels[idx] = self.current_label
                        self._update_colors()
                        print(f"  ✓ 标注 {len(idx)} 个点")
                    else:
                        print("  未选中")
                except ValueError:
                    print("  格式错误")


# =============================================================================
# 主程序
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="多视角点云生成 + 物体提取 + 交互式标注")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--load_pcd', type=str, default="/home/lmy/code/pc_dataset/pc/010_bluecup_withhandle.ply", help='加载点云 (.ply/.pcd)')

    parser.add_argument('--output', type=str, default='labeled_result.pt')
    parser.add_argument('--simple', action='store_true')
    parser.add_argument('--save_ply', type=str, default="/home/lmy/code/isaaclab5/pc_dataset_generate/pc/bluecup_handle_push/010_bluecup_withhandle.ply",
                        help='额外保存为 PLY 文件 (方便预览)')

    args = parser.parse_args()

    print(f"[INFO] 加载: {args.load_pcd}")
    pcd = o3d.io.read_point_cloud(args.load_pcd)
    if not pcd.has_colors():
        pcd.paint_uniform_color([0.7, 0.7, 0.7])

    if pcd is None or len(pcd.points) == 0:
        print("[错误] 无有效点云")
        sys.exit(1)

    print(f"[INFO] 点云: {len(pcd.points)} 个点")

    # 额外保存 PLY
    if args.save_ply:
        o3d.io.write_point_cloud(args.save_ply, pcd)
        print(f"[INFO] 已保存 PLY: {args.save_ply}")

    if args.simple:
        labeler = SimpleLabeler(pcd)
        labeler.run(args.output)
    else:
        labeler = PointCloudLabeler(pcd)
        labeler.run_interactive(args.output)

    result = labeler.get_result()
    print("\n" + "=" * 50)
    print("  标注结果:")
    print(f"    points : {result['points'].shape}")
    print(f"    labels : {result['labels'].shape}")
    if torch:
        print(f"    points_tensor: {result['points_tensor'].shape} {result['points_tensor'].dtype}")
        print(f"    labels_tensor: {result['labels_tensor'].shape} {result['labels_tensor'].dtype}")
    unique, counts = np.unique(result['labels'], return_counts=True)
    for u, c in zip(unique, counts):
        name = LABEL_NAMES[u] if u < len(LABEL_NAMES) else str(u)
        print(f"    label {u} ({name}): {c} ({c/len(result['labels'])*100:.1f}%)")
    print("=" * 50)


if __name__ == '__main__':
    main()

