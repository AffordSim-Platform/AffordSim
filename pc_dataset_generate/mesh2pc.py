"""
OBJ Mesh to Point Cloud Converter
Converts an OBJ mesh file to a point cloud with 5000-10000 points.
Uses Open3D for mesh loading, surface sampling, and downsampling.

Usage:
    python obj_to_pointcloud.py input.obj output.ply --num_points 8000
"""

import argparse
import numpy as np

try:
    import open3d as o3d
except ImportError:
    print("Please install open3d: pip install open3d")
    exit(1)

try:
    import trimesh
except ImportError:
    print("Please install trimesh: pip install trimesh")
    exit(1)


def mesh_to_pointcloud(
    input_path: str,
    output_path: str,
    num_points: int = 8000,
    visualize: bool = False,
):
    """
    Convert an OBJ mesh to a downsampled point cloud.

    Args:
        input_path:  Path to .obj file
        output_path: Path to save point cloud (.ply, .pcd, .xyz, etc.)
        num_points:  Target number of points (5000-10000 recommended)
        visualize:   Whether to show the result in a 3D viewer
    """
    # 1. Load mesh via trimesh (handles quad faces by auto-triangulating)
    tm = trimesh.load(input_path, force="mesh")
    if tm is None or len(tm.vertices) == 0:
        raise ValueError(f"Failed to load mesh from {input_path}")
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(tm.vertices)
    mesh.triangles = o3d.utility.Vector3iVector(tm.faces)

    mesh.compute_vertex_normals()
    print(f"Mesh loaded: {len(mesh.vertices)} vertices, {len(mesh.triangles)} faces")

    # 2. Sample points on mesh surface (oversample then downsample for uniformity)
    oversample_factor = 3
    pcd = mesh.sample_points_uniformly(number_of_points=num_points * oversample_factor)
    print(f"Sampled {len(pcd.points)} points from surface")

    # 3. Downsample using farthest point sampling for even distribution
    #    (falls back to voxel downsampling if FPS is unavailable)
    try:
        pcd = pcd.farthest_point_down_sample(num_points)
    except AttributeError:
        # Older Open3D versions may not have FPS; use voxel downsampling iteratively
        bbox = pcd.get_axis_aligned_bounding_box()
        diag = np.linalg.norm(bbox.get_max_bound() - bbox.get_min_bound())
        voxel_size = diag / 100.0
        while len(pcd.points) > num_points:
            pcd = pcd.voxel_down_sample(voxel_size)
            voxel_size *= 1.1
        # If we went below target, resample
        if len(pcd.points) < num_points:
            pcd = mesh.sample_points_uniformly(number_of_points=num_points)

    print(f"Downsampled to {len(pcd.points)} points")

    # 4. Save
    o3d.io.write_point_cloud(output_path, pcd)
    print(f"Point cloud saved to {output_path}")

    # 5. Optional visualization
    if visualize:
        o3d.visualization.draw_geometries(
            [pcd], window_name="Point Cloud", width=800, height=600
        )

    return pcd


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert OBJ mesh to point cloud")
    parser.add_argument("--input", default="/home/lmy/code/pc_dataset/019_cup_nohandle.obj", help="Input .obj file path")
    parser.add_argument("--output", default="/home/lmy/code/pc_dataset/pc/019_cup_nohandle.ply", help="Output point cloud path (.ply/.pcd/.xyz)")
    parser.add_argument(
        "--num_points",
        type=int,
        default=2048,
        help="Target number of points (default: 2048)",
    )
    parser.add_argument("--visualize", action="store_true", help="Show 3D preview")
    args = parser.parse_args()

    if not 5000 <= args.num_points <= 10000:
        print(f"Warning: {args.num_points} is outside recommended range [5000, 10000]")

    mesh_to_pointcloud(args.input, args.output, args.num_points, args.visualize)