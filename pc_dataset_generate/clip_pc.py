import open3d as o3d

pcd = o3d.io.read_point_cloud("full_scene.ply")

# 交互式裁剪：按 K 进入裁剪模式，画框选择，按 C 确认
print("按 K 画框，C 确认裁剪，Q 退出")
cropped = o3d.visualization.draw_geometries_with_editing([pcd])

# 或者用这个更直接的方式：
vis = o3d.visualization.VisualizerWithEditing()
vis.create_window()
vis.add_geometry(pcd)
vis.run()  # 手动框选
vis.destroy_window()

cropped = vis.get_cropped_geometry()
o3d.io.write_point_cloud("banana_only.ply", cropped)