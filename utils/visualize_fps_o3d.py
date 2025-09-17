import open3d as o3d
import numpy as np

def visualize_fps_o3d(all_points, sampled_points, save_path="fps_vis.ply"):
    # 转换为 PointCloud
    pcd_all = o3d.geometry.PointCloud()
    pcd_all.points = o3d.utility.Vector3dVector(all_points)
    pcd_all.paint_uniform_color([0.7, 0.7, 0.7])  # 浅灰色

    pcd_sample = o3d.geometry.PointCloud()
    pcd_sample.points = o3d.utility.Vector3dVector(sampled_points)
    pcd_sample.paint_uniform_color([1, 0, 0])  # 红色

    # 保存到文件
    o3d.io.write_point_cloud(save_path, pcd_all + pcd_sample)
    print(f"可视化点云已保存到: {save_path}")
