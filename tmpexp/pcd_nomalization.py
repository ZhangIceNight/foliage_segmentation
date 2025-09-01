import os
import numpy as np
import torch

def normalize_points(points, method="box"):
    """
    点云归一化
    points: np.array [N,3] 或 torch.Tensor [N,3]
    method: "box" -> [0,1]^3, "sphere" -> 平移到原点 + 最大半径=1
    return: np.array [N,3]
    """
    if isinstance(points, torch.Tensor):
        points_np = points.cpu().numpy()
    else:
        points_np = points

    if method == "box":
        min_xyz = points_np.min(axis=0)
        max_xyz = points_np.max(axis=0)
        points_norm = (points_np - min_xyz) / (max_xyz - min_xyz + 1e-8)
    elif method == "sphere":
        center = points_np.mean(axis=0)
        points_centered = points_np - center
        max_dist = np.linalg.norm(points_centered, axis=1).max()
        points_norm = points_centered / (max_dist + 1e-8)
    else:
        raise ValueError("method 必须是 'box' 或 'sphere'")

    return points_norm

def normalize_pointcloud_path(path, output_dir, method="box"):
    """
    path: 文件或文件夹路径
    output_dir: 输出文件夹
    method: "box" 或 "sphere"
    """
    os.makedirs(output_dir, exist_ok=True)

    if os.path.isfile(path):
        files = [path]
    elif os.path.isdir(path):
        files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".npy")]
    else:
        raise ValueError("path 不是文件也不是文件夹")

    for f in files:
        points = np.load(f)
        points_norm = normalize_points(points, method=method)
        out_file = os.path.join(output_dir, os.path.basename(f))
        np.save(out_file, points_norm)
        print(f"Normalized ({method}) and saved: {out_file}")

if __name__ == "__main__":
    # ---------------- 使用示例 ----------------
    input_path = "./trees_raw"     # 单文件或文件夹
    output_dir = "./trees_normalized"

    # 单位球归一化
    normalize_pointcloud_path(input_path, output_dir, method="sphere")
