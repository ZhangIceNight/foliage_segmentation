import os
import numpy as np
import torch


def read_point_cloud(file_path):
    """简单读取点云npz，返回xyz坐标"""
    data = np.load(file_path)
    point_cloud = data["xyz"].astype(np.float32)
    return point_cloud

def downsample(points, max_points=4096):
    """随机下采样到 max_points"""
    N = points.shape[0]
    if N <= max_points:
        return points
    idx = np.random.choice(N, max_points, replace=False)
    return points[idx]

def normalize(points):
    """归一化到中心在原点，坐标[-1,1]范围"""
    centroid = points.mean(axis=0)
    points = points - centroid
    scale = np.max(np.linalg.norm(points, axis=1))
    points = points / scale
    return points

def avg_distance_volume(points):
    """体积估算的平均点间距"""
    if points.shape[0] < 4:
        return 0.0
    min_xyz = points.min(axis=0)
    max_xyz = points.max(axis=0)
    volume = np.prod(max_xyz - min_xyz)
    avg_dist = (volume / points.shape[0]) ** (1/3)
    return avg_dist

def avg_distance_chamfer(points):
    """Chamfer 最近邻平均距离（每点1个邻居）"""
    if points.shape[0] < 2:
        return 0.0
    pts = torch.tensor(points, dtype=torch.float32)
    # pairwise distance
    dist_matrix = torch.cdist(pts, pts)
    # 让自己到自己的距离无穷大
    dist_matrix.fill_diagonal_(float('inf'))
    # 最近邻距离
    min_dist, _ = torch.min(dist_matrix, dim=1)
    return min_dist.mean().item()

def process_folder_mean(folder_path, max_points=4096):
    vol_list = []
    chamfer_list = []

    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if not os.path.isfile(file_path):
            continue
        points = read_point_cloud(file_path)
        points = downsample(points, max_points)
        points = normalize(points)

        vol_list.append(avg_distance_volume(points))
        chamfer_list.append(avg_distance_chamfer(points))

    # 计算所有文件的平均值
    avg_vol = np.mean(vol_list) if vol_list else 0.0
    avg_chamfer = np.mean(chamfer_list) if chamfer_list else 0.0
    return avg_vol, avg_chamfer

# 示例
folder = './data/ForestSemantic_Difficult/tiles_filtered_fps'
avg_vol, avg_chamfer = process_folder_mean(folder)
print(f"Average volume-based distance: {avg_vol:.6f}")
print(f"Average chamfer nearest-neighbor distance: {avg_chamfer:.6f}")