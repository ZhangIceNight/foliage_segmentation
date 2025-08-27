import torch
from preprocess import _load_points
import open3d as o3d
import numpy as np
import torch

def farthest_point_sampling(points, M):
    N = points.shape[0]
    device = points.device
    centroids = torch.zeros(M, dtype=torch.long, device=device)
    distances = torch.ones(N, device=device) * 1e10
    farthest = torch.randint(0, N, (1,), device=device).item()
    for i in range(M):
        centroids[i] = farthest
        centroid = points[farthest].unsqueeze(0)
        dist = torch.sum((points - centroid) ** 2, -1)
        distances = torch.min(distances, dist)
        farthest = torch.argmax(distances).item()
    return points[centroids]

def estimate_radius_kdtree(points, K=32, sample_size=500, multiplier=2.0, method="random"):
    """
    FPS/Random + KDTree
    points: torch.Tensor [N,3]
    K: int, KNN
    sample_size: int
    method: "random" or "fps"
    """
    N = points.shape[0]
    pts_np = points.cpu().numpy()

    # Open3D 点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_np)
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    # 采样
    if N > sample_size:
        if method == "random":
            idx = np.random.choice(N, sample_size, replace=False)
            sample_pts = pts_np[idx]
        elif method == "fps":
            sample_pts = farthest_point_sampling(points, sample_size).cpu().numpy()
        else:
            raise ValueError("method should be 'random' or 'fps'")
    else:
        sample_pts = pts_np

    kth_distances = []
    for pt in sample_pts:
        [_, idx_knn, dist_knn] = kdtree.search_knn_vector_3d(pt, K+1)
        kth_distances.append(np.sqrt(dist_knn[-1]))  # 第 K+1 个点距离（排除自己）

    avg_dist = np.mean(kth_distances)
    return avg_dist * multiplier


if __name__ == "__main__":
    pcd_path = '/public/wjzhang/datasets/Chinese_wood/Birch/reference_pc_White_Birch.npy'
    points, _, _ = _load_points(pcd_path)  # [N, 3] np.array
    points = torch.from_numpy(points).float()  # [N, 3] torch.Tensor
    K = 32
    r = estimate_radius_kdtree(points, K=K, sample_size=2000, multiplier=2.0, method="fps")
    print("Estimated radius:", r)
