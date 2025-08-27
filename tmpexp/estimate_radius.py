import torch
from preprocess import _load_points
import open3d as o3d
import numpy as np
import torch
from tqdm import tqdm

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


def estimate_radius_kdtree_batch(points, K=32, sample_size=500, multiplier=2.0, method="fps", batch_size=50, show_avg=False):
    """
    分批计算 KNN 半径 + 进度条 + 可选实时平均半径显示
    points: torch.Tensor [N,3]
    K: KNN
    sample_size: 采样点数量
    multiplier: 放大系数
    method: "random" or "fps"
    batch_size: 每批处理多少采样点
    show_avg: bool, 是否每批显示当前平均半径
    """
    N = points.shape[0]
    pts_np = points.cpu().numpy()

    print("Building KDTree...")
    # Open3D 点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_np)
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    print("Sampling points...")
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
    num_batches = int(np.ceil(sample_pts.shape[0] / batch_size))
    
    for i in tqdm(range(num_batches), desc="Estimating radius"):
        batch = sample_pts[i*batch_size : (i+1)*batch_size]
        for pt in batch:
            [_, idx_knn, dist_knn] = kdtree.search_knn_vector_3d(pt, K+1)
            kth_distances.append(np.sqrt(dist_knn[-1]))
        # 可选显示当前平均半径
        if show_avg:
            current_avg = np.mean(kth_distances)
            tqdm.write(f"Current average radius: {current_avg:.5f}")

    avg_dist = np.mean(kth_distances)
    return avg_dist * multiplier



if __name__ == "__main__":
    pcd_path = '/public/wjzhang/datasets/Chinese_wood/Birch/reference_pc_White_Birch.npy'
    print("Loading point cloud...")
    points, _, _ = _load_points(pcd_path)  # [N, 3] np.array
    points = torch.from_numpy(points).float()  # [N, 3] torch.Tensor
    K = 32
    r = estimate_radius_kdtree_batch(points, K=32, sample_size=500, multiplier=2.0, method="fps", batch_size=50, show_avg=True)
    print("Final estimated radius:", r)
