import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from scipy.spatial import cKDTree

def farthest_point_sampling(points, m=128):
    """
    points: (N, 3)
    m: 采样点数
    return: (m, 3) 采样点
    """
    N = points.shape[0]
    sampled_idx = np.zeros(m, dtype=np.int64)
    distances = np.full(N, np.inf)

    # 随机选一个初始点
    sampled_idx[0] = np.random.randint(0, N)
    farthest_point = points[sampled_idx[0]]

    for i in range(1, m):
        dist = np.linalg.norm(points - farthest_point, axis=1)
        distances = np.minimum(distances, dist)
        sampled_idx[i] = np.argmax(distances)
        farthest_point = points[sampled_idx[i]]

    return points[sampled_idx]

def compute_volume_distance(points, radius=0.1):
    """
    points: (N, 3)
    radius: 搜索半径
    return: (N,) 每个点的 volume distance
    """
    tree = cKDTree(points)
    counts = np.array([len(tree.query_ball_point(p, r=radius)) for p in points])
    volume = (4.0 / 3.0) * np.pi * (radius ** 3)
    vd = (volume / np.clip(counts, 1, None)) ** (1/3)
    return vd

def compute_chamfer_distance(points, radius=0.1):
    """
    points: (N, 3)
    radius: 搜索半径
    return: (N,) 每个点的 chamfer distance
    """
    tree = cKDTree(points)
    chamfer = []
    for i, p in enumerate(points):
        i+dx = tree.query_ball_point(p, r=radius)
        idx = [j for j in idx if j != i]  # 去掉自己
        if len(idx) == 0:
            chamfer.append(0.0)
        else:
            dists = np.linalg.norm(points[idx] - p, axis=1)
            chamfer.append(dists.min())
    cd = np.array(chamfer)
    return cd

def analyze_pointcloud(points, file_path, save_dir, m=128, radius=0.1):
    points = points
    print(f"total points: {points.shape[0]}")
    sampled_points = farthest_point_sampling(points, m=m)
    
    # 计算两个距离
    vd = compute_volume_distance(sampled_points, radius=radius)
    cd = compute_chamfer_distance(sampled_points, radius=radius)

    stats = {}
    for name, arr in [("volume", vd), ("chamfer", cd)]:
        stats[name] = {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr))
        }

        # 可视化分布
        plt.figure(figsize=(10,4))

        plt.subplot(1,2,1)
        sns.histplot(arr, bins=50, kde=True, color="blue")
        plt.title(f"{name} histogram")

        plt.subplot(1,2,2)
        sns.boxplot(x=arr, color="orange")
        plt.title(f"{name} boxplot")

        fname = os.path.splitext(os.path.basename(file_path))[0]
        save_path = os.path.join(save_dir, f"{fname}_{name}.png")
        plt.savefig(save_path, dpi=200)
        plt.close()
    # ---- 可视化 FPS 采样点 vs 全部点 ----
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points[:,0], points[:,1], points[:,2], s=1, c="lightgray", alpha=0.5)
    ax.scatter(sampled_points[:,0], sampled_points[:,1], sampled_points[:,2], s=20, c="red", alpha=0.9)
    ax.set_title("FPS sampling vs full cloud")

    fname = os.path.splitext(os.path.basename(file_path))[0]
    save_path = os.path.join(save_dir, f"{fname}_fps.png")
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Visualization saved to {save_path}")
    print(f"Stats: {stats}")
    return stats

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

def avg_distance_chamfer(points: np.ndarray) -> float:
    """Chamfer 最近邻平均距离（NumPy 版本，全局参考半径）"""
    if points.shape[0] < 2:
        return 0.0
    
    # pairwise distance
    diff = points[:, None, :] - points[None, :, :]   # [N, N, 3]
    dist_matrix = np.linalg.norm(diff, axis=-1)      # [N, N]
    
    # 自己到自己距离设为无穷大
    np.fill_diagonal(dist_matrix, np.inf)
    
    # 每个点的最近邻
    min_dist = np.min(dist_matrix, axis=1)           # [N]
    
    return float(np.mean(min_dist))

def analyze_directory(dir_path, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    all_stats = {}

    volume_all, chamfer_all = [], []
    print(f"Analyzing directory: {dir_path}")
    for file in tqdm(os.listdir(dir_path)):
        if not file.endswith(".npz"):
            continue
        fpath = os.path.join(dir_path, file)
        print(f"Processing file: {fpath}")
        pcd = np.load(fpath)["xyz"].astype(np.float32)
        pcd = normalize(pcd)
        avg_dc = avg_distance_chamfer(pcd)
        avg_dv = avg_distance_volume(pcd)
        print(f"  Avg Chamfer Distance: {avg_dc:.6f}, Avg Volume Distance: {avg_dv:.6f}")
        stats = analyze_pointcloud(pcd, fpath, save_dir, m=128, radius=avg_dv*2)
        all_stats[file] = stats

        volume_all.extend(np.load(fpath)["xyz"].astype(np.float32).shape[0] * [stats["volume"]["mean"]])
        chamfer_all.extend(np.load(fpath)["xyz"].astype(np.float32).shape[0] * [stats["chamfer"]["mean"]])

    # 全局统计
    global_stats = {
        "volume": {
            "mean": float(np.mean(volume_all)),
            "median": float(np.median(volume_all)),
            "std": float(np.std(volume_all)),
            "min": float(np.min(volume_all)),
            "max": float(np.max(volume_all))
        },
        "chamfer": {
            "mean": float(np.mean(chamfer_all)),
            "median": float(np.median(chamfer_all)),
            "std": float(np.std(chamfer_all)),
            "min": float(np.min(chamfer_all)),
            "max": float(np.max(chamfer_all))
        }
    }

    return all_stats, global_stats


if __name__ == "__main__":
    stats, global_stats = analyze_directory(
        dir_path="./data/ForestSemantic_Difficult/test/",
        save_dir="./local_density_results_vis"
        )

    print("全局统计：", global_stats)
