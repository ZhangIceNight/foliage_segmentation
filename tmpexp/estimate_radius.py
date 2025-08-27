import torch
from preprocess import _load_points
def estimate_radius(points, K=32, sample_size=1024, multiplier=2.0):
    """
    自动估计半径，用于局部密度计算
    points: [N, 3] torch.Tensor
    K: int, 目标邻居数
    sample_size: int, 随机采样点数（避免全量计算太慢）
    multiplier: float, 放大系数，保证邻居数 >= K
    return: float, 建议的半径 r
    """
    N = points.shape[0]
    # 采样一部分点
    if N > sample_size:
        idx = torch.randperm(N)[:sample_size]
        sample_pts = points[idx]
    else:
        sample_pts = points

    # 计算 pairwise distance
    dist = torch.cdist(sample_pts, points)  # [M, N]
    dist_sorted, _ = torch.sort(dist, dim=1)  # [M, N]

    # 取每个点到第 K 个邻居的距离
    kth_dist = dist_sorted[:, K]  # [M]

    # 取平均值作为基准
    avg_dist = kth_dist.mean()

    # 半径 = KNN 距离 * 系数
    r = avg_dist.item() * multiplier
    return r

if __name__ == "__main__":
    pcd_path = '/public/wjzhang/datasets/Chinese_wood/Birch/reference_pc_White_Birch.npy'
    points, _, _ = _load_points(pcd_path)  # [N, 3] torch.Tensor
    K = 32
    r = estimate_radius(points, K=K, sample_size=2000, multiplier=2.0)
    print("Estimated radius:", r)
