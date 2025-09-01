import os
import torch
import numpy as np
import open3d as o3d
from tqdm import tqdm

def get_patch_radius(points, ratio=0.2):
    """
    根据点云尺度自适应 patch 半径
    points: torch.Tensor [N,3]
    ratio: patch 半径占点云最大尺寸比例
    """
    min_xyz = points.min(0)[0]
    max_xyz = points.max(0)[0]
    diag = torch.norm(max_xyz - min_xyz)  # 点云对角线长度
    radius = diag * ratio
    return radius

def extract_patches(points, num_patches=5, ratio=0.2):
    """
    从单棵树点云提取局部 patch
    points: torch.Tensor [N,3]
    ratio: patch 半径占点云最大尺寸比例
    return: list of np.array patches
    """
    N = points.shape[0]
    pts_np = points.cpu().numpy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_np)

    patch_radius = get_patch_radius(points, ratio)
    patches = []
    kdtree = o3d.geometry.KDTreeFlann(pcd)  # 只建一次 KDTree

    for _ in range(num_patches):
        # 随机选择 patch 中心
        center_idx = np.random.randint(N)
        center = pts_np[center_idx]

        # KDTree 搜索邻居
        [_, idxs, _] = kdtree.search_radius_vector_3d(center, patch_radius)
        if len(idxs) == 0:
            continue  # 如果没有点，跳过
        patch = pts_np[idxs]
        patches.append(patch)
    return patches

if __name__ == "__main__":

    # ---------------- 批量处理所有树 ----------------
    input_dir = "./trees_raw"
    output_dir = "./patches"
    os.makedirs(output_dir, exist_ok=True)

    tree_files = [f for f in os.listdir(input_dir) if f.endswith(".npy")]

    for tree_file in tqdm(tree_files, desc="Generating patches"):
        points = torch.from_numpy(np.load(os.path.join(input_dir, tree_file))).float()
        patches = extract_patches(points, num_patches=5, ratio=0.2)
        # 保存每个 patch
        for i, patch in enumerate(patches):
            out_file = os.path.join(output_dir, f"{tree_file[:-4]}_patch{i}.npy")
            np.save(out_file, patch)
