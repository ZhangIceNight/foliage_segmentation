import open3d as o3d
import numpy as np
import os
import argparse
from tqdm import tqdm


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def farthest_point_sample(point, npoint=6):
    """
    Input:
        point: 点云数据, [N, D]
        npoint: 采样点数量
    Return:
        centroids: 采样后的点云索引
    """
    N, D = point.shape
    if N < npoint:
        idxes = np.hstack((np.tile(np.arange(N), npoint//N), np.random.randint(N, size=npoint%N)))
        return point[idxes, :]

    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    
    return point[centroids.astype(np.int32)]


def knn_patch(points, labels, patch_size=2048):
    """
    Input:
        points: 点云坐标 [N, 3]
        labels: 点云标签 [N]
        patch_size: 每个patch的点数
    Return:
        patches: [num_patches, patch_size, 3]
        patch_labels: [num_patches, patch_size]
    """
    # 归一化点云
    points = pc_normalize(points)
    
    # 创建KD树
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    
    # FPS采样获取中心点
    fps_points = farthest_point_sample(points)
    
    patches = []
    patch_labels = []
    
    for center in fps_points:
        # 获取K近邻点
        [_, idx, _] = kdtree.search_knn_vector_3d(center, patch_size)
        patches.append(points[idx])
        patch_labels.append(labels[idx])
        
    return np.array(patches), np.array(patch_labels)


def process_file(input_file, output_file, patch_size=2048):
    """处理单个文件"""
    # 读取npy文件
    data = np.load(input_file)
    points = data[:, :3]
    labels = data[:, -1].astype(np.int32)
    
    # 获取patches
    patches, patch_labels = knn_patch(points, labels, patch_size)
    
    # 合并坐标和标签
    output_data = []
    for patch, patch_label in zip(patches, patch_labels):
        combined = np.column_stack((patch, patch_label))
        output_data.append(combined)
    
    # 保存结果
    np.save(output_file, np.array(output_data))


def main(args):
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 获取所有npy文件
    files = [f for f in os.listdir(args.input_dir) if f.endswith('.npy')]
    print(f"找到 {len(files)} 个npy文件")
    
    # 处理所有文件
    for file in tqdm(files, desc="处理进度"):
        input_path = os.path.join(args.input_dir, file)
        output_path = os.path.join(args.output_dir, f"patch_{file}")
        process_file(input_path, output_path, args.patch_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='input', help='输入npy文件目录')
    parser.add_argument('--output_dir', type=str, default='output', help='输出patch文件目录')
    parser.add_argument('--patch_size', type=int, default=2048, help='每个patch的点数')
    args = parser.parse_args()
    
    main(args)
