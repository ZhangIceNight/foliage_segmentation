import os
import numpy as np
from torch.utils.data import Dataset

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

class LeafDatasetWholeScene(Dataset):
    def __init__(self, root, split='train', block_points=4096):
        self.root = root
        self.block_points = block_points
        self.split = split
        
        # 加载数据文件列表
        self.file_list = [f for f in os.listdir(os.path.join(root, split)) if f.endswith('.txt')]
        self.scene_points_list = []
        self.semantic_labels_list = []
        
        # 加载所有场景数据
        for file in self.file_list:
            data = np.loadtxt(os.path.join(root, split, file))
            points = data[:, :3]  # XYZ
            labels = data[:, -1]  # 标签 (0: 非叶子, 1: 叶子)
            
            points = pc_normalize(points)
            self.scene_points_list.append(points)
            self.semantic_labels_list.append(labels)

    def __len__(self):
        return len(self.scene_points_list)

    def __getitem__(self, index):
        points = self.scene_points_list[index]
        labels = self.semantic_labels_list[index]
        
        # 采样点云块
        point_idxs = np.arange(points.shape[0])
        np.random.shuffle(point_idxs)
        
        block_points = points[point_idxs[:self.block_points]]
        block_labels = labels[point_idxs[:self.block_points]]
        
        # 计算采样权重
        sample_weight = np.ones(shape=(self.block_points,))
        
        return block_points, block_labels, sample_weight, point_idxs[:self.block_points] 