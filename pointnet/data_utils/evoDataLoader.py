import os
import numpy as np
import torch
from torch.utils.data import Dataset


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

class EvoDataset(Dataset):
    def __init__(self, data_root, points_per_sample=4096, split='train'):
        self.points_per_sample = points_per_sample  # 与Group处理的点数匹配
        self.split = split
        
        # 加载数据
        self.scenes = []
        data_dir = os.path.join(data_root, split)
        for file in os.listdir(data_dir):
            if file.endswith('.npy'):
                scene = np.load(os.path.join(data_dir, file))
                # 将场景分成多个4096点的块
                n_blocks = len(scene) // self.points_per_sample
                blocks = []
                for i in range(n_blocks):
                    start_idx = i * self.points_per_sample
                    end_idx = start_idx + self.points_per_sample
                    blocks.append(scene[start_idx:end_idx])
                self.scenes.extend(blocks)
                
    def __len__(self):
        return len(self.scenes)  # 返回总block数
        
    def __getitem__(self, idx):
        points = self.scenes[idx]  # (4096, 4) - xyz和label
        
        points = points[:, :3]  # XYZ坐标
        labels = points[:, -1].astype(np.int32)  # 标签
        # 归一化点云
        points = pc_normalize(points)

        pts = torch.FloatTensor(points).transpose(0, 1)  # (3, 4096)
        label = torch.LongTensor(labels)  # (4096,)
        return pts, label