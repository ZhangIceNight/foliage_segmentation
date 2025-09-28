import json
import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pytorch_lightning import LightningDataModule
from utils import augmentations

class Scenes_Dataset(Dataset):
    def __init__(self, data_dir, split='train', num_points=1024, file_list=None, use_normalization=False, calculate_avg_dist=False, augmentations_list=[]):
        self.data_dir = data_dir
        self.split = split
        self.num_points = num_points
        self.file_list = file_list
        self.use_normalization = use_normalization
        self.augmentations = augmentations_list
        self.calculate_avg_dist = calculate_avg_dist
 
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.file_list[idx])
        data = np.load(file_path)
        point_cloud = data["xyz"].astype(np.float32)
        label = data["label"].astype(np.int64)

        N = point_cloud.shape[0]
        # 随机采样
        if N >= self.num_points:
            idxs = np.random.choice(N, self.num_points, replace=False)
        else:
            idxs = np.random.choice(N, self.num_points, replace=True)
        
        point_cloud = point_cloud[idxs]
        label = label[idxs]

        # 数据增强（仅对训练集）
        if self.split == 'train':
            if 'rotate' in self.augmentations:
                point_cloud = augmentations.random_rotate_point_cloud_y_axis(point_cloud)
            if 'scale' in self.augmentations:
                point_cloud = augmentations.random_scale_point_cloud(point_cloud)
            if 'shift' in self.augmentations:
                point_cloud = augmentations.random_shift_point_cloud(point_cloud)
            if 'dropout' in self.augmentations:
                point_cloud = augmentations.random_sample_dropout(point_cloud)
 
        # 归一化
        if self.use_normalization:
            point_cloud = augmentations.pc_normalize(point_cloud)
        
        # 计算平均邻居距离（用于 DHMamba 的高斯权重）
        if self.calculate_avg_dist:
            """体积估算的平均点间距"""
            min_xyz = point_cloud.min(axis=0)
            max_xyz = point_cloud.max(axis=0)
            volume = np.prod(max_xyz - min_xyz)
            avg_dist = (volume / point_cloud.shape[0]) ** (1/3)

        if self.split == 'train':
            return torch.as_tensor(point_cloud).float(), torch.as_tensor(label).long(), (avg_dist if self.calculate_avg_dist else 0.0)
        else:
            return torch.as_tensor(point_cloud).float(), torch.as_tensor(label).long(), (avg_dist if self.calculate_avg_dist else 0.0), file_path
class Scenes_DataModule(LightningDataModule):
    def __init__(self, data_dir, split_json_path, num_points=1024, batch_size=32, fold_idx=0, num_workers=4, use_normalization=False, augmentations_list=None, calculate_avg_dist=False, **kwargs):
        super().__init__()
        self.data_dir = data_dir
        self.split_json_path = split_json_path
        self.batch_size = batch_size
        self.num_points = num_points
        self.fold_idx = fold_idx
        self.num_workers = num_workers
        self.use_normalization = use_normalization
        self.augmentations_list = augmentations_list
        self.calculate_avg_dist = calculate_avg_dist

    def setup(self, stage=None):
        with open(self.split_json_path, "r") as f:
            splits = json.load(f)
        train_files = splits[f"fold_{self.fold_idx}"]["train"]
        val_files   = splits[f"fold_{self.fold_idx}"]["val"]
        self.train_ds = Scenes_Dataset(self.data_dir, split='train', num_points=self.num_points, file_list=train_files, use_normalization=self.use_normalization, calculate_avg_dist=self.calculate_avg_dist, augmentations_list=self.augmentations_list)
        self.val_ds = Scenes_Dataset(self.data_dir, split='val', num_points=self.num_points, file_list=val_files, use_normalization=self.use_normalization, calculate_avg_dist=self.calculate_avg_dist)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )

if __name__ == "__main__":
    data_module = Scenes_DataModule(
        data_dir="./data/ForestSemantic_Difficult/tiles_filtered_fps",
        split_json_path="./data/ForestSemantic_Difficult/split.json",
        num_points=4096,
        batch_size=8,
        num_workers=0
    )
    
 
    data_module.setup()
 
    # 测试 data_loader
    train_loader = data_module.train_dataloader()
    batch = next(iter(train_loader))
    points, labels = batch
 
    print("Point cloud batch shape: ", points.shape)   # Should be [B, N, 3]
    print("Label batch shape: ", labels.shape)         # Should be [B, ]