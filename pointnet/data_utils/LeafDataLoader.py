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
        
        # 读取文件列表
        list_filename = 'trainval_list.txt' if split == 'train' else 'test_list.txt'
        with open(os.path.join(root, 'leaf_dataset', list_filename), 'r') as f:
            self.file_list = [line.strip().replace('.txt', '.npy') for line in f.readlines()]
            
        self.scene_points_list = []
        self.semantic_labels_list = []
        
        # 加载所有场景数据
        for file in self.file_list:
            # 使用npy格式加载数据
            data = np.load(os.path.join(root, 'npy_data', file))
            points = data[:, :3]  # XYZ
            labels = data[:, -1]  # 标签
            
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
        
        return block_points, block_labels, point_idxs[:self.block_points] 

if __name__ == '__main__':
    # 测试数据路径
    data_root = 'data/'
    num_point = 4096
    
    # 初始化数据集
    train_data = LeafDatasetWholeScene(root=data_root, split='train', block_points=num_point)
    print('训练数据大小:', train_data.__len__())
    if len(train_data) > 0:
        print('单个点云数据形状:', train_data.__getitem__(0)[0].shape)
        print('单个标签数据形状:', train_data.__getitem__(0)[1].shape)
    
    # 测试数据加载器
    import torch
    import time
    import random
    
    # 设置随机种子
    manual_seed = 123
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    
    def worker_init_fn(worker_id):
        random.seed(manual_seed + worker_id)
    
    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(train_data, 
                                             batch_size=16, 
                                             shuffle=True, 
                                             num_workers=4, 
                                             pin_memory=True, 
                                             worker_init_fn=worker_init_fn)
    
    # 测试数据加载速度
    print("\n测试数据加载速度:")
    for idx in range(2):
        end = time.time()
        for i, (points, labels, point_idxs) in enumerate(train_loader):
            print('批次: {}/{} - 用时: {:.4f}s'.format(
                i+1, len(train_loader), time.time() - end))
            print(f'点云形状: {points.shape}')
            print(f'标签形状: {labels.shape}')
            print(f'索引形状: {point_idxs.shape}')
            if i == 2:  # 只测试前3个批次
                break
            end = time.time() 