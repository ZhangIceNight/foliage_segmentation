import os
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
import torch  

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
        with open(os.path.join(root, list_filename), 'r') as f:
            self.file_list = [line.strip() for line in f.readlines()]
        
        print(f"Loading {split} data...")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        # 按需加载数据
        data = np.load(os.path.join(self.root, self.file_list[index]))
        points = data[:, :3]  # XYZ坐标
        labels = data[:, -1].astype(np.int32)  # 标签
        
        # 归一化点云
        points = pc_normalize(points)
        
        # 采样处理
        if points.shape[0] > self.block_points:
            point_idxs = np.random.choice(points.shape[0], self.block_points, replace=False)
        else:
            point_idxs = np.random.choice(points.shape[0], self.block_points, replace=True)
        
        points = points[point_idxs]
        labels = labels[point_idxs]
        
        return torch.FloatTensor(points), torch.LongTensor(labels)

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
                                             batch_size=2, 
                                             shuffle=True,
                                             pin_memory=True)
    
    # 测试数据加载速度
    print("\n测试数据加载速度:")
    try:
        for idx in range(2):
            print(f"开始第 {idx+1} 轮测试...")
            end = time.time()
            
            for i, data in enumerate(train_loader):
                print(f"正在处理批次 {i+1}...")
                try:
                    points, labels = data
                    print('批次: {}/{} - 用时: {:.4f}s'.format(
                        i+1, len(train_loader), time.time() - end))
                    print(f'点云形状: {points.shape}')
                    print(f'标签形状: {labels.shape}')
                    if i == 2:  # 只测试前3个批次
                        break
                    end = time.time()
                except Exception as e:
                    print(f"处理批次时出错: {str(e)}")
                    raise e
                
    except Exception as e:
        print(f"发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
