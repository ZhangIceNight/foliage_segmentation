import numpy as np
from plyfile import PlyData
import os
from tqdm import tqdm

def process_ply_to_npy(ply_dir, save_dir):
    """
    处理PLY文件并保存为NPY格式
    格式: [x, y, z, label]
    按照每个点的data_split标志分类
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        os.makedirs(os.path.join(save_dir, 'train'))
        os.makedirs(os.path.join(save_dir, 'test'))
        
    # 遍历所有PLY文件
    for file in tqdm(os.listdir(ply_dir)):
        if not file.endswith('.ply'):
            continue
            
        ply_path = os.path.join(ply_dir, file)
        plydata = PlyData.read(ply_path)
        data = plydata['vertex'].data
        
        # 分离训练点和测试点
        train_mask = (data['data_split'] == 0) | (data['data_split'] == 1)  # 训练集和验证集
        test_mask = data['data_split'] == 2  # 测试集
        
        # 处理训练数据
        if np.any(train_mask):
            train_points = np.zeros((np.sum(train_mask), 4))
            train_points[:, 0] = data['x'][train_mask]
            train_points[:, 1] = data['y'][train_mask]
            train_points[:, 2] = data['z'][train_mask]
            train_points[:, 3] = data['gt_class'][train_mask]
            
            train_save_path = os.path.join(save_dir, 'train', file.replace('.ply', '_train.npy'))
            np.save(train_save_path, train_points)
            
        # 处理测试数据
        if np.any(test_mask):
            test_points = np.zeros((np.sum(test_mask), 4))
            test_points[:, 0] = data['x'][test_mask]
            test_points[:, 1] = data['y'][test_mask]
            test_points[:, 2] = data['z'][test_mask]
            test_points[:, 3] = data['gt_class'][test_mask]
            
            test_save_path = os.path.join(save_dir, 'test', file.replace('.ply', '_test.npy'))
            np.save(test_save_path, test_points)

if __name__ == '__main__':
    # 设置路径
    ply_dir = 'data'  # PLY文件目录
    save_dir = 'data'  # NPY保存目录
    
    # 处理文件
    process_ply_to_npy(ply_dir, save_dir)