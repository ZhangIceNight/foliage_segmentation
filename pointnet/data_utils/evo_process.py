import numpy as np
from plyfile import PlyData
import os
from tqdm import tqdm

def process_ply_to_npy(ply_dir, save_dir):
    """
    处理PLY文件并保存为NPY格式
    格式: [x, y, z, label]
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
        
        # 提取xyz坐标和标签
        points = np.zeros((len(data), 4))  # [x, y, z, label]
        points[:, 0] = data['x']  # x坐标
        points[:, 1] = data['y']  # y坐标
        points[:, 2] = data['z']  # z坐标
        points[:, 3] = data['gt_class']  # 标签
        
        # 根据data_split决定保存路径
        # data_split: 0=train, 1=val, 2=test
        if data['data_split'][0] == 2:  # 测试集
            save_path = os.path.join(save_dir, 'test', file.replace('.ply', '.npy'))
        elif data['data_split'][0] == 0 or data['data_split'][0] == 1:  # 训练集和验证集
            save_path = os.path.join(save_dir, 'train', file.replace('.ply', '.npy'))
            
        # 保存为npy文件
        np.save(save_path, points)

if __name__ == '__main__':
    # 设置路径
    ply_dir = 'data/evonpy'  # PLY文件目录
    save_dir = 'data/evonpy'  # NPY保存目录
    
    # 处理文件
    process_ply_to_npy(ply_dir, save_dir)