import numpy as np
from plyfile import PlyData
import os
from tqdm import tqdm

def process_ply_to_npy(ply_dir, save_dir):
    """
    处理PLY文件并保存为NPY格式
    格式: [x, y, z, label]
    只保留标签4和5的点,并将其映射为0和1
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
        
        # 只保留标签为4和5的点
        target_labels = np.array([4, 5])
        label_mask = np.isin(data['gt_class'], target_labels)
        data_filtered = data[label_mask]
        
        # 检查标签
        unique_labels = np.unique(data_filtered['gt_class'])
        print(f"File {file} filtered labels: {unique_labels}")
        
        # 分离训练点和测试点
        train_data = data_filtered[data_filtered['data_split'] != 2]
        test_data = data_filtered[data_filtered['data_split'] == 2]
        
        # 处理训练数据
        if len(train_data) > 0:
            train_points = np.zeros((len(train_data), 4))
            train_points[:, 0] = train_data['x']
            train_points[:, 1] = train_data['y']
            train_points[:, 2] = train_data['z']
            # 将标签4映射为0,标签5映射为1
            train_points[:, 3] = (train_data['gt_class'] == 5).astype(int)
            
            train_save_path = os.path.join(save_dir, 'train', file.replace('.ply', '_train.npy'))
            np.save(train_save_path, train_points)
            print(f"Saved {len(train_points)} train points with labels: {np.unique(train_points[:, 3], return_counts=True)}")
            
        # 处理测试数据
        if len(test_data) > 0:
            test_points = np.zeros((len(test_data), 4))
            test_points[:, 0] = test_data['x']
            test_points[:, 1] = test_data['y']
            test_points[:, 2] = test_data['z']
            # 将标签4映射为0,标签5映射为1
            test_points[:, 3] = (test_data['gt_class'] == 5).astype(int)
            
            test_save_path = os.path.join(save_dir, 'test', file.replace('.ply', '_test.npy'))
            np.save(test_save_path, test_points)
            print(f"Saved {len(test_points)} test points with labels: {np.unique(test_points[:, 3], return_counts=True)}")

if __name__ == '__main__':
    # 设置路径
    ply_dir = 'data'  # PLY文件目录
    save_dir = 'data'  # NPY保存目录
    
    # 处理文件
    process_ply_to_npy(ply_dir, save_dir)