import os
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import provider

def augment_point_cloud(points, labels):
    """对单个点云进行增强"""
    augmented_data = []
    
    # 增强版本1：旋转+抖动
    points_aug1 = points[np.newaxis, ...]  # 添加batch维度
    points_aug1 = provider.rotate_point_cloud(points_aug1)[0]  # 移除batch维度
    points_aug1 = provider.jitter_point_cloud(points_aug1[np.newaxis, ...])[0]
    augmented_data.append(np.column_stack((points_aug1, labels)))
    
    # 增强版本2：随机平移+缩放
    points_aug2 = points[np.newaxis, ...]
    points_aug2 = provider.shift_point_cloud(points_aug2)[0]
    points_aug2 = provider.random_scale_point_cloud(points_aug2[np.newaxis, ...])[0]
    augmented_data.append(np.column_stack((points_aug2, labels)))
    
    # 增强版本3：随机旋转+随机dropout
    points_aug3 = points[np.newaxis, ...]
    points_aug3 = provider.rotate_point_cloud_z(points_aug3)[0]
    points_aug3 = provider.random_point_dropout(points_aug3[np.newaxis, ...])[0]
    augmented_data.append(np.column_stack((points_aug3, labels)))
    
    return augmented_data

def process_and_split_dataset(data_dir, output_dir, train_ratio=0.8, seed=42):
    """处理数据集：增强和划分"""
    random.seed(seed)
    Path(output_dir).mkdir(exist_ok=True)
    
    # 获取所有原始npy文件
    files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
    random.shuffle(files)
    
    # 划分训练集和测试集
    train_size = int(len(files) * train_ratio)
    train_files = files[:train_size]
    test_files = files[train_size:]
    
    print(f"原始数据集大小: {len(files)}")
    print(f"训练集: {len(train_files)} 文件")
    print(f"测试集: {len(test_files)} 文件")
    
    # 处理训练集（包括增强）
    train_files_with_aug = []
    for file in tqdm(train_files, desc="处理训练集"):
        # 添加原始文件
        train_files_with_aug.append(file)
        
        # 读取数据
        data = np.load(os.path.join(data_dir, file))
        points = data[:, :3]  # XYZ坐标
        labels = data[:, -1]  # 标签
        
        # 生成增强数据
        augmented_data = augment_point_cloud(points, labels)
        
        # 保存增强数据
        for i, aug_data in enumerate(augmented_data, 1):
            aug_filename = f"Plot_3_{file[:-4]}_aug_{i}.npy"
            np.save(os.path.join(output_dir, aug_filename), aug_data)
            train_files_with_aug.append(aug_filename)
            
        # 复制原始文件到新目录
        if data_dir != output_dir:
            np.save(os.path.join(output_dir, file), data)
    
    # 复制测试集文件（不增强）
    for file in tqdm(test_files, desc="复制测试集"):
        if data_dir != output_dir:
            data = np.load(os.path.join(data_dir, file))
            np.save(os.path.join(output_dir, file), data)
    
    # 保存文件列表
    with open(os.path.join(output_dir, 'trainval_list.txt'), 'w') as f:
        f.write('\n'.join(train_files_with_aug))
        
    with open(os.path.join(output_dir, 'test_list.txt'), 'w') as f:
        f.write('\n'.join(test_files))
        
    print("\n数据集处理完成:")
    print(f"- 训练集: {len(train_files_with_aug)} 文件 (包含 {len(train_files)} 个原始文件)")
    print(f"- 测试集: {len(test_files)} 文件 (全部为原始文件)")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/public/wjzhang/datasets/Forest_Semantic/Plot_3_npy',
                        help='包含原始npy文件的目录')
    parser.add_argument('--output_dir', type=str, default='/public/wjzhang/datasets/Forest_Semantic/Plot_3_npy_augmented',
                        help='保存增强后数据的目录')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='训练集比例')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    args = parser.parse_args()
    
    process_and_split_dataset(
        args.data_dir,
        args.output_dir,
        args.train_ratio,
        args.seed
    )