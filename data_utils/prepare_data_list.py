import os
import random
import argparse

def create_data_lists(data_dir, output_dir, train_ratio=0.8, seed=42):
    """创建训练集和测试集的文件列表"""
    random.seed(seed)
    
    # 获取所有npy文件
    files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
    random.shuffle(files)
    
    # 计算训练集大小
    train_size = int(len(files) * train_ratio)
    
    # 划分数据集
    train_files = files[:train_size]
    test_files = files[train_size:]
    
    # 保存文件列表
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'trainval_list.txt'), 'w') as f:
        f.write('\n'.join(train_files))
        
    with open(os.path.join(output_dir, 'test_list.txt'), 'w') as f:
        f.write('\n'.join(test_files))
        
    print(f"数据集划分完成:")
    print(f"- 训练集: {len(train_files)} 文件")
    print(f"- 测试集: {len(test_files)} 文件")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='$HOME/datasets/LabelledPCnpy')
    parser.add_argument('--output_dir', type=str, default='$HOME/datasets/LabelledPCnpy')
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    create_data_lists(args.data_dir, args.output_dir, args.train_ratio, args.seed)