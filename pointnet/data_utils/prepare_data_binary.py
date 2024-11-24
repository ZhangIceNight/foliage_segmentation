import os
import numpy as np
from tqdm import tqdm

def convert_txt_to_npy(data_dir, output_dir):
    """将txt点云数据转换为npy格式"""
    os.makedirs(output_dir, exist_ok=True)
    
    files = [f for f in os.listdir(data_dir) if f.endswith('.txt')]
    print(f"找到 {len(files)} 个txt文件")
    
    for file in tqdm(files, desc="转换进度"):
        input_path = os.path.join(data_dir, file)
        output_path = os.path.join(output_dir, file.replace('.txt', '.npy'))
        
        # 读取txt数据
        data = np.loadtxt(input_path, delimiter=None, skiprows=1)
        data = data[:, [0,1,2,4]]
        labels = data[:, 3]
        if not np.all(np.isin(labels, [1, 2, 3])):
            raise ValueError(f"发现非法标签值，标签值必须是1、2或3")
            
        data[:, 3] = np.where(labels == 3, 0, 1)        # 保存为npy格式
        np.save(output_path, data)


if __name__ == '__main__':
    input_dir = '/public/wjzhang/datasets/wood_seg_samples/wood_seg_samples'
    output_dir = '/public/wjzhang/datasets/wood_seg_samples/wood_seg_samples_npy'
    convert_txt_to_npy(input_dir, output_dir)
    