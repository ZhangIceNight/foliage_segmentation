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
        data = np.loadtxt(input_path, delimiter=' ')
        
        # 保存为npy格式
        np.save(output_path, data)

if __name__ == '__main__':
    input_dir = '/public/wjzhang/datasets/LabelledPC'
    output_dir = '/public/wjzhang/datasets/LabelledPCnpy'
    convert_txt_to_npy(input_dir, output_dir)
    