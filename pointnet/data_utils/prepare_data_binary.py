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
        data = np.loadtxt(input_path, delimiter=',')
        if data.shape[0] < 10000:
            print("文件：", file, "跳过")
            continue
        data = data[:, [0,1,2,4]] # 只保留x,y,z,label
        labels = data[:, 3] # 获取标签
        

        # 筛选出标签为2,3,4,5的点
        valid_mask = np.isin(labels, [2, 3, 4, 5])
        data = data[valid_mask]
        labels = labels[valid_mask]
       
        # 将标签5映射为0，其他标签(2,3,4)映射为1
        data[:, 3] = np.where(labels == 5, 0, 1)
        np.save(output_path, data)


if __name__ == '__main__':
    input_dir = '/public/wjzhang/datasets/Forest_Semantic/Plot_3'
    output_dir = '/public/wjzhang/datasets/Forest_Semantic/Plot_3_npy'
    convert_txt_to_npy(input_dir, output_dir)
    