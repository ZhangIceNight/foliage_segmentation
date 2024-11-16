import laspy

import numpy as np
import os

# 读取 LAS 文件
las_path = './datas/Plot_1.las'
las = laspy.read(las_path)

# # 列出所有点数据的属性
# print("LAS 文件包含以下属性：")
# for dimension in las.point_format.dimension_names:
    # print(dimension)

# print("分类信息：", las.classification)





# 获取所有点的 Point Source ID（树的编号）
point_source_ids = las.pt_src_id

# 获取唯一的树ID（Point Source ID）
unique_tree_ids = np.unique(point_source_ids)

# 创建输出目录
output_dir = "single_trees"
os.makedirs(output_dir, exist_ok=True)

# 创建TXT输出目录
txt_output_dir = "single_trees_txt"
os.makedirs(txt_output_dir, exist_ok=True)

# 遍历每棵树的ID，提取相应的点，并保存为新的 LAS 文件和 TXT 文件
for tree_id in unique_tree_ids:
    # 获取属于当前树的所有点的索引
    tree_indices = np.where(point_source_ids == tree_id)[0]
    
    # 提取当前树的点数据
    tree_x = las.x[tree_indices]
    tree_y = las.y[tree_indices]
    tree_z = las.z[tree_indices]
    tree_intensity = las.intensity[tree_indices]
    tree_classification = las.classification[tree_indices]
    tree_pt_src_id = las.pt_src_id[tree_indices]

    # 过滤掉类别为 1 和 6 的点
    valid_indices = np.where((tree_classification != 1) & (tree_classification != 6))[0]
    
    # 提取过滤后的点数据
    tree_x = tree_x[valid_indices]
    tree_y = tree_y[valid_indices]
    tree_z = tree_z[valid_indices]
    tree_intensity = tree_intensity[valid_indices]
    tree_classification = tree_classification[valid_indices]
    tree_pt_src_id = tree_pt_src_id[valid_indices]
    
    # 创建一个新的 LAS 文件对象来存储过滤后的点
    tree_las = laspy.create(point_format=las.header.point_format, file_version=las.header.version)
    
    # 复制原始 LAS 文件的头信息
    tree_las.header = las.header
    
    # 保存过滤后的点到新 LAS 文件对象
    tree_las.x = tree_x
    tree_las.y = tree_y
    tree_las.z = tree_z
    tree_las.intensity = tree_intensity
    tree_las.classification = tree_classification
    tree_las.pt_src_id = tree_pt_src_id
    
    # 如果有 GPS 时间，也可以提取
    if 'gps_time' in las.point_format.dimension_names:
        tree_las.gps_time = las.gps_time[tree_indices][valid_indices]
    
    # 构建输出文件路径
    output_file = os.path.join(output_dir, f"tree_{tree_id}.las")
    
    # 保存LAS文件
    tree_las.write(output_file)
    print(f"保存树 ID {tree_id} 的点云至 {output_file}")

    # 保存为TXT格式
    txt_output_file = os.path.join(txt_output_dir, f"tree_{tree_id}.txt")
    with open(txt_output_file, 'w') as f:
        for i in range(len(tree_x)):
            # 将点的坐标和其他信息写入TXT文件
            line = f"{tree_x[i]}, {tree_y[i]}, {tree_z[i]}, {tree_intensity[i]}, {tree_classification[i]}, {tree_pt_src_id[i]}"
            f.write(line + "\n")
    
    print(f"保存树 ID {tree_id} 的点云至 {txt_output_file}")

print("完成所有树的拆分和保存！")
