import numpy as np

def load_point_cloud(file_path):
    """
    从txt文件中加载点云数据。文件每行包含x, y, z坐标和一个标签。
    
    参数:
        file_path (str): txt文件路径
    
    返回:
        np.ndarray: 点云坐标，形状为 (n_points, 3)
    """
    # 读取文件中的数据
    data = np.loadtxt(file_path)
    
    # 提取坐标数据，忽略标签列
    points = data[:, :3]  # 前三列为x, y, z坐标
    
    return points

