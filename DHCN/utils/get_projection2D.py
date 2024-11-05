import os
import math
import numpy as np
import open3d as o3d
from PIL import Image
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from get_patch import pc_normalize  # 导入pc_normalize函数

# 设置离屏渲染
os.environ["OPEN3D_CPU_RENDERING"] = "true"

def generate_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return path


def cut_img(image):
    """
    Image.crop(left, up, right, below)
    left：Distance of the top left corner from the left boundary
    up：Distance of the top left corner from the upper boundary
    right：Distance of the bottom right corner from the left boundary
    below：Distance of the bottom right corner from the upper boundary
    """
    ImageArray = np.array(image)
    row = ImageArray.shape[0]
    col = ImageArray.shape[1]

    x_left = row
    x_top = col
    x_right = 0
    x_bottom = 0

    for r in range(row):
        for c in range(col):
            if ImageArray[r][c][0] < 255:
                if x_top > r:
                    x_top = r
                if x_bottom < r:
                    x_bottom = r
                if x_left > c:
                    x_left = c
                if x_right < c:
                    x_right = c

    if x_left == row and x_top == col and x_right == x_bottom == 0:
        cropped = image
    else:
        cropped = image.crop((x_left - 1, x_top - 1, x_right + 1, x_bottom + 1))  # (left, upper, right, lower)
    return cropped


# Camera Rotation
def camera_rotation(points, labels, out_path, file_name):
    """修改后的camera_rotation函数,处理npy格式数据"""
    # 创建点云对象并归一化
    points = pc_normalize(points)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # 根据标签设置颜色
    colors = np.zeros((len(points), 3))
    unique_labels = np.unique(labels)
    for i, label in enumerate(unique_labels):
        mask = labels == label
        colors[mask] = plt.cm.tab10(i)[:3]  # 使用matplotlib的颜色映射
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # 创建离屏渲染器
    render = o3d.visualization.rendering.OffscreenRenderer(640, 480)
    render.scene.add_geometry("cloud", pcd)
    
    # 设置相机参数
    bounds = pcd.get_axis_aligned_bounding_box()
    center = bounds.get_center()
    extent = bounds.get_extent()
    eye = center + [0, 0, np.linalg.norm(extent)]
    up = [0, 1, 0]
    
    tmp = 0
    interval = 5.82
    use_number = [1, 4, 7, 10, 13, 16, 19, 22, 25, 28]
    
    while tmp < 60:
        tmp += 1
        # 计算旋转矩阵
        if tmp < 30:
            angle_x = 12 * interval * tmp
            angle_y = 0
        elif 30 <= tmp < 60:
            angle_x = 0
            angle_y = 12 * interval * (tmp - 30)
        elif 60 <= tmp < 90:
            angle = 12 * interval * (tmp - 60)
            angle_x = angle / math.sqrt(2)
            angle_y = angle / math.sqrt(2)
        else:
            angle = 12 * interval * (tmp - 90)
            angle_x = angle / math.sqrt(2)
            angle_y = -angle / math.sqrt(2)
            
        if tmp in use_number:
            # 应用旋转
            rot_mat = o3d.geometry.get_rotation_matrix_from_xyz(
                [math.radians(angle_y), math.radians(angle_x), 0])
            current_eye = np.dot(rot_mat, eye - center) + center
            
            # 设置相机
            render.setup_camera(60.0, current_eye, center, up)
            
            # 渲染和保存
            save_path = os.path.join(out_path, f"{file_name}_{tmp}.png")
            if not os.path.exists(save_path):
                img = render.render_to_image()
                img = cut_img(Image.fromarray(np.asarray(img)))
                img.save(save_path)


def projection(path, out_path):
    """处理npy文件并生成多视角投影"""
    files = [f for f in os.listdir(path) if f.endswith('.npy')]
    print(f"找到 {len(files)} 个npy文件")
    
    for file in tqdm(files, desc="处理进度"):
        data = np.load(os.path.join(path, file))
        points = data[:, :3]
        labels = data[:, -1].astype(np.int32)
        file_name = file.split('.npy')[0]
        camera_rotation(points, labels, out_path, file_name)


def main(config):
    path = config.path
    out_path = config.out_path
    generate_dir(path)
    generate_dir(out_path)
    projection(path, out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='input')
    parser.add_argument('--out_path', type=str, default='output')
    config = parser.parse_args()
    
    main(config)
